import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import yaml
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from transformers import AutoTokenizer, AutoModel, BertModel, BertConfig
import matplotlib.pyplot as plt
from tqdm import tqdm

# 加载配置文件
def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# 设置随机种子以确保结果可复现
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 确定设备
def get_device(device_config):
    if device_config == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_config)

# original version
# class BertFeatureExtractor:
#     def __init__(self, model_path, tokenizer, max_length, device):
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         self.device = device
#         config = load_config()
#         # 加载预训练的BERT模型 - 使用BertModel而不是AutoModel避免配置冲突
#         trust_remote_code = config['loading']['trust_remote_code']
#         local_files_only = config['loading']['local_files_only']
#         self.model = BertModel.from_pretrained(
#             model_path,
#             trust_remote_code=trust_remote_code,
#             local_files_only=local_files_only
#         ).to(device)
#         print("Successfully loaded BERT model")
        
#         # 设置模型为评估模式，禁用dropout等
#         self.model.eval()
    
#     def extract_features(self, sequences, batch_size=32):
#         """从序列中提取BERT特征"""
#         features = []
        
#         # 确保所有序列都是字符串类型
#         sequences = [str(seq) for seq in sequences]
        
#         # 检查数据类型
#         print(f"Sequences type: {type(sequences)}")
#         print(f"First sequence type: {type(sequences[0])}")
#         # print(f"First few sequences: {sequences[:3]}")
        
#         # 分批处理序列以避免OOM
#         for i in tqdm(range(0, len(sequences), batch_size), desc="提取BERT特征"):
#             batch_sequences = sequences[i:i+batch_size]
            
#             try:
#                 # 对批次进行编码
#                 encoding = self.tokenizer(
#                     batch_sequences,
#                     padding='max_length',
#                     truncation=True,
#                     max_length=self.max_length,
#                     return_tensors='pt'
#                 ).to(self.device)
                
#                 # 不计算梯度
#                 with torch.no_grad():
#                     outputs = self.model(**encoding)
#                     # 获取[CLS]标记的嵌入，代表整个序列的表示
#                     batch_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
#                     features.append(batch_features)
#             except Exception as e:
#                 print(f"Error processing batch {i}-{i+batch_size}: {e}")
#                 print(f"Problematic batch: {batch_sequences}")
#                 raise
        
#         # 合并所有批次的特征
#         return np.vstack(features)

# 特征提取类 - 用于预先提取BERT特征
class BertFeatureExtractor:
    def __init__(self, model_path, tokenizer, max_length, device):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        
        config = load_config()
        
        # 从配置文件加载推理优化相关参数
        inference_config = config.get('inference', {})
        self.batch_size = inference_config.get('batch_size', 128)  # 默认使用更大的批次大小
        self.num_workers = inference_config.get('num_workers', 4)  # 数据加载的并行工作进程数
        self.use_amp = inference_config.get('use_amp', True)  # 是否使用混合精度推理
        self.use_gradient_checkpointing = inference_config.get('use_gradient_checkpointing', False)  # 是否使用梯度检查点
        
        # 加载预训练的BERT模型
        trust_remote_code = config['loading']['trust_remote_code']
        local_files_only = config['loading']['local_files_only']
        
        # 加载模型时配置优化参数
        self.model = BertModel.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only
        ).to(device)
        
        # 如果需要梯度检查点（在推理时通常不需要，但在某些情况下可用于控制内存使用）
        if self.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        print(f"Successfully loaded BERT model with batch_size={self.batch_size}, num_workers={self.num_workers}, use_amp={self.use_amp}")
        
        # 设置模型为评估模式，禁用dropout等
        self.model.eval()
        
        # 创建AMP的scaler，用于混合精度计算
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
    
    def _create_dataloader(self, sequences):
        """创建用于批处理的DataLoader"""
        # 确保所有序列都是字符串类型
        sequences = [str(seq) for seq in sequences]
        
        # 创建简单的Dataset
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, texts):
                self.texts = texts
                
            def __len__(self):
                return len(self.texts)
                
            def __getitem__(self, idx):
                return self.texts[idx]
        
        dataset = SimpleDataset(sequences)
        
        # 创建DataLoader进行并行加载和批处理
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,  # 使用固定内存加速CPU到GPU的数据传输
            shuffle=False  # 保持顺序不变
        )
    
    def extract_features(self, sequences, batch_size=None):
        """从序列中提取BERT特征"""
        if batch_size is not None:
            self.batch_size = batch_size  # 允许在运行时覆盖配置的批次大小
            
        features = []
        
        # 创建DataLoader以实现并行数据处理
        dataloader = self._create_dataloader(sequences)
        
        # 检查数据类型
        print(f"Sequences type: {type(sequences)}")
        if len(sequences) > 0:  # 改用 len() 检查是否为空，而不是直接使用 if sequences
            print(f"First sequence type: {type(sequences[0])}")
        
        # 处理每个批次
        for batch_sequences in tqdm(dataloader, desc="提取BERT特征"):
            try:
                # 对批次进行编码
                encoding = self.tokenizer(
                    batch_sequences,
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                ).to(self.device)
                
                # 不计算梯度
                with torch.no_grad():
                    # 使用自动混合精度(AMP)进行推理
                    with torch.cuda.amp.autocast(enabled=self.use_amp):
                        outputs = self.model(**encoding)
                    
                    # 获取[CLS]标记的嵌入，代表整个序列的表示
                    batch_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    features.append(batch_features)
                    
            except Exception as e:
                print(f"Error processing batch: {e}")
                print(f"Problematic batch samples: {batch_sequences[:2]}")  # 只打印前两个样本以减少输出
                raise
        
        # 合并所有批次的特征
        return np.vstack(features)

# 优化的数据集类，使用预先提取的特征
class OptimizedDNADataset(Dataset):
    def __init__(self, bert_features, additional_features, labels):
        """
        初始化数据集
        bert_features: 从BERT提取的特征 [n_samples, bert_dim]
        additional_features: 额外特征 [n_samples, feature_dim]
        labels: 标签 [n_samples]
        """
        self.bert_features = torch.tensor(bert_features, dtype=torch.float32)
        self.additional_features = torch.tensor(additional_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'bert_features': self.bert_features[idx],
            'additional_features': self.additional_features[idx],
            'label': self.labels[idx]
        }

# 简化的分类器模型 - 只训练分类层
class DNAClassifier(nn.Module):
    def __init__(self, config):
        super(DNAClassifier, self).__init__()
        bert_dim = 768  # BERT输出维度通常是768
        feature_dim = config['model']['feature_dim']
        hidden_dim = config['model']['hidden_dim']
        dropout_rate = config['model']['dropout_rate']
        
        # 特征编码器 - 处理额外特征
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU()
        )
        
        # 分类器 - 结合BERT特征和额外特征
        self.classifier = nn.Sequential(
            nn.Linear(bert_dim + 32, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, bert_features, additional_features):
        # 对额外特征进行编码
        encoded_features = self.feature_encoder(additional_features)
        
        # 融合BERT特征和额外特征
        fused_features = torch.cat((bert_features, encoded_features), dim=1)
        
        # 分类预测
        output = self.classifier(fused_features)
        
        return output.squeeze()

# 数据加载
def load_data(file_path):
    df = pd.read_csv(file_path)
    sequences = df['sequence'].values
    # features = df[['noes_score', 'pm_score', 'seq_score', 'phastCons']].values
    features = df[['seq_score', 'phastCons']].values
    labels = df['label'].values
    
    # 检查数据
    print(f"Data shapes - Sequences: {sequences.shape}, Features: {features.shape}, Labels: {labels.shape}")
    print(f"Sequence data type: {type(sequences)}, Element type: {type(sequences[0]) if len(sequences) > 0 else 'N/A'}")
    print(f"Sample sequences: {sequences[:3]}")
    
    return sequences, features, labels

# 训练模型
def train_model(model, train_loader, val_loader, criterion, optimizer, config, device):
    patience = config['training']['patience']
    num_epochs = config['training']['num_epochs']
    
    best_val_loss = float('inf')
    patience_counter = 0
    training_history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'val_auc': []}
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # 训练阶段
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]"):
            bert_features = batch['bert_features'].to(device)
            additional_features = batch['additional_features'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播
            outputs = model(bert_features, additional_features)
            
            # 计算损失
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        
        # 验证阶段
        val_loss, val_accuracy, val_auc = evaluate_model(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val AUC: {val_auc:.4f}")
        
        # 记录训练历史
        training_history['train_loss'].append(avg_train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['val_accuracy'].append(val_accuracy)
        training_history['val_auc'].append(val_auc)
        
        # 早停策略
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_dna_classifier.pth')
            print("Model saved!")
        else:
            patience_counter += 1
            print(f"Early stopping counter: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break
    
    return training_history

# 模型评估函数
def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            bert_features = batch['bert_features'].to(device)
            additional_features = batch['additional_features'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播
            outputs = model(bert_features, additional_features)
            
            # 计算损失
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # 收集预测结果和标签
            predictions = (outputs > 0.5).float().cpu().numpy()
            all_preds.extend(predictions)
            all_labels.extend(labels.cpu().numpy())
    
    # 计算评估指标
    accuracy = accuracy_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except:
        auc = 0.5  # 如果只有一个类别，AUC计算会失败
    
    return total_loss / len(data_loader), accuracy, auc

# 可视化训练历史
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失曲线
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    # 准确率和AUC曲线
    ax2.plot(history['val_accuracy'], label='Validation Accuracy')
    ax2.plot(history['val_auc'], label='Validation AUC')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Score')
    ax2.set_title('Validation Metrics')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

# 主函数
def main():
    # 加载配置
    config = load_config()
    
    # 设置随机种子
    set_seed(config['data']['random_seed'])
    
    # 设置设备
    device = get_device(config['device'])
    print(f"Using device: {device}")
    
    # 加载DNABERT2模型和tokenizer
    model_path = config['model']['bert_model_path']
    
    # 从配置文件获取模型加载参数
    trust_remote_code = config['loading']['trust_remote_code']
    local_files_only = config['loading']['local_files_only']
    
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only
        )
        print("Tokenizer loaded successfully")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return
    
    # 加载数据
    print("Loading data...")
    data_file = config['data']['file_path']
    sequences, additional_features, labels = load_data(data_file)
    
    # 划分训练集和验证集
    train_seqs, val_seqs, train_features, val_features, train_labels, val_labels = train_test_split(
        sequences, additional_features, labels, 
        test_size=config['data']['test_size'], 
        random_state=config['data']['random_seed'],
        stratify=labels
    )
    
    print(f"Train set size: {len(train_seqs)}, Validation set size: {len(val_seqs)}")
    
    # 提取BERT特征
    print("Extracting BERT features...")
    max_seq_length = config['training']['max_seq_length']
    feature_extractor = BertFeatureExtractor(model_path, tokenizer, max_seq_length, device)
    
    # 提取训练集和验证集的BERT特征
    print("Extracting BERT features for training set...")
    train_bert_features = feature_extractor.extract_features(train_seqs)
    
    print("Extracting BERT features for validation set...")
    val_bert_features = feature_extractor.extract_features(val_seqs)
    
    print(f"BERT feature dimension: {train_bert_features.shape[1]}")
    
    # 创建优化的数据集
    train_dataset = OptimizedDNADataset(train_bert_features, train_features, train_labels)
    val_dataset = OptimizedDNADataset(val_bert_features, val_features, val_labels)
    
    # 创建数据加载器
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 创建分类器模型 (不包含BERT部分)
    model = DNAClassifier(config).to(device)
    
    # 打印模型信息
    print("Classification model architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {total_params:,}")
    
    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=float(config['training']['learning_rate']))
    
    # 训练模型
    print("Training classification model...")
    history = train_model(model, train_loader, val_loader, criterion, optimizer, config, device)
    
    # 可视化训练过程
    plot_training_history(history)
    
    # 加载最佳模型进行评估
    best_model = DNAClassifier(config).to(device)
    best_model.load_state_dict(torch.load('best_dna_classifier.pth'))
    
    val_loss, val_accuracy, val_auc = evaluate_model(best_model, val_loader, criterion, device)
    print(f"Best Model Evaluation - Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, AUC: {val_auc:.4f}")

if __name__ == "__main__":
    main()