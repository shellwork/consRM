import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import yaml
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from transformers import AutoTokenizer, AutoModel, BertModel, BertConfig
import matplotlib.pyplot as plt
from tqdm import tqdm
from imblearn.over_sampling import SMOTE
from torch.utils.data.sampler import WeightedRandomSampler

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
        bert_dim = 768  # BERT output dimension
        feature_dim = config['model']['feature_dim']
        hidden_dim = config['model']['hidden_dim']
        dropout_rate = config['model']['dropout_rate']
        
        # Enhanced feature encoder with batch normalization
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate/2)
        )
        
        # Batch normalization for BERT features
        self.bert_norm = nn.BatchNorm1d(bert_dim)
        
        # Deeper classifier with residual connections
        self.classifier_1 = nn.Sequential(
            nn.Linear(bert_dim + 64, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Residual block
        self.residual_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate/2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # Output layer
        self.classifier_2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, bert_features, additional_features):
        # Normalize BERT features
        bert_features = self.bert_norm(bert_features)
        
        # Process additional features
        encoded_features = self.feature_encoder(additional_features)
        
        # Concatenate features
        fused_features = torch.cat((bert_features, encoded_features), dim=1)
        
        # First classifier layer
        x = self.classifier_1(fused_features)
        
        # Residual connection
        residual = x
        x = self.residual_block(x)
        x = x + residual  # Add residual connection
        
        # Output layer
        output = self.classifier_2(x)
        
        return output.squeeze()

# 计算样本权重以处理不平衡数据
def calculate_class_weights(labels):
    # 计算类别频率
    class_counts = np.bincount(labels.astype(int))
    # 计算权重（少数类获得更高权重）
    weights = 1.0 / class_counts
    # 归一化权重，使其总和为len(labels)
    weights = weights * len(labels) / np.sum(weights * class_counts)
    
    # 为每个数据点分配权重
    sample_weights = np.zeros(len(labels))
    for i, label in enumerate(labels):
        sample_weights[i] = weights[int(label)]
    
    return sample_weights, weights

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
    # print(f"Sample sequences: {sequences[:3]}")
    
    # 打印类别分布情况
    unique, counts = np.unique(labels, return_counts=True)
    print("类别分布:")
    for u, c in zip(unique, counts):
        print(f"类别 {u}: {c} 样本 ({c/len(labels)*100:.2f}%)")
    
    return sequences, features, labels

# 应用SMOTE过采样技术对少数类进行采样
def apply_smote(bert_features, additional_features, labels, sampling_strategy=0.5):
    combined_features = np.hstack((bert_features, additional_features))
    
    # Use a more conservative sampling strategy with k_neighbors tuned to your dataset size
    smote = SMOTE(sampling_strategy=0.7, k_neighbors=min(5, sum(labels == 1) - 1), random_state=42)
    combined_features_resampled, labels_resampled = smote.fit_resample(combined_features, labels)
    
    # Split back into separate feature sets
    bert_features_resampled = combined_features_resampled[:, :bert_features.shape[1]]
    additional_features_resampled = combined_features_resampled[:, bert_features.shape[1]:]
    
    return bert_features_resampled, additional_features_resampled, labels_resampled

# 训练模型
def train_model(model, train_loader, val_loader, criterion, optimizer, config, device, class_weights=None):
    patience = config['training']['patience']
    num_epochs = config['training']['num_epochs']
    
    best_val_loss = float('inf')
    best_f1 = 0.0
    patience_counter = 0
    training_history = {
        'train_loss': [], 'val_loss': [], 
        'val_accuracy': [], 'val_precision': [], 
        'val_recall': [], 'val_f1': [], 'val_auc': []
    }
    
    # 设置类别权重给BCE损失函数
    if class_weights is not None and isinstance(criterion, nn.BCELoss):
        criterion = nn.BCELoss(weight=torch.tensor([class_weights[1]]).to(device))
        print(f"设置BCELoss的权重为: {class_weights[1]}")
    
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
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        val_loss = val_metrics['loss']
        val_accuracy = val_metrics['accuracy']
        val_precision = val_metrics['precision']
        val_recall = val_metrics['recall']
        val_f1 = val_metrics['f1']
        val_auc = val_metrics['auc']
        
        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_accuracy:.4f}, Val Prec: {val_precision:.4f}, "
              f"Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}")
        
        # 记录训练历史
        training_history['train_loss'].append(avg_train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['val_accuracy'].append(val_accuracy)
        training_history['val_precision'].append(val_precision)
        training_history['val_recall'].append(val_recall)
        training_history['val_f1'].append(val_f1)
        training_history['val_auc'].append(val_auc)
        
        # 早停策略
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_dna_classifier.pth')
            print(f"保存模型! 最佳验证F1: {best_f1:.4f}")
#         else:
#             patience_counter += 1
#             print(f"早停计数器: {patience_counter}/{patience}")
            
#             if patience_counter >= patience:
#                 print("早停触发!")
#                 break
    
    return training_history

# 模型评估函数
def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_probs = []  # 保存原始概率值用于计算AUC
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
            probs = outputs.cpu().numpy()
            predictions = (probs > 0.5).astype(float)
            all_probs.extend(probs)
            all_preds.extend(predictions)
            all_labels.extend(labels.cpu().numpy())
    
    # 将列表转换为数组以便计算
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # 计算评估指标
    accuracy = accuracy_score(all_labels, all_preds)
    
    # 处理特殊情况
    if len(np.unique(all_labels)) == 1:  # 只有一个类别
        precision = 0 if np.sum(all_preds) == 0 else 1.0
        recall = 0 if all_labels[0] == 1 else 1.0
        f1 = 0
        auc = 0.5
    else:
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs)
    
    # 显示混淆矩阵
    conf_matrix = confusion_matrix(all_labels, all_preds)
    print("\n混淆矩阵:")
    print(conf_matrix)
    print(f"TP: {conf_matrix[1][1]}, FP: {conf_matrix[0][1]}")
    print(f"FN: {conf_matrix[1][0]}, TN: {conf_matrix[0][0]}")
    
    return {
        'loss': total_loss / len(data_loader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

# 可视化训练历史
def plot_training_history(history):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 损失曲线
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Validation Loss')
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    
    # 准确率曲线
    axes[0, 1].plot(history['val_accuracy'], label='Accuracy')
    axes[0, 1].plot(history['val_auc'], label='AUC')
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Validation Accuracy and AUC')
    axes[0, 1].legend()
    
    # 精确度和召回率曲线
    axes[1, 0].plot(history['val_precision'], label='Precision')
    axes[1, 0].plot(history['val_recall'], label='Recall')
    axes[1, 0].set_xlabel('Epochs')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Validation Precision and Recall')
    axes[1, 0].legend()
    
    # F1分数曲线
    axes[1, 1].plot(history['val_f1'], label='F1 Score')
    axes[1, 1].set_xlabel('Epochs')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Validation F1 Score')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

# 使用加权损失函数处理不平衡数据
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean', pos_weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight = pos_weight  # Weight for positive class
    
    def forward(self, inputs, targets):
        # Apply weights to BCE loss
        if self.pos_weight is not None:
            weights = torch.ones_like(targets)
            weights[targets == 1] = self.pos_weight
            BCE_loss = nn.BCELoss(reduction='none', weight=weights)(inputs, targets)
        else:
            BCE_loss = nn.BCELoss(reduction='none')(inputs, targets)
            
        pt = torch.exp(-BCE_loss)  # Prediction confidence
        
        # Apply focal weighting
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

# 为WeightedRandomSampler创建样本权重
def create_weighted_sampler(labels):
    class_counts = np.bincount(labels.astype(int))
    class_weights = 1. / class_counts
    sample_weights = class_weights[labels.astype(int)]
    sample_weights = torch.from_numpy(sample_weights).float()
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    return sampler

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
    
    # 划分训练集和验证集 - 分层采样保持类别分布
    train_seqs, val_seqs, train_features, val_features, train_labels, val_labels = train_test_split(
        sequences, additional_features, labels, 
        test_size=config['data']['test_size'], 
        random_state=config['data']['random_seed'],
        stratify=labels  # 确保训练集和验证集有相似的类别分布
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
    
    # 处理类别不平衡 - 可选择使用SMOTE或类别权重
    # 1. 计算类别权重
    _, class_weights = calculate_class_weights(train_labels)
    print(f"类别权重: {class_weights}")
    
    # 2. 应用SMOTE过采样 - 可以根据需要调整比例
    # 取消注释以下行使用SMOTE
    # train_bert_features, train_features, train_labels = apply_smote(
    #     train_bert_features, train_features, train_labels, 
    #     sampling_strategy=0.5  # 设置少数类与多数类的比例
    # )
    
    # 创建优化的数据集
    train_dataset = OptimizedDNADataset(train_bert_features, train_features, train_labels)
    val_dataset = OptimizedDNADataset(val_bert_features, val_features, val_labels)
    
    # 创建带加权采样器的数据加载器
    batch_size = config['training']['batch_size']
    
    # 使用WeightedRandomSampler处理不平衡
    # train_sampler = create_weighted_sampler(train_labels)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    
    # 如果使用了SMOTE，就不需要加权采样器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 创建分类器模型
    model = DNAClassifier(config).to(device)
    
    # 打印模型信息
    print("Classification model architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {total_params:,}")
    
    # 定义损失函数 - 使用Focal Loss或加权BCE来处理不平衡数据
    # criterion = nn.BCELoss()  # 标准交叉熵损失
    # criterion = nn.BCELoss(weight=torch.tensor([class_weights[1]]).to(device))  # 加权BCE
    criterion = FocalLoss(alpha=0.75, gamma=2.0)  # Focal Loss
    
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=float(config['training']['learning_rate']))
    
    # 训练模型
    print("Training classification model...")
    history = train_model(model, train_loader, val_loader, criterion, optimizer, config, device, class_weights)
    
    # 可视化训练过程
    plot_training_history(history)
    
    # 加载最佳模型进行评估
    best_model = DNAClassifier(config).to(device)
    best_model.load_state_dict(torch.load('best_dna_classifier.pth'))
    
    # 对验证集进行全面评估
    val_metrics = evaluate_model(best_model, val_loader, criterion, device)
    print(f"\n最佳模型评估结果:")
    print(f"验证损失: {val_metrics['loss']:.4f}")
    print(f"准确率: {val_metrics['accuracy']:.4f}")
    print(f"精确率: {val_metrics['precision']:.4f}")
    print(f"召回率: {val_metrics['recall']:.4f}")
    print(f"F1分数: {val_metrics['f1']:.4f}")
    print(f"AUC: {val_metrics['auc']:.4f}")

if __name__ == "__main__":
    main()