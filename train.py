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
from transformers import AutoTokenizer, AutoModel
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

# 自定义数据集类
class DNASequenceDataset(Dataset):
    def __init__(self, sequences, features, labels, tokenizer, max_length):
        self.sequences = sequences
        self.features = features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        features = self.features[idx]
        label = self.labels[idx]
        
        # 使用DNABERT2的tokenizer处理序列
        encoding = self.tokenizer(
            seq,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )
        
        # 去掉batch维度
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'features': torch.tensor(features, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.float32)
        }

# 定义融合模型
class DNASequenceClassifier(nn.Module):
    def __init__(self, bert_model, config):
        super(DNASequenceClassifier, self).__init__()
        self.bert_model = bert_model
        
        # BERT输出维度通常是768
        bert_output_dim = 768
        feature_dim = config['model']['feature_dim']
        hidden_dim = config['model']['hidden_dim']
        dropout_rate = config['model']['dropout_rate']
        feature_encoder_dims = config['model']['feature_encoder']['hidden_dims']
        
        # 对额外特征的MLP编码层
        feature_encoder_layers = []
        input_dim = feature_dim
        for hidden_dim in feature_encoder_dims:
            feature_encoder_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim
        
        # 移除最后一层的Dropout
        if feature_encoder_layers:
            feature_encoder_layers = feature_encoder_layers[:-1]
            
        self.feature_encoder = nn.Sequential(*feature_encoder_layers)
        
        # 获取特征编码器的输出维度
        feature_output_dim = feature_encoder_dims[-1] if feature_encoder_dims else feature_dim
        
        # 融合后特征的分类器
        self.classifier = nn.Sequential(
            nn.Linear(bert_output_dim + feature_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, input_ids, attention_mask, features):
        # 获取BERT的输出
        bert_output = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        # 获取[CLS]标记的嵌入，代表整个序列的表示
        sequence_output = bert_output.last_hidden_state[:, 0, :]
        
        # 对额外特征进行编码
        feature_encoded = self.feature_encoder(features)
        
        # 特征融合
        fused_features = torch.cat((sequence_output, feature_encoded), dim=1)
        
        # 分类预测
        output = self.classifier(fused_features)
        
        return output.squeeze()

# 数据加载
def load_data(file_path):
    
    df = pd.read_csv(file_path)
    sequences = df['sequence'].values
    features = df[['noes_score', 'pm_score', 'seq_score', 'phastCons']].values
    labels = df['label'].values
    
    return sequences, features, labels

# 模型训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, config, device):
    patience = config['training']['patience']
    num_epochs = config['training']['num_epochs']
    
    best_val_loss = float('inf')
    patience_counter = 0
    training_history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'val_auc': []}
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            features = batch['features'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播
            outputs = model(input_ids, attention_mask, features)
            
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
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            features = batch['features'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播
            outputs = model(input_ids, attention_mask, features)
            
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
    trust_remote_code = config.get('loading', {}).get('trust_remote_code', True)
    local_files_only = config.get('loading', {}).get('local_files_only', True)
    
    # 尝试使用标准方式加载
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only
    )
    
    from transformers import BertModel
    bert_model = BertModel.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only
    )

    # 确保模型处于评估模式开始
    bert_model.eval()
    
    # 加载数据
    print("Loading data...")
    data_file = config['data']['file_path']
    sequences, features, labels = load_data(data_file)
    
    # 划分训练集和验证集
    train_seqs, val_seqs, train_features, val_features, train_labels, val_labels = train_test_split(
        sequences, features, labels, 
        test_size=config['data']['test_size'], 
        random_state=config['data']['random_seed'],
        stratify=labels
    )
    
    print(f"Train set size: {len(train_seqs)}, Validation set size: {len(val_seqs)}")
    
    # 创建数据集
    max_seq_length = config['training']['max_seq_length']
    train_dataset = DNASequenceDataset(train_seqs, train_features, train_labels, tokenizer, max_seq_length)
    val_dataset = DNASequenceDataset(val_seqs, val_features, val_labels, tokenizer, max_seq_length)
    
    # 创建数据加载器
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 创建模型
    model = DNASequenceClassifier(bert_model, config).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=float(config['training']['learning_rate']))
    
    # 训练模型
    print("Training model...")
    history = train_model(model, train_loader, val_loader, criterion, optimizer, config, device)
    
    # 可视化训练过程
    plot_training_history(history)
    
    # 加载最佳模型进行评估
    best_model = DNASequenceClassifier(bert_model, config).to(device)
    best_model.load_state_dict(torch.load('best_dna_classifier.pth'))
    
    val_loss, val_accuracy, val_auc = evaluate_model(best_model, val_loader, criterion, device)
    print(f"Best Model Evaluation - Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, AUC: {val_auc:.4f}")

if __name__ == "__main__":
    main()
