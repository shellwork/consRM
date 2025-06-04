import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import yaml
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler  # 改进：使用RobustScaler替代StandardScaler
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
import torch.nn.functional as F
class DNAClassifier(nn.Module):
    def __init__(self, config, imbalance_ratio=1.0):
        super(DNAClassifier, self).__init__()
        bert_dim = 768
        feature_dim = config['model']['feature_dim']
        hidden_dim = config['model']['hidden_dim']
        dropout_rate = config['model']['dropout_rate']
        
        # A. Sequence Features Module (对应图中上半部分)
        self.bert_norm = nn.LayerNorm(bert_dim)
        
        # 修复：使用适合单一向量的处理方式，而不是CNN
        # 因为BERT特征是 [batch_size, 768] 的单一向量，不是序列
        self.sequence_feature_processor = nn.Sequential(
            nn.Linear(bert_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.3)
        )
        
        # B. Physicochemical Features Module (对应图中下半部分)
        # 特征工程预处理
        self.feature_preprocessor = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.3)
        )
        
        # C. 特征融合和分类
        # 计算拼接后的特征维度
        total_features = 128 + 32  # sequence features + physicochemical features
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(total_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5)
        )
        
        # 最终分类器
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(32, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, bert_features, additional_features):
        batch_size = bert_features.size(0)
        
        # 确保输入维度正确
        if len(bert_features.shape) != 2:
            raise ValueError(f"Expected bert_features to be 2D [batch_size, 768], got shape {bert_features.shape}")
        if len(additional_features.shape) != 2:
            raise ValueError(f"Expected additional_features to be 2D [batch_size, feature_dim], got shape {additional_features.shape}")
        
        # A. 序列特征处理 - 直接处理BERT向量
        bert_features = self.bert_norm(bert_features)
        sequence_features = self.sequence_feature_processor(bert_features)
        
        # B. 物理化学特征处理
        physicochemical_features = self.feature_preprocessor(additional_features)
        
        # C. 特征拼接和融合
        fused_features = torch.cat([sequence_features, physicochemical_features], dim=1)
        fused_features = self.feature_fusion(fused_features)
        
        # 最终分类
        logits = self.classifier(fused_features)
        return torch.sigmoid(logits).squeeze()

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

# 改进：数据预处理函数
def improved_data_preprocessing(sequences, features, labels, config):
    """改进的数据预处理流程"""
    
    # 1. 特征标准化 - 使用RobustScaler对异常值更鲁棒
    print("Applying feature scaling...")
    scaler = RobustScaler()  # 对异常值更鲁棒
    features_scaled = scaler.fit_transform(features)
    
    print(f"原始特征范围: {features.min(axis=0)} ~ {features.max(axis=0)}")
    print(f"缩放后特征范围: {features_scaled.min(axis=0)} ~ {features_scaled.max(axis=0)}")
    
    # 2. 分层采样确保类别分布
    train_seqs, val_seqs, train_features, val_features, train_labels, val_labels = train_test_split(
        sequences, features_scaled, labels, 
        test_size=config['data']['test_size'], 
        random_state=config['data']['random_seed'],
        stratify=labels
    )
    
    # 3. 打印详细的类别分布信息
    print("\n=== 类别分布分析 ===")
    unique_train, counts_train = np.unique(train_labels, return_counts=True)
    unique_val, counts_val = np.unique(val_labels, return_counts=True)
    
    print("训练集类别分布:")
    for u, c in zip(unique_train, counts_train):
        print(f"  类别 {u}: {c} 样本 ({c/len(train_labels)*100:.2f}%)")
    
    print("验证集类别分布:")
    for u, c in zip(unique_val, counts_val):
        print(f"  类别 {u}: {c} 样本 ({c/len(val_labels)*100:.2f}%)")
    
    # 计算不平衡比率
    pos_ratio = np.sum(train_labels) / len(train_labels)
    imbalance_ratio = (1 - pos_ratio) / pos_ratio if pos_ratio > 0 else float('inf')
    print(f"正样本比例: {pos_ratio:.4f}, 不平衡比率: {imbalance_ratio:.2f}:1")
    
    return (train_seqs, val_seqs, train_features, val_features, 
            train_labels, val_labels, scaler, imbalance_ratio)

# 数据加载
def load_data(file_path):
    df = pd.read_csv(file_path)
    sequences = df['sequence'].values
    # features = df[['noes_score', 'pm_score', 'seq_score', 'phastCons']].values
    features = df[['noes_score', 'seq_score', 'phastCons']].values
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
def apply_smote(bert_features, additional_features, labels, sampling_strategy=0.3):
    combined_features = np.hstack((bert_features, additional_features))
    
    # Use a more conservative sampling strategy with k_neighbors tuned to your dataset size
    smote = SMOTE(sampling_strategy=0.7, k_neighbors=min(5, sum(labels == 1) - 1), random_state=42)
    combined_features_resampled, labels_resampled = smote.fit_resample(combined_features, labels)
    
    # Split back into separate feature sets
    bert_features_resampled = combined_features_resampled[:, :bert_features.shape[1]]
    additional_features_resampled = combined_features_resampled[:, bert_features.shape[1]:]
    
    return bert_features_resampled, additional_features_resampled, labels_resampled

# 改进：自适应Focal Loss
class AdaptiveFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, pos_weight=None, label_smoothing=0.0):
        super(AdaptiveFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        # 标签平滑
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # 计算BCE损失
        if self.pos_weight is not None:
            bce_loss = nn.functional.binary_cross_entropy(
                inputs, targets, reduction='none'
            )
            # 应用正样本权重
            weights = torch.where(targets >= 0.5, self.pos_weight, 1.0)
            bce_loss = bce_loss * weights
        else:
            bce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        
        # Focal loss调制
        p_t = torch.where(targets >= 0.5, inputs, 1 - inputs)
        alpha_t = torch.where(targets >= 0.5, 
                             self.alpha if self.alpha else 1.0, 
                             1 - self.alpha if self.alpha else 1.0)
        
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss
        
        return focal_loss.mean()

# 改进：阈值优化函数
def find_optimal_threshold(model, val_loader, device):
    """找到最优的分类阈值"""
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            bert_features = batch['bert_features'].to(device)
            additional_features = batch['additional_features'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(bert_features, additional_features)
            all_probs.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # 测试不同阈值
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1 = 0
    best_threshold = 0.5
    
    print("\n阈值优化结果:")
    for threshold in thresholds:
        preds = (all_probs > threshold).astype(int)
        f1 = f1_score(all_labels, preds, zero_division=0)
        recall = recall_score(all_labels, preds, zero_division=0)
        precision = precision_score(all_labels, preds, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
        
        if threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:  # 只打印关键阈值
            print(f"  阈值 {threshold:.2f}: F1={f1:.4f}, Recall={recall:.4f}, Precision={precision:.4f}")
    
    print(f"最优阈值: {best_threshold:.2f}, 最佳F1: {best_f1:.4f}")
    return best_threshold

# 改进：训练模型函数
def train_model(model, train_loader, val_loader, config, device, imbalance_ratio):
    """改进的训练流程"""
    
    # 1. 自适应损失函数参数
    pos_weight = torch.tensor([min(imbalance_ratio, 10.0)]).to(device)  # 限制最大权重
    criterion = AdaptiveFocalLoss(
        alpha=0.7,  # 稍微偏向少数类
        gamma=1.5,  # 降低gamma避免过度聚焦
        pos_weight=pos_weight,
        label_smoothing=0.05  # 轻微标签平滑
    )
    
    # 2. 优化器 - 使用AdamW with weight decay
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=float(config['training']['learning_rate']),
        weight_decay=1e-4,  # L2正则化
        betas=(0.9, 0.999)
    )
    
    # 3. 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    # 4. 训练循环
    num_epochs = config['training']['num_epochs']
    best_f1 = 0.0
    patience_counter = 0
    patience = config['training']['patience']
    
    training_history = {
        'train_loss': [], 'val_loss': [], 
        'val_accuracy': [], 'val_precision': [], 
        'val_recall': [], 'val_f1': [], 'val_auc': []
    }
    
    print(f"开始训练 - 使用权重: {pos_weight.item():.2f}")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        train_preds = []
        train_labels_list = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]"):
            bert_features = batch['bert_features'].to(device)
            additional_features = batch['additional_features'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(bert_features, additional_features)
            loss = criterion(outputs, labels)
            
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            # 收集训练预测用于监控
            with torch.no_grad():
                preds = (outputs > 0.5).cpu().numpy()
                train_preds.extend(preds)
                train_labels_list.extend(labels.cpu().numpy())
        
        avg_train_loss = total_loss / len(train_loader)
        
        # 训练集指标
        train_f1 = f1_score(train_labels_list, train_preds)
        train_recall = recall_score(train_labels_list, train_preds)
        
        # 验证阶段
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train: Loss={avg_train_loss:.4f}, F1={train_f1:.4f}, Recall={train_recall:.4f}")
        print(f"  Val:   Loss={val_metrics['loss']:.4f}, F1={val_metrics['f1']:.4f}, "
              f"Recall={val_metrics['recall']:.4f}, Precision={val_metrics['precision']:.4f}")
        
        # 学习率调度
        scheduler.step(val_metrics['f1'])
        
        # 记录历史
        training_history['train_loss'].append(avg_train_loss)
        training_history['val_loss'].append(val_metrics['loss'])
        training_history['val_accuracy'].append(val_metrics['accuracy'])
        training_history['val_precision'].append(val_metrics['precision'])
        training_history['val_recall'].append(val_metrics['recall'])
        training_history['val_f1'].append(val_metrics['f1'])
        training_history['val_auc'].append(val_metrics['auc'])
        
        # 早停和模型保存
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            patience_counter = 0
            torch.save(model.state_dict(), 'best_improved_classifier.pth')
            print(f"  ✓ 保存最佳模型! F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  早停触发! 最佳F1: {best_f1:.4f}")
                break
    
    return training_history

# 改进：模型评估函数
def evaluate_model(model, data_loader, criterion, device, threshold=0.5):
    """改进的模型评估，支持阈值调整"""
    model.eval()
    total_loss = 0
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            bert_features = batch['bert_features'].to(device)
            additional_features = batch['additional_features'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(bert_features, additional_features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            all_probs.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs > threshold).astype(int)
    
    # 计算指标
    metrics = {
        'loss': total_loss / len(data_loader),
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'auc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
    }
    
    # 打印混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        print(f"    混淆矩阵: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        if tp + fn > 0:
            print(f"    真正召回率: {tp/(tp+fn):.4f}")
    
    return metrics

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
    def __init__(self, alpha=None, gamma=2.0, pos_weight=None, label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        # 标签平滑
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # 计算BCE损失
        if self.pos_weight is not None:
            bce_loss = nn.functional.binary_cross_entropy(
                inputs, targets, reduction='none'
            )
            # 应用正样本权重
            weights = torch.where(targets >= 0.5, self.pos_weight, 1.0)
            bce_loss = bce_loss * weights
        else:
            bce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        
        # Focal loss调制
        p_t = torch.where(targets >= 0.5, inputs, 1 - inputs)
        alpha_t = torch.where(targets >= 0.5, 
                             self.alpha if self.alpha else 1.0, 
                             1 - self.alpha if self.alpha else 1.0)
        
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss
        
        return focal_loss.mean()
    
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
    
    # 使用改进的数据预处理 - 应用第二个代码的改进方案
    print("Applying improved data preprocessing...")
    (train_seqs, val_seqs, train_features, val_features, 
     train_labels, val_labels, scaler, imbalance_ratio) = improved_data_preprocessing(
        sequences, additional_features, labels, config
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
    
    # 处理类别不平衡 - 根据类别分布情况决定是否应用SMOTE
    pos_ratio = np.sum(train_labels) / len(train_labels)
    print(f"训练集正样本比例: {pos_ratio:.4f}")

    if pos_ratio < 0.5:  # 当正样本比例小于0.5时启用SMOTE
        print("检测到类别不平衡，正在应用SMOTE进行数据增强...")
        train_bert_features, train_features, train_labels = apply_smote(
            train_bert_features, train_features, train_labels, 
            sampling_strategy=0.7  # 设置少数类与多数类的比例
        )

        print(f"After SMOTE - Training set size: {len(train_labels)}")
        unique_after_smote, counts_after_smote = np.unique(train_labels, return_counts=True)
        print("SMOTE后训练集类别分布:")
        for u, c in zip(unique_after_smote, counts_after_smote):
            print(f"  类别 {u}: {c} 样本 ({c/len(train_labels)*100:.2f}%)")
    else:
        print(f"类别分布相对平衡（正样本比例: {pos_ratio:.4f}），跳过SMOTE数据增强")
    
    # 创建优化的数据集
    train_dataset = OptimizedDNADataset(train_bert_features, train_features, train_labels)
    val_dataset = OptimizedDNADataset(val_bert_features, val_features, val_labels)
    
    # 创建数据加载器 - 使用SMOTE后就不需要加权采样器
    batch_size = config['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 创建改进的分类器模型 - 使用第二个代码的ImprovedDNAClassifier
    print("Creating improved DNA classifier...")
    model = DNAClassifier(config, imbalance_ratio).to(device)
    
    # 打印模型信息
    print("Improved classification model architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Class imbalance ratio: {imbalance_ratio:.2f}:1")
    
    # 使用改进的训练函数
    print("Starting improved training process...")
    history = train_model(model, train_loader, val_loader, config, device, imbalance_ratio)
    
    # 可视化训练过程
    print("Plotting training history...")
    plot_training_history(history)
    
    # 加载最佳模型进行评估
    print("Loading best model for evaluation...")
    best_model = DNAClassifier(config, imbalance_ratio).to(device)
    best_model.load_state_dict(torch.load('best_improved_classifier.pth'))
    
    # 进行阈值优化
    print("Finding optimal threshold...")
    optimal_threshold = find_optimal_threshold(best_model, val_loader, device)
    
    # 使用最优阈值对验证集进行全面评估
    print("Evaluating model with optimal threshold...")
    
    # 创建临时的损失函数用于评估
    pos_weight = torch.tensor([min(imbalance_ratio, 10.0)]).to(device)
    eval_criterion = AdaptiveFocalLoss(
        alpha=0.7,
        gamma=1.5,
        pos_weight=pos_weight,
        label_smoothing=0.05
    )
    
    # 使用最优阈值评估
    val_metrics = evaluate_model(best_model, val_loader, eval_criterion, device, threshold=optimal_threshold)
    
    print(f"\n=== 最终评估结果 (阈值={optimal_threshold:.2f}) ===")
    print(f"验证损失: {val_metrics['loss']:.4f}")
    print(f"准确率: {val_metrics['accuracy']:.4f}")
    print(f"精确率: {val_metrics['precision']:.4f}")
    print(f"召回率: {val_metrics['recall']:.4f}")
    print(f"F1分数: {val_metrics['f1']:.4f}")
    print(f"AUC: {val_metrics['auc']:.4f}")
    
    # 保存预处理器和最优阈值
    print("Saving preprocessing components...")
    torch.save({
        'scaler': scaler,
        'optimal_threshold': optimal_threshold,
        'imbalance_ratio': imbalance_ratio,
        'config': config
    }, 'preprocessing_components.pth')
    
    print("Training completed successfully!")
    return best_model, history, optimal_threshold, scaler

if __name__ == "__main__":
    main()