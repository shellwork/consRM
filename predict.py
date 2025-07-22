import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import logging
from tqdm import tqdm
import os
import argparse
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# 从训练代码中导入必要的类和函数
# 假设训练代码文件名为 train.py，可以通过以下方式导入
# from train import DNABERT2ConservationModel, DNAConservationDataset, load_and_preprocess_data

# 由于我们在同一路径下，直接复用类定义
# 这里重新定义需要的类，实际使用时建议将公共类抽取到单独的模块中

from torch.utils.data import Dataset

class DNAConservationDataset(Dataset):
    """DNA保守性预测数据集 - 复用训练代码中的定义"""
    
    def __init__(self, sequences, genomic_features, labels, tokenizer, max_length=512):
        self.sequences = sequences
        self.genomic_features = genomic_features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 自动检测组学特征维度
        self.genomic_feature_dim = genomic_features.shape[1] if len(genomic_features.shape) > 1 else genomic_features.shape[0]

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = str(self.sequences[idx])
        genomic_feat = torch.tensor(self.genomic_features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long) if self.labels is not None else torch.tensor(-1, dtype=torch.long)
        
        # 对DNA序列进行tokenization
        encoding = self.tokenizer(
            sequence,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'genomic_features': genomic_feat,
            'labels': label
        }

class DNABERT2ConservationModel(nn.Module):
    """基于DNABERT2的DNA保守性预测模型 - 复用训练代码中的定义"""
    
    def __init__(self, model_path='./embedded_model', 
                 genomic_feature_dim=4, num_classes=2, 
                 hidden_dim=256, dropout_rate=0.1,
                 trust_remote_code=True, local_files_only=True):
        super(DNABERT2ConservationModel, self).__init__()

        from transformers import BertModel
        # 加载本地预训练的DNABERT2模型
        self.dnabert2 = BertModel.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only
        )
        self.dnabert2_dim = self.dnabert2.config.hidden_size  # 通常是768

        self.genomic_feature_dim = genomic_feature_dim

        # 组学特征处理层
        if genomic_feature_dim > 0:
            self.genomic_proj = nn.Sequential(
                nn.Linear(genomic_feature_dim, max(64, genomic_feature_dim * 16)),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(max(64, genomic_feature_dim * 16), 128),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            fusion_dim = self.dnabert2_dim + 128
        else:
            self.genomic_proj = None
            fusion_dim = self.dnabert2_dim

        # MLP分类器
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, input_ids, attention_mask, genomic_features):
        # 通过DNABERT2获取序列特征
        dnabert_outputs = self.dnabert2(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # 使用[CLS] token的表示作为序列特征
        sequence_features = dnabert_outputs.last_hidden_state[:, 0, :]

        # 处理组学特征
        if self.genomic_feature_dim > 0 and genomic_features is not None:
            genomic_processed = self.genomic_proj(genomic_features)
            # 特征融合
            fused_features = torch.cat([sequence_features, genomic_processed], dim=1)
        else:
            # 只使用序列特征
            fused_features = sequence_features

        # 分类
        logits = self.classifier(fused_features)

        return logits

def load_and_preprocess_data(data_path, genomic_feature_cols=None, has_labels=True):
    """加载和预处理数据 - 复用并修改训练代码中的函数以支持无标签数据"""
    df = pd.read_csv(data_path)
    
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"First few rows:")
    print(df.head())
    
    # 提取序列
    sequences = df.iloc[:, 0].values  # 第一列是序列
    
    # 提取标签（如果存在）
    if has_labels:
        labels = df.iloc[:, -1].values  # 最后一列是标签
        # 提取组学特征
        if genomic_feature_cols is None:
            genomic_features = df.iloc[:, 1:-1].values
        else:
            genomic_features = df.iloc[:, genomic_feature_cols].values
    else:
        labels = None
        # 提取组学特征
        if genomic_feature_cols is None:
            genomic_features = df.iloc[:, 1:].values
        else:
            genomic_features = df.iloc[:, genomic_feature_cols].values
    
    genomic_feature_dim = genomic_features.shape[1]
    
    print(f"Sequences shape: {sequences.shape}")
    print(f"Genomic features shape: {genomic_features.shape}")
    print(f"Genomic feature dimension: {genomic_feature_dim}")
    if labels is not None:
        print(f"Labels shape: {labels.shape}")
    
    return sequences, genomic_features, labels, genomic_feature_dim

class DNABERT2Predictor:
    """DNABERT2推理预测器"""
    
    def __init__(self, model_path, checkpoint_path, device, scaler=None):
        self.device = device
        self.scaler = scaler
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # 从检查点中获取模型配置信息
        model_state_dict = checkpoint['model_state_dict']
        
        # 从状态字典推断模型配置
        genomic_feature_dim = self._infer_genomic_dim(model_state_dict)
        
        # 创建模型
        self.model = DNABERT2ConservationModel(
            model_path=model_path,
            genomic_feature_dim=genomic_feature_dim,
            num_classes=2,
            hidden_dim=256,
            dropout_rate=0.1,
            trust_remote_code=True,
            local_files_only=True
        )
        
        # 加载模型权重
        self.model.load_state_dict(model_state_dict)
        self.model.to(device)
        self.model.eval()
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        
        print(f"Model loaded successfully from {checkpoint_path}")
        print(f"Genomic feature dimension: {genomic_feature_dim}")
        
    def _infer_genomic_dim(self, state_dict):
        """从状态字典推断基因组特征维度"""
        # 查找genomic_proj的第一层来推断输入维度
        for key in state_dict.keys():
            if 'genomic_proj.0.weight' in key:
                return state_dict[key].shape[1]
        return 0  # 如果没有找到，说明没有使用基因组特征
    
    def predict_batch(self, data_loader, return_probabilities=True):
        """批量预测"""
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Predicting"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                genomic_features = batch['genomic_features'].to(self.device)
                labels = batch['labels']
                
                # 前向传播
                logits = self.model(input_ids, attention_mask, genomic_features)
                
                # 获取预测结果
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
                # 如果有标签，也收集标签
                if labels[0].item() != -1:  # -1表示没有真实标签
                    all_labels.extend(labels.numpy())
        
        results = {
            'predictions': np.array(all_preds),
            'probabilities': np.array(all_probs) if return_probabilities else None
        }
        
        if all_labels:
            results['true_labels'] = np.array(all_labels)
            
        return results
    
    def predict_single(self, sequence, genomic_features, max_length=512):
        """单个样本预测"""
        # 标准化基因组特征（如果有scaler）
        if self.scaler is not None:
            genomic_features = self.scaler.transform([genomic_features])[0]
        
        # 创建单个样本的数据集
        dataset = DNAConservationDataset(
            [sequence], 
            [genomic_features], 
            [None],  # 没有标签
            self.tokenizer, 
            max_length
        )
        
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        # 进行预测
        results = self.predict_batch(data_loader)
        
        return {
            'prediction': results['predictions'][0],
            'probability': results['probabilities'][0] if results['probabilities'] is not None else None
        }
    
    def evaluate(self, data_loader):
        """评估模型性能（需要有真实标签）"""
        results = self.predict_batch(data_loader)
        
        if 'true_labels' not in results:
            raise ValueError("Cannot evaluate without true labels")
        
        true_labels = results['true_labels']
        predictions = results['predictions']
        probabilities = results['probabilities'][:, 1]  # 正类概率
        
        # 计算各种指标
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='binary'
        )
        auc = roc_auc_score(true_labels, probabilities)
        
        # 混淆矩阵
        cm = confusion_matrix(true_labels, predictions)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': cm
        }
        
        return metrics

def main():
    data_path = 'data/eval/c1.csv'
    model_path = 'embed_model'
    checkpoint_path = 'best_dnabert2_model_1_1.pth'
    output_path = 'phast4way_m6A.csv'
    batch_size = 64
    max_length = 512
    has_labels = True  # 如果您的数据包含真实标签，请设置为 True
    scaler_path = None 

    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 加载scaler（如果提供）
    scaler = None
    if scaler_path and os.path.exists(scaler_path):
        import pickle
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        logger.info(f"Scaler loaded from {scaler_path}")
    
    # 加载数据
    logger.info("Loading data...")
    sequences, genomic_features, labels, genomic_feature_dim = load_and_preprocess_data(
        data_path, 
        has_labels=has_labels
    )
    
    # 如果有scaler，标准化特征
    if scaler is not None:
        genomic_features = scaler.transform(genomic_features)
        logger.info("Genomic features standardized using provided scaler")
    
    print(f"\n=== Dataset Info ===")
    print(f"Total samples: {len(sequences)}")
    print(f"Genomic feature dimension: {genomic_feature_dim}")
    if labels is not None:
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        print(f"Label distribution: {dict(zip(unique_labels.tolist(), label_counts.tolist()))}")
    
    # 创建预测器
    logger.info("Loading model...")
    predictor = DNABERT2Predictor(
        model_path=model_path,
        checkpoint_path=checkpoint_path,
        device=device,
        scaler=scaler
    )
    
    # 创建数据集和数据加载器
    dataset = DNAConservationDataset(
        sequences, genomic_features, labels, 
        predictor.tokenizer, max_length
    )
    
    data_loader = DataLoader(
        dataset, batch_size=batch_size, 
        shuffle=False, num_workers=4
    )
    
    # 进行推理
    logger.info("Starting inference...")
    results = predictor.predict_batch(data_loader)
    
    # 如果有真实标签，进行评估
    if has_labels and labels is not None:
        logger.info("Evaluating model performance...")
        metrics = predictor.evaluate(data_loader)
        
        print(f"\n=== Evaluation Results ===")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1']:.4f}")
        print(f"AUC: {metrics['auc']:.4f}")
        print(f"Confusion Matrix:")
        print(metrics['confusion_matrix'])
    
    # 保存预测结果
    logger.info(f"Saving predictions to {output_path}")
    
    # 读取原始数据以获取列名
    original_df = pd.read_csv(data_path)
    
    # 准备输出数据
    output_data = {
        'sequence': sequences,
        'predicted_label': results['predictions'],
        'probability_class_0': results['probabilities'][:, 0],
        'probability_class_1': results['probabilities'][:, 1]
    }
    
    # 如果有真实标签，也包含在输出中
    if has_labels and labels is not None:
        output_data['true_label'] = labels
    
    # 添加原始的基因组特征到输出，使用原始列名
    feature_columns = original_df.columns[1:].tolist()  # 除第一列（序列）外的所有列
    if has_labels:
        feature_columns = feature_columns[:-1]  # 如果有标签，去掉最后一列
    
    for i, col_name in enumerate(feature_columns):
        if i < genomic_features.shape[1]:
            output_data[col_name] = genomic_features[:, i]
    
    # 保存为CSV
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_path, index=False)
    
    print(f"\n=== Prediction Summary ===")
    print(f"Total predictions: {len(results['predictions'])}")
    pred_counts = np.bincount(results['predictions'])
    for i, count in enumerate(pred_counts):
        print(f"Predicted class {i}: {count} samples ({count/len(results['predictions'])*100:.1f}%)")
    
    logger.info("Inference completed!")

if __name__ == "__main__":
    main()