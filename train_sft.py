import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import logging
from tqdm import tqdm
import os
import swanlab

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DNAConservationDataset(Dataset):
    """DNA保守性预测数据集"""
    
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
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
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
    """基于DNABERT2的DNA保守性预测模型"""
    
    def __init__(self, model_path='./embedded_model', 
             genomic_feature_dim=4, num_classes=2, 
             hidden_dim=256, dropout_rate=0.1,
             trust_remote_code=True, local_files_only=True):
        super(DNABERT2ConservationModel, self).__init__()

        # 加载本地预训练的DNABERT2模型
        self.dnabert2 = BertModel.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only
        )
        self.dnabert2_dim = self.dnabert2.config.hidden_size  # 通常是768

        self.genomic_feature_dim = genomic_feature_dim

        # 组学特征处理层 - 根据特征数量动态调整
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

        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化新增层的权重"""
        for module in [self.genomic_proj, self.classifier]:
            if module is not None:
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        nn.init.zeros_(layer.bias)
    
    def forward(self, input_ids, attention_mask, genomic_features):
        # 通过DNABERT2获取序列特征
        dnabert_outputs = self.dnabert2(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # 使用[CLS] token的表示作为序列特征
        sequence_features = dnabert_outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]

        # 处理组学特征
        if self.genomic_feature_dim > 0 and genomic_features is not None:
            genomic_processed = self.genomic_proj(genomic_features)  # [batch_size, 128]
            # 特征融合
            fused_features = torch.cat([sequence_features, genomic_processed], dim=1)
        else:
            # 只使用序列特征
            fused_features = sequence_features

        # 分类
        logits = self.classifier(fused_features)

        return logits

class DNABERT2Trainer:
    """DNABERT2微调训练器"""
    
    def __init__(self, model, train_loader, val_loader, device, 
                 learning_rate=2e-5, weight_decay=0.01):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # 设置不同的学习率：DNABERT2使用较小学习率，新增层使用较大学习率
        dnabert_params = []
        new_params = []
        
        for name, param in model.named_parameters():
            if 'dnabert2' in name:
                dnabert_params.append(param)
            else:
                new_params.append(param)
        
        self.optimizer = optim.AdamW([
            {'params': dnabert_params, 'lr': learning_rate},
            {'params': new_params, 'lr': learning_rate * 10}  # 新增层使用更大学习率
        ], weight_decay=weight_decay)
        
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # 记录学习率信息
        self.bert_lr = learning_rate
        self.new_layer_lr = learning_rate * 10
        
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        batch_losses = []
        
        pbar = tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}")
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            genomic_features = batch['genomic_features'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            logits = self.model(input_ids, attention_mask, genomic_features)
            loss = self.criterion(logits, labels)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            batch_loss = loss.item()
            total_loss += batch_loss
            batch_losses.append(batch_loss)
            
            # 收集预测结果
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{batch_loss:.4f}'})
            
            # 记录每个batch的损失
            global_step = epoch * len(self.train_loader) + batch_idx
            swanlab.log({
                "train/batch_loss": batch_loss,
                "train/learning_rate_bert": self.optimizer.param_groups[0]['lr'],
                "train/learning_rate_new": self.optimizer.param_groups[1]['lr'],
            }, step=global_step)
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy, batch_losses
    
    def validate(self, epoch):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Validating Epoch {epoch+1}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                genomic_features = batch['genomic_features'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits = self.model(input_ids, attention_mask, genomic_features)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                
                # 收集预测结果
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # 正类概率
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary'
        )
        auc = roc_auc_score(all_labels, all_probs)
        auprc = average_precision_score(all_labels, all_probs) # <--- 新增: 计算AUPRC
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'auprc': auprc  # <--- 新增: 将AUPRC添加到返回结果中
        }
    
    def train(self, num_epochs, save_path=None):
        """完整训练流程"""
        best_val_f1 = 0
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # 训练
            train_loss, train_acc, batch_losses = self.train_epoch(epoch)
            
            # 验证
            val_metrics = self.validate(epoch)
            
            # 更新学习率
            old_lr_bert = self.optimizer.param_groups[0]['lr']
            old_lr_new = self.optimizer.param_groups[1]['lr']
            self.scheduler.step(val_metrics['loss'])
            new_lr_bert = self.optimizer.param_groups[0]['lr']
            new_lr_new = self.optimizer.param_groups[1]['lr']
            
            # 记录epoch级别的指标
            swanlab.log({
                "epoch": epoch + 1,
                "train/epoch_loss": train_loss,
                "train/epoch_accuracy": train_acc,
                "val/loss": val_metrics['loss'],
                "val/accuracy": val_metrics['accuracy'],
                "val/precision": val_metrics['precision'],
                "val/recall": val_metrics['recall'],
                "val/f1": val_metrics['f1'],
                "val/auc": val_metrics['auc'],
                "val/auprc": val_metrics['auprc'], # <--- 新增: 记录AUPRC
                "learning_rate/bert": new_lr_bert,
                "learning_rate/new_layers": new_lr_new,
                "best_val_f1": max(best_val_f1, val_metrics['f1'])
            }, step=epoch)
            
            # 记录学习率变化
            if new_lr_bert != old_lr_bert:
                logger.info(f"Learning rate changed - BERT: {old_lr_bert:.2e} -> {new_lr_bert:.2e}, New layers: {old_lr_new:.2e} -> {new_lr_new:.2e}")
            
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            # <--- 修改: 日志输出中增加AUPRC
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}, "
                        f"Val Acc: {val_metrics['accuracy']:.4f}, "
                        f"Val F1: {val_metrics['f1']:.4f}, "
                        f"Val AUC: {val_metrics['auc']:.4f}, "
                        f"Val AUPRC: {val_metrics['auprc']:.4f}")
            
            # 保存最佳模型
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                if save_path:
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'val_f1': best_val_f1,
                        'val_metrics': val_metrics,
                        'train_loss': train_loss,
                        'train_acc': train_acc
                    }
                    
                    # 直接覆盖保存模型文件
                    torch.save(checkpoint, save_path)
                    
                    logger.info(f"Best model saved with F1: {best_val_f1:.4f}")
            
            print("-" * 60)
        
        # 记录最终的最佳指标
        swanlab.log({"final_best_f1": best_val_f1})


def load_and_preprocess_data(data_path, genomic_feature_cols=None):
    """加载数据"""
    df = pd.read_csv(data_path)
    
    sequences = df.iloc[:, 0].values
    labels = df.iloc[:, -1].values
    
    if genomic_feature_cols is None:
        genomic_features = df.iloc[:, 1:-1].values
    else:
        genomic_features = df.iloc[:, genomic_feature_cols].values
    
    genomic_feature_dim = genomic_features.shape[1]
    
    # 输出数据统计信息
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    print(f"\n=== Dataset Statistics ===")
    print(f"Total samples: {len(sequences)}")
    print(f"Genomic feature dimension: {genomic_feature_dim}")
    print(f"Label distribution: {dict(zip(unique_labels.tolist(), label_counts.tolist()))}")
    print(f"Class balance ratio: {label_counts.min() / label_counts.max():.3f}")
    
    # 注意：不再进行StandardScaler处理，也不返回scaler
    return sequences, genomic_features, labels, genomic_feature_dim

def main():
    # 配置参数
    config = {
        'data_path': 'data/c1.csv',  # 替换为你的数据文件路径
        'model_path': './embed_model',  # 本地模型路径
        'genomic_feature_cols': None,
        'max_length': 512,
        'batch_size': 32,
        'num_epochs': 15,
        'learning_rate': 3e-5,
        'weight_decay': 0.01,
        'hidden_dim': 256,
        'dropout_rate': 0.1,
        'test_size': 0.2,
        'random_state': 42,
        'save_path': 'best_dnabert2_model_1_1.pth',
        'trust_remote_code': True,
        'local_files_only': True,
        # SwanLab 配置
        'experiment_name': 'del_phast4way_1:1',
        'project_name': 'consRM',
        'description': 'formal training in 1:1 label'
    }
    
    # 初始化SwanLab
    swanlab.init(
        project=config['project_name'],
        experiment_name=config['experiment_name'],
        description=config['description'],
        config=config
    )
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 输出系统信息到控制台
    print(f"\n=== Device ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 加载数据
    logger.info("Loading data...")
    sequences, genomic_features, labels, genomic_feature_dim = load_and_preprocess_data(
        config['data_path'], 
        genomic_feature_cols=config.get('genomic_feature_cols', None)
    )
    
    # 1. 先划分训练集和验证集
    logger.info("Splitting data...")
    (train_seq, val_seq, 
     train_genomic_raw, val_genomic_raw, # 使用raw后缀表示未处理
     train_labels, val_labels) = train_test_split(
        sequences, genomic_features, labels,
        test_size=config['test_size'],
        random_state=config['random_state'],
        stratify=labels
    )

    # 2. 然后在划分后的数据上进行标准化
    train_genomic = train_genomic_raw
    val_genomic = val_genomic_raw
    
    if genomic_feature_dim > 0:
        logger.info("Scaling genomic features...")
        scaler = StandardScaler()
        
        # 使用训练集来拟合scaler
        train_genomic = scaler.fit_transform(train_genomic_raw)
        
        # 使用同一个scaler来转换验证集
        val_genomic = scaler.transform(val_genomic_raw)
        
        # test_genomic = scaler.transform(test_genomic_raw)
    
    # 输出数据划分信息到控制台
    print(f"\n=== Data Split ===")
    print(f"Training samples: {len(train_seq)}")
    print(f"Validation samples: {len(val_seq)}")
    print(f"Training positive ratio: {np.mean(train_labels):.3f}")
    print(f"Validation positive ratio: {np.mean(val_labels):.3f}")
    
    # 加载tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config['model_path'],
        trust_remote_code=config['trust_remote_code'],
        local_files_only=config['local_files_only']
    )
    
    # 创建数据集时使用处理后的特征
    train_dataset = DNAConservationDataset(
        train_seq, train_genomic, train_labels, 
        tokenizer, config['max_length']
    )
    val_dataset = DNAConservationDataset(
        val_seq, val_genomic, val_labels, 
        tokenizer, config['max_length']
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], 
        shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], 
        shuffle=False, num_workers=4
    )
    
    # 输出数据加载器信息到控制台
    print(f"\n=== DataLoader Info ===")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Total training steps: {len(train_loader) * config['num_epochs']}")
    
    # 创建模型
    logger.info("Creating model...")
    model = DNABERT2ConservationModel(
        model_path=config['model_path'],
        genomic_feature_dim=genomic_feature_dim,  # 使用检测到的维度
        num_classes=2,
        hidden_dim=config['hidden_dim'],
        dropout_rate=config['dropout_rate'],
        trust_remote_code=config['trust_remote_code'],
        local_files_only=config['local_files_only']
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # 输出模型信息到控制台
    print(f"\n=== Model Architecture ===")
    print(f"Genomic feature dimension: {genomic_feature_dim}")
    print(f"DNABERT2 hidden dimension: {model.dnabert2_dim}")
    print(f"Hidden dimension: {config['hidden_dim']}")
    print(f"Dropout rate: {config['dropout_rate']}")
    
    # 创建训练器
    trainer = DNABERT2Trainer(
        model, train_loader, val_loader, device,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # 输出优化器配置到控制台
    print(f"\n=== Optimizer Configuration ===")
    print(f"BERT learning rate: {trainer.bert_lr:.2e}")
    print(f"New layers learning rate: {trainer.new_layer_lr:.2e}")
    print(f"Weight decay: {config['weight_decay']}")
    print(f"Scheduler: ReduceLROnPlateau")
    print(f"Training epochs: {config['num_epochs']}")
    
    # 开始训练
    logger.info("Starting training...")
    trainer.train(config['num_epochs'], config['save_path'])
    
    logger.info("Training completed!")
    
    # 完成实验
    swanlab.finish()

if __name__ == "__main__":
    main()