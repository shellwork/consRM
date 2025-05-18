import os
import torch
import pandas as pd
import yaml
import numpy as np
import sys
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader

# 从训练文件中导入必要的类
def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

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
class DNASequenceClassifier(torch.nn.Module):
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
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim
        
        # 移除最后一层的Dropout
        if feature_encoder_layers:
            feature_encoder_layers = feature_encoder_layers[:-1]
            
        self.feature_encoder = torch.nn.Sequential(*feature_encoder_layers)
        
        # 获取特征编码器的输出维度
        feature_output_dim = feature_encoder_dims[-1] if feature_encoder_dims else feature_dim
        
        # 融合后特征的分类器
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(bert_output_dim + feature_output_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(hidden_dim // 2, 1),
            torch.nn.Sigmoid()
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

def predict(model, data_loader, device):
    """使用训练好的模型进行预测"""
    model.eval()
    all_probs = []
    all_preds = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            features = batch['features'].to(device)
            
            # 前向传播
            outputs = model(input_ids, attention_mask, features)
            
            # 收集预测概率和预测结果
            probs = outputs.cpu().numpy()
            predictions = (outputs > 0.5).float().cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(predictions)
    
    return np.array(all_probs), np.array(all_preds)

def load_model_and_tokenizer(config, device):
    """加载模型和tokenizer"""
    model_path = config['model']['bert_model_path']
    
    # 从配置文件获取模型加载参数
    trust_remote_code = config.get('loading', {}).get('trust_remote_code', True)
    local_files_only = config.get('loading', {}).get('local_files_only', True)
    
    try:
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
    except Exception as e:
        print(f"标准加载方式失败，尝试替代方法: {e}")
        
        # 替代方法：直接从本地模型目录加载
        import sys
        import os
        
        # 将模型路径添加到系统路径
        sys.path.insert(0, model_path)
        
        # 尝试从模型目录导入所需模块
        try:
            # 先尝试加载tokenizer
            if os.path.exists(os.path.join(model_path, "tokenizer_config.json")):
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path, 
                    trust_remote_code=trust_remote_code,
                    local_files_only=local_files_only
                )
            else:
                # 如果没有tokenizer配置，尝试使用标准BERT tokenizer
                from transformers import BertTokenizer
                tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            
            # 尝试直接导入和加载模型
            try:
                sys.path.insert(0, os.path.join(model_path))
                from modeling_bert import BertModel as CustomBertModel
                bert_model = CustomBertModel.from_pretrained(model_path)
            except ImportError:
                # 如果没有自定义modeling_bert，尝试使用标准BertModel但跳过配置检查
                from transformers import BertModel
                bert_config_file = os.path.join(model_path, "config.json")
                from transformers import PretrainedConfig
                config_obj = PretrainedConfig.from_json_file(bert_config_file)
                bert_model = BertModel(config_obj)
        except Exception as e2:
            print(f"替代方法也失败: {e2}")
            raise ValueError(f"无法加载模型。原始错误: {e}, 替代错误: {e2}")
    
    # 确保模型处于评估模式开始
    bert_model.eval()
    return bert_model, tokenizer

def load_data(file_path):
    """
    加载数据并预处理
    """
    df = pd.read_csv(file_path)
    
    # 检查序列列名，可能是'seq'或'sequence'
    seq_column = 'sequence' if 'sequence' in df.columns else 'seq'
    if seq_column not in df.columns:
        raise ValueError(f"未找到序列列。需要'seq'或'sequence'列。")
    
    sequences = df[seq_column].values
    
    # 检查特征列名
    if all(col in df.columns for col in ['noes_score', 'pm_score', 'seq_score', 'phastCons']):
        features = df[['noes_score', 'pm_score', 'seq_score', 'phastCons']].values
    elif all(col in df.columns for col in ['feature1', 'feature2', 'feature3', 'feature4']):
        features = df[['feature1', 'feature2', 'feature3', 'feature4']].values
    else:
        print("警告: 未找到所需的特征列。将使用零向量作为特征")
        features = np.zeros((len(sequences), 4))
    
    # 检查是否有标签列
    if 'label' in df.columns:
        labels = df['label'].values
    else:
        # 如果没有标签列，用0填充
        labels = np.zeros(len(sequences))
    
    return sequences, features, labels, 'label' in df.columns

def main():
    # 加载配置
    config = load_config()
    
    # 设置设备
    device = get_device(config['device'])
    print(f"Using device: {device}")
    
    # 获取输入参数
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    else:
        test_file = input("请输入测试数据文件路径: ")
    
    if not os.path.exists(test_file):
        print(f"错误: 文件 {test_file} 不存在!")
        return
    
    # 加载要进行预测的数据
    print(f"加载数据: {test_file}")
    sequences, features, labels, has_labels = load_data(test_file)
    
    # 加载DNABERT2模型和tokenizer
    print("加载模型和tokenizer...")
    bert_model, tokenizer = load_model_and_tokenizer(config, device)
    
    # 创建数据集
    max_seq_length = config['training']['max_seq_length']
    test_dataset = DNASequenceDataset(sequences, features, labels, tokenizer, max_seq_length)
    
    # 创建数据加载器
    batch_size = config['training']['batch_size']
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 加载训练好的模型
    if len(sys.argv) > 2:
        model_path = sys.argv[2]
    else:
        model_path = input("请输入训练好的模型路径 (默认: best_dna_classifier.pth): ")
        if not model_path:
            model_path = 'best_dna_classifier.pth'
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 不存在!")
        return
    
    # 创建模型
    model = DNASequenceClassifier(bert_model, config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"模型已加载: {model_path}")
    
    # 进行预测
    print("预测中...")
    probabilities, predictions = predict(model, test_loader, device)
    
    # 输出结果
    # 检查序列列名，可能是'seq'或'sequence'
    df = pd.read_csv(test_file)
    seq_column = 'sequence' if 'sequence' in df.columns else 'seq'
    
    result_df = pd.DataFrame({
        seq_column: sequences,
        'predicted_probability': probabilities,
        'predicted_label': predictions
    })
    
    # 如import os
import torch
import pandas as pd
import yaml
import numpy as np
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader

# 从训练文件中导入必要的类
from dna_classification_model import DNASequenceClassifier, DNASequenceDataset, load_config, get_device

def predict(model, data_loader, device):
    """使用训练好的模型进行预测"""
    model.eval()
    all_probs = []
    all_preds = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            features = batch['features'].to(device)
            
            # 前向传播
            outputs = model(input_ids, attention_mask, features)
            
            # 收集预测概率和预测结果
            probs = outputs.cpu().numpy()
            predictions = (outputs > 0.5).float().cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(predictions)
    
    return np.array(all_probs), np.array(all_preds)

def main():
    # 加载配置
    config = load_config()
    
    # 设置设备
    device = get_device(config['device'])
    print(f"Using device: {device}")
    
    # 加载要进行预测的数据
    test_file = input("请输入测试数据文件路径: ")
    if not os.path.exists(test_file):
        print(f"错误: 文件 {test_file} 不存在!")
        return
        
    df = pd.read_csv(test_file)
    sequences = df['seq'].values
    
    # 检查是否有特征列
    feature_cols = ['feature1', 'feature2', 'feature3', 'feature4']
    if all(col in df.columns for col in feature_cols):
        features = df[feature_cols].values
    else:
        print("警告: 未找到所有特征列，将使用零向量作为特征")
        features = np.zeros((len(sequences), 4))
    
    # 加载标签（如果有）
    if 'label' in df.columns:
        labels = df['label'].values
        print("找到标签列，将计算预测准确率")
    else:
        labels = np.zeros(len(sequences))  # 仅占位
        print("未找到标签列，将仅输出预测结果")
    
    # 加载DNABERT2模型和tokenizer
    model_path = config['model']['bert_model_path']
    print(f"加载模型和tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    bert_model = AutoModel.from_pretrained(model_path)
    
    # 创建数据集
    max_seq_length = config['training']['max_seq_length']
    test_dataset = DNASequenceDataset(sequences, features, labels, tokenizer, max_seq_length)
    
    # 创建数据加载器
    batch_size = config['training']['batch_size']
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 加载训练好的模型
    model_path = input("请输入训练好的模型路径 (默认: best_dna_classifier.pth): ")
    if not model_path:
        model_path = 'best_dna_classifier.pth'
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 不存在!")
        return
    
    # 创建模型
    model = DNASequenceClassifier(bert_model, config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"模型已加载: {model_path}")
    
    # 进行预测
    print("预测中...")
    probabilities, predictions = predict(model, test_loader, device)
    
    # 输出结果
    result_df = pd.DataFrame({
        'seq': sequences,
        'predicted_probability': probabilities,
        'predicted_label': predictions
    })
    
    # 如果有真实标签，计算准确率
    if 'label' in df.columns:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        accuracy = accuracy_score(labels, predictions)
        try:
            auc = roc_auc_score(labels, probabilities)
        except:
            auc = 0.5
        
        print(f"预测准确率: {accuracy:.4f}")
        print(f"ROC AUC: {auc:.4f}")
        
        # 添加真实标签到结果
        result_df['true_label'] = labels
    
    # 保存预测结果
    output_file = input("请输入保存预测结果的文件名 (默认: prediction_results.csv): ")
    if not output_file:
        output_file = 'prediction_results.csv'
    
    result_df.to_csv(output_file, index=False)
    print(f"预测结果已保存到: {output_file}")
    
    # 显示部分结果
    print("\n前5个预测结果示例:")
    print(result_df.head())

if __name__ == "__main__":
    main()
