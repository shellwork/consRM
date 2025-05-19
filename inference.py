import os
import torch
import pandas as pd
import yaml
import numpy as np
from transformers import AutoTokenizer, AutoModel, BertModel, BertConfig
from torch.utils.data import Dataset, DataLoader

# 从训练文件中导入必要的类
from train import DNAClassifier, OptimizedDNADataset, load_config, get_device, BertFeatureExtractor

def predict(model, data_loader, device):
    """使用训练好的模型进行预测"""
    model.eval()
    all_probs = []
    all_preds = []
    
    with torch.no_grad():
        for batch in data_loader:
            bert_features = batch['bert_features'].to(device)
            additional_features = batch['additional_features'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播
            outputs = model(bert_features, additional_features)
            
            # 收集预测结果和标签
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
    test_file = "./data/c.csv"
    if not os.path.exists(test_file):
        print(f"错误: 文件 {test_file} 不存在!")
        return
        
    df = pd.read_csv(test_file)
    sequences = df['sequence'].values
    # features = df[['noes_score', 'pm_score', 'seq_score', 'phastCons']].values
    features = df[['noes_score', 'seq_score', 'phastCons']].values
    
    # 加载标签（如果有）
    if 'label' in df.columns:
        labels = df['label'].values
        print("找到标签列，将计算预测准确率")
    else:
        labels = np.zeros(len(sequences))  # 仅占位
        print("未找到标签列，将仅输出预测结果")
    
    # 加载DNABERT2模型和tokenizer
    model_path = config['model']['bert_model_path']
    max_seq_length = config['training']['max_seq_length']
    trust_remote_code = config['loading']['trust_remote_code']
    local_files_only = config['loading']['local_files_only']
    print(f"加载模型和tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only
        )
    print("Tokenizer loaded successfully")
    
    feature_extractor = BertFeatureExtractor(model_path, tokenizer, max_seq_length, device)
    bert_features = feature_extractor.extract_features(sequences)
    test_dataset = OptimizedDNADataset(bert_features, features, labels)
    
    # 创建数据加载器
    batch_size = config['training']['batch_size']
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 加载训练好的模型
    model_path = 'best_dna_classifier.pth'
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 不存在!")
        return
    
    # 创建模型
    model = DNAClassifier(config).to(device)
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
    output_file = 'data/prediction_results.csv'
    
    result_df.to_csv(output_file, index=False)
    print(f"预测结果已保存到: {output_file}")
    
    # 显示部分结果
    print("\n前5个预测结果示例:")
    print(result_df.head())

if __name__ == "__main__":
    main()
