import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, 
                           roc_auc_score, average_precision_score, 
                           classification_report, confusion_matrix)
from sklearn.pipeline import Pipeline
from collections import Counter
import itertools
import warnings
warnings.filterwarnings('ignore')


def extract_kmer_features(sequences, k=4):
    """提取k-mer特征"""
    print(f"提取 {k}-mer 特征...")
    
    # 生成所有可能的k-mer
    nucleotides = ['A', 'T', 'G', 'C']
    all_kmers = [''.join(p) for p in itertools.product(nucleotides, repeat=k)]
    
    features = []
    for seq in sequences:
        # 计算每个序列的k-mer频率
        kmer_counts = Counter([seq[i:i+k] for i in range(len(seq)-k+1) if len(seq[i:i+k]) == k])
        # 转换为特征向量
        feature_vector = [kmer_counts.get(kmer, 0) for kmer in all_kmers]
        features.append(feature_vector)
    
    return np.array(features), all_kmers

def extract_basic_sequence_features(sequences):
    """提取基础序列特征"""
    print("提取基础序列特征...")
    
    features = []
    for seq in sequences:
        seq_len = len(seq)
        if seq_len == 0:
            features.append([0] * 8)
            continue
            
        # 计算碱基组成
        a_count = seq.count('A') / seq_len
        t_count = seq.count('T') / seq_len
        g_count = seq.count('G') / seq_len
        c_count = seq.count('C') / seq_len
        
        # GC含量
        gc_content = g_count + c_count
        
        # AT含量
        at_content = a_count + t_count
        
        # 序列长度
        seq_length = seq_len
        
        # 序列复杂度（简单估计：不同碱基的数量）
        unique_bases = len(set(seq))
        
        features.append([a_count, t_count, g_count, c_count, gc_content, at_content, seq_length, unique_bases])
    
    return np.array(features)

def check_data_quality(sequences, labels):
    """检查数据质量"""
    print("\n" + "="*50)
    print("数据质量检查")
    print("="*50)
    
    # 基本统计
    print(f"总样本数: {len(sequences)}")
    print(f"总标签数: {len(labels)}")
    
    # 检查序列长度
    seq_lengths = [len(seq) for seq in sequences]
    print(f"序列长度统计:")
    print(f"  最小长度: {min(seq_lengths)}")
    print(f"  最大长度: {max(seq_lengths)}")
    print(f"  平均长度: {np.mean(seq_lengths):.1f}")
    print(f"  中位数长度: {np.median(seq_lengths):.1f}")
    
    # 检查序列唯一性
    unique_sequences = len(set(sequences))
    duplicate_count = len(sequences) - unique_sequences
    print(f"序列唯一性:")
    print(f"  唯一序列数: {unique_sequences}")
    print(f"  重复序列数: {duplicate_count}")
    print(f"  重复比例: {duplicate_count/len(sequences)*100:.2f}%")
    
    # 检查标签分布
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    print(f"标签分布:")
    for label, count in zip(unique_labels, label_counts):
        print(f"  标签 {label}: {count} ({count/len(labels)*100:.2f}%)")
    
    # 类别平衡
    balance_ratio = label_counts.min() / label_counts.max()
    print(f"类别平衡比: {balance_ratio:.3f}")
    
    # 检查序列相似性（抽样检查）
    print("检查序列相似性（抽样前100个序列）...")
    similarity_threshold = 0.9
    high_similarity_pairs = 0
    sample_size = min(100, len(sequences))
    
    from difflib import SequenceMatcher
    for i in range(sample_size):
        for j in range(i+1, sample_size):
            similarity = SequenceMatcher(None, sequences[i], sequences[j]).ratio()
            if similarity > similarity_threshold:
                high_similarity_pairs += 1
                if high_similarity_pairs <= 5:  # 只打印前5对
                    print(f"  高相似序列对 {i}-{j}: 相似度 {similarity:.3f}")
    
    print(f"高相似度序列对数 (>{similarity_threshold}): {high_similarity_pairs}")
    
    # 警告
    if duplicate_count > 0:
        print("⚠️  警告: 发现重复序列!")
    if balance_ratio < 0.1:
        print("⚠️  警告: 类别严重不平衡!")
    if high_similarity_pairs > 10:
        print("⚠️  警告: 发现大量高相似度序列!")
    
    print("="*50)
    return {
        'duplicate_count': duplicate_count,
        'balance_ratio': balance_ratio,
        'high_similarity_pairs': high_similarity_pairs
    }

def load_and_preprocess_data(data_path):
    """加载和预处理数据（与原代码保持一致）"""
    print(f"加载数据: {data_path}")
    df = pd.read_csv(data_path)
    
    # 提取序列和标签（与原代码一致）
    sequences = df.iloc[:, 0].values  # 第一列是序列
    labels = df.iloc[:, -1].values    # 最后一列是标签
    
    # 数据清洗
    valid_indices = []
    for i, seq in enumerate(sequences):
        # 检查序列是否有效
        if isinstance(seq, str) and len(seq) > 0:
            # 检查是否只包含DNA碱基
            if all(base in 'ATGC' for base in seq.upper()):
                valid_indices.append(i)
    
    sequences = [str(sequences[i]).upper() for i in valid_indices]
    labels = labels[valid_indices]
    
    print(f"有效样本数: {len(sequences)}")
    
    return sequences, labels

def train_baseline_models(X_train, X_val, y_train, y_val, feature_names="features"):
    """训练多个基线模型"""
    print(f"\n训练基线模型 ({feature_names})...")
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Logistic Regression (Balanced)': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        'Logistic Regression (L1)': LogisticRegression(random_state=42, max_iter=1000, penalty='l1', solver='liblinear'),
        'Logistic Regression (L2 Strong)': LogisticRegression(random_state=42, max_iter=1000, C=0.1),
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n训练 {name}...")
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]  # 正类概率
        
        # 计算指标
        accuracy = accuracy_score(y_val, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='binary')
        auc = roc_auc_score(y_val, y_prob)
        auprc = average_precision_score(y_val, y_prob)
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'auprc': auprc
        }
        
        print(f"  准确率: {accuracy:.4f}")
        print(f"  精确率: {precision:.4f}")
        print(f"  召回率: {recall:.4f}")
        print(f"  F1分数: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")
        print(f"  AUPRC: {auprc:.4f}")
        
        # 如果AUC或AUPRC异常高，给出警告
        if auc > 0.95 or auprc > 0.95:
            print(f"  ⚠️  警告: {name} 的 AUC({auc:.4f}) 或 AUPRC({auprc:.4f}) 异常高!")
    
    return results

# 在数据划分后添加相似性检查
def check_set_similarity(train_seq, val_seq, threshold=0.95):
    """检查训练集和验证集之间的序列相似性"""
    from difflib import SequenceMatcher
    similar_count = 0
    
    print("\n检查训练集-验证集序列相似性...")
    for i, train_seq_i in enumerate(train_seq):
        for j, val_seq_j in enumerate(val_seq):
            if SequenceMatcher(None, train_seq_i, val_seq_j).ratio() > threshold:
                similar_count += 1
                if similar_count <= 5:  # 打印前5个例子
                    print(f"  高相似序列对: 训练[{i}] vs 验证[{j}]")
    
    print(f"相似度>{threshold}的序列对数: {similar_count}")
    print(f"占总验证集比例: {similar_count/len(val_seq):.2%}")
    return similar_count



def main():
    # 配置
    data_path = 'data/c1.csv'  # 请替换为你的数据路径
    test_size = 0.2
    random_state = 42
    
    print("DNA保守性预测 - 基线模型测试")
    print("="*60)
    
    # 加载数据
    sequences, labels = load_and_preprocess_data(data_path)
    
    # 数据质量检查
    quality_info = check_data_quality(sequences, labels)
    
    # 数据划分（与原代码一致）
    train_seq, val_seq, train_labels, val_labels = train_test_split(
        sequences, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )
    
    print(f"\n数据划分:")
    print(f"训练集: {len(train_seq)} 样本")
    print(f"验证集: {len(val_seq)} 样本")
    print(f"训练集正样本比例: {np.mean(train_labels):.3f}")
    print(f"验证集正样本比例: {np.mean(val_labels):.3f}")
    
    # # 在数据划分后调用
    # similar_count = check_set_similarity(train_seq, val_seq)
    
    # 1. 基础序列特征
    print("\n" + "="*60)
    print("测试1: 基础序列特征")
    print("="*60)
    
    basic_train_features = extract_basic_sequence_features(train_seq)
    basic_val_features = extract_basic_sequence_features(val_seq)
    
    # 标准化
    scaler_basic = StandardScaler()
    basic_train_features_scaled = scaler_basic.fit_transform(basic_train_features)
    basic_val_features_scaled = scaler_basic.transform(basic_val_features)
    
    basic_results = train_baseline_models(
        basic_train_features_scaled, basic_val_features_scaled,
        train_labels, val_labels, "基础序列特征"
    )
    
    # 2. 3-mer特征
    print("\n" + "="*60)
    print("测试2: 3-mer特征")
    print("="*60)
    
    kmer3_train_features, kmer3_names = extract_kmer_features(train_seq, k=3)
    kmer3_val_features, _ = extract_kmer_features(val_seq, k=3)
    
    # 标准化
    scaler_kmer3 = StandardScaler()
    kmer3_train_features_scaled = scaler_kmer3.fit_transform(kmer3_train_features)
    kmer3_val_features_scaled = scaler_kmer3.transform(kmer3_val_features)
    
    kmer3_results = train_baseline_models(
        kmer3_train_features_scaled, kmer3_val_features_scaled,
        train_labels, val_labels, "3-mer特征"
    )
    
    # 3. 4-mer特征
    print("\n" + "="*60)
    print("测试3: 4-mer特征")
    print("="*60)
    
    kmer4_train_features, kmer4_names = extract_kmer_features(train_seq, k=4)
    kmer4_val_features, _ = extract_kmer_features(val_seq, k=4)
    
    # 标准化
    scaler_kmer4 = StandardScaler()
    kmer4_train_features_scaled = scaler_kmer4.fit_transform(kmer4_train_features)
    kmer4_val_features_scaled = scaler_kmer4.transform(kmer4_val_features)
    
    kmer4_results = train_baseline_models(
        kmer4_train_features_scaled, kmer4_val_features_scaled,
        train_labels, val_labels, "4-mer特征"
    )
    
    # 4. 组合特征
    print("\n" + "="*60)
    print("测试4: 组合特征 (基础 + 3-mer)")
    print("="*60)
    
    combined_train_features = np.hstack([basic_train_features_scaled, kmer3_train_features_scaled])
    combined_val_features = np.hstack([basic_val_features_scaled, kmer3_val_features_scaled])
    
    combined_results = train_baseline_models(
        combined_train_features, combined_val_features,
        train_labels, val_labels, "组合特征"
    )
    
    # 汇总结果
    print("\n" + "="*80)
    print("基线模型结果汇总")
    print("="*80)
    
    all_results = {
        "基础特征": basic_results,
        "3-mer特征": kmer3_results,
        "4-mer特征": kmer4_results,
        "组合特征": combined_results
    }
    
    # 找出最高的AUC和AUPRC
    max_auc = 0
    max_auprc = 0
    
    for feature_type, results in all_results.items():
        print(f"\n{feature_type}:")
        for model_name, metrics in results.items():
            auc = metrics['auc']
            auprc = metrics['auprc']
            print(f"  {model_name}: AUC={auc:.4f}, AUPRC={auprc:.4f}")
            max_auc = max(max_auc, auc)
            max_auprc = max(max_auprc, auprc)
    
    # 最终判断
    print("\n" + "="*80)
    print("数据异常检测结果")
    print("="*80)
    
    print(f"最高 AUC: {max_auc:.4f}")
    print(f"最高 AUPRC: {max_auprc:.4f}")
    
    # 判断标准
    suspicious_auc = max_auc > 0.90
    suspicious_auprc = max_auprc > 0.90
    
    if suspicious_auc or suspicious_auprc:
        print("\n🚨 数据异常警告!")
        print("简单的线性模型就能达到很高的性能，这通常表明:")
        print("1. 数据存在泄露（重复序列、相似序列等）")
        print("2. 任务可能比预期简单")
        print("3. 标签质量可能有问题")
        print("\n建议:")
        print("- 检查数据中是否有重复或高度相似的序列")
        print("- 重新审视数据收集和标注过程")
        print("- 使用更严格的交叉验证方法")
        print("- 考虑使用独立的测试集")
    else:
        print("\n✅ 基线结果正常")
        print("简单模型的性能在合理范围内，DNABERT2的高性能可能是真实的。")
    
    # 数据质量总结
    print(f"\n数据质量问题总结:")
    print(f"- 重复序列数: {quality_info['duplicate_count']}")
    print(f"- 类别平衡比: {quality_info['balance_ratio']:.3f}")
    print(f"- 高相似序列对: {quality_info['high_similarity_pairs']}")

if __name__ == "__main__":
    main()