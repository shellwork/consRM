# 数据泄露检查代码
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def check_data_leakage(data_path):
    """检查数据集中是否存在重复样本或泄露"""
    
    print("=== 数据泄露检查 ===")
    
    # 1. 加载原始数据
    df = pd.read_csv(data_path)
    sequences = df.iloc[:, 0].values
    labels = df.iloc[:, -1].values
    
    print(f"总样本数: {len(sequences)}")
    print(f"标签分布: {np.bincount(labels)}")
    
    # 2. 检查序列重复
    unique_sequences = set(sequences)
    print(f"唯一序列数: {len(unique_sequences)}")
    print(f"重复序列数: {len(sequences) - len(unique_sequences)}")
    
    if len(unique_sequences) != len(sequences):
        print("⚠️  发现重复序列！")
        
        # 找出重复的序列
        seq_counts = {}
        for seq in sequences:
            seq_counts[seq] = seq_counts.get(seq, 0) + 1
        
        duplicates = {seq: count for seq, count in seq_counts.items() if count > 1}
        print(f"重复序列示例: {list(duplicates.items())[:5]}")
    
    # 3. 检查序列长度分布
    seq_lengths = [len(seq) for seq in sequences]
    print(f"序列长度统计:")
    print(f"  最小长度: {min(seq_lengths)}")
    print(f"  最大长度: {max(seq_lengths)}")
    print(f"  平均长度: {np.mean(seq_lengths):.2f}")
    print(f"  长度标准差: {np.std(seq_lengths):.2f}")
    
    # 4. 模拟你的数据划分
    train_seq, val_seq, train_labels, val_labels = train_test_split(
        sequences, labels,
        test_size=0.2,
        random_state=42,  # 使用相同的随机种子
        stratify=labels
    )
    
    # 5. 检查训练集和验证集之间的重叠
    train_set = set(train_seq)
    val_set = set(val_seq)
    overlap = train_set.intersection(val_set)
    
    print(f"\n=== 训练/验证集重叠检查 ===")
    print(f"训练集样本数: {len(train_seq)}")
    print(f"验证集样本数: {len(val_seq)}")
    print(f"重叠样本数: {len(overlap)}")
    
    if len(overlap) > 0:
        print("🚨 发现训练集和验证集重叠！")
        print(f"重叠样本示例: {list(overlap)[:3]}")
    else:
        print("✅ 训练集和验证集无重叠")
    
    # 6. 检查相似序列
    print(f"\n=== 序列相似性检查 ===")
    def hamming_distance(s1, s2):
        if len(s1) != len(s2):
            return float('inf')
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))
    
    # 随机选择一些验证集序列，检查与训练集的最小距离
    sample_val_seqs = np.random.choice(val_seq, min(50, len(val_seq)), replace=False)
    sample_train_seqs = np.random.choice(train_seq, min(100, len(train_seq)), replace=False)
    
    min_distances = []
    for val_seq_sample in sample_val_seqs:
        distances = [hamming_distance(val_seq_sample, train_seq_sample) 
                    for train_seq_sample in sample_train_seqs]
        min_distances.append(min(distances))
    
    print(f"验证集序列与训练集最小距离统计:")
    print(f"  平均最小距离: {np.mean(min_distances):.2f}")
    print(f"  最小距离的最小值: {min(min_distances)}")
    print(f"  距离为0的数量: {sum(1 for d in min_distances if d == 0)}")
    
    return {
        'total_samples': len(sequences),
        'unique_sequences': len(unique_sequences),
        'duplicates': len(sequences) - len(unique_sequences),
        'train_val_overlap': len(overlap),
        'min_distance_stats': {
            'mean': np.mean(min_distances),
            'min': min(min_distances),
            'zero_distance_count': sum(1 for d in min_distances if d == 0)
        }
    }

# 使用方法
if __name__ == "__main__":
    # 替换为你的数据文件路径
    results = check_data_leakage('data/c1.csv')
    print(f"\n=== 检查结果汇总 ===")
    print(results)