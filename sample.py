import pandas as pd
import os

# --- 配置参数 ---
# 请在这里修改为你实际的文件路径和参数

# 输入文件
FILE1_PATH = "data/negative_all.csv"  # 替换为第一个CSV文件的路径
FILE2_PATH = "data/positive_all.csv" # 替换为第二个CSV文件的路径

# 输出文件
OUTPUT_FILE_PATH = "data/c.csv" # 替换为希望保存的输出文件路径

# 从每个文件抽取的行数
NUM_SAMPLES_FILE1 = 500  # 从第一个文件抽取的行数
NUM_SAMPLES_FILE2 = 500  # 从第二个文件抽取的行数

# 随机种子 (用于可复现的抽样和打乱顺序，如果不需要可设为None)
RANDOM_SEED = 42
# --- 配置结束 ---

def sample_and_merge_csvs(file1_path, num_samples1,
                          file2_path, num_samples2,
                          output_path, random_seed=None):
    """
    从两个CSV文件中分别随机抽取行，合并它们，打乱顺序，并保存到新的CSV文件。
    """
    sampled_dfs = [] # 用于存放从各个文件抽样得到的DataFrame

    # 处理第一个文件
    print(f"--- 处理文件 1: {file1_path} ---")
    try:
        df1 = pd.read_csv(file1_path)
        print(f"成功读取文件1，包含 {len(df1)} 行。")

        if len(df1) == 0:
            print(f"警告: 文件1 '{file1_path}' 为空或只有表头，不进行抽样。")
        elif num_samples1 <= 0:
            print(f"警告: 文件1 的抽样数量 ({num_samples1}) 无效，不进行抽样。")
        else:
            actual_samples1 = min(num_samples1, len(df1))
            if actual_samples1 < num_samples1:
                print(f"警告: 文件1 请求抽样 {num_samples1} 行，但只有 {len(df1)} 行可用。将抽取 {actual_samples1} 行。")
            
            sample1 = df1.sample(n=actual_samples1, random_state=random_seed)
            sampled_dfs.append(sample1)
            print(f"从文件1中抽取了 {len(sample1)} 行。")

    except FileNotFoundError:
        print(f"错误: 文件1 '{file1_path}' 未找到。")
    except pd.errors.EmptyDataError:
         print(f"错误: 文件1 '{file1_path}' 为空，无法读取。")
    except Exception as e:
        print(f"处理文件1 '{file1_path}' 时发生错误: {e}")

    # 处理第二个文件
    print(f"\n--- 处理文件 2: {file2_path} ---")
    try:
        df2 = pd.read_csv(file2_path)
        print(f"成功读取文件2，包含 {len(df2)} 行。")

        if len(df2) == 0:
            print(f"警告: 文件2 '{file2_path}' 为空或只有表头，不进行抽样。")
        elif num_samples2 <= 0:
            print(f"警告: 文件2 的抽样数量 ({num_samples2}) 无效，不进行抽样。")
        else:
            actual_samples2 = min(num_samples2, len(df2))
            if actual_samples2 < num_samples2:
                print(f"警告: 文件2 请求抽样 {num_samples2} 行，但只有 {len(df2)} 行可用。将抽取 {actual_samples2} 行。")

            sample2 = df2.sample(n=actual_samples2, random_state=random_seed)
            sampled_dfs.append(sample2)
            print(f"从文件2中抽取了 {len(sample2)} 行。")

    except FileNotFoundError:
        print(f"错误: 文件2 '{file2_path}' 未找到。")
    except pd.errors.EmptyDataError:
         print(f"错误: 文件2 '{file2_path}' 为空，无法读取。")
    except Exception as e:
        print(f"处理文件2 '{file2_path}' 时发生错误: {e}")

    # 合并和打乱
    if not sampled_dfs:
        print("\n没有从任何文件中抽取到数据，无法生成输出文件。")
        return

    print("\n--- 合并和打乱抽样数据 ---")
    # pd.concat 会处理列名可能不完全一致的情况 (取并集，缺失处为NaN)
    # 如果两个CSV文件的列名和顺序不一致，这里需要特别注意
    # 理想情况下，两个CSV应有兼容的列结构
    merged_df = pd.concat(sampled_dfs, ignore_index=True)
    
    if merged_df.empty:
        print("合并后的DataFrame为空，不生成输出文件。")
        return

    # 检查列是否一致 (可选，但推荐)
    if len(sampled_dfs) == 2: # 只有当两个文件都成功抽样了才比较
        cols1 = set(sampled_dfs[0].columns)
        cols2 = set(sampled_dfs[1].columns)
        if cols1 != cols2:
            print("警告: 两个输入文件的列名不完全一致。合并后的文件将包含所有列，缺失值将以NaN填充。")
            print(f"文件1独有的列: {cols1 - cols2}")
            print(f"文件2独有的列: {cols2 - cols1}")


    # 打乱合并后的数据
    # frac=1 表示抽取所有行 (即打乱)
    # reset_index(drop=True) 重新生成索引
    shuffled_df = merged_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    print(f"合并并打乱后共得到 {len(shuffled_df)} 行。")

    # 保存到输出文件
    try:
        # 确保输出目录存在 (如果需要)
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"已创建输出目录: {output_dir}")

        shuffled_df.to_csv(output_path, index=False)
        print(f"\n成功! 合并后的数据已保存到: {output_path}")
    except Exception as e:
        print(f"写入输出文件 '{output_path}' 时发生错误: {e}")

if __name__ == "__main__":
    # 检查配置的抽样数量是否为负数，虽然函数内部也会处理，但这里可以提前提示
    if NUM_SAMPLES_FILE1 < 0 or NUM_SAMPLES_FILE2 < 0 :
        print("错误：配置中的抽样数量 (NUM_SAMPLES_FILE1 或 NUM_SAMPLES_FILE2) 不能为负数。请修改配置。")
    else:
        sample_and_merge_csvs(
            FILE1_PATH, NUM_SAMPLES_FILE1,
            FILE2_PATH, NUM_SAMPLES_FILE2,
            OUTPUT_FILE_PATH, RANDOM_SEED
        )