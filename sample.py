import pandas as pd
import os

# --- 配置参数 ---
# 请在这里修改为你实际的文件路径和参数

# 输入文件
FILE1_PATH = "data/negative_all.csv"  # 替换为第一个CSV文件的路径
FILE2_PATH = "data/positive_all.csv" # 替换为第二个CSV文件的路径

# 输出文件
OUTPUT_FILE_PATH = "data/c1.csv" # 替换为希望保存的输出文件路径

# 从每个文件抽取的行数
# 设置为 -1 表示抽取该文件的所有行
# 设置为 0 表示不抽取该文件的任何行
# 设置为正数 N 表示随机抽取 N 行
NUM_SAMPLES_FILE1 = 1600   # 从第一个文件抽取的行数 (例如：抽取所有行)
NUM_SAMPLES_FILE2 = 1600 # 从第二个文件抽取的行数 (例如：抽取500行)

# 要从最终合并文件中删除的列名列表 (如果不需要删除列，请设置为空列表 [])
# COLUMNS_TO_DELETE = ['noes_score', 'pm_score', 'seq_score', 'phastCons'] # 例如: ['多余列A', '临时数据列']
COLUMNS_TO_DELETE = ['noes_score', 'pm_score']
# 随机种子 (用于可复现的抽样和打乱顺序，如果不需要可设为None)
RANDOM_SEED = 42
# --- 配置结束 ---

def sample_merge_process_csvs(file1_path, num_samples1,
                              file2_path, num_samples2,
                              output_path, columns_to_delete, random_seed=None):
    """
    从两个CSV文件中分别抽取行（可指定全部行），合并它们，打乱顺序，
    删除指定列，并保存到新的CSV文件。
    """
    sampled_dfs = [] # 用于存放从各个文件抽样得到的DataFrame

    # --- 处理第一个文件 ---
    print(f"--- 处理文件 1: {file1_path} ---")
    try:
        df1 = pd.read_csv(file1_path)
        print(f"成功读取文件1，包含 {len(df1)} 行。")

        if len(df1) == 0:
            print(f"警告: 文件1 '{file1_path}' 为空或只有表头，不进行处理。")
        elif num_samples1 == -1: # 抽取所有行
            sampled_dfs.append(df1)
            print(f"从文件1中获取了全部 {len(df1)} 行。")
        elif num_samples1 <= 0: # 抽样数量为0或无效负数
            print(f"警告: 文件1 的抽样数量 ({num_samples1}) 无效或为0，不进行抽样。")
        else: # 随机抽样指定数量的行
            actual_samples1 = min(num_samples1, len(df1))
            if actual_samples1 < num_samples1:
                print(f"警告: 文件1 请求抽样 {num_samples1} 行，但只有 {len(df1)} 行可用。将抽取 {actual_samples1} 行。")
            
            sample1 = df1.sample(n=actual_samples1, random_state=random_seed)
            sampled_dfs.append(sample1)
            print(f"从文件1中随机抽取了 {len(sample1)} 行。")

    except FileNotFoundError:
        print(f"错误: 文件1 '{file1_path}' 未找到。")
    except pd.errors.EmptyDataError:
        print(f"错误: 文件1 '{file1_path}' 为空，无法读取。")
    except Exception as e:
        print(f"处理文件1 '{file1_path}' 时发生错误: {e}")

    # --- 处理第二个文件 ---
    print(f"\n--- 处理文件 2: {file2_path} ---")
    try:
        df2 = pd.read_csv(file2_path)
        print(f"成功读取文件2，包含 {len(df2)} 行。")

        if len(df2) == 0:
            print(f"警告: 文件2 '{file2_path}' 为空或只有表头，不进行处理。")
        elif num_samples2 == -1: # 抽取所有行
            sampled_dfs.append(df2)
            print(f"从文件2中获取了全部 {len(df2)} 行。")
        elif num_samples2 <= 0: # 抽样数量为0或无效负数
            print(f"警告: 文件2 的抽样数量 ({num_samples2}) 无效或为0，不进行抽样。")
        else: # 随机抽样指定数量的行
            actual_samples2 = min(num_samples2, len(df2))
            if actual_samples2 < num_samples2:
                print(f"警告: 文件2 请求抽样 {num_samples2} 行，但只有 {len(df2)} 行可用。将抽取 {actual_samples2} 行。")

            sample2 = df2.sample(n=actual_samples2, random_state=random_seed)
            sampled_dfs.append(sample2)
            print(f"从文件2中随机抽取了 {len(sample2)} 行。")

    except FileNotFoundError:
        print(f"错误: 文件2 '{file2_path}' 未找到。")
    except pd.errors.EmptyDataError:
        print(f"错误: 文件2 '{file2_path}' 为空，无法读取。")
    except Exception as e:
        print(f"处理文件2 '{file2_path}' 时发生错误: {e}")

    # --- 合并和打乱 ---
    if not sampled_dfs:
        print("\n没有从任何文件中获取到数据，无法生成输出文件。")
        return

    print("\n--- 合并和打乱数据 ---")
    merged_df = pd.concat(sampled_dfs, ignore_index=True)
    
    if merged_df.empty:
        print("合并后的DataFrame为空，不生成输出文件。")
        return

    # 检查列是否一致 (可选，但推荐)
    if len(sampled_dfs) == 2 and sampled_dfs[0] is not None and sampled_dfs[1] is not None:
        # 确保两个df都实际存在数据
        cols1_df = sampled_dfs[0]
        cols2_df = sampled_dfs[1]
        if not cols1_df.empty and not cols2_df.empty:
            cols1 = set(cols1_df.columns)
            cols2 = set(cols2_df.columns)
            if cols1 != cols2:
                print("警告: 两个输入文件的列名不完全一致。合并后的文件将包含所有列，缺失值将以NaN填充。")
                if cols1 - cols2:
                    print(f"  文件1独有的列 (或第一个有效文件): {cols1 - cols2}")
                if cols2 - cols1:
                    print(f"  文件2独有的列 (或第二个有效文件): {cols2 - cols1}")
    elif len(sampled_dfs) == 1 and sampled_dfs[0] is not None:
        print("提示: 只有一个文件的数据被处理。")


    shuffled_df = merged_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    print(f"合并并打乱后共得到 {len(shuffled_df)} 行。")

    # --- 删除指定的列 ---
    if columns_to_delete and not shuffled_df.empty:
        print("\n--- 删除指定列 ---")
        existing_columns_in_df = [col for col in columns_to_delete if col in shuffled_df.columns]
        missing_columns_in_df = [col for col in columns_to_delete if col not in shuffled_df.columns]

        if missing_columns_in_df:
            print(f"警告：以下指定要删除的列在合并后的数据中不存在: {', '.join(missing_columns_in_df)}")

        if existing_columns_in_df:
            shuffled_df = shuffled_df.drop(columns=existing_columns_in_df, axis=1)
            print(f"已从合并后的数据中删除列: {', '.join(existing_columns_in_df)}")
        elif columns_to_delete: # 仅当用户尝试删除某些列但都未找到时提示
             print("没有有效的列从合并后的数据中被删除 (指定的列均未找到)。")
    elif columns_to_delete and shuffled_df.empty:
        print("警告: 合并后的数据为空，无法执行列删除操作。")


    # --- 保存到输出文件 ---
    try:
        # 确保输出目录存在 (如果需要)
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"已创建输出目录: {output_dir}")

        shuffled_df.to_csv(output_path, index=False, encoding='utf-8-sig') # 使用 utf-8-sig 编码以更好兼容Excel
        print(f"\n成功! 处理后的数据已保存到: {output_path}")
    except Exception as e:
        print(f"写入输出文件 '{output_path}' 时发生错误: {e}")

if __name__ == "__main__":
    # 检查配置的抽样数量是否为无效负数 (允许-1表示所有行)
    if (NUM_SAMPLES_FILE1 < -1) or \
       (NUM_SAMPLES_FILE2 < -1) :
        print("错误：配置中的抽样数量 (NUM_SAMPLES_FILE1 或 NUM_SAMPLES_FILE2) 不能为小于-1的负数。请修改配置 (-1 表示所有行)。")
    else:
        sample_merge_process_csvs(
            FILE1_PATH, NUM_SAMPLES_FILE1,
            FILE2_PATH, NUM_SAMPLES_FILE2,
            OUTPUT_FILE_PATH,
            COLUMNS_TO_DELETE, # 新增参数
            RANDOM_SEED
        )