import pandas as pd
import argparse
import os

def sample_csv_rows(input_file, output_file, num_samples, random_seed=None):
    """
    从CSV文件中随机抽取指定数量的行，并保存到新的CSV文件。

    参数:
    input_file (str): 输入的CSV文件路径。
    output_file (str): 输出的CSV文件路径。
    num_samples (int): 要抽取的行数。
    random_seed (int, optional): 随机数种子，用于可复现的抽样。默认为None。
    """
    try:
        # 1. 读取CSV文件
        print(f"正在读取文件: {input_file} ...")
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"错误: 输入文件 '{input_file}' 未找到。")
        return
    except pd.errors.EmptyDataError:
        print(f"错误: 输入文件 '{input_file}' 为空。")
        # 如果原始文件为空，创建一个空的输出文件（或者只含表头，如果能确定）
        # 这里我们假设如果原始文件都无法解析表头，就直接返回
        return
    except Exception as e:
        print(f"读取CSV文件 '{input_file}' 时发生错误: {e}")
        return

    # 2. 检查请求的样本数量
    if num_samples <= 0:
        print("错误: 抽样数量必须大于0。")
        return

    if len(df) == 0:
        print(f"警告: 输入文件 '{input_file}' 中没有数据行 (可能只有表头或为空)。")
        # 如果只有表头，则创建一个包含相同表头但没有数据的新CSV
        try:
            pd.DataFrame(columns=df.columns).to_csv(output_file, index=False)
            print(f"已创建空的CSV文件（仅含表头）到: {output_file}")
        except Exception as e:
            print(f"写入空的CSV文件 '{output_file}' 时发生错误: {e}")
        return


    if num_samples > len(df):
        print(f"警告: 请求的抽样数量 ({num_samples}) 大于文件中的总行数 ({len(df)})。")
        print(f"将抽取所有 {len(df)} 行。")
        sampled_df = df.copy() # 复制所有行
    else:
        # 3. 随机抽样
        # df.sample() 会随机抽取行
        # random_state 参数用于可复现性，如果提供了种子
        print(f"正在从 {len(df)} 行中随机抽取 {num_samples} 行...")
        sampled_df = df.sample(n=num_samples, random_state=random_seed)

    # 4. 保存到新的CSV文件
    try:
        print(f"正在将抽取的行保存到: {output_file} ...")
        sampled_df.to_csv(output_file, index=False) # index=False 避免写入pandas的索引
        print(f"成功! {len(sampled_df)} 行已保存到 '{output_file}'。")
    except Exception as e:
        print(f"写入CSV文件 '{output_file}' 时发生错误: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从CSV文件中随机抽取指定数量的行到新的CSV文件。")
    parser.add_argument("input_file", help="输入的CSV文件路径。")
    parser.add_argument("output_file", help="输出的CSV文件路径。")
    parser.add_argument("num_samples", type=int, help="要抽取的行数 (超参数)。")
    parser.add_argument("--seed", type=int, default=None, help="随机数种子 (可选, 用于可复现的抽样)。")

    args = parser.parse_args()

    sample_csv_rows(args.input_file, args.output_file, args.num_samples, args.seed)