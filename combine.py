import pandas as pd

def concatenate_csv_files(file1_path, file2_path, output_file_path):
  """
  拼接两个表头相同的 CSV 文件。

  参数:
    file1_path (str): 第一个 CSV 文件的路径。
    file2_path (str): 第二个 CSV 文件的路径。
    output_file_path (str): 合并后输出的 CSV 文件的路径。
  """
  try:
    # 读取两个 CSV 文件到 pandas DataFrame
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    # 检查表头是否相同 (可选步骤，但建议)
    if not df1.columns.equals(df2.columns):
      print("警告：两个 CSV 文件的表头不完全相同。请检查文件内容。")
      # 如果希望在表头不同时停止操作，可以取消下面一行的注释
      # return

    # 拼接两个 DataFrame
    # ignore_index=True 会重新生成索引，避免索引重复
    concatenated_df = pd.concat([df1, df2], ignore_index=True)

    # 将合并后的 DataFrame 保存到新的 CSV 文件
    concatenated_df.to_csv(output_file_path, index=False) # index=False 避免将 DataFrame 的索引写入 CSV

    print(f"文件 '{file1_path}' 和 '{file2_path}' 已成功合并到 '{output_file_path}'")

  except FileNotFoundError:
    print("错误：一个或两个 CSV 文件未找到。请检查文件路径。")
  except Exception as e:
    print(f"发生错误：{e}")

# --- 使用示例 ---
if __name__ == "__main__":
  # 假设你的 CSV 文件名为 'dataset1.csv' 和 'dataset2.csv'
  # 并且它们与你的 Python 脚本在同一个目录下

  # 定义输入文件路径
  csv_file1 = 'data/negative_all.csv'
  csv_file2 = 'data/positive_all.csv'

  # 定义输出文件路径
  output_csv_file = 'data/combine.csv'

  # 调用函数来拼接 CSV 文件
  concatenate_csv_files(csv_file1, csv_file2, output_csv_file)
