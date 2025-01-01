import pandas as pd
from collections import Counter

# 读取CSV文件
file_path = './paper/medcial/small.csv'  # 修改为你的实际文件路径
df = pd.read_csv(file_path)

# 按出版年份统计论文数量
year_counts = df['Publication Year'].value_counts().sort_index()

# 打印每年论文数量
for year, count in year_counts.items():
    print(f"Year {year}: {count} papers")