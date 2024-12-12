这是openrank杯2024比赛，This is AI的仓库。
这是一个简单的数据集可视化工具，使用的数据集是Seaborn自带的tips，后面可以直接更换：
# 数据分析与可视化（pandas、matplotlib、seaborn、plotly、bokeh）
## 环境要求
- Python 3.x
- pandas
- seaborn
- matplotlib
- plotly
- bokeh
- Jupyter Notebook（可选，但推荐用于交互式分析）
## 代码示例
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from bokeh.io import output_notebook, show
from bokeh.plotting import figure
import plotly.express as px

# 在Jupyter中启用内联显示
%matplotlib inline
output_notebook()

# 1. 数据加载与初步查看（使用seaborn自带tips示例数据）
tips = sns.load_dataset("tips")
print("数据集前五行：")
display(tips.head())

print("\n数据基本信息：")
display(tips.info())

# 基本统计描述
print("\n数据描述统计：")
display(tips.describe())

# 2. 数据清洗与特征处理
categorical_cols = ['sex', 'smoker', 'day', 'time']
for col in categorical_cols:
    tips[col] = tips[col].astype('category')

# 新增特征列：tip_rate = tip/total_bill（小费率）
tips['tip_rate'] = tips['tip'] / tips['total_bill']

# 3. 使用matplotlib进行基础可视化 - 账单金额分布直方图
plt.figure(figsize=(8, 6))
plt.hist(tips['total_bill'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Total Bill')
plt.xlabel('Total Bill')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.show()

# 4. 使用seaborn进行可视化

# 4.1 箱线图：不同日期下的消费金额分布
plt.figure(figsize=(8,6))
sns.boxplot(x='day', y='total_bill', data=tips, palette='Set3')
plt.title('Total Bill Distribution Across Days')
plt.show()

# 4.2 散点图 + 回归线：小费率与账单金额的关系
plt.figure(figsize=(8,6))
sns.regplot(x='total_bill', y='tip_rate', data=tips, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title('Tip Rate vs Total Bill')
plt.show()

# 4.3 热力图：变量间相关性
corr = tips[['total_bill','tip','tip_rate','size']].corr()
plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()

# 5. 使用Plotly实现交互式可视化
fig = px.scatter(
    tips, 
    x='total_bill', y='tip', 
    color='smoker', 
    symbol='sex',
    size='size', 
    title='Interactive Scatter Plot (Plotly): Total Bill vs Tip',
    hover_data=['day','time']
)
fig.show()

# 6. 使用Bokeh进行交互式可视化
p = figure(
    title="Bokeh Interactive Plot: Tip Rate vs Total Bill",
    x_axis_label='Total Bill',
    y_axis_label='Tip Rate',
    plot_width=600,
    plot_height=400,
    tools="pan,wheel_zoom,box_zoom,reset,hover"
)
p.circle(
    tips['total_bill'], 
    tips['tip_rate'], 
    size=7, 
    color="navy", 
    alpha=0.5
)
show(p)

