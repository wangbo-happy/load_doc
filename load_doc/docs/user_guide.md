# 用户指南

## 简介

欢迎使用数据处理与分析工具包。该工具包提供了一套用于数据加载、清洗、转换、分析和可视化的工具，帮助您更高效地处理数据科学工作流程。

## 安装

使用pip安装该包：

```bash
pip install -r requirements.txt
```

## 快速开始

以下是一个简单的使用示例：

```python
import pandas as pd
from src.data_processor.data_loader import load_csv
from src.data_processor.data_cleaner import handle_missing_values
from src.analysis.descriptive_stats import get_basic_stats
from src.visualization.plot_factory import PlotFactory

# 加载数据
df = load_csv('data/raw/example.csv')

# 处理缺失值
df_clean = handle_missing_values(df, strategy='fill')

# 获取基本统计信息
stats = get_basic_stats(df_clean)
print(stats)

# 创建可视化
plot_factory = PlotFactory(style='seaborn')
histogram = plot_factory.histogram(df_clean, column='age')
histogram.savefig('output/age_histogram.png')
```

## 模块概述

### 数据处理 (data_processor)

#### 数据加载 (data_loader)

提供从各种来源加载数据的功能：

- `load_csv`: 从CSV文件加载数据
- `load_excel`: 从Excel文件加载数据
- `load_database`: 从数据库加载数据

```python
from src.data_processor.data_loader import load_csv

df = load_csv('data/raw/example.csv', encoding='utf-8')
```

#### 数据清洗 (data_cleaner)

提供处理缺失值和异常值的功能：

- `handle_missing_values`: 处理缺失值（删除、填充、插值）
- `detect_outliers`: 检测异常值
- `fix_data_types`: 修复数据类型问题

```python
from src.data_processor.data_cleaner import handle_missing_values, detect_outliers

# 填充缺失值
df_clean = handle_missing_values(df, strategy='fill')

# 检测异常值
outliers = detect_outliers(df_clean, method='iqr')
```

#### 数据转换 (data_transformer)

提供数据标准化和特征工程功能：

- `normalize`: 数据标准化（Z-score、Min-Max等）
- `encode_categorical`: 编码分类变量
- `create_time_features`: 从日期字段创建时间特征
- `reduce_dimensions`: 降维处理

```python
from src.data_processor.data_transformer import normalize, encode_categorical

# 标准化数值列
df_norm = normalize(df, method='z-score')

# 对分类变量进行独热编码
df_encoded = encode_categorical(df, columns=['category'], method='one-hot')
```

### 分析 (analysis)

#### 描述性统计 (descriptive_stats)

提供基本统计分析功能：

- `get_basic_stats`: 计算基本统计量
- `get_categorical_stats`: 对分类变量进行统计
- `check_normality`: 检查正态性
- `get_quantiles`: 计算分位数

```python
from src.analysis.descriptive_stats import get_basic_stats, get_categorical_stats

# 获取数值变量的统计信息
stats = get_basic_stats(df)

# 获取分类变量的统计信息
cat_stats = get_categorical_stats(df, columns=['category'])
```

#### 假设检验 (hypothesis_test)

提供统计假设检验功能：

- `t_test_one_sample`: 单样本t检验
- `t_test_two_sample`: 双样本t检验
- `paired_t_test`: 配对t检验
- `one_way_anova`: 单因素方差分析
- `chi_square_test`: 卡方检验
- `mann_whitney_test`: Mann-Whitney U检验

```python
from src.analysis.hypothesis_test import t_test_two_sample, chi_square_test

# 进行双样本t检验
result = t_test_two_sample(group1_data, group2_data)
print(f"P值: {result['p_value']}, 是否拒绝原假设: {result['reject_null']}")

# 进行卡方检验
chi_result = chi_square_test(observed_freq_table)
```

#### 相关性分析 (correlation_analysis)

提供变量关系分析功能：

- `calculate_correlation_matrix`: 计算相关矩阵
- `calculate_correlation_significance`: 计算相关系数的显著性
- `get_top_correlations`: 获取最强相关关系
- `point_biserial_correlation`: 点二列相关
- `calculate_vif`: 计算方差膨胀因子

```python
from src.analysis.correlation_analysis import calculate_correlation_matrix, get_top_correlations

# 计算相关矩阵
corr_matrix = calculate_correlation_matrix(df, method='pearson')

# 获取前10个最强相关关系
top_corr = get_top_correlations(df, n=10)
```

### 可视化 (visualization)

#### 图表工厂 (plot_factory)

提供统一的可视化接口：

- `histogram`: 直方图
- `density_plot`: 密度图
- `scatter_plot`: 散点图
- `correlation_matrix`: 相关矩阵热图
- `bar_plot`: 条形图
- `pie_chart`: 饼图
- `line_plot`: 折线图
- `box_plot`: 箱线图
- `violin_plot`: 小提琴图
- `heatmap`: 热图
- `pair_plot`: 配对图

```python
from src.visualization.plot_factory import PlotFactory

# 创建图表工厂
plot_factory = PlotFactory(style='seaborn', figsize=(10, 6))

# 创建直方图
hist = plot_factory.histogram(df, column='age', bins=20, kde=True)

# 创建散点图
scatter = plot_factory.scatter_plot(df, x='age', y='income', hue='gender')

# 创建相关矩阵可视化
corr_plot = plot_factory.correlation_matrix(df)

# 保存图表
plot_factory.save_plot(hist, 'age_distribution', folder='output', formats=['png', 'pdf'])
```

## 配置

可以通过编辑`config/config.yaml`文件来配置工具包的行为：

```yaml
# 示例配置
data:
  default_encoding: 'utf-8'
  missing_values: ['NA', 'N/A', '', 'null']

visualization:
  default_style: 'seaborn'
  default_palette: 'viridis'
  default_dpi: 300
```

## 进阶使用

### 自定义数据处理流程

您可以创建自己的数据处理流程：

```python
from src.data_processor.data_loader import load_csv
from src.data_processor.data_cleaner import handle_missing_values, detect_outliers, fix_data_types
from src.data_processor.data_transformer import normalize, encode_categorical

def my_preprocessing_pipeline(file_path):
    # 加载数据
    df = load_csv(file_path)
    
    # 处理缺失值
    df = handle_missing_values(df, strategy='fill')
    
    # 检测并移除异常值
    outliers = detect_outliers(df)
    for col in outliers.columns:
        df = df[~outliers[col]]
    
    # 修复数据类型
    type_dict = {'age': 'int', 'income': 'float', 'date': 'datetime'}
    df = fix_data_types(df, type_dict)
    
    # 标准化数值特征
    df = normalize(df)
    
    # 编码分类特征
    df = encode_categorical(df, columns=['category', 'region'])
    
    return df

# 使用自定义流程
df_processed = my_preprocessing_pipeline('data/raw/example.csv')
```

### 使用可视化创建报告

创建综合数据报告：

```python
import matplotlib.pyplot as plt
from src.data_processor.data_loader import load_csv
from src.analysis.descriptive_stats import get_basic_stats
from src.analysis.correlation_analysis import calculate_correlation_matrix
from src.visualization.plot_factory import PlotFactory

def create_data_report(file_path, output_folder):
    # 加载数据
    df = load_csv(file_path)
    
    # 创建图表工厂
    plot_factory = PlotFactory(style='seaborn')
    
    # 创建基本统计数据
    stats = get_basic_stats(df)
    
    # 创建一系列可视化
    for col in df.select_dtypes(include='number').columns:
        hist = plot_factory.histogram(df, column=col)
        plot_factory.save_plot(hist, f"{col}_histogram", folder=output_folder)
    
    # 创建相关矩阵可视化
    corr_matrix = calculate_correlation_matrix(df)
    corr_plot = plot_factory.heatmap(corr_matrix, title='Correlation Matrix')
    plot_factory.save_plot(corr_plot, "correlation_matrix", folder=output_folder)
    
    # 创建散点图矩阵
    pair_plot = plot_factory.pair_plot(df)
    plot_factory.save_plot(pair_plot, "pair_plot", folder=output_folder)
    
    # 保存统计数据
    stats.to_csv(f"{output_folder}/basic_stats.csv")
    
    print(f"报告已生成至 {output_folder} 文件夹")

# 创建报告
create_data_report('data/raw/example.csv', 'reports')
```

## 故障排除

### 常见问题

1. **导入错误**：
   确保已安装所有依赖项: `pip install -r requirements.txt`

2. **内存错误**：
   处理大型数据集时，考虑使用数据分块或减少内存使用：
   ```python
   df = load_csv('large_file.csv', chunksize=10000)
   ```

3. **可视化问题**：
   如果图表未显示或保存失败，检查matplotlib后端设置，尤其是在无头服务器上：
   ```python
   import matplotlib
   matplotlib.use('Agg')  # 使用非交互式后端
   ```

## 获取帮助

如果您在使用过程中遇到任何问题，请查阅API参考文档或联系开发团队获取支持。 