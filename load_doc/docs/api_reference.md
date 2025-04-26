# API 参考文档

## 数据处理 (data_processor)

### 数据加载 (data_loader)

#### load_csv

```python
load_csv(file_path: str, **kwargs) -> pd.DataFrame
```

从CSV文件加载数据。

**参数**:
- `file_path`: CSV文件路径
- `**kwargs`: 传递给pd.read_csv的其他参数

**返回**:
- 包含加载数据的DataFrame

#### load_excel

```python
load_excel(file_path: str, sheet_name: Union[str, int, list, None] = 0, **kwargs) -> Union[pd.DataFrame, dict]
```

从Excel文件加载数据。

**参数**:
- `file_path`: Excel文件路径
- `sheet_name`: 要加载的工作表名称或索引，如果为None则加载所有工作表
- `**kwargs`: 传递给pd.read_excel的其他参数

**返回**:
- 包含加载数据的DataFrame或DataFrame字典（如果加载多个工作表）

#### load_database

```python
load_database(connection_string: Optional[str] = None, config_file: Optional[str] = None, query: str = "SELECT * FROM main_table") -> pd.DataFrame
```

从数据库加载数据。

**参数**:
- `connection_string`: 数据库连接字符串。如果为None，将使用config_file
- `config_file`: 数据库配置文件路径。默认为"config/database.ini"
- `query`: 要执行的SQL查询。默认选择"main_table"中的所有内容

**返回**:
- 包含查询结果的DataFrame

### 数据清洗 (data_cleaner)

#### handle_missing_values

```python
handle_missing_values(df: pd.DataFrame, strategy: str = 'drop', columns: Optional[List[str]] = None, fill_values: Optional[Dict[str, Any]] = None) -> pd.DataFrame
```

处理DataFrame中的缺失值。

**参数**:
- `df`: 输入DataFrame
- `strategy`: 处理缺失值的策略 ('drop', 'fill', 'interpolate')
- `columns`: 应用策略的列列表。如果为None，应用于所有列
- `fill_values`: 用于填充缺失值的列:值对的字典（与'fill'策略一起使用）

**返回**:
- 处理了缺失值的DataFrame

#### detect_outliers

```python
detect_outliers(df: pd.DataFrame, columns: Optional[List[str]] = None, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame
```

在DataFrame中检测异常值。

**参数**:
- `df`: 输入DataFrame
- `columns`: 检查异常值的列列表。如果为None，检查所有数值列
- `method`: 异常值检测方法 ('iqr', 'zscore')
- `threshold`: 异常值检测阈值 (IQR使用1.5，Z-score通常为3)

**返回**:
- 布尔掩码DataFrame，其中True表示异常值

#### fix_data_types

```python
fix_data_types(df: pd.DataFrame, type_dict: Dict[str, str]) -> pd.DataFrame
```

修复DataFrame中列的数据类型。

**参数**:
- `df`: 输入DataFrame
- `type_dict`: 将列名映射到目标数据类型的字典（例如，{'age': 'int', 'price': 'float', 'date': 'datetime'}）

**返回**:
- 数据类型已更正的DataFrame

### 数据转换 (data_transformer)

#### normalize

```python
normalize(df: pd.DataFrame, columns: Optional[List[str]] = None, method: str = 'z-score') -> pd.DataFrame
```

标准化DataFrame中的数值数据。

**参数**:
- `df`: 输入DataFrame
- `columns`: 要标准化的列列表。如果为None，标准化所有数值列
- `method`: 标准化方法 ('z-score', 'min-max')

**返回**:
- 包含标准化值的DataFrame

#### encode_categorical

```python
encode_categorical(df: pd.DataFrame, columns: List[str], method: str = 'one-hot', drop_first: bool = False) -> pd.DataFrame
```

编码DataFrame中的分类变量。

**参数**:
- `df`: 输入DataFrame
- `columns`: 要编码的分类列列表
- `method`: 编码方法 ('one-hot', 'label', 'ordinal')
- `drop_first`: 是否在one-hot编码中删除第一个类别

**返回**:
- 包含编码分类变量的DataFrame

#### create_time_features

```python
create_time_features(df: pd.DataFrame, date_column: str, features: List[str] = ['year', 'month', 'day', 'dayofweek']) -> pd.DataFrame
```

从日期列创建基于时间的特征。

**参数**:
- `df`: 输入DataFrame
- `date_column`: 包含日期时间值的列的名称
- `features`: 要创建的时间特征列表（选项：'year', 'month', 'day', 'hour', 'minute', 'dayofweek', 'quarter', 'weekofyear', 'dayofyear'）

**返回**:
- 包含额外基于时间的特征的DataFrame

#### reduce_dimensions

```python
reduce_dimensions(df: pd.DataFrame, columns: Optional[List[str]] = None, n_components: int = 2, method: str = 'pca') -> pd.DataFrame
```

使用PCA或其他方法降低数据维度。

**参数**:
- `df`: 输入DataFrame
- `columns`: 用于降维的列列表。如果为None，使用所有数值列
- `n_components`: 要降至的组件数量
- `method`: 降维方法 ('pca', 'tsne', etc.)

**返回**:
- 具有降低维度的DataFrame

## 分析 (analysis)

### 描述性统计 (descriptive_stats)

#### get_basic_stats

```python
get_basic_stats(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame
```

计算数值列的基本描述性统计信息。

**参数**:
- `df`: 输入DataFrame
- `columns`: 要分析的列列表。如果为None，分析所有数值列

**返回**:
- 包含每列基本统计信息的DataFrame

#### get_categorical_stats

```python
get_categorical_stats(df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]
```

计算分类列的描述性统计信息。

**参数**:
- `df`: 输入DataFrame
- `columns`: 要分析的分类列列表。如果为None，分析所有非数值列

**返回**:
- 将列名映射到具有其统计信息的DataFrame的字典

### 假设检验 (hypothesis_test)

#### t_test_one_sample

```python
t_test_one_sample(data: Union[List, np.ndarray, pd.Series], popmean: float = 0, alpha: float = 0.05) -> Dict
```

执行单样本t检验。

**参数**:
- `data`: 样本数据
- `popmean`: 要检验的预期总体均值
- `alpha`: 显著性水平

**返回**:
- 包含检验结果的字典

#### t_test_two_sample

```python
t_test_two_sample(data1: Union[List, np.ndarray, pd.Series], data2: Union[List, np.ndarray, pd.Series], equal_var: bool = True, alpha: float = 0.05) -> Dict
```

执行双样本t检验。

**参数**:
- `data1`: 第一个样本数据
- `data2`: 第二个样本数据
- `equal_var`: 是否假设等方差
- `alpha`: 显著性水平

**返回**:
- 包含检验结果的字典

### 相关性分析 (correlation_analysis)

#### calculate_correlation_matrix

```python
calculate_correlation_matrix(df: pd.DataFrame, columns: Optional[List[str]] = None, method: str = 'pearson') -> pd.DataFrame
```

计算选定列的相关矩阵。

**参数**:
- `df`: 输入DataFrame
- `columns`: 包含在相关分析中的列列表。如果为None，包括所有数值列
- `method`: 相关方法 ('pearson', 'spearman', 或 'kendall')

**返回**:
- 包含相关矩阵的DataFrame

## 可视化 (visualization)

### 图表工厂 (plot_factory)

#### PlotFactory

```python
PlotFactory(style: str = 'default', palette: str = 'viridis', figsize: Tuple[int, int] = (10, 6), dpi: int = 100)
```

用于创建各种类型图表的工厂类。

**参数**:
- `style`: 图表样式 ('default', 'ggplot', 'seaborn', etc.)
- `palette`: 图表的调色板
- `figsize`: 默认图形大小 (宽度, 高度)，单位为英寸
- `dpi`: 图形分辨率的点每英寸数

#### PlotFactory.create_plot

```python
create_plot(plot_type: str, df: pd.DataFrame, **kwargs) -> plt.Figure
```

基于指定类型创建图表。

**参数**:
- `plot_type`: 要创建的图表类型
- `df`: 包含要绘制数据的DataFrame
- `**kwargs`: 特定图表类型的其他参数

**返回**:
- Matplotlib Figure对象

#### PlotFactory.histogram

```python
histogram(df: pd.DataFrame, column: str, bins: int = 30, kde: bool = True, title: Optional[str] = None, xlabel: Optional[str] = None, ylabel: str = 'Frequency', show_stats: bool = True) -> plt.Figure
```

为数值列创建直方图。

**参数**:
- `df`: 包含数据的DataFrame
- `column`: 要绘制的列
- `bins`: 直方图的箱数
- `kde`: 是否叠加核密度估计
- `title`: 图表标题
- `xlabel`: X轴标签
- `ylabel`: Y轴标签
- `show_stats`: 是否在图表上显示基本统计信息

**返回**:
- Matplotlib Figure对象 