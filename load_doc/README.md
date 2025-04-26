# 数据处理与分析工具包

一个全面的数据处理、分析和可视化工具包，旨在简化数据科学工作流程。

## 功能特点

本工具包提供以下核心功能：

- **数据加载**：从CSV、Excel和数据库轻松加载数据
- **数据清洗**：处理缺失值、检测异常值、修复数据类型
- **数据转换**：标准化数值、编码分类变量、创建时间特征
- **统计分析**：描述性统计、假设检验、相关性分析
- **数据可视化**：通过统一接口创建各种专业图表

## 安装

### 先决条件

- Python 3.8+
- pip包管理器

### 安装步骤

1. 克隆仓库：

```bash
git clone https://github.com/yourusername/data_analysis_toolkit.git
cd data_analysis_toolkit
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 安装包（可选）：

```bash
pip install -e .
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
plot_factory.save_plot(histogram, 'age_distribution', folder='outputs')
```

更多示例和详细用法，请参阅 [用户指南](docs/user_guide.md)。

## 项目结构

```
data_analysis_toolkit/
├── src/                  # 源代码
│   ├── data_processor/   # 数据处理模块
│   ├── analysis/         # 统计分析模块
│   └── visualization/    # 可视化模块
├── tests/                # 单元测试
├── docs/                 # 文档
├── config/               # 配置文件
├── data/                 # 数据目录
│   ├── raw/              # 原始数据
│   └── processed/        # 处理后的数据
├── requirements.txt      # 依赖列表
├── setup.py              # 安装脚本
└── README.md             # 本文档
```

## 文档

详细文档：

- [用户指南](docs/user_guide.md)
- [API参考](docs/api_reference.md)
- [架构设计](docs/architecture.md)

## 贡献

欢迎贡献！请阅读 [贡献指南](CONTRIBUTING.md) 了解如何参与项目。

## 许可证

本项目采用 MIT 许可证 - 详情请参阅 [LICENSE](LICENSE) 文件。

## 联系方式

如有问题或建议，请通过以下方式联系：

- 电子邮件：your.email@example.com
- GitHub Issues：[提交问题](https://github.com/yourusername/data_analysis_toolkit/issues) 