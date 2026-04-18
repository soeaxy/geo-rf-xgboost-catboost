# geo-rf-xgboost-catboost

[English](./README.md) | [简体中文](./README.zh-CN.md)

一个基于树模型的地球化学回归工作流，支持可选的空间分区建模。

## 当前能力

这个仓库现在可以直接使用随仓附带的 `Sn.xlsx` 和 `Ta.xlsx` 作为示例数据运行完整流程：

- 不再依赖本地硬编码绝对路径
- 使用统一的公共流水线模块处理数据读取、训练、预测和指标输出
- 仅当数据中存在坐标列时才启用 Geo 空间模型
- 示例数据默认输出为 CSV；如果安装了 `geopandas`，仍可保留 shapefile 预测导出能力
- 绘图脚本读取真实生成的指标和预测结果，不再依赖写死的示例值

## 快速开始

1. 安装基础依赖：

```bash
pip install -r requirements.txt
```

2. 生成仓库内置示例数据集：

```bash
python 0_dataprepare.py
```

3. 训练当前环境中可用的模型：

```bash
python 1_train_geo_rf_xgb_catboost.py
```

如果没有安装 `xgboost` 或 `catboost`，脚本会自动跳过对应模型，并输出跳过原因。

4. 对示例数据生成预测结果：

```bash
python 3_generate_prediction_of_train.py
```

5. 生成评估图件：

```bash
python 4_plot_scatter.py --target Sn
python 4_plot_scatter_all_in_one.py --target Ta
python 5_plot_radar.py
```

## 输入格式

当前流水线支持以下输入：

- `.xlsx` 和 `.csv`：表格型工作流
- `.shp`：空间型工作流，需安装 `geopandas`

对仓库自带示例而言，`0_dataprepare.py` 会把 `Sn.xlsx` 和 `Ta.xlsx` 合并为 `data/input/example_dataset.csv`。

## 输出结果

运行生成的结果会写入 `results/`：

- `results/models/`：训练后的模型文件
- `results/metrics/model_metrics.csv`：评估指标汇总
- `results/metadata/training_metadata.json`：用于后续预测复用的特征、目标和模型元数据
- `results/predictions/`：留出集预测结果和全量数据预测结果
- `results/feature_importance/`：特征重要性 CSV 与 PNG 图

绘图输出位于 `figures/`。
