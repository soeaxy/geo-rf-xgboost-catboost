import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error, r2_score
import geopandas as gpd
# ---- 1. 指标计算函数 ----


def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # R2
    r2 = r2_score(y_true, y_pred)
    # RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # Bias
    bias = np.mean(y_pred - y_true)
    # RPD
    rpd = np.std(y_true) / rmse if rmse != 0 else np.nan
    # Lin's Concordance Correlation Coefficient (LCCC)
    cor = np.corrcoef(y_true, y_pred)[0, 1]
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    lccc = (2 * cor * np.std(y_true) * np.std(y_pred)) / \
        (var_true + var_pred + (mean_true - mean_pred) ** 2)
    return r2, rmse, rpd, lccc, bias

# ---- 2. 论文风格高质量散点图函数 ----


def plot_regression_scatter(y_true, y_pred, xlabel, ylabel, title, save_path=None):
    # 指标
    r2, rmse, rpd, lccc, bias = calculate_metrics(y_true, y_pred)
    # 拟合直线
    slope, intercept, _, _, _ = linregress(y_true, y_pred)
    fit_line = slope * np.array(y_true) + intercept

    # 画图
    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=y_true, y=y_pred, color='salmon',
                    edgecolor=None, s=32, alpha=0.7)
    plt.plot(y_true, fit_line, color='red', lw=2, label='Model trend')
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val],
             'k--', lw=1.5, label='1:1 line')

    # 指标文本
    metrics_text = (
        r"$R^2$ = {:.2f}".format(r2) + '\n' +
        "RPD = {:.2f}".format(rpd) + '\n' +
        "RMSE = {:.2f}".format(rmse) + '\n' +
        "LCCC = {:.2f}".format(lccc) + '\n' +
        "bias = {:.2f}".format(bias)
    )
    plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
             fontsize=13, va='top', ha='left', bbox=dict(facecolor='white', edgecolor='gray', alpha=0.6))

    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=15)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# ---- 3. 批量循环出图（gdf, 元素与模型列表） ----


def batch_regression_plots(gdf, elements, model_types, save_dir='figures'):
    import os
    os.makedirs(save_dir, exist_ok=True)
    for elem in elements:
        if elem not in gdf.columns:
            print(f"{elem} not found in columns, skip.")
            continue
        for model_type in model_types:
            pred_col = f"{elem}_{model_type}"
            if pred_col not in gdf.columns:
                continue
            plot_regression_scatter(
                y_true=gdf[elem],
                y_pred=gdf[pred_col],
                xlabel=f"Measured {elem}",
                ylabel=f"Predicted {elem}",
                title=f"Measured vs Predicted {elem} ({model_type.upper()})",
                save_path=os.path.join(
                    save_dir, f"{elem}_{model_type}_regression.png")
            )

# ----------------- 使用示例 -----------------
# 假设你的gdf已经有真实值和预测值字段，且
elements = ['Sn', 'Ta']
model_types = ['rf', 'xgb', 'cat', 'geo_rf', 'geo_xgb', 'geo_cat']

# === 1. 配置路径与参数 ===
INPUT_SHP_PATH = rf"results\prediction_shp\train_pred.shp"

# 复制shp及其所有依赖文件到目标文件夹
dst_dir = r"data\data4prediction"
os.makedirs(dst_dir, exist_ok=True)
base, ext = os.path.splitext(INPUT_SHP_PATH)
for suffix in [
    '.shp', '.shx', '.dbf', '.prj', '.cpg', '.qix', '.fix', '.sbn',
    '.sbx', '.aih', '.ain', '.atx', '.ixs', '.mxs', '.xml'
]:
    src_file = base + suffix
    if os.path.exists(src_file):
        shutil.copy(src_file, os.path.join(
            dst_dir, os.path.basename(src_file)))
dst_path = os.path.join(dst_dir, os.path.basename(INPUT_SHP_PATH))

gdf = gpd.read_file(dst_path)
batch_regression_plots(gdf, elements, model_types)

# 或单独画一组
# plot_regression_scatter(
#     y_true=gdf['Sn'],
#     y_pred=gdf['Sn_rf'],
#     xlabel='Measured Sn',
#     ylabel='Predicted Sn',
#     title='Measured vs Predicted Sn (RF Model)'
# )
