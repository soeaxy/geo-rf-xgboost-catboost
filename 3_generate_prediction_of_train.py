import os
import shutil
import geopandas as gpd
import numpy as np
from joblib import load
import matplotlib as mpl
import matplotlib.pyplot as plt

# === 1. 配置路径与参数 ===
INPUT_SHP_PATH = rf"E:\01项目相关\2023\赣南稀土矿\code_gannan_ree\shp\2020_XT.shp"

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
print(f"Input file copied to: {dst_path}")

OUTPUT_SHP_PATH = rf"results\prediction_shp\train_pred.shp"
os.makedirs(os.path.dirname(OUTPUT_SHP_PATH), exist_ok=True)
MODEL_DIR = r"results\models"  # 需与训练阶段一致

ELEMENTS = ['Sn', 'Ta']  # 预测元素
FEATURES = [
    'VV', 'RVI4SI', 'R3', 'MTCI', 'MCARI1', 'DVIre', 'NDVIre2n',
    'NDVIre3n', 'MTVI', 'IREC', 'MSI', 'TNDV', 'TCARI_OSAV',
    'MCARI_OSAV', 'RDVI_3', 'H_c'
]
MODEL_TYPES = ['geo_rf', 'geo_xgb', 'geo_cat', 'rf', 'xgb', 'cat']

# === 2. 递归空间分区模型预测函数 ===


def predict_geo_model(model_dict, X, coords):
    """
    递归空间分区模型预测
    :param model_dict: dict, 递归模型结构
    :param X: ndarray, 特征数组
    :param coords: ndarray, 空间坐标
    :return: ndarray, 预测结果
    """
    if 'model' in model_dict:
        return model_dict['model'].predict(X)
    partitions = model_dict['kmeans'].predict(coords)
    predictions = np.zeros(X.shape[0])
    for partition_label in np.unique(partitions):
        idx = partitions == partition_label
        X_sub, coords_sub = X[idx], coords[idx]
        predictions[idx] = predict_geo_model(
            model_dict['partitions'][partition_label], X_sub, coords_sub)
    return predictions


# === 3. 加载数据和预测 ===
gdf = gpd.read_file(dst_path)
coords = np.array(list(zip(gdf.geometry.x, gdf.geometry.y)))
features = gdf[FEATURES].values

for elem in ELEMENTS:
    for model_type in MODEL_TYPES:
        model_path = os.path.join(
            MODEL_DIR, f'{elem}_{model_type}_model.joblib')
        col_name = f'{elem}_{model_type}'  # 新增字段名
        if not os.path.exists(model_path):
            print(f"未找到模型文件: {model_path}, 跳过")
            continue
        print(f'加载模型: {model_path}')
        model = load(model_path)
        try:
            if model_type.startswith('geo_'):
                y_pred = predict_geo_model(model, features, coords)
            else:
                y_pred = model.predict(features)
            gdf[col_name] = y_pred
        except Exception as e:
            print(f"模型 {col_name} 预测失败: {e}")

# === 4. 输出预测结果为shp文件 ===
gdf.to_file(OUTPUT_SHP_PATH, encoding='utf-8')
print(f"\n预测完成，结果已保存到: {OUTPUT_SHP_PATH}")

# === 5. 绘制真实值 vs 预测值的散点图 ===
mpl.rcParams['font.family'] = 'Times New Roman'

for elem in ELEMENTS:
    plt.figure(figsize=(12, 6))
    true_col = elem
    if true_col not in gdf.columns:
        print(f"未找到目标列: {true_col}，跳过绘图")
        continue
    for model_type in MODEL_TYPES:
        pred_col = f"{elem}_{model_type}"
        if pred_col not in gdf.columns:
            continue
        plt.scatter(gdf[true_col], gdf[pred_col],
                    label=model_type, alpha=0.6, s=20)
    plt.xlabel(f"True {elem}")
    plt.ylabel(f"Predicted {elem}")
    plt.title(f"{elem} True Data vs Prediction Data")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
