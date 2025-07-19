import os
import shutil
import geopandas as gpd
import numpy as np
from joblib import load
from rasterio.transform import from_origin
import rasterio

# === 配置路径 ===
TARGET = 'xinfeng2'  # 修改为你的实际名称
INPUT_SHP_PATH = rf"D:\Work\code\geo-rf-xgboost-catboost\data\data4predictions\{TARGET}_net5m_label_Clip.shp"

# 复制输入文件到 data/data4prediction 目录
dst_dir = r"data\data4prediction"
os.makedirs(dst_dir, exist_ok=True)
dst_path = os.path.join(dst_dir, os.path.basename(INPUT_SHP_PATH))
# 复制shp及其所有关联文件（如shx, dbf, prj等）
base, ext = os.path.splitext(INPUT_SHP_PATH)
for suffix in ['.shp', '.shx', '.dbf', '.prj', '.cpg', '.qix', '.fix', '.sbn', '.sbx', '.aih', '.ain', '.atx', '.ixs', '.mxs', '.xml']:
    src_file = base + suffix
    if os.path.exists(src_file):
        shutil.copy(src_file, os.path.join(dst_dir, os.path.basename(src_file)))
print(f"Input file copied to: {dst_path}")


OUTPUT_SHP_PATH = rf"results\prediction\{TARGET}_pred.shp"
os.makedirs(os.path.dirname(OUTPUT_SHP_PATH), exist_ok=True)
MODEL_DIR = r"results\models"  # 与训练阶段保持一致

# ====== 配置模型与特征 ======
ELEMENTS = ['Sn', 'Ta']  # 可按你的需求修改
FEATURES = [
    'VV', 'RVI4SI', 'R3', 'MTCI', 'MCARI1', 'DVIre', 'NDVIre2n',
    'NDVIre3n', 'MTVI', 'IREC', 'MSI', 'TNDV', 'TCARI_OSAV',
    'MCARI_OSAV', 'RDVI_3', 'H_c'
]
MODEL_TYPES = ['geo_rf', 'geo_xgb', 'geo_cat', 'rf', 'xgb', 'cat']


def predict_geo_model(model_dict, X, coords):
    """递归空间分区模型预测。"""
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


# ====== 主流程 ======
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
        if model_type.startswith('geo_'):
            y_pred = predict_geo_model(model, features, coords)
        else:
            y_pred = model.predict(features)
        gdf[col_name] = y_pred
        # 保存为tif

        # 假设数据为规则网格，提取分辨率和范围
        xs = np.sort(np.unique(gdf.geometry.x))
        ys = np.sort(np.unique(gdf.geometry.y))
        pixel_size_x = np.min(np.diff(xs)) if len(xs) > 1 else 1
        pixel_size_y = np.min(np.diff(ys)) if len(ys) > 1 else 1
        width = len(xs)
        height = len(ys)
        # 构建二维数组
        arr = np.full((height, width), np.nan, dtype=np.float32)
        x_to_idx = {x: i for i, x in enumerate(xs)}
        y_to_idx = {y: i for i, y in enumerate(ys)}
        for i, geom in enumerate(gdf.geometry):
            x, y = geom.x, geom.y
            xi = x_to_idx.get(x)
            yi = y_to_idx.get(y)
            if xi is not None and yi is not None:
                arr[height - 1 - yi, xi] = y_pred[i]  # y轴反向
        # 设置地理变换
        transform = from_origin(xs[0] - pixel_size_x / 2, ys[-1] + pixel_size_y / 2, pixel_size_x, pixel_size_y)
        tif_dir = os.path.join("results", "prediction_tif")
        os.makedirs(tif_dir, exist_ok=True)
        tif_path = os.path.join(tif_dir, f"{TARGET}_{elem}_{model_type}.tif")
        with rasterio.open(
            tif_path,
            'w',
            driver='GTiff',
            height=arr.shape[0],
            width=arr.shape[1],
            count=1,
            dtype=arr.dtype,
            crs=gdf.crs,
            transform=transform,
            nodata=np.nan
        ) as dst:
            dst.write(arr, 1)
        print(f"已保存tif: {tif_path}")

# 输出为新的shp文件
gdf.to_file(OUTPUT_SHP_PATH, encoding='utf-8')
print(f"\n预测完成，结果已保存到: {OUTPUT_SHP_PATH}")
