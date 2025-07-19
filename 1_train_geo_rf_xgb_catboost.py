import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from joblib import dump
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
import shutil
import warnings
import os
from joblib import load
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.rcParams['font.family'] = 'Times New Roman'
# ========== 配置 ==========
# 需要分析的元素（可自行添加/修改）
ELEMENTS = ['Sn', 'Ta']  # 例如 ['Sn', 'Ta']

# 输入数据文件
INPUT_SHP = r"E:\01项目相关\2023\赣南稀土矿\code_gannan_ree\shp\2020_XT.shp"
INPUT_COPY_DIR = 'data/input'
os.makedirs(INPUT_COPY_DIR, exist_ok=True)
shp_name = os.path.basename(INPUT_SHP)
shp_base = os.path.splitext(INPUT_SHP)[0]
# 自动查找所有与shp同名的文件（包括shp.xml等扩展名）
for file in os.listdir(os.path.dirname(INPUT_SHP)):
    if os.path.splitext(file)[0] == os.path.splitext(shp_name)[0]:
        src_file = os.path.join(os.path.dirname(INPUT_SHP), file)
        dst_file = os.path.join(INPUT_COPY_DIR, file)
        if not os.path.exists(dst_file):
            shutil.copy(src_file, dst_file)
shp_copy_path = os.path.join(INPUT_COPY_DIR, shp_name)

# ========== 模型与指标文件夹 ==========
MODEL_DIR = 'results/models'
METRIC_DIR = 'results/metrics'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRIC_DIR, exist_ok=True)

# ========== 模型选择与参数 ==========


def train_local_model(X, y, model_type='rf'):
    if model_type == 'rf':
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        model = RandomForestRegressor(random_state=42)
    elif model_type == 'lgbm':
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [-1, 10, 20],
            'num_leaves': [31, 50],
            'min_child_samples': [20, 30]
        }
        model = LGBMRegressor(random_state=42, verbose=-1)
    elif model_type == 'xgb':
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 1.0]
        }
        model = XGBRegressor(random_state=42, verbosity=0)
    elif model_type == 'cat':
        param_grid = {
            'iterations': [100, 200],
            'depth': [6, 10],
            'learning_rate': [0.05, 0.1]
        }
        model = CatBoostRegressor(random_state=42, verbose=0)
    else:
        raise ValueError(
            "model_type must be 'rf', 'lgbm', 'xgb', or 'cat'")
    grid_search = GridSearchCV(
        model, param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
    grid_search.fit(X, y)
    return grid_search.best_estimator_


def spatial_partition(coords, n_partitions=2):
    kmeans = KMeans(n_clusters=n_partitions, random_state=42)
    partitions = kmeans.fit_predict(coords)
    return partitions


def geo_aware_model(X, y, coords, depth=0, max_depth=2, model_type='rf'):
    if depth >= max_depth or len(y) < 50:
        model = train_local_model(X, y, model_type)
        return {'model': model}
    partitions = spatial_partition(coords, n_partitions=2)
    partition_models = {}
    for partition_label in np.unique(partitions):
        idx = partitions == partition_label
        X_sub, y_sub, coords_sub = X[idx], y[idx], coords[idx]
        model_info = geo_aware_model(
            X_sub, y_sub, coords_sub, depth + 1, max_depth, model_type)
        partition_models[partition_label] = model_info
    return {'partitions': partition_models, 'kmeans': KMeans(n_clusters=2, random_state=42).fit(coords)}


def predict_geo_model(model_dict, X, coords):
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


# ========== 主流程 ==========
# 数据加载
data = gpd.read_file(INPUT_SHP)
features = data[['VV', 'RVI4SI', 'R3', 'MTCI', 'MCARI1', 'DVIre', 'NDVIre2n',
                 'NDVIre3n', 'MTVI', 'IREC', 'MSI', 'TNDV', 'TCARI_OSAV',
                 'MCARI_OSAV', 'RDVI_3', 'H_c', 'MRAI2']]
coords = np.array(list(zip(data.geometry.x, data.geometry.y)))

# 遍历每个元素
for elem in ELEMENTS:
    if elem not in data.columns:
        print(f"Warning: {elem} 不在数据列中，跳过。")
        continue

    print(f"\n====== 处理元素: {elem} ======")
    target = data[elem]

    X_train, X_test, y_train, y_test, coords_train, coords_test = train_test_split(
        features, target, coords, test_size=0.2, random_state=42)

    # --------- 递归空间分区模型 ----------
    for model_type in ['rf', 'xgb', 'cat']:
        print(f"\n=== 递归空间分区+{model_type.upper()} ===")
        geo_model = geo_aware_model(
            X_train.values, y_train.values, coords_train, max_depth=3, model_type=model_type)
        y_pred = predict_geo_model(geo_model, X_test.values, coords_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        print(
            f'[Geo-{model_type.upper()}] MSE: {mse:.3f}, RMSE: {rmse:.3f}, R2: {r2:.3f}, MAE: {mae:.3f}')

        # 保存模型
        model_save_path = os.path.join(
            MODEL_DIR, f'{elem}_geo_{model_type}_model.joblib')
        dump(geo_model, model_save_path)

        # 保存所有递归空间分区模型的指标到同一个txt文件
        all_metrics_path = os.path.join(METRIC_DIR, 'all_models_metrics_geo.txt')
        with open(all_metrics_path, 'a', encoding='utf-8') as f:
            f.write(f'元素: {elem}, 模型: Geo-{model_type.upper()}\n')
            f.write(f'MSE: {mse:.6f}\n')
            f.write(f'RMSE: {rmse:.6f}\n')
            f.write(f'R2: {r2:.6f}\n')
            f.write(f'MAE: {mae:.6f}\n')
            f.write('-' * 40 + '\n')

    # --------- 普通四种模型 ----------
    for model_type in ['rf', 'xgb', 'cat']:
        print(f"\n=== 普通{model_type.upper()} ===")
        model = train_local_model(X_train, y_train, model_type)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        print(
            f'[Base-{model_type.upper()}] MSE: {mse:.3f}, RMSE: {rmse:.3f}, R2: {r2:.3f}, MAE: {mae:.3f}')

        # 保存模型
        model_save_path = os.path.join(
            MODEL_DIR, f'{elem}_{model_type}_model.joblib')
        dump(model, model_save_path)

        # 保存所有模型的指标到同一个txt文件
        all_metrics_path = os.path.join(METRIC_DIR, 'all_models_metrics.txt')
        with open(all_metrics_path, 'a', encoding='utf-8') as f:
            f.write(f'元素: {elem}, 模型: {model_type.upper()}\n')
            f.write(f'MSE: {mse:.6f}\n')
            f.write(f'RMSE: {rmse:.6f}\n')
            f.write(f'R2: {r2:.6f}\n')
            f.write(f'MAE: {mae:.6f}\n')
            f.write('-' * 40 + '\n')

print("\n全部元素处理完毕。")


FIGURE_DIR = 'figures'
os.makedirs(FIGURE_DIR, exist_ok=True)

def plot_feature_importance(importances, feature_names, title, save_path):
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.barh(range(len(importances)), importances[indices], align='center')
    plt.yticks(range(len(importances)), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.gca().invert_yaxis()  # 让重要性大的排在最上面
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.close()

for elem in ELEMENTS:
    for model_type in ['rf', 'xgb', 'cat']:
        # 普通模型
        model_path = os.path.join(MODEL_DIR, f'{elem}_{model_type}_model.joblib')
        if os.path.exists(model_path):
            model = None
            try:
                model = load(model_path)
            except Exception:
                continue
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'get_feature_importance'):
                importances = model.get_feature_importance()
            else:
                continue
            save_path = os.path.join(FIGURE_DIR, f'{elem}_{model_type}_feature_importance.png')
            plot_feature_importance(importances, features.columns,
                                    f'Feature Importance of {elem} based on {model_type.upper()} ', save_path)
        # 递归空间分区模型
        geo_model_path = os.path.join(MODEL_DIR, f'{elem}_geo_{model_type}_model.joblib')
        if os.path.exists(geo_model_path):
            geo_model = None
            try:
                geo_model = load(geo_model_path)
            except Exception:
                continue
            # 递归提取所有叶子模型的特征重要性并平均
            def collect_leaf_importances(model_dict):
                if 'model' in model_dict:
                    model = model_dict['model']
                    if hasattr(model, 'feature_importances_'):
                        return [model.feature_importances_]
                    elif hasattr(model, 'get_feature_importance'):
                        return [model.get_feature_importance()]
                    else:
                        return []
                else:
                    importances = []
                    for sub in model_dict['partitions'].values():
                        importances += collect_leaf_importances(sub)
                    return importances
            importances_list = collect_leaf_importances(geo_model)
            if importances_list:
                importances = np.mean(importances_list, axis=0)
                save_path = os.path.join(FIGURE_DIR, f'{elem}_geo_{model_type}_feature_importance.png')
                plot_feature_importance(
                    importances, features.columns, f'GEO_{model_type.upper()} based Feature Importance of {elem}', save_path)
