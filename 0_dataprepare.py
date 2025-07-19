import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
import numpy as np
import os

# === 配置路径 ===

name = "xunwu3"  # 修改为你的实际名称
shp_path = rf"E:\06文档相关\李媛论文资料\3_论文处理过程\2_结果\2020_result_RF\X_136\{name}\{name}_net5m_label_Clip.shp"
pred_csv_path = rf"E:\01项目相关\2023\赣南稀土矿\code_gannan_ree\prediction\{name}_5m.csv"

# 读取shp和csv
gdf = gpd.read_file(shp_path)
# 读取预测结果csv
pred_df = pd.read_csv(pred_csv_path)

# 确保ID列为字符串类型，便于合并
gdf['Id'] = gdf['Id'].astype(str)
pred_df['FID'] = pred_df['FID'].astype(str)

shp_id_col = 'Id'
csv_id_col = 'FID'

# 检查ID列是否存在
if shp_id_col not in gdf.columns:
    raise ValueError(f"Shapefile中未找到预期的ID列，请检查实际列名：{list(gdf.columns)}")
if csv_id_col not in pred_df.columns:
    raise ValueError(f"CSV中未找到预期的ID列，请检查实际列名：{list(pred_df.columns)}")

print(f"Shapefile使用字段 '{shp_id_col}', CSV使用字段 '{csv_id_col}' 进行合并。")

# 只保留shp的Id和geometry列
gdf_reduced = gdf[['Id', 'geometry']]

# 合并属性表，只保留shp的Id和geometry，csv的所有列
merged_gdf = gdf_reduced.merge(pred_df, left_on='Id', right_on=csv_id_col, how='left')
# 去掉最后的12列
if merged_gdf.shape[1] > 12:
    merged_gdf = merged_gdf.iloc[:, :-12]

features = [
    'VV', 'RVI4SI', 'R3', 'MTCI', 'MCARI1', 'DVIre', 'NDVIre2n',
    'NDVIre3n', 'MTVI', 'IREC', 'MSI', 'TNDV', 'TCARI_OSAV',
    'MCARI_OSAV', 'RDVI_3', 'H_c'
]

# 输出shp路径
output_shp_dir = os.path.join("data", "data4predictions")
os.makedirs(output_shp_dir, exist_ok=True)
output_shp_path = os.path.join(output_shp_dir, os.path.basename(shp_path))
print(output_shp_path)
# 只保留FID, geometry 和 features
# cols_to_save = ['FID', 'geometry'] + [f for f in features if f in merged_gdf.columns]
merged_gdf.to_file(output_shp_path, encoding='utf-8')


