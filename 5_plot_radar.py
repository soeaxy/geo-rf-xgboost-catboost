import numpy as np
import os
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'

models = [
    "RF", "XGB", "CAT",
    "Geo-RF", "Geo-XGB", "Geo-CAT"
]
metrics_names = ['R2', 'RMSE', 'MAE']

sn_data = [
    [0.618, 7.09, 4.38],
    [0.571, 7.51, 4.85],
    [0.837, 4.64, 3.32],
    [0.707, 6.21, 3.12],
    [0.730, 5.97, 3.01],
    [0.845, 4.52, 2.61],
]
ta_data = [
    [0.676, 1.40, 0.94],
    [0.588, 1.57, 0.96],
    [0.841, 0.98, 0.67],
    [0.756, 1.21, 0.59],
    [0.733, 1.27, 0.60],
    [0.855, 0.93, 0.56],
]

def normalize(data, metric_idx):
    arr = np.array(data)[:, metric_idx]
    if metric_idx == 0:
        arr_norm = (arr - arr.min()) / (arr.max() - arr.min())
    else:
        arr_norm = (arr.max() - arr) / (arr.max() - arr.min())
    return arr_norm

sn_norm = np.array([normalize(sn_data, i) for i in range(3)]).T
ta_norm = np.array([normalize(ta_data, i) for i in range(3)]).T

def plot_radar_subplot(ax, data, title):
    N = data.shape[1]
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    for idx, (d, label) in enumerate(zip(data, models)):
        values = d.tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=label)
        ax.fill(angles, values, alpha=0.15)
    ax.set_thetagrids(np.degrees(angles[:-1]), metrics_names)
    ax.set_title(title, size=16)
    ax.set_ylim(0, 1.05)

fig, axs = plt.subplots(1, 2, figsize=(14, 7), subplot_kw=dict(polar=True))
plot_radar_subplot(axs[0], sn_norm, "Performance Comparison for Sn Prediction")
plot_radar_subplot(axs[1], ta_norm, "Performance Comparison for Ta Prediction")
axs[1].legend(loc='upper right', bbox_to_anchor=(1.35, 1.05))
plt.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig(os.path.join('figures', 'Performance_Comparison_Sn_Ta.png'), dpi=600, bbox_inches='tight')
plt.show()
