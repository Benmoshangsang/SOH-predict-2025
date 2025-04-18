import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from matplotlib.colors import to_rgb

# Define file paths
file_paths = {
    'Present': r'C:\Users\LiaoGuanfu\Downloads\RUL_prediction-main (1)\RUL_prediction-main\测试效果\6.8消融实验\交叉6.30\Present.csv',
    'Without gru': r'C:\Users\LiaoGuanfu\Downloads\RUL_prediction-main (1)\RUL_prediction-main\测试效果\6.8消融实验\交叉6.30\Model without gru.csv',
    'Without cnn': r'C:\Users\LiaoGuanfu\Downloads\RUL_prediction-main (1)\RUL_prediction-main\测试效果\6.8消融实验\交叉6.30\Model without cnn.csv',
    'Without attention': r'C:\Users\LiaoGuanfu\Downloads\RUL_prediction-main (1)\RUL_prediction-main\测试效果\6.8消融实验\交叉6.30\Model without attention.csv',
    'Without dropout': r'C:\Users\LiaoGuanfu\Downloads\RUL_prediction-main (1)\RUL_prediction-main\测试效果\6.8消融实验\交叉6.30\Model without dropout.csv',
}

# Load all the CSV files into a dictionary of dataframes
dfs = {model: pd.read_csv(path) for model, path in file_paths.items()}

# Add a 'Model' column to each dataframe and concatenate them into a single dataframe
for model, df in dfs.items():
    df['Model'] = model

combined_df = pd.concat(dfs.values())

# Define function to calculate evaluation metrics
def calculate_metrics(df):
    metrics = {}
    predictions = df['Predictions']
    true_values = df['True Values']
    
    metrics['MAE'] = mean_absolute_error(true_values, predictions)
    metrics['MSE'] = mean_squared_error(true_values, predictions)
    metrics['MAPE'] = np.mean(np.abs((true_values - predictions) / true_values)) * 100
    metrics['RMSE'] = np.sqrt(metrics['MSE'])
    metrics['R2'] = r2_score(true_values, predictions)
    metrics['CRMSD'] = np.sqrt(np.mean((predictions - true_values - np.mean(predictions - true_values)) ** 2))
    metrics['MAD'] = np.median(np.abs(predictions - true_values))
    metrics['nRMSE'] = metrics['RMSE'] / (true_values.max() - true_values.min())
    
    return metrics

# Calculate metrics for each model
metrics_dict = {}
models = combined_df['Model'].unique()

for model in models:
    model_df = combined_df[combined_df['Model'] == model]
    metrics_dict[model] = calculate_metrics(model_df)

metrics_df = pd.DataFrame(metrics_dict).T

# Normalize the metrics for better comparison
metrics_df_normalized = (metrics_df - metrics_df.min()) / (metrics_df.max() - metrics_df.min())

# 调整颜色亮度的函数
def adjust_brightness(color, factor):
    rgb = to_rgb(color)
    return tuple(max(min(c * factor, 1.0), 0.0) for c in rgb)

# 设置调整亮度的因子
brightness_factor = 1.2

# 定义颜色
base_colors = {
    'Present': 'blue',
    'Without gru': 'orange',
    'Without cnn': 'green',
    'Without attention': 'red',
    'Without dropout': 'purple'
}

# 调整颜色亮度
adjusted_colors = {key: adjust_brightness(color, brightness_factor) for key, color in base_colors.items()}

# Plot bar chart for each metric
fig, ax = plt.subplots(figsize=(7 / 2.54, 4.85 / 2.54), dpi=300)  # 设置图片大小为6cm x 6cm，分辨率为300dpi
metrics_df_normalized.plot(kind='bar', ax=ax, color=[adjusted_colors[model] for model in metrics_df_normalized.index])

ax.set_ylabel('Value', fontsize=5, fontname='Times New Roman')
ax.set_xticklabels(ax.get_xticklabels(), fontsize=5, fontname='Times New Roman', rotation=45, ha='right', rotation_mode='anchor')
ax.set_yticklabels(ax.get_yticklabels(), fontsize=4, fontname='Times New Roman')
ax.tick_params(axis='x', pad=0)  # 调整横坐标标签距离

# Remove x-axis tick marks
ax.tick_params(axis='x', length=0)

# Minimize the legend box
ax.legend(loc='upper left', fontsize=4, frameon=False)

plt.tight_layout(pad=0.2)
plt.savefig('bar_chart_metrics.png', format='png', bbox_inches='tight', pad_inches=0)
plt.savefig('bar_chart_metrics.pdf', format='pdf', bbox_inches='tight', pad_inches=0)
plt.savefig('bar_chart_metrics.tiff', format='tiff', bbox_inches='tight', pad_inches=0)
plt.show()

# Plot radar chart for each model
labels = metrics_df.columns
num_vars = len(labels)

# Compute angle of each axis
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # The plot is circular, so we need to "complete the loop"

fig, ax = plt.subplots(figsize=(4 / 2.54, 4 / 2.54), dpi=300, subplot_kw=dict(polar=True))  # 设置图片大小为6cm x 6cm，分辨率为300dpi

for model, color in zip(metrics_df.index, [adjusted_colors[model] for model in metrics_df.index]):
    values = metrics_df_normalized.loc[model].tolist()
    values += values[:1]
    ax.fill(angles, values, color=color, alpha=0.5)
    ax.plot(angles, values, color=color, linewidth=0.5, label=model)

ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=5, fontname='Times New Roman')
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize=5)

plt.tight_layout(pad=0.2)
plt.savefig('radar_chart_metrics.png', format='png', bbox_inches='tight', pad_inches=0)
plt.savefig('radar_chart_metrics.pdf', format='pdf', bbox_inches='tight', pad_inches=0)
plt.savefig('radar_chart_metrics.tiff', format='tiff', bbox_inches='tight', pad_inches=0)
plt.show()
