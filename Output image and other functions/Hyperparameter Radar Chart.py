import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from matplotlib.colors import to_rgb

# Define file paths
file_paths = {
    'UGABO': r'C:\Users\LiaoGuanfu\Downloads\RUL_prediction-main (1)\RUL_prediction-main\测试效果\6.3对比实验\cnn-lstm-attention\超函数对比\交叉\UGABO.csv',
    'Bayesian Optimization': r'C:\Users\LiaoGuanfu\Downloads\RUL_prediction-main (1)\RUL_prediction-main\测试效果\6.3对比实验\cnn-lstm-attention\超函数对比\交叉\Bayesian Optimization.csv',
    'PSO': r'C:\Users\LiaoGuanfu\Downloads\RUL_prediction-main (1)\RUL_prediction-main\测试效果\6.3对比实验\cnn-lstm-attention\超函数对比\交叉\PSO.csv',
    'Random Search': r'C:\Users\LiaoGuanfu\Downloads\RUL_prediction-main (1)\RUL_prediction-main\测试效果\6.3对比实验\cnn-lstm-attention\超函数对比\交叉\Random Search.csv',
    'TPE': r'C:\Users\LiaoGuanfu\Downloads\RUL_prediction-main (1)\RUL_prediction-main\测试效果\6.3对比实验\cnn-lstm-attention\超函数对比\交叉\TPE.csv'
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
    
    return metrics

# Calculate metrics for each model
metrics_dict = {}
models = combined_df['Model'].unique()

for model in models:
    model_df = combined_df[combined_df['Model'] == model]
    metrics_dict[model] = calculate_metrics(model_df)

metrics_df = pd.DataFrame(metrics_dict).T

# Adjust brightness factor
def adjust_brightness(color, factor):
    rgb = to_rgb(color)
    return tuple(max(min(c * factor, 1.0), 0.0) for c in rgb)

brightness_factor = 1.2

# Define colors
colors = {
    'UGABO': 'blue',
    'Bayesian Optimization': 'orange',
    'PSO': 'green',
    'Random Search': 'red',
    'TPE': 'purple'
}

adjusted_colors = {key: adjust_brightness(color, brightness_factor) for key, color in colors.items()}

# Plot bar chart for each metric
fig, ax = plt.subplots(figsize=(7 / 2.54, 4.83 / 2.54), dpi=300)
metrics_df.plot(kind='bar', ax=ax, color=[adjusted_colors[model] for model in metrics_df.index])

ax.set_ylabel('Value', fontsize=5, fontname='Times New Roman')
ax.set_xticklabels(ax.get_xticklabels(), fontsize=5, fontname='Times New Roman', rotation=45, ha='right', rotation_mode='anchor')
ax.set_yticklabels(ax.get_yticklabels(), fontsize=4, fontname='Times New Roman')
ax.tick_params(axis='x', pad=0)

ax.tick_params(axis='x', length=0)

ax.legend(loc='upper left', fontsize=4, frameon=False)

plt.tight_layout(pad=0.2)
plt.savefig('bar_chart_metrics_updated.pdf', format='pdf', bbox_inches='tight', pad_inches=0)
plt.savefig('bar_chart_metrics_updated.tiff', format='tiff', bbox_inches='tight', pad_inches=0)
plt.show()

# Plot radar chart for each model
labels = metrics_df.columns
num_vars = len(labels)

angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(6 / 2.54, 5 / 2.54), dpi=300, subplot_kw=dict(polar=True))

for model, color in zip(metrics_df.index, [adjusted_colors[model] for model in metrics_df.index]):
    values = metrics_df.loc[model].tolist()
    values += values[:1]
    ax.fill(angles, values, color=color, alpha=0.5)
    ax.plot(angles, values, color=color, linewidth=0.5, label=model)

ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=5, fontname='Times New Roman', va='top')

for label, angle in zip(ax.get_xticklabels(), angles):
    x = np.cos(angle)
    y = np.sin(angle)
    label.set_horizontalalignment('center' if -0.5 < x < 0.5 else 'left' if x < -0.5 else 'right')
    label.set_verticalalignment('top' if y < -0.5 else 'bottom' if y > 0.5 else 'center')

ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize=5)

plt.tight_layout(pad=0.2)
plt.savefig('radar_chart_metrics_updated.pdf', format='pdf', bbox_inches='tight', pad_inches=0)
plt.savefig('radar_chart_metrics_updated.tiff', format='tiff', bbox_inches='tight', pad_inches=0)
plt.show()
