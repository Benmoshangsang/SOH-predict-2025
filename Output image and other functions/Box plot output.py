import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define file paths
file_paths = {
    'UGABO': r'C:\Users\LiaoGuanfu\Downloads\RUL_prediction-main (1)\RUL_prediction-main\测试效果\6.3对比实验\cnn-lstm-attention\超函数对比\切片\UGABO.csv',
    'Bayesian ': r'C:\Users\LiaoGuanfu\Downloads\RUL_prediction-main (1)\RUL_prediction-main\测试效果\6.3对比实验\cnn-lstm-attention\超函数对比\切片\Bayesian Optimization.csv',
    'PSO': r'C:\Users\LiaoGuanfu\Downloads\RUL_prediction-main (1)\RUL_prediction-main\测试效果\6.3对比实验\cnn-lstm-attention\超函数对比\切片\PSO.csv',
    'Random ': r'C:\Users\LiaoGuanfu\Downloads\RUL_prediction-main (1)\RUL_prediction-main\测试效果\6.3对比实验\cnn-lstm-attention\超函数对比\切片\Random Search.csv',
    'TPE': r'C:\Users\LiaoGuanfu\Downloads\RUL_prediction-main (1)\RUL_prediction-main\测试效果\6.3对比实验\cnn-lstm-attention\超函数对比\切片\TPE.csv'
}

# Load all the CSV files into a dictionary of dataframes
dfs = {model: pd.read_csv(path) for model, path in file_paths.items()}

# Add a 'Model' column to each dataframe and concatenate them into a single dataframe
for model, df in dfs.items():
    df['Model'] = model

combined_df = pd.concat(dfs.values())

# Calculate the prediction error for each model
combined_df['Error'] = combined_df['Predictions'] - combined_df['True Values']

# Create the figure with the exact desired size in centimeters
fig, ax = plt.subplots(figsize=(8.2 / 2.54, 6.5 / 2.54), dpi=400)  # 7cm width and 4cm height

# Create a boxplot grouped by model
box = combined_df.boxplot(column='Error', by='Model', grid=False, patch_artist=True, return_type='dict', 
                          boxprops=dict(facecolor='lightblue', color='blue', linewidth=1),
                          medianprops=dict(color='red', linewidth=1),
                          whiskerprops=dict(color='blue', linewidth=1),
                          capprops=dict(color='blue', linewidth=1),
                          flierprops=dict(marker='o', color='blue', markersize=2, markeredgewidth=0.5),  # Ensure uniform markeredgewidth
                          ax=ax)

# Remove the automatic 'Boxplot grouped by Model' title and plot title
plt.suptitle('')
plt.title('')  # Clear any title

# Set labels for axes with tight label padding
ax.set_ylabel('Prediction Error', fontsize=6, fontname='Times New Roman', labelpad=-1)  # Adjust labelpad to move it closer to the y-axis numbers
ax.set_xlabel('', fontsize=6, fontname='Times New Roman', labelpad=2)

# Move x-axis labels closer to the axis
ax.xaxis.set_tick_params(labelsize=6, pad=-1)  # Adjusted the pad to -1 to move labels closer
ax.yaxis.set_tick_params(labelsize=6, pad=1)

# Adjust layout to remove excess space
plt.subplots_adjust(left=0.15, right=0.95, top=0.85, bottom=0.25)

# Save with tight bounding box to remove excess white space
plt.savefig('box_plot_metrics_no_title_updated.pdf', format='pdf', bbox_inches='tight', pad_inches=0)
plt.savefig('box_plot_metrics_no_title_updated.tiff', format='tiff', bbox_inches='tight', pad_inches=0)
plt.show()
