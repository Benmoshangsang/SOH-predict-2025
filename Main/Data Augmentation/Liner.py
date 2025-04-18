# -*- coding: utf-8 -*-
"""
Created on Thu May  2 20:41:35 2024

@author: LiaoGuanfu
"""

import pandas as pd
import numpy as np
import os

def linear_interpolate_and_fill(input_folder, output_folder, file_name):
    # 构建输入输出路径
    input_path = os.path.join(input_folder, file_name)
    output_path = os.path.join(output_folder, 'Linear_Interpolated_' + file_name)
    
    # 检查文件是否存在
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"No such file: '{input_path}'")
    
    # 加载数据
    df = pd.read_csv(input_path)
    
    # 指定要插值的列和需要填充的列
    interp_columns = ['voltage_battery', 'current_battery', 'temp_battery', 'current_load', 'voltage_load', 'time']
    fill_columns = ['amb_temp', 'cycle', 'date_time']
    
    # 创建新的索引
    original_indices = df.index
    new_indices = np.linspace(original_indices.min(), original_indices.max(), 2*len(df)-1)

    # 对数值列进行插值
    df_interp = df.set_index(original_indices).reindex(new_indices).interpolate(method='linear')
    
    # 对非数值列使用前向填充
    df_interp[fill_columns] = df.set_index(original_indices)[fill_columns].reindex(new_indices).fillna(method='pad')

    # 重设索引
    df_interp.reset_index(drop=True, inplace=True)

    # 保存插值后的数据
    df_interp.to_csv(output_path, index=False)
    print(f"Interpolated data has been saved to {output_path}")

input_folder = r'C:\Users\LiaoGuanfu\Downloads\RUL_prediction-main (1)\RUL_prediction-main\data\NASA\charge\test'
output_folder = r'C:\Users\LiaoGuanfu\Downloads\RUL_prediction-main (1)\RUL_prediction-main\data\NASA\charge\test'
file_name = 'B0018_charge.csv'

# 调用函数执行插值和填充
linear_interpolate_and_fill(input_folder, output_folder, file_name)