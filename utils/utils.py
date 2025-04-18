from sklearn.preprocessing import MinMaxScaler
import numpy as np
from pandas import read_csv, DataFrame
import matplotlib.pyplot as plt
import os


def preprocess(dataset):
    #preprocess 函数
#目的：该函数的目的是对数据集的每个特征进行归一化处理。
#参数：dataset，一个三维数组，通常是一个时间序列数据集，其中包括多个特征。
#过程：对于数据集的每个特征（dataset 的第二维），使用 MinMaxScaler 对该特征下的所有数据进行归一化处理，使其值位于指定的范围（默认为0到1之间）。
#返回值：返回归一化后的数据集和对应的归一化器（scalers）字典。这个字典以特征索引为键，以对应的 MinMaxScaler 实例为值，可用于后续的逆归一化操作。
    scalers = {}
    for i in range(dataset.shape[1]):
        scalers[i] = MinMaxScaler(feature_range=(0, 1))
        dataset[:, i, :] = scalers[i].fit_transform(dataset[:, i, :])
    return dataset, scalers

#extract_VIT_capacity 函数
#目的：该函数用于从给定的电池充放电数据集中提取特征并创建训练样本。
#参数：
#x_datasets：包含电池充电过程数据的集合。
#y_datasets：包含电池放电过程中电池容量数据的集合。
#seq_len：序列长度，指定要在每个样本中包含多少个时间步的数据。
#hop：跳跃值，决定从一个样本到下一个样本时跳过多少个时间步。
#sample：采样频率。
#extract_all、extract_c_only、extract_v_only、extract_i_only、extract_t_only、extract_vit_only、extract_vc_only：这些参数用于指定需要提取的特征类型（如仅电压、仅电流、电压+电流+温度等）。
#过程：这个函数将遍历 x_datasets 和 y_datasets 中的数据，根据指定的参数提取特征，创建包含电压、电流、温度和电池容量的输入向量，以及对应的目标电池容量值。
#返回值：函数返回几个列表：
#x：包含处理后的输入特征向量。
#y：目标电池容量值。
#z：循环索引，指示每个样本对应的充电循环。
#SS：存储每个特征的归一化器（scaler），用于后续的数据处理。
#VITC：临时输入向量，可能用于中间计算步骤。
def extract_VIT_capacity(x_datasets=None, y_datasets=None, seq_len=5, hop=1, sample=10, extract_all=True,
                         extract_c_only=False, extract_v_only=False, extract_i_only=False, extract_t_only=False,
                         extract_vit_only=False, extract_vc_only=False):
    x = []  # VITC = inputs voltage, current, temperature (in vector) + capacity (in scalar)
    y = []  # target capacity (in scalar)
    z = []  # cycle index
    SS = []  # scaler
    VITC = []  # temporary input
#遍历充电和放电数据集：通过 zip(x_datasets, y_datasets) 将充电数据集 (x_datasets) 和放电数据集 (y_datasets) 打包在一起，然后遍历每一对充电和放电数据。

#加载和清理充电数据 (x_data)：

#使用 read_csv(x_data).dropna() 加载充电数据文件并删除任何含有缺失值的行。
#仅保留 'cycle', 'voltage_battery', 'current_battery', 'temp_battery' 这四个列，分别代表充电周期、电池电压、电池电流和电池温度。
#将 'cycle' 列的每个值增加1，可能是为了调整周期编号使之从1开始，或者是为了数据处理上的某种需要。
#计算唯一充电周期的数量，存储在 x_len 中。
#加载和清理放电数据 (y_data)：

#使用 read_csv(y_data).dropna() 加载放电数据文件并删除任何含有缺失值的行。
#生成 'cycle_idx' 列，值从1开始递增，表示放电周期的索引。
#保留 'capacity', 'cycle_idx' 这两列，分别代表每个放电周期的电池容量和周期索引。
#将 DataFrame 转换为 numpy 数组，并将数组中的数据类型转换为 float32。
#计算放电数据的长度，存储在 y_len 中。
#计算数据长度 (data_len)：

#使用公式 np.int32(np.floor((y_len - seq_len - 1) / hop)) + 1 来计算基于给定的序列长度 (seq_len) 和跳跃值 (hop) 能够生成的数据样本数量。这个计算考虑了序列长度和跳跃值对最终能够生成的样本数量的影响。
    for x_data, y_data in zip(x_datasets, y_datasets):
        # Load VIT from charging profile
        x_df = read_csv(x_data).dropna()
        x_df = x_df[['cycle', 'voltage_battery', 'current_battery', 'temp_battery']]
        x_df['cycle'] = x_df['cycle'] + 1
        x_len = len(x_df.cycle.unique())  # - seq_len

        # Load capacity from discharging profile
        y_df = read_csv(y_data).dropna()
        y_df['cycle_idx'] = y_df.index + 1
        y_df = y_df[['capacity', 'cycle_idx']]
        y_df = y_df.values  # Convert pandas dataframe to numpy array
        y_df = y_df.astype('float32')  # Convert values to float
        y_len = len(y_df)  # - seq_len

        data_len = np.int32(np.floor((y_len - seq_len - 1) / hop)) + 1
        #遍历放电数据集：对于 y_datasets 中的每个元素（即每个放电周期），执行以下操作。

#选取相应的充电数据：

#通过在充电数据 x_df 中选取与当前放电周期相对应的周期（cy），来筛选出相关的充电数据。
#对电压、电流和温度数据进行采样和平均处理：

#对于每种测量类型（电压、电流、温度），首先根据参数 sample 对数据进行等间隔采样。
#如果数据长度不是 sample 的整数倍，则去除末尾多余的数据点，以使得数据长度变为 sample 的整数倍。
#然后，将数据重塑成二维数组，每 sample 个数据点形成一行。
#对每行数据（即每个 sample 区间内的数据）计算平均值，得到每个测量类型的采样平均值。
#提取电池容量数据：

#从 y_df 数组中直接提取对应于当前放电周期的电池容量值。
#根据指定条件组合特征：

#根据函数参数（extract_c_only, extract_v_only, extract_i_only, extract_t_only, extract_all, extract_vit_only, extract_vc_only）决定如何组合上述处理过的电压、电流、温度和容量数据。
#如果指定了只提取某一种数据（例如仅电容、仅电压等），则只将相应的数据添加到 VITC 列表中。
#如果指定了提取所有数据或组合部分数据（如电压、电流和温度），则将相应的数据通过 np.concatenate 函数组合后添加到 VITC 列表中。
#结果收集：

#所有处理和组合完成后的数据被收集在 VITC 列表中，该列表最终可用于机器学习模型的训练或其他分析。
        for i in range(y_len):
            cy = x_df.cycle.unique()[i]
            df = x_df.loc[x_df['cycle'] == cy]
            # Voltage measured
            le = len(df['voltage_battery']) % sample
            vTemp = df['voltage_battery'].to_numpy()
            if le != 0:
                vTemp = vTemp[0:-le]
            vTemp = np.reshape(vTemp, (len(vTemp) // sample, -1))  # , order="F")
            vTemp = vTemp.mean(axis=0)
            # Current measured
            iTemp = df['current_battery'].to_numpy()
            if le != 0:
                iTemp = iTemp[0:-le]
            iTemp = np.reshape(iTemp, (len(iTemp) // sample, -1))  # , order="F")
            iTemp = iTemp.mean(axis=0)
            # Temperature measured
            tTemp = df['temp_battery'].to_numpy()
            if le != 0:
                tTemp = tTemp[0:-le]
            tTemp = np.reshape(tTemp, (len(tTemp) // sample, -1))  # , order="F")
            tTemp = tTemp.mean(axis=0)
            # Capacity measured
            cap = np.array([y_df[i, 0]])
            # Combined
            if extract_c_only:
                VITC_inp = cap
                VITC.append(VITC_inp)
            elif extract_v_only:
                VITC_inp = vTemp
                VITC.append(VITC_inp)
            elif extract_i_only:
                VITC_inp = iTemp
                VITC.append(VITC_inp)
            elif extract_t_only:
                VITC_inp = tTemp
                VITC.append(VITC_inp)
            elif extract_all:
                VITC_inp = np.concatenate((vTemp, iTemp, tTemp, cap))
                VITC.append(VITC_inp)
            elif extract_vit_only:
                VITC_inp = np.concatenate((vTemp, iTemp, tTemp))
                VITC.append(VITC_inp)
            elif extract_vc_only:
                VITC_inp = np.concatenate((vTemp, cap))
                VITC.append(VITC_inp)

        # Normalize using MinMaxScaler
        #创建 DataFrame 并转换为 Numpy 数组：

#首先，使用 DataFrame(VITC) 将之前准备的 VITC 列表（包含电压、电流、温度和容量的组合特征）转换为 Pandas 的 DataFrame 对象。
#然后，通过 .values 方法将 DataFrame 转换为 Numpy 数组 df_VITC，以便进行进一步的处理。
#扩展数据维度并进行归一化处理：

#通过 df_VITC[:, :, np.newaxis] 语句将数组扩展一个新的维度，这通常是为了满足特定机器学习库或模型输入数据的形状要求。新的维度使得每个特征值都在其独立的数组中。
#调用 preprocess 函数对这个扩展维度后的数组进行归一化处理。preprocess 函数为每个特征计算并应用了一个 MinMaxScaler，这样每个特征值都被归一化到了0和1之间的范围。归一化可以帮助加快模型训练过程，并提高模型的收敛速度和性能。
#preprocess 函数返回两个对象：归一化后的数据 scaled_x 和用于归一化的 scaler 对象。scaler 对象可以在后续的预测或模型评估中用于逆归一化，以便将预测结果转换回原始的数据范围。
#调整数据类型和形状：

#将 scaled_x 数组的数据类型转换为 float32。这一步通常是为了确保数据类型与机器学习模型的输入要求相匹配，同时 float32 类型比默认的 float64 类型占用更少的内存，有助于减少计算资源消耗。
#通过 [:, :, 0] 语句移除扩展的维度，因为归一化处理完成后，不再需要这个额外的维度。这样 scaled_x 的形状就被调整为适合模型输入的二维数组形式。
        df_VITC = DataFrame(VITC).values
        scaled_x, scaler = preprocess(df_VITC[:, :, np.newaxis])
        scaled_x = scaled_x.astype('float32')[:, :, 0]  # Convert values to float

        # Create input data
        for i in range(data_len):
            x.append(scaled_x[(hop * i):(hop * i + seq_len), :])
            y.append(scaled_x[hop * i + seq_len, -1])
            # z.append(y_df[hop*i+seq_len, 1])
        SS.append(scaler)
        # import pdb; pdb.set_trace()
    return np.array(x), np.array(y)[:, np.newaxis], SS
#plot_loss 函数
#目的：绘制模型在训练过程中的损失值，包括训练损失和验证损失。
#参数：
#history：模型训练过程中返回的历史对象，包含了训练和验证过程的损失值。
#save_dir：保存图像的根目录。
#model_dir：模型相关文件的目录，用于在 save_dir 下进一步分类保存图像。
#过程：
#使用 matplotlib.pyplot 创建一个图像，设置大小为 8x4 英寸。
#从 history.history['loss'] 和 history.history['val_loss'] 中提取训练损失和验证损失数据，分别绘制为两条曲线。
#设置图像的标题、轴标签，并添加图例。
#将绘制的损失曲线图保存到 save_dir/model_dir 目录下，文件名为 train_loss.png。
#plot_pred 函数
#目的：绘制模型的预测结果与真实值的对比图。
#参数：
#predict：模型的预测结果。
#true：实际的真实值。
#save_dir：保存图像的根目录。
#model_dir：模型相关文件的目录。
#name：保存的图像文件名（不包含文件扩展名）。
#过程：
#将 predict 和 true 数组重塑为一维数组，以便能够在图中正确绘制。
#使用 matplotlib.pyplot 创建一个新的图像，设置大小为 12x4 英寸，分辨率为 150 DPI。
#绘制 predict 和 true 作为两条曲线，分别标注为“Prediction”和“True”。
#设置图像的轴标签和图例，并调整字体大小以便清晰显示。
#将绘制的预测结果对比图保存到 save_dir/model_dir 目录下，文件名为由参数 name 指定的值加上文件扩展名 .png。

def plot_loss(history, save_dir, model_dir):
    # Plot model loss
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(save_dir, model_dir, 'train_loss.png'))


def plot_pred(predict, true, save_dir, model_dir, name):  # Plot test prediction
    predict = predict.reshape(predict.shape[0])
    true = true.reshape(true.shape[0])
    plt.figure(figsize=(12, 4), dpi=150)
    plt.plot(predict, label='Prediction')
    plt.plot(true, label='True')
    plt.xlabel('Number of Cycle', fontsize=13)
    plt.ylabel('Discharge Capacity (Ah)', fontsize=13)
    plt.legend(loc='upper right', fontsize=12)
    plt.savefig(os.path.join(save_dir, model_dir, name + '.png'))
