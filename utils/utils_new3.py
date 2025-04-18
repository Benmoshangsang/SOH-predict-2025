import numpy as np
from sklearn.preprocessing import MinMaxScaler

from pandas import read_csv, DataFrame
#参数：

#dataset：这是函数的输入参数，预期是一个numpy数组或者类似数组的数据结构，其中包含了需要进行归一化处理的数据。
#过程：

#首先，创建了一个 MinMaxScaler 实例，这是来自 sklearn.preprocessing 的一个归一化工具。通过设置 feature_range=(0, 1) 参数，这个实例被配置为将所有特征值缩放到0到1的范围内。
#然后，使用这个 MinMaxScaler 实例的 fit_transform 方法对输入的 dataset 进行归一化处理。这个方法首先计算数据的最小值和最大值，然后使用这些值将数据缩放到0到1的范围。这个过程不仅返回归一化后的数据 (scaled)，同时也让 MinMaxScaler 实例内部学习到了数据的最小值和最大值，这对于未来的数据逆变换（即从归一化状态恢复到原始状态）是必要的。
#返回值：

#scaled：归一化后的数据集，其所有特征值都被缩放到了0和1之间。
#scalers：已经根据输入数据 dataset 被训练（fit）的 MinMaxScaler 实例。这个实例保存了进行数据变换所需的最小值和最大值，可以用于将新数据或归一化后的数据逆变换回原始的尺度。
def preprocess(dataset):
    scalers = MinMaxScaler(feature_range=(0, 1))
    scaled = scalers.fit_transform(dataset)
    return scaled, scalers
#函数名称：extract_VIT_capacity
#参数：
#x_datasets：电池充电过程的数据集路径列表。
#y_datasets：电池放电过程的数据集路径列表。
#seq_len：序列长度，表示在每个输入数据样本中要考虑的连续周期数。
#hop：每次移动的周期数，用于确定连续样本之间的重叠程度。
#sample：采样间隔，指定在提取特征时每隔多少个数据点取一个数据点。
#v、II、t、c：布尔类型参数，分别指示是否提取电压、电流、温度和容量特征。
#功能说明
#目的：该函数的主要目的是处理电池测试数据，按照指定的参数提取关键的物理量（电压、电流、温度、容量）作为特征，以便用于后续的数据分析或机器学习模型训练。

#数据处理流程：

#数据加载：使用 pandas 的 read_csv 函数分别加载 x_datasets 和 y_datasets 中指定的电池充电和放电数据文件。
#数据预处理：可能包括去除缺失值、选择特定的列（如电压、电流、温度和循环编号）、过滤掉某些不需要的数据（例如循环编号为0的数据），以及重置数据框架的索引。
#特征提取：根据函数参数（v、II、t、c）决定哪些特征需要被提取。对于每种物理量，可能需要进行进一步的处理，如采样、计算平均值等，以减少数据的维度并提取有用的信息。
#序列化处理：根据参数 seq_len 和 hop，从处理后的特征数据中构造输入序列，这些输入序列可以用作机器学习模型的训练样本。
#局部变量：

#V、I、T、C：分别用于暂存电压、电流、温度和容量的特征数据。
#x、y：用于存储最终生成的输入特征集和目标变量集。
#SS：可能用于存储数据标准化或归一化过程中使用的缩放器（scalers），以便于将来的数据转换或逆转换。

def extract_VIT_capacity(x_datasets, y_datasets, seq_len, hop, sample, v=False, II=False, t=False, c=False, slicing=False, filp=False):
    V = []
    I = []
    T = []
    C = []

    x = []
    y = []

#遍历充电和放电数据集：通过 zip(x_datasets, y_datasets) 同时遍历充电数据集 (x_datasets) 和放电数据集 (y_datasets) 中的文件路径。

#处理充电数据 (x_data)：

#使用 read_csv(x_data) 从文件加载充电数据，然后通过 .dropna() 去除任何含有缺失值的行。
#仅选择 'cycle', 'voltage_battery', 'current_battery', 'temp_battery' 这四列，分别代表充电周期、电池电压、电池电流和电池温度。
#通过 x_df = x_df[x_df['cycle'] != 0] 过滤掉周期编号为0的数据，可能是因为某些原因第0周期的数据不适用于分析或模型训练。
#重置 DataFrame 的索引，并去除旧的索引列。
#处理放电数据 (y_data)：

#同样地，从文件加载放电数据，并去除含有缺失值的行。
#选择 'capacity', 'cycle_idx' 列，并将 cycle_idx 设置为数据的索引加1，代表放电周期的唯一标识。
#将 DataFrame 转换为 Numpy 数组，并将数据类型转换为 float32，这对于后续的数值计算和模型训练更为适宜。
#计算数据长度 (data_len)：

#根据放电数据的长度 (y_len)、序列长度 (seq_len) 和步长 (hop) 计算出可以生成的数据序列的数量。这里使用的公式 np.int32(np.floor((y_len - seq_len - 1) / hop)) + 1 考虑了每个序列包含的周期数和在生成连续序列时每次跳过的周期数。
    for x_data, y_data in zip(x_datasets, y_datasets):
        # Load VIT from charging profile
        x_df = read_csv(x_data).dropna()
        x_df = x_df[['cycle', 'voltage_battery', 'current_battery', 'temp_battery']]
        x_df = x_df[x_df['cycle'] != 0]  # cycle ke-0 tidak masuk
        x_df = x_df.reset_index().drop(columns="index")

        x_df = read_csv(x_data).dropna()
        x_df = x_df[['cycle', 'voltage_battery', 'current_battery', 'temp_battery']]
        x_df = x_df[x_df['cycle'] != 0]  # cycle ke-0 tidak masuk
        x_len = len(x_df.cycle.unique())  # - seq_len
        if slicing:
            x_slicing = int(x_len * 0.8)
            x_start = np.random.randint(1, x_len-x_slicing)
            slicing_x_list= x_df.cycle.unique()[x_start:x_start+x_slicing]
            bool_index = x_df['cycle'].isin(slicing_x_list)
            x_df = x_df[bool_index]
        x_df = x_df.reset_index().drop(columns="index")


        # Load capacity from discharging profile
        y_df = read_csv(y_data).dropna()
        y_df = y_df[['capacity']]
        y_df = y_df.values  # Convert pandas dataframe to numpy array
        y_df = y_df.astype('float32')  # Convert values to float

        if slicing:
            y_df = y_df[x_start:x_start+x_slicing]
        y_len = len(y_df)
        data_len = np.int32(np.floor((y_len - seq_len - 1) / hop)) + 1
#遍历放电数据集：对于放电数据集中的每一个数据点（即每一个放电周期），执行以下步骤。

#提取和处理容量数据（C）：
#从放电数据 (y_df) 中提取容量值，并将其加入到容量列表 C 中。
#使用 preprocess 函数对 C 进行归一化处理，然后转换数据类型为 float32。这是为了保证数据在后续处理或模型训练中的一致性和准确性。
#计算需要去除的数据点数量：根据给定的采样间隔 (sample)，计算在对电压、电流和温度进行等间隔采样时需要去除的数据点数量，以确保每个采样的维度一致。

#条件性提取电压、电流和温度特征：

#电压（V）：如果选项 v 被设置为 True，则对电压数据进行等间隔采样，计算每个采样区间的平均值，并将结果加入到电压列表 V 中。然后，对 V 进行归一化处理，并转换数据类型为 float32。
#电流（I）：如果选项 II（表示电流）被设置为 True，则对电流数据执行与电压相同的处理步骤，并将处理结果加入到电流列表 I 中。
#温度（T）：如果选项 t 被设置为 True，则对温度数据执行与电压相同的处理步骤，并将处理结果加入到温度列表 T 中。
#数据重塑和平均：对于每种物理量（电压、电流、温度），在去除多余的数据点后，将数据重塑为一个二维数组，其中每行包含 sample 个数据点。然后计算每行的平均值，得到采样后的特征值。

#归一化处理：对每种提取的特征（电压、电流、温度、容量）使用 preprocess 函数进行归一化处理，以保证数据的标准化，有利于后续的机器学习模型训练。
        for i in range(y_len):
            if filp:
                cy = x_df.cycle.unique()[-i-2]
                df = x_df.loc[x_df['cycle'] == cy]
                cap = np.array([y_df[-i-1, 0]])
            else:
                cy = x_df.cycle.unique()[i]
                df = x_df.loc[x_df['cycle'] == cy]
                cap = np.array([y_df[i, 0]])
            C.append(cap)
            df_C = DataFrame(C).values
            scaled_C, scaler_C = preprocess(df_C)
            scaled_C = scaled_C.astype('float32')[:, :]

            le = len(df['voltage_battery']) % sample
            if v:
                # Voltage measured
                vTemp = df['voltage_battery'].to_numpy()
                if le != 0:
                    vTemp = vTemp[0:-le] 
                vTemp = np.reshape(vTemp, (len(vTemp) // sample, -1)) #, order="F")
                vTemp = vTemp.mean(axis=0)
                V.append(vTemp)
                df_V = DataFrame(V).values
                scaled_V, scaler = preprocess(df_V)
                scaled_V = scaled_V.astype('float32')[:, :]

            elif II:
                # Current measured
                iTemp = df['current_battery'].to_numpy()
                if le != 0:
                    iTemp = iTemp[0:-le]
                iTemp = np.reshape(iTemp, (len(iTemp) // sample, -1)) #, order="F")
                iTemp = iTemp.mean(axis=0)
                I.append(iTemp)
                df_I = DataFrame(I).values
                scaled_I, scaler = preprocess(df_I)
                scaled_I = scaled_I.astype('float32')[:, :]

            elif t:
                # Temperature measured
                tTemp = df['temp_battery'].to_numpy()
                if le != 0:
                    tTemp = tTemp[0:-le]
                tTemp = np.reshape(tTemp, (len(tTemp) // sample, -1)) #, order="F")
                tTemp = tTemp.mean(axis=0)
                T.append(tTemp)
                df_T = DataFrame(T).values
                scaled_T, scaler = preprocess(df_T)
                scaled_T = scaled_T.astype('float32')[:, :]
#构建训练数据（x）：
#基于选定的特征创建输入序列：根据函数参数（v、II、t、c），选择相应的特征（电压、电流、温度、容量）来创建机器学习模型的输入数据。

#如果选择了电压（v），则从归一化后的电压数据（scaled_V）中按指定的步长（hop）和序列长度（seq_len）创建序列。
#如果选择了电流（II），则采用类似的方法处理电流数据（scaled_I）。
#如果选择了温度（t），则处理温度数据（scaled_T）。
#如果选择了容量（c），则处理容量数据（scaled_C）。
#序列创建逻辑：对于所选的特征，代码遍历计算得出的数据长度（data_len），每次迭代根据步长（hop）和序列长度（seq_len）从对应的归一化数据中切片，创建一系列用于训练的输入序列，并将这些序列添加到列表 x 中。

#准备目标数据（y）：
#对于每个输入序列，从归一化的容量数据（scaled_C）中提取相应的目标值，即在序列之后的容量值。这意味着模型的任务是基于序列预测接下来的容量值。
#遍历数据长度（data_len），每次迭代从归一化的容量数据中提取一个目标值，并添加到列表 y 中。
#返回值：
#最后，函数将输入数据序列 x 和对应的目标值 y 转换为 Numpy 数组，并返回这些数组以及用于归一化容量数据的缩放器（scaler_C）。
        if v:
            for i in range(data_len):
                x.append(scaled_V[(hop * i):(hop * i + seq_len)])
        elif II:
            for i in range(data_len):
                x.append(scaled_I[(hop * i):(hop * i + seq_len)])
        elif t:
            for i in range(data_len):
                x.append(scaled_T[(hop * i):(hop * i + seq_len)])
        elif c:
            for i in range(data_len):
                x.append(scaled_C[(hop * i):(hop * i + seq_len)])

        for i in range(data_len):
            y.append(scaled_C[hop * i + seq_len])
    return np.array(x), np.array(y), scaler_C