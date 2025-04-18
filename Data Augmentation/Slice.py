import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import json
import matplotlib.pyplot as plt
import re

import utils_xin as utils
import utils as utilss
import param_separated as pr

# Setting the seed for reproducibility
SEED = 12345
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["CUDA_VISIBLE_DEVICES"] = str(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

def flip_datasets(path):
    """Flip the datasets vertically and save them with a new name."""
    files = os.listdir(path)
    for file in files:
        if file.endswith('.csv'):
            full_path = os.path.join(path, file)
            data = pd.read_csv(full_path)
            flipped_data = data.iloc[::-1].reset_index(drop=True)
            new_filename = file.replace('.csv', '_flipped.csv')
            flipped_data.to_csv(os.path.join(path, new_filename), index=False)

def prepare_datasets(base_path):
    """Prepare dataset file paths after flipping them."""
    sub_dirs = ['charge/train', 'discharge/train', 'charge/test', 'discharge/test']
    dataset_files = {}
    for sub_dir in sub_dirs:
        full_path = os.path.join(base_path, sub_dir)
        flip_datasets(full_path)  # Flip datasets
        files = [os.path.join(full_path, f) for f in os.listdir(full_path) if 'flipped' in f]
        key = sub_dir.replace('/', '_')
        dataset_files[key] = files
    return dataset_files
def plot_predictions(true_data, predicted_data, title='Predicted vs Actual Values', num_points=100):
    plt.figure(figsize=(10, 6))
    plt.plot(true_data[:num_points], label='Actual Value', color='blue', marker='o')
    plt.plot(predicted_data[:num_points], label='Predicted Value', color='red', alpha=0.7, marker='x')
    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig('D:\\filename.png')  # Save the figure before displaying it
    plt.show()


def error_distribution(true_values, predicted_values):
    errors = np.abs(true_values - predicted_values)
    error_ranges = [(0, 0.01), (0.01, 0.05), (0.05, 0.1), (0.1, 0.5), (0.5, np.max(errors))]
    counts = []
    for min_error, max_error in error_ranges:
        count = np.sum((errors >= min_error) & (errors < max_error))
        counts.append(count)
    return error_ranges, counts

def plot_error_distribution(error_ranges, counts, title='Error Distribution'):
    labels = [f"{min_error}-{max_error}" for min_error, max_error in error_ranges]
    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts, color='skyblue')
    plt.title(title)
    plt.xlabel('Error Range')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()  # It's a good practice to call tight_layout before saving
    plt.savefig('D:\\error_distribution.png')  # Save the figure before displaying it
    plt.show()

def main():
   
    base_path = r'C:\Users\LiaoGuanfu\Downloads\RUL_prediction-main (1)\RUL_prediction-main\data\NASA'
    datasets = prepare_datasets(base_path)  # Prepare and get flipped datasets

    # Extract data for training, validation, and testing
    train_x_files = datasets['charge_train']
    train_y_files = datasets['discharge_train']
    test_x_files = datasets['charge_test']
    test_y_files = datasets['discharge_test']

    # Create KFold splits
    folds = KFold(n_splits=pr.k, shuffle=True, random_state=SEED).split(train_x_files)
    for j, (train_idx, val_idx) in enumerate(folds):
        train_x_data = [train_x_files[i] for i in train_idx]
        train_y_data = [train_y_files[i] for i in train_idx]
        val_x_data = [train_x_files[i] for i in val_idx]
        val_y_data = [train_y_files[i] for i in val_idx]

        # Correctly unpack all returned values from the utility function
        trainX_lstm, trainY_lstm, SS_tr_lstm = utils.extract_VIT_capacity(train_x_data, train_y_data, pr.seq_len_lstm, pr.hop, pr.sample)
        valX_lstm, valY_lstm, SS_val_lstm = utils.extract_VIT_capacity(val_x_data, val_y_data, pr.seq_len_lstm, pr.hop, pr.sample)
        testX_lstm, testY_lstm, SS_tt_lstm = utils.extract_VIT_capacity(test_x_files, test_y_files, pr.seq_len_lstm, pr.hop, pr.sample)


    # Loading flipped datasets

    # Define the test data arrays used in the model
    test_x_data = test_x_files
    test_y_data = test_y_files

    

    folds = list(KFold(n_splits=pr.k, shuffle=True, random_state=pr.random).split(train_x_files))
    for j, (train_idx, val_idx) in enumerate(folds):
        train_x_data = [train_x_files[i] for i in train_idx]
        train_y_data = [train_y_files[i] for i in train_idx]
        val_x_data = [train_x_files[i] for i in val_idx]
        val_y_data = [train_y_files[i] for i in val_idx]

        # Extracting and preparing datasets
        trainX_lstm, trainY_lstm, SS_tr_lstm = utils.extract_VIT_capacity(train_x_data, train_y_data, pr.seq_len_lstm, pr.hop, pr.sample, c=True)
        valX_lstm, valY_lstm, SS_val_lstm = utils.extract_VIT_capacity(val_x_data, val_y_data, pr.seq_len_lstm, pr.hop, pr.sample, c=True)
        testX_lstm, testY_lstm, SS_tt_lstm = utils.extract_VIT_capacity(test_x_files, test_y_files, pr.seq_len_lstm, pr.hop, pr.sample, c=True)
        
        print('Flipped train X:', train_x_data)
        print('Flipped train Y:', train_y_data)
        print('Flipped val X:', val_x_data)
        print('Flipped val Y:', val_y_data)
        print('Flipped test X:', test_x_files)
        print('Flipped test Y:', test_y_files)

        trainX_lstm, trainY_lstm, SS_tr_lstm = utils.extract_VIT_capacity(train_x_data, train_y_data, pr.seq_len_lstm,
                                                                          pr.hop, pr.sample,
                                                                          c=True)
        valX_lstm, valY_lstm, SS_val_lstm = utils.extract_VIT_capacity(val_x_data, val_y_data, pr.seq_len_lstm, pr.hop,
                                                                       pr.sample,
                                                                       c=True)
        testX_lstm, testY_lstm, SS_tt_lstm = utils.extract_VIT_capacity(test_x_data, test_y_data, pr.seq_len_lstm,
                                                                        pr.hop, pr.sample,
                                                                        c=True)
        print('Input shape: {}'.format(trainX_lstm.shape))

        v_trainX_cnn, v_trainY_cnn, v_SS_tr_cnn = utils.extract_VIT_capacity(train_x_data, train_y_data, pr.seq_len_cnn,
                                                                             pr.hop, pr.sample,
                                                                             v=True)
        v_valX_cnn, v_valY_cnn, v_SS_val_cnn = utils.extract_VIT_capacity(val_x_data, val_y_data, pr.seq_len_cnn,
                                                                          pr.hop, pr.sample,
                                                                          v=True)
        v_testX_cnn, v_testY_cnn, v_SS_tt_cnn = utils.extract_VIT_capacity(test_x_data, test_y_data, pr.seq_len_cnn,
                                                                           pr.hop, pr.sample,
                                                                           v=True)
        print('Input shape: {}'.format(v_trainX_cnn.shape))

        i_trainX_cnn, i_trainY_cnn, i_SS_tr_cnn = utils.extract_VIT_capacity(train_x_data, train_y_data, pr.seq_len_cnn,
                                                                             pr.hop, pr.sample,
                                                                             II=True)
        i_valX_cnn, i_valY_cnn, i_SS_val_cnn = utils.extract_VIT_capacity(val_x_data, val_y_data, pr.seq_len_cnn,
                                                                          pr.hop, pr.sample,
                                                                          II=True)
        i_testX_cnn, i_testY_cnn, i_SS_tt_cnn = utils.extract_VIT_capacity(test_x_data, test_y_data, pr.seq_len_cnn,
                                                                           pr.hop, pr.sample,
                                                                           II=True)
        print('Input shape: {}'.format(i_trainX_cnn.shape))

        t_trainX_cnn, t_trainY_cnn, t_SS_tr_cnn = utils.extract_VIT_capacity(train_x_data, train_y_data, pr.seq_len_cnn,
                                                                             pr.hop, pr.sample,
                                                                             t=True)
        t_valX_cnn, t_valY_cnn, t_SS_val_cnn = utils.extract_VIT_capacity(val_x_data, val_y_data, pr.seq_len_cnn,
                                                                          pr.hop, pr.sample,
                                                                          t=True)
        t_testX_cnn, t_testY_cnn, t_SS_tt_cnn = utils.extract_VIT_capacity(test_x_data, test_y_data, pr.seq_len_cnn,
                                                                           pr.hop, pr.sample,
                                                                           t=True)
        print('Input shape: {}'.format(t_trainX_cnn.shape))

        input_CNN_v = Input(shape=(pr.seq_len_cnn, v_trainX_cnn.shape[-1]), name="CNN_Input_V")
        input_CNN_i = Input(shape=(pr.seq_len_cnn, i_trainX_cnn.shape[-1]), name="CNN_Input_i")
        input_CNN_t = Input(shape=(pr.seq_len_cnn, t_trainX_cnn.shape[-1]), name="CNN_Input_t")

        input_LSTM = Input(shape=(pr.seq_len_lstm, trainX_lstm.shape[-1]), name="LSTM_Input")

        LSTM_layer = LSTM(32,
                          activation='tanh',
                          return_sequences=True,
                          name="LSTM_layer")(input_LSTM)

        CNN_layer_v = Conv1D(32, 5, activation='relu',
                             strides=1, padding="same",
                             name="CNN_layer_v")(input_CNN_v)

        CNN_layer_i = Conv1D(32, 5, activation='relu',
                             strides=1, padding="same",
                             name="CNN_layer_i")(input_CNN_i)

        CNN_layer_t = Conv1D(32, 5, activation='relu',
                             strides=1, padding="same",
                             name="CNN_layer_t")(input_CNN_t)

        concat_cnn = concatenate([CNN_layer_v, CNN_layer_i, CNN_layer_t])

        CNN_fusion = Conv1D(32, 5, activation='relu',
                            strides=1, padding='same',
                            name="CNN_fusion")(concat_cnn)

        concat = concatenate([LSTM_layer, CNN_fusion])

        flat = Flatten()(concat)
        output = Dense(32, activation='relu', name="Predictor")(flat)
        output = Dense(1, name="Output")(output)

        model = Model(inputs=[input_LSTM, input_CNN_v, input_CNN_i, input_CNN_t], outputs=[output])

        optim = Adam(learning_rate=pr.lr)

        loss = Huber(delta=2)
        model.compile(loss=loss, optimizer=optim)

        history = model.fit(x=[trainX_lstm, v_trainX_cnn, i_trainX_cnn, t_trainX_cnn],
                            y=[trainY_lstm, v_trainY_cnn, i_trainY_cnn, t_trainY_cnn],
                            validation_data=([valX_lstm, v_valX_cnn, i_valX_cnn, t_valX_cnn],
                                             [valY_lstm, v_valY_cnn, i_valY_cnn, t_valY_cnn]),
                            batch_size=pr.batch_size,
                            epochs=pr.epochs)

        save_dir = pr.save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_dir = pr.model_dir + '_k' + str(j + 1)
        if not os.path.exists(os.path.join(save_dir, model_dir)):
            os.makedirs(os.path.join(save_dir, model_dir))

        model.save(save_dir + model_dir + "/saved_model_and_weight")
        print("bobot dan model tersimpan")

        val_loss = []
        val_results = model.evaluate([valX_lstm, v_valX_cnn, i_valX_cnn, t_valX_cnn],
                                     [valY_lstm, v_valY_cnn, i_valY_cnn, t_valY_cnn])
        val_loss.append(val_results)
        print('Val loss:', val_results)

        test_loss = []
        results = model.evaluate([testX_lstm, v_testX_cnn, i_testX_cnn, t_testX_cnn],
                                 [testY_lstm, v_testY_cnn, i_testY_cnn, t_testY_cnn])
        test_loss.append(results)
        print('Test loss:', results)

        valPredict = model.predict([valX_lstm, v_valX_cnn, i_valX_cnn, t_valX_cnn])
        testPredict = model.predict([testX_lstm, v_testX_cnn, i_testX_cnn, t_testX_cnn])

        inv_valY = SS_val_lstm.inverse_transform(v_valY_cnn)
        inv_valPredict = SS_val_lstm.inverse_transform(valPredict)

        inv_testY = SS_tt_lstm.inverse_transform(v_testY_cnn)
        inv_testPredict = SS_tt_lstm.inverse_transform(testPredict)

        test_mae = mean_absolute_error(inv_testY, inv_testPredict)
        test_mse = mean_squared_error(inv_testY, inv_testPredict)
        test_mape = mean_absolute_percentage_error(inv_testY, inv_testPredict)
        test_rmse = np.sqrt(mean_squared_error(inv_testY, inv_testPredict))
        print('\nTest Mean Absolute Error: %f MAE' % test_mae)
        print('Test Mean Square Error: %f MSE' % test_mse)
        print('Test Mean Absolute Percentage Error: %f MAPE' % test_mape)
        print('Test Root Mean Squared Error: %f RMSE' % test_rmse)

        with open(os.path.join(save_dir, model_dir, 'eval_metrics.txt'), 'w') as f:
            f.write('Train data: ')
            f.write(json.dumps(train_x_data))
            f.write('\nVal data: ')
            f.write(json.dumps(val_x_data))
            f.write('\nTest data: ')
            f.write(json.dumps(test_x_data))
            f.write('\n\nTest Mean Absolute Error: ')
            f.write(json.dumps(str(test_mae)))
            f.write('\nTest Mean Square Error: ')
            f.write(json.dumps(str(test_mse)))
            f.write('\nTest Mean Absolute Percentage Error: ')
            f.write(json.dumps(str(test_mape)))
            f.write('\nTest Root Mean Squared Error: ')
            f.write(json.dumps(str(test_rmse)))

        testPred_file = open(os.path.join(save_dir, model_dir, 'test_predict.txt'), 'w')
        for row in inv_testPredict:
            np.savetxt(testPred_file, row)
        testPred_file.close()

        testY_file = open(os.path.join(save_dir, model_dir, 'test_true.txt'), 'w')
        for row in inv_testY:
            np.savetxt(testY_file, row)
        testY_file.close()

        utilss.plot_loss(history, save_dir, model_dir)
        utilss.plot_pred(inv_valPredict, inv_valY, save_dir, model_dir, "val_pred")
        utilss.plot_pred(inv_testPredict, inv_testY, save_dir, model_dir, "test_pred")

        # Plot predictions and error distribution
        plot_predictions(inv_valY, inv_valPredict, title='Validation: Predicted vs Actual Values')
        plot_predictions(inv_testY, inv_testPredict, title='Test: Predicted vs Actual Values')

        error_ranges, counts = error_distribution(inv_testY, inv_testPredict)
        plot_error_distribution(error_ranges, counts, title='Test Set Error Distribution')

if __name__ == "__main__":
    main() 