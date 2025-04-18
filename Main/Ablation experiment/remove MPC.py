import random as rn
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import psutil
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten, Attention, Dropout, BatchNormalization
from tensorflow.keras import Model
from tensorflow.keras.layers import concatenate, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import json
import re
import utils_new3 as utils
import utils as utilss
import param_separated as pr
import pandas as pd
from tensorflow.keras.regularizers import l2
import tensorflow_probability as tfp
from scipy.optimize import minimize

# Set random seeds for reproducibility
SEED = 12345
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
rn.seed(SEED)
tf.random.set_seed(SEED)
os.environ["CUDA_VISIBLE_DEVICES"] = str(SEED)

def align_batches_by_min_length(*arrays):
    """Safely align all non-empty and valid arrays to the shortest length, with debug printing."""
    valid_arrays = []
    for i, arr in enumerate(arrays):
        if arr is None:
            print(f"[WARNING] Array {i} is None — skipped.")
            continue
        if not isinstance(arr, np.ndarray):
            print(f"[WARNING] Array {i} is not a NumPy array — converting.")
            arr = np.array(arr)
        if arr.shape[0] == 0:
            print(f"[WARNING] Array {i} is empty — skipped.")
            continue
        print(f"[INFO] Array {i} — shape before alignment: {arr.shape}")
        valid_arrays.append(arr)

    if not valid_arrays:
        raise ValueError("All arrays are None or empty!")

    min_len = min(arr.shape[0] for arr in valid_arrays)
    print(f"[INFO] Minimum aligned length: {min_len}")

    aligned_arrays = [arr[:min_len] for arr in valid_arrays]

    for i, arr in enumerate(aligned_arrays):
        print(f"[INFO] Array {i} — shape after alignment: {arr.shape}")

    return aligned_arrays


def save_predictions_and_true_values(predictions, true_values, file_name):
    """Safely save predictions and true values to CSV, aligned by shortest length."""
    predictions = predictions.flatten()
    true_values = true_values.flatten()

    # Auto-align to shortest length
    min_len = min(len(predictions), len(true_values))
    predictions = predictions[:min_len]
    true_values = true_values[:min_len]

    df = pd.DataFrame({
        'Predictions': predictions,
        'True Values': true_values
    })
    df.to_csv(file_name, index=False)


def plot_predictions(true_data, predicted_data, cycle_values, num_points=100):
    true_data = true_data.flatten()
    predicted_data = predicted_data.flatten()
    cycle_values = cycle_values[:min(len(cycle_values), len(true_data), len(predicted_data))]
    true_data = true_data[:len(cycle_values)]
    predicted_data = predicted_data[:len(cycle_values)]

    plt.figure(figsize=(7 / 2.54, 3.8 / 2.54), dpi=300)
    plt.plot(cycle_values[:num_points], true_data[:num_points], label='Actual Value', color='blue', linewidth=1, alpha=0.7)
    plt.plot(cycle_values[:num_points], predicted_data[:num_points], label='Predicted Value', color='red', alpha=0.7, linewidth=1)
    plt.fill_between(cycle_values[:num_points], true_data[:num_points], predicted_data[:num_points], color='gray', alpha=0.2)
    plt.xlabel('Cycle', fontsize=6, fontname='Times New Roman')
    plt.ylabel('Capacity (Ahr)', fontsize=6, fontname='Times New Roman')
    plt.legend(fontsize=6, frameon=False)
    plt.xticks(fontsize=6, fontname='Times New Roman')
    plt.yticks(fontsize=6, fontname='Times New Roman')
    plt.tight_layout(pad=0.2)
    plt.savefig('D:\\filename.pdf', format='pdf')
    plt.savefig('D:\\filename.tiff', format='tiff')


def error_distribution(true_values, predicted_values):
    true_values = true_values.flatten()
    predicted_values = predicted_values.flatten()
    min_len = min(len(true_values), len(predicted_values))
    true_values = true_values[:min_len]
    predicted_values = predicted_values[:min_len]
    errors = np.abs(true_values - predicted_values)
    error_ranges = [(0, 0.01), (0.01, 0.05), (0.05, 0.1), (0.1, 0.5), (0.5, np.max(errors))]
    counts = [np.sum((errors >= min_e) & (errors < max_e)) for min_e, max_e in error_ranges]
    return error_ranges, counts


def plot_error_distribution(error_ranges, counts):
    labels = [f"{min_error}-{max_error}" for min_error, max_error in error_ranges]
    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts, color='skyblue')
    plt.xlabel('Error Range')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout(pad=0.2)
    plt.savefig('D:\\error_distribution.png')


def plot_error_histogram(errors):
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, color='skyblue')
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.tight_layout(pad=0.2)
    plt.savefig('D:\\error_histogram.png')


def plot_time_series_errors(errors):
    plt.figure(figsize=(10, 6))
    plt.plot(errors, color='skyblue')
    plt.xlabel('Sample Index')
    plt.ylabel('Prediction Error')
    plt.tight_layout(pad=0.2)
    plt.savefig('D:\\time_series_errors.png')


def plot_metrics(history, metric):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history[metric], label='Train ' + metric)
    plt.plot(history.history['val_' + metric], label='Validation ' + metric)
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.legend()
    plt.tight_layout(pad=0.2)
    plt.savefig(f'D:\\{metric}_history.png')


def plot_evaluation_metrics(metrics):
    labels = ['MAE', 'MSE', 'MAPE', 'RMSE', 'R2', 'CRMSD', 'MAD', 'nRMSE']
    values = [metrics['test_mae'], metrics['test_mse'], metrics['test_mape'], metrics['test_rmse'], metrics['test_r2'], metrics['test_crmsd'], metrics['test_mad'], metrics['test_nrmse']]
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color='skyblue')
    plt.xlabel('Metrics')
    plt.ylabel('Value')
    plt.tight_layout(pad=0.2)
    plt.savefig('D:\\evaluation_metrics.png')


def post_process(predictions, threshold=0.05):
    for i in range(1, len(predictions) - 1):
        if (predictions[i] - predictions[i-1] > threshold) and (predictions[i] - predictions[i+1] > threshold):
            predictions[i] = (predictions[i-1] + predictions[i+1]) / 2
    return predictions


def calculate_metrics(true_values, predicted_values):
    """Compute standard regression metrics safely with length alignment."""
    # Ensure consistent length
    min_len = min(len(true_values), len(predicted_values))
    true_values = true_values[:min_len]
    predicted_values = predicted_values[:min_len]

    # Flatten if needed
    true_values = true_values.flatten()
    predicted_values = predicted_values.flatten()

    # Calculate metrics
    mae = mean_absolute_error(true_values, predicted_values)
    mse = mean_squared_error(true_values, predicted_values)
    mape = mean_absolute_percentage_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_values, predicted_values)
    crmsd = np.sqrt(np.mean((predicted_values - true_values.mean())**2))
    mad = np.mean(np.abs(predicted_values - true_values.mean()))
    nrmse = rmse / (true_values.max() - true_values.min())

    return mae, mse, mape, rmse, r2, crmsd, mad, nrmse


def plot_resource_usage(total_time, memory_usage):
    plt.figure(figsize=(7 / 2.54, 3.8 / 2.54), dpi=300)
    metrics = ['Total Time (s)', 'Memory Usage (MB)']
    values = [total_time, memory_usage]
    plt.bar(metrics, values, color='skyblue')
    plt.ylabel('Value')
    plt.xticks(fontsize=6, fontname='Times New Roman')
    plt.yticks(fontsize=6, fontname='Times New Roman')
    plt.tight_layout(pad=0.2)
    plt.savefig('D:\\resource_usage.png', format='png')


def build_model(lstm_units=512, learning_rate=0.0001):
    input_LSTM = Input(shape=(2, 10), name="LSTM_Input")

    LSTM_layer = LSTM(lstm_units, activation='tanh', return_sequences=True)(input_LSTM)
    dropout_lstm = Dropout(0.4)(LSTM_layer)
    batch_norm_lstm = BatchNormalization()(dropout_lstm)

    flat = Flatten()(batch_norm_lstm)
    dense1 = Dense(512, activation='relu')(flat)
    dropout_dense = Dropout(0.4)(dense1)
    batch_norm_dense = BatchNormalization()(dropout_dense)
    output = Dense(1)(batch_norm_dense)

    model = Model(inputs=input_LSTM, outputs=output)

    optim = Adam(learning_rate=learning_rate)

    model.compile(loss='mae', optimizer=optim, metrics=['mae'])

    return model


param_grid = {
    'lstm_units': [256, 512, 1024],
    'learning_rate': [1e-5, 1e-4, 1e-3, 1e-2],
    'batch_size': [16, 32, 64],
    'epochs': [100, 200, 300, 400]
}

# Main function for data processing, model training, and evaluation
def main():
    start_time = time.time()
    process = psutil.Process(os.getpid())
    
    pth = pr.pth
    pth = r'C:\Users\LiaoGuanfu\Downloads\RUL_prediction-main (1)\RUL_prediction-main\data\NASA'
    shuffle = True
    filp = False
    slicing = False

    if shuffle:
        x_files = [os.path.join(pth, 'charge/total', f) for f in os.listdir(os.path.join(pth, 'charge/total'))]
        y_files = [os.path.join(pth, 'discharge/total', f) for f in os.listdir(os.path.join(pth, 'discharge/total'))]
        train_x_files = rn.sample(x_files, 3)
        selected_index = [x_files.index(item) for item in train_x_files]
        train_y_files = [y_files[i] for i in selected_index]
        train_x_files.sort(key=lambda f: int(re.sub(r'\D', '', f)))
        train_y_files.sort(key=lambda f: int(re.sub(r'\D', '', f)))
        test_x_data = list(set(x_files) - set(train_x_files))
        test_y_data = list(set(y_files) - set(train_y_files))
    else:
        train_x_files = [os.path.join(pth, 'charge/train', f) for f in os.listdir(os.path.join(pth, 'charge/train'))]
        train_x_files.sort(key=lambda f: int(re.sub(r'\D', '', f)))
        train_y_files = [os.path.join(pth, 'discharge/train', f) for f in os.listdir(os.path.join(pth, 'discharge/train'))]
        train_y_files.sort(key=lambda f: int(re.sub(r'\D', '', f)))
        test_x_data = [os.path.join(pth, 'charge/test', f) for f in os.listdir(os.path.join(pth, 'charge/test'))]
        test_y_data = [os.path.join(pth, 'discharge/test', f) for f in os.listdir(os.path.join(pth, 'discharge/test'))]

    print("train X:", train_x_files)
    print("train Y:", train_y_files)

    folds = list(KFold(n_splits=pr.k, shuffle=True, random_state=pr.random).split(train_x_files))

    total_mae = 0
    total_mse = 0
    total_mape = 0
    total_rmse = 0
    total_r2 = 0
    total_crmsd = 0
    total_mad = 0
    total_nrmse = 0
    num_folds = len(folds)

    for j, (train_idx, val_idx) in enumerate(folds):
        print('\nFold', j + 1)
        train_x_data = [train_x_files[train_idx[i]] for i in range(len(train_idx))]
        train_y_data = [train_y_files[train_idx[i]] for i in range(len(train_idx))]
        val_x_data = [train_x_files[val_idx[i]] for i in range(len(val_idx))]
        val_y_data = [train_y_files[val_idx[i]] for i in range(len(val_idx))]
        print("train X:", train_x_data)
        print("train y:", train_y_data)
        print("val X:", val_x_data)
        print("val y", val_y_data)
        print("test: x", test_x_data)
        print("test: y", test_y_data)

        # Ensure all sequences have the same length by padding the shorter ones
        trainX_lstm, trainY_lstm, SS_tr_lstm = utils.extract_VIT_capacity(train_x_data, train_y_data, pr.seq_len_lstm, pr.hop, pr.sample, c=True, slicing=slicing, filp=filp)
        valX_lstm, valY_lstm, SS_val_lstm = utils.extract_VIT_capacity(val_x_data, val_y_data, pr.seq_len_lstm, pr.hop, pr.sample, c=True, slicing=slicing, filp=filp)
        testX_lstm, testY_lstm, SS_tt_lstm = utils.extract_VIT_capacity(test_x_data, test_y_data, pr.seq_len_lstm, pr.hop, pr.sample, c=True, slicing=slicing, filp=filp)

        # Ensure equal lengths of all datasets
        trainX_lstm, trainY_lstm = align_batches_by_min_length(trainX_lstm, trainY_lstm)
        valX_lstm, valY_lstm = align_batches_by_min_length(valX_lstm, valY_lstm)
        testX_lstm, testY_lstm = align_batches_by_min_length(testX_lstm, testY_lstm)

        # Build model
        model = build_model(lstm_units=512, learning_rate=0.0001)

        history = model.fit(
            x=trainX_lstm,
            y=trainY_lstm,
            validation_data=(valX_lstm, valY_lstm),
            batch_size=16,
            epochs=500,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)]
        )

        save_dir = pr.save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_dir = pr.model_dir + '_k' + str(j + 1)
        if not os.path.exists(os.path.join(save_dir, model_dir)):
            os.makedirs(os.path.join(save_dir, model_dir))

        # Save the model using the Keras native format
        model.save(os.path.join(save_dir, model_dir, "saved_model_and_weight.keras"))
        print("Model and weights saved")

        val_loss = []
        val_results = model.evaluate(valX_lstm, valY_lstm)
        val_loss.append(val_results)
        print('Val loss:', val_results)

        test_loss = []
        results = model.evaluate(testX_lstm, testY_lstm)
        test_loss.append(results)
        print('Test loss:', results)

        valPredict = model.predict(valX_lstm)
        testPredict = model.predict(testX_lstm)

        inv_valY = SS_val_lstm.inverse_transform(valY_lstm)
        inv_valPredict = SS_val_lstm.inverse_transform(valPredict)
        inv_testY = SS_tt_lstm.inverse_transform(testY_lstm)
        inv_testPredict = SS_tt_lstm.inverse_transform(testPredict)

        inv_valPredict = post_process(inv_valPredict)
        inv_testPredict = post_process(inv_testPredict)

        save_predictions_and_true_values(inv_testPredict, inv_testY, 'cnn_lstm_predictions.csv')

        max_cycle = len(inv_testY)

        cycle_values = np.arange(0, max_cycle)

        test_mae, test_mse, test_mape, test_rmse, test_r2, test_crmsd, test_mad, test_nrmse = calculate_metrics(inv_testY, inv_testPredict)

        print('\nTest Mean Absolute Error: %f MAE' % test_mae)
        print('Test Mean Square Error: %f MSE' % test_mse)
        print('Test Mean Absolute Percentage Error: %f MAPE' % test_mape)
        print('Test Root Mean Squared Error: %f RMSE' % test_rmse)
        print('Test R2: %f' % test_r2)
        print('Test CRMSD: %f' % test_crmsd)
        print('Test MAD: %f' % test_mad)
        print('Test nRMSE: %f' % test_nrmse)

        total_mae += test_mae
        total_mse += test_mse
        total_mape += test_mape
        total_rmse += test_rmse
        total_r2 += test_r2
        total_crmsd += test_crmsd
        total_mad += test_mad
        total_nrmse += test_nrmse

    avg_mae = total_mae / num_folds
    avg_mse = total_mse / num_folds
    avg_mape = total_mape / num_folds
    avg_rmse = total_rmse / num_folds
    avg_r2 = total_r2 / num_folds
    avg_crmsd = total_crmsd / num_folds
    avg_mad = total_mad / num_folds
    avg_nrmse = total_nrmse / num_folds

    print('\nAverage Test Mean Absolute Error: %f MAE' % avg_mae)
    print('Average Test Mean Square Error: %f MSE' % avg_mse)
    print('Average Test Mean Absolute Percentage Error: %f MAPE' % avg_mape)
    print('Average Test Root Mean Squared Error: %f RMSE' % avg_rmse)
    print('Average Test R2: %f' % avg_r2)
    print('Average Test CRMSD: %f' % avg_crmsd)
    print('Average Test MAD: %f' % avg_mad)
    print('Average Test nRMSE: %f' % avg_nrmse)

    end_time = time.time()
    total_time = end_time - start_time
    memory_usage = process.memory_info().rss / (1024 ** 2)  # Convert to MB

    print(f"\nTotal training time: {total_time:.2f} seconds")
    print(f"Memory usage: {memory_usage:.2f} MB")
    plot_resource_usage(total_time, memory_usage)


if __name__ == "__main__":
    main()
