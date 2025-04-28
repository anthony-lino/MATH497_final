import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import random
import tensorflow as tf
import os

# 1. 设置随机种子确保可复现性
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
os.environ['PYTHONHASHSEED'] = str(42)

# 2. 定义运行实验的函数
def run_experiment(data_file, results_folder):
    # 读入数据
    df = pd.read_csv(data_file)
    data = df[['open', 'high', 'low', 'close']].values

    # 归一化
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    # 创建时间序列
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(X), np.array(y)

    seq_length = 60
    X, y = create_sequences(data, seq_length)

    # 划分 75%训练集, 20%验证集, 5%测试集
    total_samples = len(X)
    train_size = int(0.75 * total_samples)
    val_size = int(0.20 * total_samples)

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

    # 建立LSTM模型
    model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(seq_length, 4)),
    Dense(4)
    ])

    model.compile(optimizer='adam', loss='mse')

    # 训练模型
    history = model.fit(
        X_train, y_train,
        epochs=1000,
        batch_size=32,
        validation_data=(X_val, y_val)
    )

    # 预测结果
    y_train_pred = model.predict(X_train)
    y_train_pred_original = scaler.inverse_transform(y_train_pred)
    y_train_original = scaler.inverse_transform(y_train)

    y_val_pred = model.predict(X_val)
    y_val_pred_original = scaler.inverse_transform(y_val_pred)
    y_val_original = scaler.inverse_transform(y_val)

    y_test_pred = model.predict(X_test)
    y_test_pred_original = scaler.inverse_transform(y_test_pred)
    y_test_original = scaler.inverse_transform(y_test)

    # 保存预测结果
    train_df = pd.DataFrame(np.hstack([y_train_original, y_train_pred_original]), columns=['True_Open', 'True_High', 'True_Low', 'True_Close', 'Predicted_Open', 'Predicted_High', 'Predicted_Low', 'Predicted_Close'])
    val_df = pd.DataFrame(np.hstack([y_val_original, y_val_pred_original]), columns=['True_Open', 'True_High', 'True_Low', 'True_Close', 'Predicted_Open', 'Predicted_High', 'Predicted_Low', 'Predicted_Close'])
    test_df = pd.DataFrame(np.hstack([y_test_original, y_test_pred_original]), columns=['True_Open', 'True_High', 'True_Low', 'True_Close', 'Predicted_Open', 'Predicted_High', 'Predicted_Low', 'Predicted_Close'])

    train_df.to_csv(f'{results_folder}/train_predictions.csv', index=False)
    val_df.to_csv(f'{results_folder}/val_predictions.csv', index=False)
    test_df.to_csv(f'{results_folder}/test_predictions.csv', index=False)

    # 计算评估指标
    def calculate_metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        return mae, mse, rmse, r2

    train_mae, train_mse, train_rmse, train_r2 = calculate_metrics(y_train_original, y_train_pred_original)
    val_mae, val_mse, val_rmse, val_r2 = calculate_metrics(y_val_original, y_val_pred_original)
    test_mae, test_mse, test_rmse, test_r2 = calculate_metrics(y_test_original, y_test_pred_original)

    # 保存评估指标到文本文件
    with open(f'{results_folder}/metrics.txt', 'w') as f:
        f.write(f"Train Metrics - MAE: {train_mae}, MSE: {train_mse}, RMSE: {train_rmse}, R2: {train_r2}\n")
        f.write(f"Validation Metrics - MAE: {val_mae}, MSE: {val_mse}, RMSE: {val_rmse}, R2: {val_r2}\n")
        f.write(f"Test Metrics - MAE: {test_mae}, MSE: {test_mse}, RMSE: {test_rmse}, R2: {test_r2}\n")

    # 绘制四宫格图
    def plot_predictions(y_true, y_pred, title, save_path):
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        labels = ['Open Price', 'High Price', 'Low Price', 'Close Price']
        for i, ax in enumerate(axs.flatten()):
            ax.plot(y_true[:, i], label='True')
            ax.plot(y_pred[:, i], label='Predicted')
            ax.set_title(f'{title} - {labels[i]}')
            ax.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()

    plot_predictions(y_train_original, y_train_pred_original, 'Train', f'{results_folder}/train_predictions.png')
    plot_predictions(y_val_original, y_val_pred_original, 'Validation', f'{results_folder}/val_predictions.png')
    plot_predictions(y_test_original, y_test_pred_original, 'Test', f'{results_folder}/test_predictions.png')

# 3. 运行实验 for different datasets
datasets = ['USD_CNY_5_years.csv', 'USD_EUR_5_years.csv', 'USD_JPY_5_years.csv']
for dataset in datasets:
    results_folder = f'Results_LSTM/{dataset.split(".")[0]}'  # 创建不同的文件夹保存结果
    os.makedirs(results_folder, exist_ok=True)
    run_experiment(dataset, results_folder)
