import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping

# 设置随机种子以确保可复现性
def set_random_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)

# 1. 定义模型创建函数
def create_model(learning_rate=0.001, batch_size=32, lstm_units=64, dropout_rate=0.2, num_layers=1, seq_length=60):
    model = Sequential()
    model.add(LSTM(units=lstm_units, return_sequences=(num_layers > 1), input_shape=(seq_length, 4)))
    
    for _ in range(1, num_layers):
        model.add(LSTM(units=lstm_units, return_sequences=True))
    
    model.add(Dropout(dropout_rate))
    model.add(Dense(4))  # 预测 open, high, low, close 四个价格
    
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    
    return model

# 2. 加载数据
def load_data(data_file):
    df = pd.read_csv(data_file)
    data = df[['open', 'high', 'low', 'close']].values
    
    # 归一化
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    
    return data, scaler

# 3. 创建时间序列
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# 4. 运行网格搜索
def run_grid_search(data_file, param_grid):
    data, scaler = load_data(data_file)
    
    # 创建时间序列
    seq_length = 60  # 默认值，GridSearch 会覆盖这个值
    X, y = create_sequences(data, seq_length)
    
    # 划分数据集
    total_samples = len(X)
    train_size = int(0.75 * total_samples)
    val_size = int(0.20 * total_samples)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]
    
    # 初始化最好的分数和参数
    best_score = float('inf')
    best_params = None
    
    # 开始网格搜索
    for learning_rate in param_grid['learning_rate']:
        for batch_size in param_grid['batch_size']:
            for lstm_units in param_grid['lstm_units']:
                for dropout_rate in param_grid['dropout_rate']:
                    for num_layers in param_grid['num_layers']:
                        for seq_length in param_grid['seq_length']:
                            for epochs in param_grid['epochs']:
                                
                                # 为每个网格设置随机种子
                                set_random_seeds(42)
                                
                                # 创建模型
                                model = create_model(learning_rate, batch_size, lstm_units, dropout_rate, num_layers, seq_length)
                                
                                # 训练模型
                                early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                                history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), verbose=0, callbacks=[early_stopping])
                                
                                # 在验证集上评估模型
                                val_loss = model.evaluate(X_val, y_val, verbose=0)
                                
                                # 如果验证损失更小，则更新最佳参数和分数
                                if val_loss < best_score:
                                    best_score = val_loss
                                    best_params = {
                                        'learning_rate': learning_rate,
                                        'batch_size': batch_size,
                                        'lstm_units': lstm_units,
                                        'dropout_rate': dropout_rate,
                                        'num_layers': num_layers,
                                        'seq_length': seq_length,
                                        'epochs': epochs
                                    }
                                    
    return best_params, best_score

# 5. 保存最佳参数和评分到文件
def save_best_params(best_params, best_score, filename="best_params.txt"):
    with open(filename, 'w') as f:
        f.write(f"Best parameters found: {best_params}\n")
        f.write(f"Best validation loss (score): {best_score}\n")

# 6. 示例：使用不同数据集运行网格搜索
datasets = ['USD_CNY_5_years.csv', 'USD_EUR_5_years.csv', 'USD_JPY_5_years.csv']

# 定义网格搜索的超参数
param_grid = {
    'learning_rate': [0.001, 0.01, 0.0001],
    'batch_size': [16, 32, 64, 128],
    'lstm_units': [16, 32, 64, 128],
    'dropout_rate': [0.1, 0.2, 0.3, 0.4],
    'num_layers': [1, 2, 3, 4],
    'seq_length': [30, 60, 90, 120],
    'epochs': [50, 100, 200, 500, 1000]
}

# 为每个数据集运行网格搜索
for dataset in datasets:
    best_params, best_score = run_grid_search(dataset, param_grid)
    save_best_params(best_params, best_score, filename=f"best_params_{dataset}.txt")
    print(f"Best parameters for {dataset}: {best_params}")
    print(f"Best validation loss for {dataset}: {best_score}")
