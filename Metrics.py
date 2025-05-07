import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import random
import tensorflow as tf
import os

# 1. Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
os.environ['PYTHONHASHSEED'] = str(42)

results_folder = 'Results_LSTM/USD_JPY_5_years'

train_df = pd.read_csv(f'{results_folder}/train_predictions.csv')
val_df = pd.read_csv(f'{results_folder}/val_predictions.csv')
test_df = pd.read_csv(f'{results_folder}/test_predictions.csv')

def extract_true_pred(df):
    y_true = df[['True_Open', 'True_High', 'True_Low', 'True_Close']].values
    y_pred = df[['Predicted_Open', 'Predicted_High', 'Predicted_Low', 'Predicted_Close']].values
    return y_true, y_pred

y_train_original, y_train_pred_original = extract_true_pred(train_df)
y_val_original, y_val_pred_original = extract_true_pred(val_df)
y_test_original, y_test_pred_original = extract_true_pred(test_df)

# Calculate evaluation metrics
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2

train_mae, train_mse, train_rmse, train_r2 = calculate_metrics(y_train_original, y_train_pred_original)
val_mae, val_mse, val_rmse, val_r2 = calculate_metrics(y_val_original, y_val_pred_original)
test_mae, test_mse, test_rmse, test_r2 = calculate_metrics(y_test_original, y_test_pred_original)

# Save metrics to a text file
with open(f'{results_folder}/metrics.txt', 'w') as f:
    f.write(f"Train Metrics - MAE: {train_mae}, MSE: {train_mse}, RMSE: {train_rmse}, R2: {train_r2}\n")
    f.write(f"Validation Metrics - MAE: {val_mae}, MSE: {val_mse}, RMSE: {val_rmse}, R2: {val_r2}\n")
    f.write(f"Test Metrics - MAE: {test_mae}, MSE: {test_mse}, RMSE: {test_rmse}, R2: {test_r2}\n")

# Plot 4-panel prediction figures
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
