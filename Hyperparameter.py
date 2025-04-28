import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping

# Set random seeds for reproducibility
def set_random_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)

# 1. Define the model creation function
def create_model(learning_rate=0.001, batch_size=32, lstm_units=64, dropout_rate=0.2, num_layers=1, seq_length=60):
    model = Sequential()
    model.add(LSTM(units=lstm_units, return_sequences=(num_layers > 1), input_shape=(seq_length, 4)))
    
    for _ in range(1, num_layers):
        model.add(LSTM(units=lstm_units, return_sequences=True))
    
    model.add(Dropout(dropout_rate))
    model.add(Dense(4))  # Predicting open, high, low, close prices
    
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    
    return model

# 2. Load data
def load_data(data_file):
    df = pd.read_csv(data_file)
    data = df[['open', 'high', 'low', 'close']].values
    
    # Normalize the data
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    
    return data, scaler

# 3. Create time series sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# 4. Run grid search
def run_grid_search(data_file, param_grid):
    data, scaler = load_data(data_file)
    
    # Create time series sequences
    seq_length = 60  # Default value, will be overwritten in GridSearch
    X, y = create_sequences(data, seq_length)
    
    # Split the dataset
    total_samples = len(X)
    train_size = int(0.75 * total_samples)
    val_size = int(0.20 * total_samples)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]
    
    # Initialize best score and best parameters
    best_score = float('inf')
    best_params = None
    
    # Start grid search
    for learning_rate in param_grid['learning_rate']:
        for batch_size in param_grid['batch_size']:
            for lstm_units in param_grid['lstm_units']:
                for dropout_rate in param_grid['dropout_rate']:
                    for num_layers in param_grid['num_layers']:
                        for seq_length in param_grid['seq_length']:
                            for epochs in param_grid['epochs']:
                                
                                # Set random seeds for each grid configuration
                                set_random_seeds(42)
                                
                                # Create the model
                                model = create_model(learning_rate, batch_size, lstm_units, dropout_rate, num_layers, seq_length)
                                
                                # Train the model
                                early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                                history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), verbose=0, callbacks=[early_stopping])
                                
                                # Evaluate the model on validation set
                                val_loss = model.evaluate(X_val, y_val, verbose=0)
                                
                                # Update the best parameters if current loss is lower
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

# 5. Save the best parameters and score to a file
def save_best_params(best_params, best_score, filename="best_params.txt"):
    with open(filename, 'w') as f:
        f.write(f"Best parameters found: {best_params}\n")
        f.write(f"Best validation loss (score): {best_score}\n")

# 6. Example: Run grid search on different datasets
datasets = ['USD_CNY_5_years.csv', 'USD_EUR_5_years.csv', 'USD_JPY_5_years.csv']

# Define the hyperparameter grid for grid search
param_grid = {
    'learning_rate': [0.001, 0.01, 0.0001],
    'batch_size': [16, 32, 64, 128],
    'lstm_units': [16, 32, 64, 128],
    'dropout_rate': [0.1, 0.2, 0.3, 0.4],
    'num_layers': [1, 2, 3, 4],
    'seq_length': [30, 60, 90, 120],
    'epochs': [50, 100, 200, 500, 1000]
}

# Run grid search for each dataset
for dataset in datasets:
    best_params, best_score = run_grid_search(dataset, param_grid)
    save_best_params(best_params, best_score, filename=f"best_params_{dataset}.txt")
    print(f"Best parameters for {dataset}: {best_params}")
    print(f"Best validation loss for {dataset}: {best_score}")
