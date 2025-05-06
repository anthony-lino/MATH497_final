# %% 
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from itertools import product
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")  # To suppress convergence warnings

# Load your data
df = pd.read_csv("USD_CNY_5_years.csv", parse_dates=['date'], index_col='date')
series = df['close']  # Select your univariate time series

# Split into train/test
train_size = int(len(series) * 0.8)
train, test = series[:train_size], series[train_size:]

# Define parameter ranges
p_values = range(1, 11)
d_values = range(0, 3)
q_values = range(1, 11)

best_score = float('inf')
best_order = None

# Grid search
for p, d, q in product(p_values, d_values, q_values):
    try:
        model = ARIMA(train, order=(p, d, q))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=len(test))
        mse = mean_squared_error(test, forecast)

        if mse < best_score:
            best_score = mse
            best_order = (p, d, q)

    except Exception:
        continue

print(f"Best ARIMA order: {best_order}")
print(f"Best MSE on test set: {best_score}")
