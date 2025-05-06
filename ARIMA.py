# %% 
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from itertools import product
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings("ignore")  # To suppress convergence warnings

# Load your data
df_USD_CNY = pd.read_csv("USD_CNY_5_years.csv", parse_dates=['date'], index_col='date')
df_USD_EUR = pd.read_csv("USD_EUR_5_years.csv", parse_dates=['date'], index_col='date')
df_USD_JPY = pd.read_csv("USD_JPY_5_years.csv", parse_dates=['date'], index_col='date')
dfs = [df_USD_CNY, df_USD_EUR, df_USD_JPY]

types = ['close','open','high','low']

# Define parameter ranges
p_values = range(0, 4)
d_values = range(0, 3)
q_values = range(0, 4)

# Function to calculate metrics
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

results = []
best_aic = float("inf")
best_bic = float("inf")
best_aic_order = None
best_bic_order = None

for p, d, q, df,type in product(p_values, d_values, q_values, dfs, types):
    try:
        model = ARIMA(df[type], order=(p, d, q))
        model_fit = model.fit()

        aic = model_fit.aic
        bic = model_fit.bic

        results.append({'p': p, 'd': d, 'q': q, 'aic': aic, 'bic': bic})

        if aic < best_aic:
            best_aic = aic
            best_aic_order = (p, d, q)
            print("New best AIC:", best_aic, "with parameters:", best_aic_order)
        else:
            print("Parameters:", (p, d, q), "with AIC:", aic)

        if bic < best_bic:
            best_bic = bic
            best_bic_order = (p, d, q)
    except Exception as e:
        results.append({'p': p, 'd': d, 'q': q, 'aic': None, 'bic': None})
        print("Failed to fit model:", (p, d, q), "Error:", e)
        continue

# Save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("arima_grid_search_results_info_criterion.csv", index=False)

