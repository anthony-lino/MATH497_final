# %% 
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from itertools import product
from sklearn.metrics import mean_squared_error
import warnings
import joblib

warnings.filterwarnings("ignore")  # To suppress convergence warnings

# Load your data
df = pd.read_csv("USD_CNY_5_years.csv", parse_dates=['date'], index_col='date')
series = df['close']  # Select your univariate time series

# Define parameter ranges
p_values = range(1, 11)
d_values = range(0, 3)
q_values = range(1, 11)

results = []
best_aic = float("inf")
best_bic = float("inf")
best_aic_order = None
best_bic_order = None

for p, d, q in product(p_values, d_values, q_values):
    try:
        model = ARIMA(series, order=(p, d, q))
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

print(f"Best ARIMA order by AIC: {best_aic_order} with AIC: {best_aic}")
print(f"Best ARIMA order by BIC: {best_bic_order} with BIC: {best_bic}")