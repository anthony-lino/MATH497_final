import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from itertools import product
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import math

warnings.filterwarnings("ignore")  # To suppress convergence warnings

# Load your data
df_USD_CNY = pd.read_csv("USD_CNY_5_years.csv", parse_dates=['date'], index_col='date')
df_USD_EUR = pd.read_csv("USD_EUR_5_years.csv", parse_dates=['date'], index_col='date')
df_USD_JPY = pd.read_csv("USD_JPY_5_years.csv", parse_dates=['date'], index_col='date')
dfs = {'USD_CNY': df_USD_CNY, 'USD_EUR': df_USD_EUR, 'USD_JPY': df_USD_JPY}

types = ['close', 'open', 'high', 'low']

# Define parameter ranges
p_values = range(0, 5)
d_values = range(0, 3)
q_values = range(0, 5)

# Function to calculate metrics
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

# Dictionary to store all results
all_results = []

# Process each currency pair and price type
for currency_pair, df in dfs.items():
    for type in types:
        print(f"\nProcessing {currency_pair} - {type}")
        series = df[type]
        
        # Create train/validation/test splits (75%/20%/5%)
        n = len(series)
        train_size = int(n * 0.75)
        val_size = int(n * 0.20)
        
        train_data = series[:train_size]
        val_data = series[train_size:train_size+val_size]
        test_data = series[train_size+val_size:]
        
        print(f"Split sizes - Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")
        
        # Test different ARIMA parameters
        for p, d, q in product(p_values, d_values, q_values):
            try:
                print(f"Fitting ARIMA({p},{d},{q}) for {currency_pair} {type}")
                
                # Train the model on training data
                model = ARIMA(train_data, order=(p, d, q))
                model_fit = model.fit()
                
                # Get information criteria
                aic = model_fit.aic
                bic = model_fit.bic
                
                # Evaluate on training data
                train_predictions = model_fit.predict(start=0, end=len(train_data)-1)
                train_metrics = calculate_metrics(train_data, train_predictions)
                
                # Forecast for validation set
                val_predictions = model_fit.forecast(steps=len(val_data))
                val_metrics = calculate_metrics(val_data.values, val_predictions)
                
                # Retrain model on training + validation data for final test
                full_train_data = series[:train_size+val_size]
                full_model = ARIMA(full_train_data, order=(p, d, q))
                full_model_fit = full_model.fit()
                
                # Forecast for test set
                test_predictions = full_model_fit.forecast(steps=len(test_data))
                test_metrics = calculate_metrics(test_data.values, test_predictions)
                
                # Store results
                result = {
                    'Currency': currency_pair,
                    'Type': type,
                    'p': p, 'd': d, 'q': q,
                    'AIC': aic,
                    'BIC': bic,
                    'Train_MAE': train_metrics['MAE'],
                    'Train_MSE': train_metrics['MSE'],
                    'Train_RMSE': train_metrics['RMSE'],
                    'Train_R2': train_metrics['R2'],
                    'Val_MAE': val_metrics['MAE'],
                    'Val_MSE': val_metrics['MSE'],
                    'Val_RMSE': val_metrics['RMSE'],
                    'Val_R2': val_metrics['R2'],
                    'Test_MAE': test_metrics['MAE'],
                    'Test_MSE': test_metrics['MSE'],
                    'Test_RMSE': test_metrics['RMSE'],
                    'Test_R2': test_metrics['R2']
                }
                
                all_results.append(result)
                print(f"Successful fit with metrics: Train RMSE={train_metrics['RMSE']:.4f}, Val RMSE={val_metrics['RMSE']:.4f}, Test RMSE={test_metrics['RMSE']:.4f}")
                
            except Exception as e:
                print(f"Failed to fit model ARIMA({p},{d},{q}) for {currency_pair} {type}. Error: {str(e)}")
                all_results.append({
                    'Currency': currency_pair,
                    'Type': type,
                    'p': p, 'd': d, 'q': q,
                    'AIC': None,
                    'BIC': None,
                    'Error': str(e)
                })

# Convert results to DataFrame
results_df = pd.DataFrame(all_results)

# Find best models based on validation RMSE
best_models = []
for currency_pair in dfs.keys():
    for type in types:
        subset = results_df[(results_df['Currency'] == currency_pair) & (results_df['Type'] == type)]
        if not subset.empty and 'Val_RMSE' in subset.columns:
            # Filter out rows with NaN values in Val_RMSE
            valid_subset = subset.dropna(subset=['Val_RMSE'])
            if not valid_subset.empty:
                best_idx = valid_subset['Val_RMSE'].idxmin()
                best_model = valid_subset.loc[best_idx]
                best_models.append({
                    'Currency': currency_pair,
                    'Type': type,
                    'Best_Model': f"ARIMA({int(best_model['p'])},{int(best_model['d'])},{int(best_model['q'])})",
                    'Val_RMSE': best_model['Val_RMSE'],
                    'Test_RMSE': best_model['Test_RMSE'],
                    'Test_R2': best_model['Test_R2']
                })

# Convert best models to DataFrame
best_models_df = pd.DataFrame(best_models)

# Save results
results_df.to_csv("arima_evaluation_results_full.csv", index=False)
best_models_df.to_csv("arima_best_models.csv", index=False)

print("\nAnalysis complete. Results saved to CSV files.")
print("\nBest models by validation RMSE:")
print(best_models_df.to_string())