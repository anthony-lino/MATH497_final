# %%
import pandas as pd
from sklearn.metrics import mean_squared_error

# Load CSV
df = pd.read_csv("USD_JPY_5_years/test_predictions.csv")  # Replace with your actual file path

# Metrics to compare
metrics = ['Close','High','Low','Open']

# Calculate and print MSE for each metric
for metric in metrics:
    true_col = f'True_{metric}'
    pred_col = f'Predicted_{metric}'
    
    mse = mean_squared_error(df[true_col], df[pred_col])
    print(f"MSE for {metric}: {mse:.6f}")