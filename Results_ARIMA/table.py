# %%
import pandas as pd

df = pd.read_csv("arima_evaluation_results_full.csv")
# %%

df.groupby(['Currency', 'Type']).apply(lambda x: x.loc[x['Test_MSE'].idxmin()]).reset_index(drop=True).to_csv("arima_evaluation_results_best_MSE.csv", index=False)