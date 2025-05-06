import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Define the data
data = pd.read_csv("arima_grid_search_results_info_criterion.csv")
# Create DataFrame
df = pd.DataFrame(data)

# Create labels for x-axis
df['model'] = df.apply(lambda row: f"({row['p']:.0f},{row['d']:.0f},{row['q']:.0f})", axis=1)

# df_d = df[df['p']==0]
# df_d = df[df['q']==0]
# Set up the figure with appropriate size
plt.figure(figsize=(10, 6))

# Create positions for the points
x = np.arange(len(df['model']))

# Create the scatter plot with lines
plt.plot(x, df['aic'], 'o-', linewidth=2, markersize=8, label='AIC', color='steelblue')
plt.plot(x, df['bic'], 's-', linewidth=2, markersize=8, label='BIC', color='darkorange')

# Add labels, title and custom x-axis tick labels
plt.xlabel('ARIMA Models (p,d,q)', fontsize=12)
plt.ylabel('Information Criteria Values', fontsize=12)
plt.title('AIC and BIC Values for Different ARIMA Models', fontsize=14)
plt.xticks(x, df['model'], fontsize=11)

# Add a grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Add value labels for each point
for i, (aic_val, bic_val) in enumerate(zip(df['aic'], df['bic'])):
    plt.text(i, aic_val - 1, f'{aic_val:.1f}', ha='center', va='top', fontsize=9)
    plt.text(i, bic_val + 1, f'{bic_val:.1f}', ha='center', va='bottom', fontsize=9)

# Add a legend
plt.legend(loc='best')

# Set y-axis limits to better show the differences
lowest_val = min(df['aic'].min(), df['bic'].min())
highest_val = max(df['aic'].max(), df['bic'].max())
margin = (highest_val - lowest_val) * 0.05
plt.ylim(lowest_val - margin, highest_val + margin)

# Tighten the layout
plt.tight_layout()

# Show the plot
plt.show()

# If you want to save the figure
plt.savefig('arima_info_model_comparison.png', dpi=300, bbox_inches='tight')