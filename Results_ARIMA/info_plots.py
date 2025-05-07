# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Define the data
data = pd.read_csv("arima_evaluation_results_full.csv")
# Create DataFrame
df = pd.DataFrame(data)

df = df[(df['Currency'] == 'USD_CNY') & (df['Type'] == 'close')]

# Create labels for x-axis
df['model'] = df.apply(lambda row: f"({row['p']:.0f},{row['d']:.0f},{row['q']:.0f})", axis=1)
df['p'] = df['p'].astype(np.int64)
df['q'] = df['q'].astype(np.int64)
df['d'] = df['d'].astype(np.int64)
df['AIC'] = df['AIC'].astype(np.int64)
df['BIC'] = df['BIC'].astype(np.int64)

# Set up figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Define line styles for different d values
line_styles = {0: '-', 1: '--', 2: '-.', 3: ':'}
markers = {0: 'o', 1: 's', 2: '^', 3: 'D'}

# ========================= PLOT 1: p=0 =========================
p0_df = df[df['p'] == 0].sort_values(by=['d', 'q'])

# Create plot for p=0, varying q on x-axis, separate lines for each d
for d in range(4):
    d_data = p0_df[p0_df['d'] == d]
    
    # Plot AIC for this d value (blue lines)
    axes[0].plot(d_data['q'], d_data['AIC'], 
                 color='blue', linestyle=line_styles[d], marker=markers[d],
                 label=f'AIC (d={d})')
    
    # Plot BIC for this d value (red lines)
    axes[0].plot(d_data['q'], d_data['BIC'], 
                 color='red', linestyle=line_styles[d], marker=markers[d],
                 label=f'BIC (d={d})')

# Configure first subplot
axes[0].set_title('p=0', fontsize=14)
axes[0].set_xlabel('q Parameter', fontsize=12)
axes[0].set_ylabel('Information Criteria Values', fontsize=12)
axes[0].grid(True, linestyle='--', alpha=0.7)
axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)

# ========================= PLOT 2: q=0 =========================
q0_df = df[df['q'] == 0].sort_values(by=['d', 'p'])

# Create plot for q=0, varying p on x-axis, separate lines for each d
for d in range(4):
    d_data = q0_df[q0_df['d'] == d]
    
    # Plot AIC for this d value (blue lines)
    axes[1].plot(d_data['p'], d_data['AIC'], 
                 color='blue', linestyle=line_styles[d], marker=markers[d],
                 label=f'AIC (d={d})')
    
    # Plot BIC for this d value (red lines)
    axes[1].plot(d_data['p'], d_data['BIC'], 
                 color='red', linestyle=line_styles[d], marker=markers[d],
                 label=f'BIC (d={d})')

# Configure second subplot
axes[1].set_title('q=0', fontsize=14)
axes[1].set_xlabel('p Parameter', fontsize=12)
axes[1].set_ylabel('Information Criteria Values', fontsize=12)
axes[1].grid(True, linestyle='--', alpha=0.7)
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)

# Common y-limit for both plots for better comparison
y_min = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0])
y_max = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
axes[0].set_ylim(y_min, y_max)
axes[1].set_ylim(y_min, y_max)

# Add a custom legend to clearly show line styles
fig.text(0.5, 0.01, 'USD-CNY, market close\nLine styles: solid (d=0), dashed (d=1), dashdot (d=2), dotted (d=3)', 
         ha='center', fontsize=12)


plt.tight_layout()
plt.subplots_adjust(bottom=0.25)  # Make room for legend below plots
plt.show()
plt.savefig('arima_info_model_comparison.png', dpi=300, bbox_inches='tight')