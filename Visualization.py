import pandas as pd
import matplotlib.pyplot as plt

# 配置
pairs = ['USD_EUR', 'USD_CNY', 'USD_JPY']
price_types = ['open', 'high', 'low', 'close']
titles = {'open': 'Open Price', 'high': 'High Price', 'low': 'Low Price', 'close': 'Close Price'}
colors = {'open': 'red', 'high': 'orange', 'low': 'blue', 'close': 'green'}

# 创建画布：3 行 × 4 列
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 12), sharex=False)
fig.suptitle('Forex Exchange Rates (Past 5 Years)', fontsize=16)

# 遍历每个货币对与特征
for i, pair in enumerate(pairs):
    df = pd.read_csv(f'{pair}_5_years.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    for j, price_type in enumerate(price_types):
        ax = axes[i, j]
        ax.plot(df['date'], df[price_type], label=f'{pair} - {price_type}',
                color=colors[price_type])
        ax.set_title(f'{pair} - {titles[price_type]}', fontsize=10)
        ax.set_xlabel('Date')
        ax.set_ylabel('Rate')
        ax.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('forex_3x4_plot.png', dpi=300)
plt.show()
