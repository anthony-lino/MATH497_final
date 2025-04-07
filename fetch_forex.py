import os
import requests
import pandas as pd
from datetime import datetime, timedelta

API_KEY = 'IB9KTHT25OD9TEDO'
BASE_URL = 'https://www.alphavantage.co/query?'

def fetch_forex_data(from_currency, to_currency, start_date, end_date):
    function = 'FX_DAILY'
    outputsize = 'full'

    params = {
        'function': function,
        'from_symbol': from_currency,
        'to_symbol': to_currency,
        'apikey': API_KEY,
        'outputsize': outputsize,
        'datatype': 'json'
    }

    response = requests.get(BASE_URL, params=params)
    data = response.json()

    if 'Time Series FX (Daily)' not in data:
        raise ValueError(f"Error fetching data: {data.get('Note', data.get('Information', 'Unknown error'))}")

    date_list = []
    for date, ohlc in data['Time Series FX (Daily)'].items():
        if start_date <= date <= end_date:
            date_list.append({
                'date': date,
                'open': float(ohlc['1. open']),
                'high': float(ohlc['2. high']),
                'low': float(ohlc['3. low']),
                'close': float(ohlc['4. close'])
            })

    df = pd.DataFrame(date_list)
    return df

def save_forex_data(df, from_currency, to_currency):
    filename = f"{from_currency}_{to_currency}_5_years.csv"
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

def main():
    currency_pairs = [
        ('USD', 'EUR'),
        ('USD', 'CNY'),
        ('USD', 'JPY')
    ]

    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=5*365)

    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    for from_currency, to_currency in currency_pairs:
        print(f"Fetching {from_currency}/{to_currency} data...")
        df = fetch_forex_data(from_currency, to_currency, start_date_str, end_date_str)
        save_forex_data(df, from_currency, to_currency)

if __name__ == '__main__':
    main()
