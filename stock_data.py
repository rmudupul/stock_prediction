import pandas as pd
import numpy as np
import yfinance as yf
import os
import contextlib

# Set default limit and offset values
limit = None  # Default to process all symbols if no limit is provided
offset = 0    # Default starting offset

timeframes = {
    'daily': '1d',
    'weekly': '1wk',
    'monthly': '1mo',
    '3mo': '3mo',
    '6mo': '6mo',
    '1y': '1y',
    '5y': '5y'
}

# Timeframes for different data resolutions
timeframe_sma_ema = {
    'daily': {
        'SMA': [10, 50, 200],
        'EMA': [12, 26, 50]
    },
    'weekly': {
        'SMA': [4, 13, 52],
        'EMA': [9, 26, 52]
    },
    'monthly': {
        'SMA': [3, 12, 24],
        'EMA': [6, 12]
    },
    '3mo': {
        'SMA': [1, 4, 8],
        'EMA': [1, 4, 8]
    },
    '6mo': {
        'SMA': [1, 2, 4],
        'EMA': [1, 2, 4]
    },
    '1y': {
        'SMA': [1, 2, 5],
        'EMA': [1, 2, 5]
    },
    '5y': {
        'SMA': [2, 5, 10],
        'EMA': [2, 5, 10]
    }
}

# Function to calculate SMAs
def calculate_sma(data, timeframes):
    for period in timeframes:
        # Use the minimum of the available days or the defined period for SMA
        data[f'SMA_{period}'] = data['Close'].rolling(window=period, min_periods=1).mean()
    return data

# Function to calculate EMAs
def calculate_ema(data, timeframes):
    for period in timeframes:
        data[f'EMA_{period}'] = data['Close'].ewm(span=period, adjust=False).mean()
    return data

# Function to calculate simple and log returns
def calculate_returns(data):
    # Simple returns
    data['Simple Return'] = data['Close'].pct_change()
    
    # Logarithmic returns
    data['Log Return'] = np.log(data['Close'] / data['Close'].shift(1))
    
    return data

# Function to calculate historical volatility with forward-fill for empty values
def calculate_volatility(data, timeframe):
    # Set the appropriate number of periods in a year depending on the timeframe
    periods_in_year = {
        'daily': 252,     # 252 trading days in a year
        'weekly': 52,     # 52 weeks in a year
        'monthly': 12,    # 12 months in a year
        '3mo': 4,         # 4 quarters in a year (3-month periods)
        '6mo': 2,         # 2 half-year periods in a year
        '1y': 1,          # 1 year
        '5y': 1/5         # 1 year spans 5 years (inverse)
    }
    
    # Default window size for rolling volatility is set to 21 periods (can be adjusted if needed)
    window_size = 21

    # Use the number of periods per year specific to the timeframe
    annualization_factor = np.sqrt(periods_in_year[timeframe])

    # Calculate daily (or respective timeframe) volatility and annualize it
    def volatility_calculation(x):
        if len(x) < 2:  # Less than 2 points, we can't calculate standard deviation
            return np.nan
        return np.std(x, ddof=0) * annualization_factor

    data['Volatility'] = data['Log Return'].rolling(window=window_size, min_periods=1).apply(
        volatility_calculation, raw=True
    )

    # Fill missing values with available data if there are less than 21 data points
    for i in range(len(data)):
        if pd.isna(data['Volatility'].iloc[i]) and i >= window_size:
            available_data = data['Log Return'].iloc[i-window_size+1:i+1]
            if len(available_data) > 0:
                data['Volatility'].iloc[i] = volatility_calculation(available_data)

    return data
    # Set the appropriate number of periods in a year depending on the timeframe
    periods_in_year = {
        'daily': 252,     # 252 trading days in a year
        'weekly': 52,     # 52 weeks in a year
        'monthly': 12,    # 12 months in a year
        '3mo': 4,         # 4 quarters in a year (3-month periods)
        '6mo': 2,         # 2 half-year periods in a year
        '1y': 1,          # 1 year
        '5y': 1/5         # 1 year spans 5 years (inverse)
    }
    
    # Default window size for rolling volatility is set to 21 periods (can be adjusted if needed)
    window_size = 21

    # Use the number of periods per year specific to the timeframe
    annualization_factor = np.sqrt(periods_in_year[timeframe])

    # Calculate daily (or respective timeframe) volatility and annualize it
    data['Volatility'] = data['Log Return'].rolling(window=window_size, min_periods=1).apply(
            lambda x: np.std(x, ddof=0) * annualization_factor, raw=True
        )
    
    # Fill NaN or 0 values with the last available non-null value
    data['Volatility'] = data['Volatility'].replace(0, np.nan).fillna(method='ffill')

    return data



# Function to calculate MACD for all timeframes
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    data['MACD'] = data['Close'].ewm(span=short_window, adjust=False).mean() - data['Close'].ewm(span=long_window, adjust=False).mean()
    data['Signal Line'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()  # Signal line
    data['Histogram'] = data['MACD'] - data['Signal Line']
    return data

# Function to calculate RSI
def calculate_rsi(data, period=14):
    # Calculate the difference in price from the previous step
    delta = data['Close'].diff()

    # Calculate gains (positive deltas) and losses (negative deltas)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Initialize average gains and losses
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()

    # Use the first value of average gain/loss to smooth the remaining values
    for i in range(period, len(data)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i]) / period

    # Calculate the relative strength (RS)
    rs = avg_gain / avg_loss

    # Calculate RSI
    data['RSI'] = 100 - (100 / (1 + rs))

    return data


# Function to calculate Bollinger Bands
def calculate_bollinger_bands(data, window=20, num_std_dev=2):
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    data['Middle Band'] = data['Close'].rolling(window=window, min_periods=1).mean()
    
    # Use NumPy's std
    rolling_std = data['Close'].rolling(window=window, min_periods=1).apply(lambda x: np.std(x, ddof=0))  # ddof=0 for population std deviation

    data['Upper Band'] = data['Middle Band'] + (rolling_std * num_std_dev)
    data['Lower Band'] = data['Middle Band'] - (rolling_std * num_std_dev)

    return data


# Ensure the 'historical data' directory exists
output_dir = 'hist_data'
os.makedirs(output_dir, exist_ok=True)

# Load the list of symbols and clean data
data = pd.read_csv("http://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt", sep='|')
data_clean = data[data['Test Issue'] == 'N']
symbols = data_clean['NASDAQ Symbol'].tolist()

# Adjust the limit and range
limit = limit if limit else len(symbols)
end = min(offset + limit, len(symbols))
is_valid = [False] * len(symbols)

# Force silencing of verbose API output
with open(os.devnull, 'w') as devnull:
    with contextlib.redirect_stdout(devnull):
        for i in range(offset, end):
            s = symbols[i]
            
            # Download data for different timeframes
            for timeframe, period in timeframes.items():
                data = yf.download(s, period='max', interval=period)
                
                # Skip if no data is found
                if data.empty:
                    continue

                # Calculate additional features
                data = calculate_returns(data)
                data = calculate_volatility(data, timeframe)

                # Calculate SMAs and EMAs for this timeframe
                data = calculate_sma(data, timeframe_sma_ema[timeframe]['SMA'])
                data = calculate_ema(data, timeframe_sma_ema[timeframe]['EMA'])

                # Calculate MACD for all timeframes
                data = calculate_macd(data)

                # Calculate RSI for all timeframes (commonly 14 days)
                data = calculate_rsi(data)

                # Calculate Bollinger Bands
                data = calculate_bollinger_bands(data)

                # Save the data for each timeframe
                data.to_csv(f'{output_dir}/{s}_{timeframe}.csv')

                # Save dividends, splits, and other data for the symbol
                stock = yf.Ticker(s)

                # Uncomment to save other data if needed
                # dividends = stock.dividends
                # if not dividends.empty:
                #     dividends.to_csv(f'{output_dir}/{s}_dividends.csv')

                # splits = stock.splits
                # if not splits.empty:
                #     splits.to_csv(f'{output_dir}/{s}_splits.csv')

                # quarterly_earnings = stock.quarterly_earnings  # Quarterly earnings data
                # if quarterly_earnings is not None and not quarterly_earnings.empty:
                #     quarterly_earnings.to_csv(f'{output_dir}/{s}_quarterly_earnings.csv')

                # financials = stock.financials  # Yearly financials
                # if financials is not None and not financials.empty:
                #     financials.to_csv(f'{output_dir}/{s}_financials.csv')

                # quarterly_financials = stock.quarterly_financials  # Quarterly financials
                # if quarterly_financials is not None and not quarterly_financials.empty:
                #     quarterly_financials.to_csv(f'{output_dir}/{s}_quarterly_financials.csv')

                # recommendations = stock.recommendations  # Analyst recommendations
                # if recommendations is not None and not recommendations.empty:
                #     recommendations.to_csv(f'{output_dir}/{s}_recommendations.csv')

            # Mark the symbol as valid if data was saved
            is_valid[i] = True

print('Total number of valid symbols downloaded = {}'.format(sum(is_valid)))
