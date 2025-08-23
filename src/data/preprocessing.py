import yfinance as yf
import math
import numpy as np
import pandas as pd
import ta

# training data preprocessing
def preprocess(stock):
    data = yf.download(train, period='20y', auto_adjust=True)
    df = data.xs(train, level=1, axis=1).copy()
    close = data['Close'][train]
    log_close = np.log(close)
    returns = log_close.diff()
    volatility = returns.rolling(20).std()
    df['Log return'] = returns
    df['Log close'] = log_close
    df['Volatility'] = volatility
    # may want TA later
    # df = ta.add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
    return df

def segment(df, begin, end):
    return df.loc[begin:end]

# train = 'AAPL'
# test = 'AAPL'
train = 'SPY'
test = 'SPY'

train_df = preprocess(train)
test_df = preprocess(test)
spy_df = preprocess('SPY')

train_df = segment(train_df, begin = pd.Timestamp("2010-01-01"), end = pd.Timestamp("2019-01-01"))
test_df = segment(test_df, begin = pd.Timestamp("2019-03-01"), end = pd.Timestamp("2025-01-01"))
spy_df = segment(spy_df, begin = pd.Timestamp("2022-01-01"), end = pd.Timestamp("2025-01-01"))