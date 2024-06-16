import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from statsmodels.stats.weightstats import ztest

## Getting stock price data
#
def prepPrices(price_history, n_days,  mov_avg=0, target_col='Close'):
        
    # Filling NaNs with the most recent values for any missing data
    prices = price_history.fillna(method='ffill')
    
    # Getting the N Day Moving Average and rounding the values for some light data preprocessing
    if mov_avg>0:
        prices['MA'] = prices[[target_col]].rolling(
        window=mov_avg).mean().apply(lambda x: round(x, 2))
    # Dropping the Nans
    prices.dropna(inplace=True)
    #ignore time in index to merge with other datasets later
    if isinstance(price_history.index, pd.DatetimeIndex):
        prices.index=[pd.to_datetime(str(x).split('T')[0]) for x in prices.index.values]
    else:
       prices.index=[pd.to_datetime(str(x).split()[0]) for x in prices.index.values]
    prices.index=prices.index.tz_localize(None)
 
    return prices

def getStockPrices(stock, n_days, mov_avg, target_col='Close'):
    """
    Gets stock prices from now to N days ago and training amount will be in addition 
    to the number of days to train.
    """
    
    # Designating the Ticker
    ticker = yf.Ticker(stock)

    # Getting all price history
    price_history = ticker.history(period=f"{n_days}d")
    
    # Check on length
    #if len(price_history)<n_days+training_days+mov_avg:
    #    return pd.DataFrame(), price_history
    
    prices=prepPrices(price_history, n_days, mov_avg, target_col)

    return price_history, prices

#additional stocks
#indicators WITH LAG
def prepare_data_feat(prices, training_days,n_lags, target_col='MA', extra_features=['RSI']):
    #split prices into train and test
    dataset_train, dataset_test=prices.iloc[:training_days], prices.iloc[training_days:]
    cols=[target_col]+extra_features
    sc_extra={}
    features_scaled={}
    for f in cols:#extra_features:
        sc_extra[f]=MinMaxScaler(feature_range = (0, 1))
        features_scaled[f]=sc_extra[f].fit_transform(dataset_train[[f]].values)
    #Creating a data structure with n_lags timesteps and 1 output
    X_train = []
    y_train = []
    for i in range(n_lags,  len(features_scaled[target_col])):#len(training_set_scaled)):
        x_train=features_scaled[target_col][i-n_lags:i].squeeze()
        for f in extra_features:
            x_train=np.hstack((x_train, features_scaled[f][i-n_lags:i].squeeze()))
        X_train.append(x_train)
        y_train.append(features_scaled[target_col][i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    # Reshape for LSTM to include the new dimension
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1] // len(cols), len(cols)))

    #prepare test set
    real_stock_price = dataset_test[[target_col]]
    dataset_total = pd.concat((dataset_train[cols], dataset_test[cols]), axis = 0)
    dataset_total = dataset_total[len(dataset_total) - len(dataset_test) - n_lags:]
    for f in cols:
        dataset_total[f]=sc_extra[f].transform(dataset_total[[f]])

    X_test = []
    for i in range(n_lags, len(dataset_total)):
        x_test=dataset_total[target_col][i-n_lags:i].values
        for f in extra_features:
            x_test=np.hstack((x_test, dataset_total[f][i-n_lags:i].values))
        X_test.append(x_test)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]// len(cols), (len(extra_features)+1)))

    return X_train, y_train, X_test, real_stock_price, sc_extra[target_col]


def prepare_data_feat_horizon(prices, training_days, n_lags, forecast_horizon, target_col='MA', extra_features=['RSI']):
    """
    Prepares data for LSTM model considering a window of n_lags past observations to create X_train and X_test.
    y_train is prepared for long-term forecasting with a specified forecast horizon.

    Args:
        prices (pd.DataFrame): DataFrame containing the historical price data.
        training_days (int): Number of days to use for training.
        n_lags (int): Number of past observations to use as features.
        forecast_horizon (int): Number of days ahead to forecast.
        target_col (str): The target column to forecast.
        extra_features (list): List of additional features to include.

    Returns:
        tuple: (X_train, y_train, X_test, real_stock_price, sc_target) where:
            - X_train (np.ndarray): Training features.
            - y_train (np.ndarray): Training targets.
            - X_test (np.ndarray): Testing features.
            - real_stock_price (pd.DataFrame): Real stock prices for comparison.
            - sc_target (MinMaxScaler): Scaler used for the target column.
    """
    # Split prices into train and test
    dataset_train, dataset_test = prices.iloc[:training_days], prices.iloc[training_days:]
    cols = [target_col] + extra_features
    sc_extra = {}
    features_scaled = {}

    # Scale features
    for f in cols:
        sc_extra[f] = MinMaxScaler(feature_range=(0, 1))
        features_scaled[f] = sc_extra[f].fit_transform(dataset_train[[f]].values)

    # Create a data structure with n_lags timesteps and forecast_horizon output
    X_train = []
    y_train = []
    for i in range(n_lags, len(features_scaled[target_col]) - forecast_horizon):
        x_train = features_scaled[target_col][i - n_lags:i].squeeze()
        for f in extra_features:
            x_train = np.hstack((x_train, features_scaled[f][i - n_lags:i].squeeze()))
        X_train.append(x_train)
        y_train.append(features_scaled[target_col][i:i + forecast_horizon, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    # Reshape for LSTM to include the new dimension
    X_train = np.reshape(X_train, (X_train.shape[0], n_lags, len(cols)))

    # Prepare test set
    real_stock_price = dataset_test[['Open', 'Close', 'MA']]
    dataset_total = pd.concat((dataset_train[cols], dataset_test[cols]), axis=0)
    dataset_total = dataset_total[len(dataset_total) - len(dataset_test) - n_lags:]

    for f in cols:
        dataset_total[f] = sc_extra[f].transform(dataset_total[[f]])

    X_test = []
    for i in range(n_lags, len(dataset_total) - forecast_horizon):
        x_test = dataset_total[target_col][i - n_lags:i].values
        for f in extra_features:
            x_test = np.hstack((x_test, dataset_total[f][i - n_lags:i].values))
        X_test.append(x_test)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], n_lags, len(cols)))

    return X_train, y_train, X_test, real_stock_price, sc_extra[target_col]

def prepare_data_feat_step(prices, training_days, n_lags, step, target_col='MA', extra_features=['RSI']):
    """
    Prepares data for LSTM model considering a window of n_lags past observations to create X_train and X_test.
    y_train is prepared for long-term forecasting with a specified forecast horizon.

    Args:
        prices (pd.DataFrame): DataFrame containing the historical price data.
        training_days (int): Number of days to use for training.
        n_lags (int): Number of past observations to use as features.
        step (int): The day to forecast in future. If ==0 means that the next day will be predicted
        target_col (str): The target column to forecast.
        extra_features (list): List of additional features to include.

    Returns:
        tuple: (X_train, y_train, X_test, real_stock_price, sc_target) where:
            - X_train (np.ndarray): Training features.
            - y_train (np.ndarray): Training targets.
            - X_test (np.ndarray): Testing features.
            - real_stock_price (pd.DataFrame): Real stock prices for comparison.
            - sc_target (MinMaxScaler): Scaler used for the target column.
    """
    # Split prices into train and test
    dataset_train, dataset_test = prices.iloc[:training_days], prices.iloc[training_days:]
    cols = [target_col] + extra_features
    sc_extra = {}
    features_scaled = {}

    # Scale features
    for f in cols:
        sc_extra[f] = MinMaxScaler(feature_range=(0, 1))
        features_scaled[f] = sc_extra[f].fit_transform(dataset_train[[f]].values)

    # Create a data structure with n_lags timesteps and step output
    X_train = []
    y_train = []
    for i in range(n_lags, len(features_scaled[target_col]) - step):
        x_train = features_scaled[target_col][i - n_lags:i].squeeze()
        for f in extra_features:
            x_train = np.hstack((x_train, features_scaled[f][i - n_lags:i].squeeze()))
        X_train.append(x_train)
        #y_train.append(dataset_train[target_col].iloc[i + step])
        y_train.append(features_scaled[target_col][i + step])
    X_train, y_train = np.array(X_train), np.array(y_train)
    # Reshape for LSTM to include the new dimension
    X_train = np.reshape(X_train, (X_train.shape[0], n_lags, len(cols)))

    # Prepare test set
    real_stock_price = dataset_test[[target_col]]
    dataset_total = pd.concat((dataset_train[cols], dataset_test[cols]), axis=0)
    dataset_total = dataset_total[len(dataset_total) - len(dataset_test) - n_lags:]

    for f in cols:
        dataset_total[f] = sc_extra[f].transform(dataset_total[[f]])

    X_test = []
    for i in range(n_lags, len(dataset_total) - step):
        x_test = dataset_total[target_col][i - n_lags:i].values
        for f in extra_features:
            x_test = np.hstack((x_test, dataset_total[f][i - n_lags:i].values))
        X_test.append(x_test)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], n_lags, len(cols)))

    return X_train, y_train, X_test, real_stock_price, sc_extra


def calculate_rsi(data, window=14):
    """
    Calculate the Relative Strength Index (RSI) for a given data set and window period.
    :param data: Pandas Series of stock prices.
    :param window: The period over which to calculate RSI. Commonly used period is 14.
    :return: Pandas Series containing the RSI values.
    """
    # Calculate price differences
    delta = data.diff()

    # Make two series: one for gains and one for losses
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    # Calculate the exponential moving average of gains and losses
    avg_gain = gain.ewm(com=window-1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window-1, min_periods=window).mean()

    # Calculate the RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
