
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
from collections import namedtuple
from sklearn.preprocessing import LabelEncoder


def encode_labels(data, columns):
    for column in columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
    return data


def train_test_split(data, test_size, y_name):
    train_test = namedtuple('train_test', ['x_train', 'x_test', 'y_train', 'y_test'])
    split_row = len(data) - int(test_size * len(data))
    train_data = data.iloc[:split_row]
    test_data = data.iloc[split_row:]
    return train_test(x_train=train_data.drop(y_name, axis=1).to_numpy(), x_test=test_data.drop(y_name, axis=1).to_numpy(), y_train=train_data[y_name].to_numpy(), y_test=test_data[y_name].to_numpy())

def train_test_split_numpy (data, test_size, y_index=-1):
    train_test = namedtuple('train_test', ['x_train', 'x_test', 'y_train', 'y_test'])
    split_row = int(len(data) * (1 - test_size))
    x_train = np.array(data[:split_row, :y_index].tolist())
    x_test = np.array(data[split_row:, :y_index].tolist())
    y_train = np.array(data[:split_row, y_index].tolist())
    y_test = np.array(data[split_row:, y_index].tolist())
    return train_test(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)

def get_ohlc_data(tickers, start_date='2017-01-01'):
    ohlc_data = {}
    for ticker in tickers:
        try:
            stock_data = yf.download(ticker, start=start_date)
            ohlc_data[ticker] = stock_data
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    return ohlc_data

import numpy as np
import scipy.stats as stats
import pandas as pd

def compute_stats(series: pd.Series, name: str):
    stats_dict = {
        'Name': name,
        'Minimum': np.min(series),
        'Q1': series.quantile(0.25),
        'Median': np.median(series),
        'Q3': series.quantile(0.75),
        'Maximum': np.max(series),
        'Range': np.ptp(series),
        'Mean': np.mean(series),
        'Standard deviation': np.std(series),
        'Sum': np.sum(series),
        'Coefficient of variation': stats.variation(series),
        'Median Absolute Deviation': stats.median_abs_deviation(series),
        'Kurtosis': stats.kurtosis(series),
        'Skewness': stats.skew(series)
    }    
    return stats_dict


def getWeights(d,lags):
    # return the weights from the series expansion of the differencing operator
    # for real orders d and up to lags coefficients
    w=[1]
    for k in range(1,lags):
        w.append(-w[-1]*((d-k+1))/k)
    w=np.array(w).reshape(-1,1) 
    return w




def ts_differencing(series, order, lag_cutoff):
    # return the time series resulting from (fractional) differencing
    # for real orders order up to lag_cutoff coefficients
    
    weights=getWeights(order, lag_cutoff)
    res=0
    for k in range(lag_cutoff):
        res += weights[k]*series.shift(k).fillna(0)
    return res[lag_cutoff:] 




from itertools import product
import polars as pl
from sklearn.preprocessing import MinMaxScaler

class TimeSeries:
    def __init__(self, ticker: str, date: str = '2017-01-01'):
        self.data = get_ohlc_data([ticker],date)[ticker].reset_index()
        self.close_price = self.data['Close'].copy()
        self.date = self.data['Date'].copy()
        self.ticker = ticker
        self.predictions = {}

    def construct_returns(self):
        self.data['Returns'] = self.data['Close'].pct_change()
        return self
    
    def profitability (self, column, threshold):
        self.data['Profitable'] = (self.data[column] > threshold).astype(int)
        return self
    
    def construct_technical_indicators(self, indicators: list, windows: list):
        for func, window in product(indicators, windows):
            series = func(self.data, window)
            self.data[series.name] = series
        return self
    
    def lag_column(self, column, skip_lags,n_lags):
        self.data = pl.DataFrame(self.data)
        self.data = self.data.with_columns([pl.col(column).shift(lag).alias(f"{column}_lag{lag}") for lag in range(skip_lags, n_lags + 1)]).to_pandas()
        return self

   

    def drop_columns(self,cols):
        if isinstance(cols, str):
            cols = [cols]
        if not set(cols).issubset(self.data.columns):
            print(self.data.columns)
            raise ValueError(f"Some of the {cols} not present in the data")
        self.data = self.data.drop(columns=cols)
        return self
    

    def pca(self, explained_variance_threshold, target):
        # Perform PCA transformation
        pca = PCA()
        pca.fit(self.data.drop(target, axis=1).to_numpy())
        # Calculate cumulative explained variance ratio
        cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
        num_components = np.argmax(cumulative_variance_ratio >= explained_variance_threshold) + 1
        # Transform the original matrix using the selected number of components
        transformed_features = pca.transform(self.data.drop(target, axis=1).to_numpy())[:, :num_components]
        self.data = transformed_features
        print(f"Number of components explaining at least {explained_variance_threshold*100}% of the variation:", num_components)
        print("Shape of transformed features matrix:", transformed_features.shape)
        return self

    def apply_min_max_scaling (self):
        scaler_input = MinMaxScaler(feature_range=(-1, 1))
        scaler_input.fit(self.data)
        self.data = scaler_input.transform(self.data)
        return self
    
    def train_test_split(self, test_size, target):    
        if isinstance(self.data, pd.DataFrame):
            self.modelling_data = train_test_split(data=self.data,test_size=test_size,y_name=target)
        else:
            self.modelling_data = train_test_split_numpy(data=self.data,test_size=test_size)
        return self

    def encode_labels(self, columns_start_with: str):
        self.data = self.data.dropna()
        self.data = encode_labels(self.data, [column for column in self.data.columns if column.startswith(columns_start_with)])
        return self
    
    def shift_column(self, column, n_shifts, drop_nulls):
        if n_shifts > 0:
            suffix = 'Lagged'
        else:
            suffix = 'Ahead'       
        self.data[f"{column}_{suffix}_{abs(n_shifts)}"] = self.data[column].shift(n_shifts)
        if drop_nulls:
            self.data = self.data.dropna()
        return self
    
    def add_index(self):
        self.data = self.data.reset_index()
        return self

    def plot_correlation_matrix(self):        
        plt.figure(figsize=(10,10))
        sns.heatmap(self.data.dropna().corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=2)
        plt.show()
        return self
    
    def dropna(self):
        self.data = self.data.dropna()
        return self

    def get_reshaped_X_for_LSTM (self, which_data, min_max_scale=False):
        if min_max_scale:
            scaler_input = MinMaxScaler(feature_range=(-1, 1))
            scaler_input.fit(self.modelling_data.x_train)
            if which_data == 'train':
                return np.expand_dims(scaler_input.transform(self.modelling_data.x_train), axis=1)
            else:
                return np.expand_dims(scaler_input.transform(self.modelling_data.x_test), axis=1)
        if which_data == 'train':
            return np.expand_dims(self.modelling_data.x_train, axis=1)
        else:
            return np.expand_dims(self.modelling_data.x_test, axis=1)    
    

    def predict(self, model, model_name: str):
        self.predictions[model_name] = model.predict(self.modelling_data.x_test)
        return self 

    @property
    def input_feature_shape (self):
        return 1, self.modelling_data.x_train.shape[1]

def min_max_scale(time_series, which_data='x'):
    scaler_input = MinMaxScaler(feature_range=(-1, 1))
    if which_data == 'x':
        scaler_input.fit(time_series.modelling_data.x_train)
        return scaler_input.transform(time_series.modelling_data.x_train), scaler_input.transform(time_series.modelling_data.x_test)
    else:
        y_train = time_series.modelling_data.y_train.reshape(len(time_series.modelling_data.y_train), 1)
        y_test= time_series.modelling_data.y_test.reshape(len(time_series.modelling_data.y_test), 1)
        scaler_input.fit(y_train)
        return scaler_input.transform(y_train), scaler_input.transform(y_test)




from tsfresh.feature_extraction import feature_calculators
import numpy as np
from tsfresh.feature_extraction import feature_calculators

def calculate_summary_statistics(time_series):
    """
    Calculate summary statistics for the given time series.

    Parameters:
    - time_series (list or array-like): The input time series.

    Returns:
    - dict: A dictionary containing summary statistics.
    """

    # Absolute Sum of Changes
    abs_sum_of_changes = feature_calculators.absolute_sum_of_changes(time_series)
    abs_max = feature_calculators.absolute_maximum(time_series)
    # Descriptive Statistics on Autocorrelation
    autocorrelation = feature_calculators.autocorrelation(time_series, lag=100)

    # Approximate Entropy
    approximate_entropy = feature_calculators.approximate_entropy(time_series, m=30, r=0.7 * np.std(time_series))

    # ADF Statistic (Augmented Dickey-Fuller test)
    adf_statistic = feature_calculators.augmented_dickey_fuller(time_series, param=[{"attr": "pvalue", "autolag": "AIC"}])

    # Create a dictionary to store the results
    summary_stats = {
        "Abs Maximum": abs_max,
        'Abs Sum of Changes': abs_sum_of_changes,
        'Autocorrelation': autocorrelation,
        'Approximate Entropy': approximate_entropy,
        'ADF Stat (p Value)': adf_statistic
    }

    return summary_stats
def inverse_transform_predictions (ticker: TimeSeries, preds: np.ndarray):
    scaler_input = MinMaxScaler(feature_range=(-1, 1))
    scaler_input.fit(ticker.modelling_data.y_train.reshape(len(ticker.modelling_data.y_train), 1))
    return scaler_input.inverse_transform(preds.reshape(-1, 1)).flatten()


def get_actual_and_predicted (ticker: TimeSeries, model):
    actual_and_predicted = namedtuple('ActualAndPredicted', ['actual', 'predicted'])
    preds = model.model.predict(ticker.get_reshaped_X_for_LSTM(which_data='test', min_max_scale=True)).flatten()
    true = ticker.modelling_data.y_test
    return actual_and_predicted(true, inverse_transform_predictions(ticker,preds))

def plot_actual_vs_predicted (ticker: TimeSeries, model, title):
    ActualAndPredicted = namedtuple('ActualAndPredicted', ['actual', 'predicted'])
    preds = model.model.predict(ticker.get_reshaped_X_for_LSTM(which_data='test', min_max_scale=True)).flatten()
    true = ticker.modelling_data.y_test
    idx = range(len(preds))
    from visualisation import plot_multiline_chart
    plot_multiline_chart([(idx, inverse_transform_predictions(ticker, preds), "Predicted"), (idx, true, 'Actual')], title, 'Index', 'Price')
    return ActualAndPredicted(true, inverse_transform_predictions(ticker,preds))


def plot_actual_vs_predicted_mlp (ticker: TimeSeries, model, title):
    ActualAndPredicted = namedtuple('ActualAndPredicted', ['actual', 'predicted'])
    preds = model.model.predict(ticker.modelling_data.x_test).flatten()
    true = ticker.modelling_data.y_test
    idx = range(len(preds))
    from visualisation import plot_multiline_chart
    plot_multiline_chart([(idx, inverse_transform_predictions(ticker, preds), "Predicted"), (idx, true, 'Actual')], title, 'Index', 'Price')
    return ActualAndPredicted(true, inverse_transform_predictions(ticker,preds))