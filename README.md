# Stock prediction project plan

# Goal:

Test different state of art time series forecasting models for stock price prediction combining it with additional data from news feed to define the winner and build trading strategy on top.

# SCOPE: AMZN, TSLA, AAPL stocks

# Success metrics:

MAPE for 1 day, 1 week of prediction

Define the trading strategy for the best model/ensemble of models

### 1. Stock Price Data Gathering with Yahoo Finance

**Steps:**

- Use the `yfinance` library in Python to collect historical stock prices.
- Ensure that the data includes open, high, low, close, volume, and adjusted close prices.

**Code Example:**

```python
import yfinance as yf

# Define the stock ticker and time period
ticker = 'AAPL'
start_date = '2010-01-01'
end_date = '2023-12-31'

# Fetch historical data
stock_data = yf.download(ticker, start=start_date, end=end_date)

# Preview the data
print(stock_data.head())

```

### 2. Data Gathering from [The GDELT Project](https://www.gdeltproject.org/) for Sentiment Analysis

**Steps:**

- Use https://github.com/alex9smith/gdelt-doc-api to collect news articles related to the stocks.
- Extract relevant financial news within the timeframe of the historical stock price data.

**Code Example:**

```python
from gdeltdoc import GdeltDoc, Filters

f = Filters(
    keyword = "Amazon amzn",
    start_date = "2023-05-10",
    end_date = "2024-05-11"
)

gd = GdeltDoc()

# Search for articles matching the filters
articles = gd.article_search(f)

# Get a timeline of the number of articles matching the filters
timeline = gd.timeline_search("timelinevol", f)
```

### 3. Prepare Data

### 3.a. Transform Time Series Data for Supervised Learning Models

**Steps:**

- Generate features such as moving averages, trading volume, and other technical indicators like RSI.
- Structure the data to include lagged features for the supervised models like XGBoost and LSTM.

**Code Example:**

```python
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
        y_train.append(features_scaled[target_col][i + step, 0])
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
    for i in range(n_lags, len(dataset_total) - step):
        x_test = dataset_total[target_col][i - n_lags:i].values
        for f in extra_features:
            x_test = np.hstack((x_test, dataset_total[f][i - n_lags:i].values))
        X_test.append(x_test)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], n_lags, len(cols)))

    return X_train, y_train, X_test, real_stock_price, sc_extra[target_col]

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
    
    # Add more technical indicators like MACD (Moving Average Convergence Divergence) and Bollinger Bands for richer feature sets.
```

### 3.b. Transform Time Series Data for Time Series Prediction Models

**Steps:**

- Prepare the data in a sequential format suitable for time series models from packages like NeuralForecast and Darts.

**Code Example:**

```python
pythonCopy code
from darts import TimeSeries
from darts.models import ExponentialSmoothing

# Convert to TimeSeries
series = TimeSeries.from_dataframe(stock_data, 'Date', 'Close')

# Prepare data for modeling
train, val = series.split_before(0.8)

# Initialize model
model = ExponentialSmoothing()

# Train model
model.fit(train)

# Forecast
forecast = model.predict(len(val))
print(forecast)

```

### 3.c. Transform News Data into Sentiment Data Using FinBERT

**Steps:**

- Apply FinBERT model to the news articles to generate sentiment scores.
- Integrate sentiment scores with stock price data.

Tokenize and analyze the sentiment of each article using FinBERT.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load FinBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

# Function to get sentiment
def get_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=1)
    sentiment = torch.argmax(scores, dim=1).item()
    return sentiment  # 0: Negative, 1: Neutral, 2: Positive

articles=news_data[...]
# Analyze sentiments
sentiments = [get_sentiment(article) for article in articles]

# Combine articles with their sentiments
article_sentiments = list(zip(articles, sentiments))

```

- **Integrate Sentiment Data with Stock Prices:**
    - Combine the sentiment scores with historical stock price data for further analysis.
    
    ```python
    # Convert news data to DataFrame
    news_df = pd.DataFrame(article_sentiments, columns=['Article', 'Sentiment'])
    news_df['Date'] = [article['publishedAt'][:10] for article in news_data]
    
    # Aggregate sentiments by date
    sentiment_daily = news_df.groupby('Date')['Sentiment'].mean().reset_index()
    
    # Merge stock data with sentiment data
    stock_data.reset_index(inplace=True)
    merged_data = pd.merge(stock_data, sentiment_daily, how='left', left_on='Date', right_on='Date')
    merged_data.fillna(0, inplace=True)
    
    # Preview the merged data
    print(merged_data.head())
    
    ```
    

### 4. Training the Models

      **Data Splitting**

- Split the dataset into training , validation and testing sets. (70%, 15%, 15% or before the last three months and 50/50% )
- Train the models using the data older than the most recent three months.

**Model Training**

- Train your models on the training set. Use the validation set to tune hyperparameters and avoid overfitting.
    
    ```python
    from keras.models import Sequential
    from keras.layers import LSTM, Dense
    
    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(timesteps, features)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))
    # Implement early stopping and model checkpoints to prevent overfitting.
    
    #Train Darts and NeuralForecast models with covariants
    
    #XGBoost for its robustness and ability to handle non-linear relationships.
    
    # Apply NBEATS for time series forecasting with a focus on scalability.
    
    # Implement NHITS to handle hierarchical time series forecasting.
    
    # Use TFT for capturing both short-term and long-term temporal patterns.
    
    # Develop a regression model to predict stock prices using sentiment scores as features.
    
    ```
    

### 4. Testing the Models

**Initial Testing**

- Evaluate the models' performance on daily and weekly forecasts using Mean Absolute Percentage Error (MAPE). (add Mean Squared Error (MSE), or Root Mean Squared Error (RMSE) optionally)

**Autoregressive Forecasting (optional)**

- Implement autoregressive forecasting where the predicted values for the next step are fed back into the model as inputs for further predictions.

# References

https://medium.com/@redeaddiscolll/integrating-sentiment-analysis-in-stock-price-forecasting-with-deep-learning-techniques-bb5f84fd59f6

https://unit8co.github.io/darts/index.html

https://nixtlaverse.nixtla.io/neuralforecast/index.html

https://www.sciencedirect.com/science/article/pii/S1544612324002575

https://www.insightbig.com/post/stock-market-sentiment-prediction-with-openai-and-python

https://www.gdeltproject.org/
