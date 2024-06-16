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

### 2. Data Gathering from [The GDELT Project](https://www.gdeltproject.org/) for Sentiment Analysis

**Steps:**

- Use https://github.com/alex9smith/gdelt-doc-api to collect news articles related to the stocks.
- Extract relevant financial news within the timeframe of the historical stock price data.

### 3. Prepare Data

### 3.a. Transform Time Series Data for Supervised Learning Models

**Steps:**

- Generate features such as moving averages, trading volume, and other technical indicators like RSI.
- Structure the data to include lagged features for the supervised models like XGBoost and LSTM.

### 3.b. Transform Time Series Data for Time Series Prediction Models

**Steps:**

- Prepare the data in a sequential format suitable for time series models from packages like NeuralForecast and Darts.

### 3.c. Transform News Data into Sentiment Data Using FinBERT

**Steps:**

- Apply FinBERT model to the news articles to generate sentiment scores.
- Integrate sentiment scores with stock price data.

Tokenize and analyze the sentiment of each article using FinBERT.

- **Integrate Sentiment Data with Stock Prices:**
    - Combine the sentiment scores with historical stock price data for further analysis.

### 4. Training the Models

      **Data Splitting**

- Split the dataset into training , validation and testing sets. (70%, 15%, 15% or before the last three months and 50/50% )
- Train the models using the data older than the most recent three months.

**Model Training**

- Train your models on the training set. Use the validation set to tune hyperparameters and avoid overfitting.

      **LSTM model**
      **XGBoost model**
      **NBEATS model**
      **TFT model**
      **REgression model with sentiment data**
    
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
