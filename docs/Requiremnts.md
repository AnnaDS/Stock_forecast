# Stock Price Forecasting Tool Requirements

## Title and Description
- **Title**: "Stock Price Forecasting Tool"
- **Description**: "Forecast future stock prices for TSLA, AMZN, and AAPL using historical data and various prediction models. Compare the outcomes of multiple models simultaneously."

## User Interface Elements
### Dropdown Menu for Stock Selection
- **Label**: "Select Stock"
- **Options**: "TSLA", "AMZN", "AAPL"

### Checklist for Prediction Models
- **Label**: "Select Prediction Models"
- **Options**:
  - "LSTM"
  - "XGboost"
  - "NBEATS"
  - "NHITS"
  - "TFT"

### Checklist for Additional Features
- **Label**: "Select Additional Features"
- **Options**:
  - "RSI"
  - "MACD"
  - "Sentiment analysis"

### Dropdown Menu for Forecast Horizon
- **Label**: "Select Forecast Horizon"
- **Options**:
  - "1 Month"
  - "2 Months"
  - "3 Months"

### Button for Generating Forecast
- **Label**: "Generate Forecast"

## Output Section
### Forecast Results
- **Elements**:
  - Area to display historical data , forecasted stock price charts with comparisons of selected models
  - Option to download forecast data as CSV
  - Table for accuracy metrics, including Mean Absolute Percentage Error (MAPE)

## Technical Requirements
- The web app should be built using Streamlit.
- It should pull historical stock data from a reliable source like Yahoo Finance using the `yfinance` library.
- Models should be pre-trained and loaded from saved files (e.g., `.h5` for Keras models).
- The app should handle custom date ranges effectively with appropriate date pickers.
- It should provide real-time feedback and error handling for incorrect inputs or unavailable data.
- The interface must be responsive and user-friendly.
- The app should allow the comparison of forecast outcomes from multiple models on the same chart.

## Design and Usability
- Follow web design best practices to ensure accessibility and ease of use.
- Ensure the interface is clean, with clear labels and sufficient spacing between elements.
- Test the interface with potential users to gather feedback and make improvements.
