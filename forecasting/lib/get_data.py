import yfinance as yf
import pandas as pd
## Getting stock price data
#
def prepPrices(price_history,  mov_avg=0, target_col='Close'):
        
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

def getStockPrices(stock, history_len, mov_avg, target_col='Close'):
    """
    Gets stock prices from now to N days ago and training amount will be in addition 
    to the number of days to train.
    """
    
    # Designating the Ticker
    ticker = yf.Ticker(stock)

    # Getting all price history
    price_history = ticker.history(period=history_len)
    
    # Check on length
    #if len(price_history)<n_days+training_days+mov_avg:
    #    return pd.DataFrame(), price_history
    
    prices=prepPrices(price_history, mov_avg, target_col)

    return price_history, prices
