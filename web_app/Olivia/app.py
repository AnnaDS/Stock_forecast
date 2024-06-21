import streamlit as st 
import datetime

######### initial setup 

companyNames = ['TESLA (TSLA)', 'AMAZON (AMZN)', 'APPLE (APPL)']
forecastHorizon = ['1 month', '2 months', '3 months']
st.set_page_config(page_title='SPFT@streamlit', layout='wide')
st.title(':orange[Stock Price Forecasting Tool]')

############ SIDEBAR for setting the parameters ##########################
with st.sidebar:
    # Dropdown menu to select the type of stock 
    selectStock = st.selectbox('Select :orange[Stock]', options=companyNames, index=None)
    
    st.write('Select :orange[Prediction] Models')
    lstmCheckbox = st.checkbox('LSTM')
    xgboostCheckbox = st.checkbox('XGboost')
    nbeatsCheckbox = st.checkbox('NBEATS')
    nhitsCheckbox = st.checkbox('NHITS')
    tftCheckbox = st.checkbox('TFT')
    
    st.write('Select Additional :orange[Features]')
    rsiCheckbox = st.checkbox('RSI')
    macdCheckbox = st.checkbox('MACD')
    saCheckbox = st.checkbox('Sentiment Analysis')
    
    # Dropdown menu to select timeframe to predict stock prices
    selectForecast = st.selectbox('Select :orange[Forecast] Horizon', options=forecastHorizon, index=None)
    
    forecastButton = st.button('Generate Forecast')

######### About the app ###############
markdown_about_msg = f"""
        
        Forecast future stock price for :blue[{selectStock}]
    
        Compare the outcomes of multiple models simultaneously

    """
if (selectStock == 'TESLA (TSLA)' or selectStock == 'AMAZON (AMZN)' or selectStock == 'APPLE (APPL)'): 
    st.markdown(markdown_about_msg)
else: 
    st.markdown('Please select stock to get started')

#st.download_button(
#    label=f'Download {selectStock} Forecast'
#    data=
#)


























