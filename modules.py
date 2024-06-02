# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 15:10:00 2024

@author: Ndubisi M. Uzoegbu
"""

import joblib
import pandas as pd
import pandas_ta as ta
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the trained models and scaler
lin_reg = joblib.load('linear_regression_model1.pkl')
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')


# Function to add technical indicators to the new input data
def add_technical_indicators(df, column):
    df['SMA'] = ta.sma(df[column], length=30)
    df['EMA'] = ta.ema(df[column], length=30)
    df['RSI'] = ta.rsi(df[column], length=14)
    macd = ta.macd(df[column])
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_signal'] = macd['MACDs_12_26_9']
    df['MACD_hist'] = macd['MACDh_12_26_9']
    return df

# Function to add lagged features to the new input data
def add_lagged_features(df, column, num_lags):
    for lag in range(30, num_lags + 1):
        df[f'{column}_lag{lag}'] = df[column].shift(lag)
    return df

# Function to preprocess new data
def preprocess_new_data(new_data):
    new_data['datetime'] = pd.to_datetime(new_data['Date'].astype(str) + new_data['Time'].astype(str).str.zfill(6), format='%Y%m%d%H%M%S')
    new_data.set_index('datetime', inplace=True)
    new_data['Date'] = pd.to_datetime(new_data['Date'], format='%Y%m%d')
    new_data['Year'] = new_data['Date'].dt.year
    new_data['Month'] = new_data['Date'].dt.month
    new_data['Day'] = new_data['Date'].dt.day
    new_data['Weekday'] = new_data['Date'].dt.weekday  # Monday=0, Sunday=6
    
    def parse_time(time_str):
        time = datetime.strptime(time_str, '%H%M%S')
        hour = time.hour
        minute = time.minute
        period = 0 if hour < 12 else 1
        hour = hour if hour <= 12 else hour - 12
        return hour, minute, period

    new_data['Hour'], new_data['Minute'], new_data['Period'] = zip(*new_data['Time'].apply(lambda x: parse_time(str(x).zfill(6))))
    
    new_data = new_data.drop(columns=['Date', 'Time', 'Ticker', 'Open', 'High', 'Low'])
    
    new_data = add_technical_indicators(new_data, 'Close')
    new_data = add_lagged_features(new_data, 'Close', 60)
    
    new_data = new_data.dropna()
    
    return new_data

# Function to predict the price using trained models
def forecast_price(new_data):
    new_data_preprocessed = preprocess_new_data(new_data)
    X_new = new_data_preprocessed.drop(columns=['Close'])
    X_new_scaled = scaler.transform(X_new)
    X_new_pca = pca.transform(X_new_scaled)
    
    # Predict using linear regression model
    y_pred_lin = lin_reg.predict(X_new_pca)
    return y_pred_lin
