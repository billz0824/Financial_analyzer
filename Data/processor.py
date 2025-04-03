import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from .downloader import downlaod_data

#################
# CONFIGURATION #
#################

class Baseline_CFG:
    target_col = 'neg_log_return'
    other_targets = ['High', 'Low', 'Volume', 'Open', 'Close']
    scaler = MinMaxScaler()
    imputer = SimpleImputer(strategy='median')
    lags = [1, 10, 20, 50, 100]

###################
# LOAD STOCK DATA #
###################

def get_data(stock, start, end):
    path = downlaod_data(stock, start, end)

    # processing due to noisy download 
    raw = pd.read_csv(path, header=None)
    raw.columns = raw.iloc[0]
    raw = raw[2:].copy()  # Skip first two rows (tickers + column labels)

    # Clean column names and set proper Date index
    raw.columns.name = None
    raw.rename(columns={raw.columns[0]: 'Date'}, inplace=True)

    # Strip whitespace and drop any accidental empty rows
    raw['Date'] = raw['Date'].astype(str).str.strip()
    raw = raw[raw['Date'].str.match(r'\d{4}-\d{2}-\d{2}')]
    raw['Date'] = pd.to_datetime(raw['Date'], errors='coerce')
    raw = raw.set_index('Date')
    raw = raw[raw.index.notnull()]
    raw = raw.apply(pd.to_numeric, errors='coerce')

    return raw

###########################
# BASE FEATURE GENERATION #
###########################

def baseline_feature_generator(dataset):
    # Momentum-based Indicators
    dataset['momentum'] = dataset['Close'].pct_change().shift(1)
    dataset['log_momentum'] = np.log1p(dataset['momentum'])

    # Moving Averages and Exponential Moving Averages
    dataset['ma7'] = dataset['Close'].rolling(window=7).mean().shift(1)
    dataset['ma21'] = dataset['Close'].rolling(window=21).mean().shift(1)
    dataset['26ema'] = dataset['Close'].ewm(span=26, adjust=False).mean().shift(1)
    dataset['12ema'] = dataset['Close'].ewm(span=12, adjust=False).mean().shift(1)
    dataset['MACD'] = dataset['12ema'] - dataset['26ema']
    dataset['20sd'] = dataset['Close'].rolling(window=20).std().shift(1)
    dataset['upper_band'] = dataset['ma21'] + (dataset['20sd'] * 2)
    dataset['lower_band'] = dataset['ma21'] - (dataset['20sd'] * 2)
    dataset['ema'] = dataset['Close'].ewm(com=0.5).mean().shift(1)

    # RSI Calculation
    delta = dataset['Close'].diff(1).shift(2)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean().shift(1)
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean().shift(1)
    rs = gain / (loss + 1e-6)
    dataset['RSI'] = 100 - (100 / (1 + rs))

    # ATR Calculation
    dataset['High-Low'] = dataset['High'].shift(1) - dataset['Low'].shift(1)
    dataset['High-PClose'] = abs(dataset['High'].shift(1) - dataset['Close'].shift(2))
    dataset['Low-PClose'] = abs(dataset['Low'].shift(1) - dataset['Close'].shift(2))
    dataset['TR'] = dataset[['High-Low', 'High-PClose', 'Low-PClose']].max(axis=1)
    dataset['ATR'] = dataset['TR'].rolling(window=14).mean().shift(1)

    # Williams %R
    dataset['WilliamsR'] = (
        (dataset['High'].rolling(window=14).max().shift(2) - dataset['Close'].shift(1)) /
        (dataset['High'].rolling(window=14).max().shift(2) - dataset['Low'].rolling(window=14).min().shift(2) + 1e-6)
    ).shift(1) * -100

    # OBV Calculation
    dataset['OBV'] = (np.sign(dataset['Close'].diff().shift(2)) * dataset['Volume'].shift(2)).fillna(0).cumsum().shift(1)

    # Rate of Change (ROC)
    dataset['ROC'] = dataset['Close'].pct_change(periods=12).shift(2) * 100

    # Stochastic Oscillator (%K, %D)
    dataset['%K'] = (
        (dataset['Close'].shift(1) - dataset['Low'].rolling(14).min().shift(1)) /
        (dataset['High'].rolling(14).max().shift(1) - dataset['Low'].rolling(14).min().shift(1) + 1e-6)
    ) * 100
    dataset['%D'] = dataset['%K'].rolling(3).mean().shift(1)

    # TARGET: Negative Log Return
    dataset['neg_log_return'] = -np.log(dataset['Close'].shift(-1) / dataset['Close'])
    return dataset.fillna(0)

###########################
# LAGS FEATURE GENERATION #
###########################

def baseline_lags_generator(dataset, lags=Baseline_CFG.lags):
    lagged = dataset.copy()
    for feature in dataset.columns:
        for lag in lags:
            lagged[f'{feature}_lag{lag}'] = dataset[feature].shift(lag)
    return lagged.fillna(0)


#######################
# BASIC PREPROCESSING #
#######################

def baseline_preprocessing(stock, start, end, scaler=Baseline_CFG.scaler, test_ratio=0.15, target=Baseline_CFG.target_col):
    dataset = get_data(stock, start, end)
    dataset = baseline_feature_generator(dataset)
    dataset = baseline_lags_generator(dataset)

    # Drop rows with NaN in target
    dataset = dataset.dropna(subset=[target])

    # Split into training and testing sets
    training, testing = train_test_split(dataset, test_size=test_ratio, shuffle=True)

    # Drop forward-looking information
    X_df = training.drop(columns=[target] + Baseline_CFG.other_targets)
    y = training[target]

    # Scale features
    X = scaler.fit_transform(X_df)

    # Further split into train/val sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_ratio, shuffle=False)

    return X_df, X_train, y_train, X_val, y_val, testing, scaler
