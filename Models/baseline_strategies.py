import pandas as pd
import numpy as np


##############################
# Define Baseline Strategies #
##############################

class Strategies:

    # Initiaize amount of funds
    def __init__(self, funds=10000):
        self.funds = funds

    def naive_follow(self, df):
        funds = self.funds
        df['Signal'] = (df['Close'] > df['Open']).astype(int)
        df['Signal'] = df['Signal'].shift(1)
        df['Daily_PnL'] = df['Signal'] * ((df['Close'].shift(-1) - df['Open'].shift(-1)) * (funds // df['Open'].shift(-1)))
        df['Daily_PnL'] = df['Daily_PnL'].fillna(0)
        df['Funds'] = funds + df['Daily_PnL'].cumsum()
        return df

    def buy_and_hold(self, df):
        funds = self.funds
        shares_bought = funds // df['Open'].iloc[0]
        remaining_funds = funds % df['Open'].iloc[0]
        df['Portfolio_Value'] = shares_bought * df['Close'] + remaining_funds
        df['Daily_PnL'] = df['Portfolio_Value'].diff().fillna(0)
        df['Funds'] = df['Portfolio_Value']
        return df

    def momentum_strategy(self, df):
        funds = self.funds
        df['Signal'] = ((df['12ema'] < df['26ema']) & (df['12ema'].shift(-1) >= df['26ema'].shift(-1))).astype(int)
        df['Signal'] = df['Signal'].shift(1)
        df['Daily_PnL'] = df['Signal'] * ((df['Close'].shift(-1) - df['Open'].shift(-1)) * (funds // df['Open'].shift(-1)))
        df['Daily_PnL'] = df['Daily_PnL'].fillna(0)
        df['Funds'] = funds + df['Daily_PnL'].cumsum()
        return df

    def model_based_naive(self, model, test_df, scaler, drop_columns):
        funds = self.funds
        X_test = test_df.drop(columns=drop_columns).fillna(0)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

        y_pred = model.predict(X_test)
        test_df['Signal'] = pd.Series((y_pred < 0).astype(int).flatten(), index=test_df.index).shift(1)
        test_df['Daily_PnL'] = test_df['Signal'] * (np.exp(-test_df['neg_log_return']) - 1) * funds
        test_df['Daily_PnL'] = test_df['Daily_PnL'].fillna(0)
        test_df['Funds'] = funds + test_df['Daily_PnL'].cumsum()
        return test_df
