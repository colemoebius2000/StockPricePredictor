import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import date
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


# retrieves historical data of ticker
# from Jan 2018 to current day
# predicts the price of ticker 2 months out or 5% of the data set

def getData(ticker):
    tk = yf.Ticker(ticker)
    print(tk.info)

    company = tk.info['shortName']
    print("You chose: " + company)
    currDate = date.today()
    df = yf.download(ticker, '2018-01-01', currDate)
    df['Adj Close'].plot()
    plt.title(company + " Historical Price")
    plt.xlabel('Year')
    plt.ylabel('USD')
    plt.show()

    # creates training and testing data 70/30 split
    df.reset_index(level=0, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)

    df['hlPct'] = (df['High'] - df['Low']) / df['Low'] * 100.0
    df['pctChange'] = (df['Close'] - df['Open']) / df['Open'] * 100.0
    df = df[['hlPct', 'pctChange', 'Close', 'Volume']]

    forecast_out = int(math.ceil(0.05 * len(df)))  # forecasting out 5% of the entire dataset
    df['label'] = df['Close'].shift(-forecast_out)

    # trainSet, testSet = train_test_split(df, test_size=0.3)
    # normalizes sets
    scaler = StandardScaler()
    X = np.array(df.drop(['label'], 1))
    scaler.fit(X)
    X = scaler.transform(X)
    X_Predictions = X[-forecast_out:]  # data to be predicted
    X = X[:-forecast_out]  # data to be trained

    df.dropna(inplace=True)
    y = np.array(df['label'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_confidence = lr.score(X_test, y_test)

    rg = Ridge()
    rg.fit(X_train, y_train)

    rg_confidence = rg.score(X_test, y_test)

    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    rf_confidence = rf.score(X_test, y_test)

    svr = SVR()
    svr.fit(X_train, y_train)
    svr_confidence = svr.score(X_test, y_test)

    names = ['Linear Regression', 'Ridge', 'Random Forest', 'SVR']
    columns = ['model', 'accuracy']
    scores = [lr_confidence, rg_confidence, rf_confidence, svr_confidence]
    alg_vs_score = pd.DataFrame([[x, y] for x, y in zip(names, scores)], columns=columns)

    alg_vs_score.plot(kind='bar', x='model', y='accuracy', color=['blue', 'red', 'yellow', 'orange'])
    plt.show()

    forecast_set = rf.predict(X_Predictions)
    datelist = pd.date_range(currDate, periods=len(forecast_set))
    forecastDf = pd.DataFrame(forecast_set, index=datelist, columns=['Predictions'])
    lastDate = datelist[len(datelist) - 1]
    df2 = yf.download(ticker, '2018-01-01', currDate)
    df_all = pd.merge(df2['Close'], forecastDf['Predictions'], how='outer', left_index=True, right_index=True)

    df_all.plot(figsize=(10, 6))
    plt.title(company + " Price Prediction")
    plt.xlabel('Year')
    plt.ylabel('USD')
    for x, y in zip(df_all.index, df_all['Predictions']):
        if x == lastDate:
            label = "{:.2f}".format(y)
            l2 = x.strftime("%b %d %Y")
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 10), ha='center')
            plt.annotate(l2, (x, y), textcoords="offset points", xytext=(0, 20), ha='center')
    plt.show()


def run():
    print()
    print("Welcome. This program gives a 2 month forecast of a stock using a time series"
          " of it's historical close data from 2018 to the current date.")
    print("It compares several machine learning models to make the most accurate forecast.")
    ticker = input("\n          Enter stock ticker to continue: ")

    getData(ticker)


if __name__ == "__main__":
    run()
