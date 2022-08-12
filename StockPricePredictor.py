import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import date
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR
import streamlit as st


# retrieves historical data of ticker
# from Jan 2018 to current day
# predicts the price of ticker 2 months out or 5% of the data set

def getData(ticker):
    plt.rcdefaults()
    plt.rcParams.update({'figure.facecolor':"#2c4d56"})
    tk = yf.Ticker(ticker)
    print(tk.info)
    # need try catch
    company = tk.info['shortName']
    print("You chose: " + company)
    currDate = date.today()
    data = yf.download(ticker, '2018-01-01', currDate)

    # creates training and testing data 67/33 split
    data.reset_index(level=0, inplace=True)
    data['Date'] = pd.to_datetime(data['Date'], infer_datetime_format=True)

    df = data[['Close']]

    forecast_out = int(math.ceil(0.05 * len(df)))  # forecasting out 5% of the entire dataset
    df['forecasted'] = df['Close'].shift(-forecast_out)

    # trainSet, testSet = train_test_split(df, test_size=0.3)
    # normalizes sets
    scaler = StandardScaler()
    X = np.array(df.drop(['forecasted'], 1))
    scaler.fit(X)
    X = scaler.transform(X)
    X_Predictions = X[-forecast_out:]  # data to be predicted
    X = X[:-forecast_out]  # data to be trained
    df.dropna(inplace=True)

    y = np.array(df['forecasted'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

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
    columns = ['Model', 'Accuracy']
    scores = [lr_confidence, rg_confidence, rf_confidence, svr_confidence]
    alg_vs_score = pd.DataFrame([[x, y] for x, y in zip(names, scores)], columns=columns)

    alg_vs_score.plot(kind='bar', x='Model', y='Accuracy', color=['blue', 'red', 'yellow', 'orange'])


    with st.container():
        fig_col1, fig_col2 = st.columns(2)
        with fig_col1:
            st.title("Performance")
            plt.xlabel("Model")
            plt.ylabel("Accuracy")
            plt.legend('', frameon=False)
            st.pyplot(plt.gcf())
        if (alg_vs_score['Accuracy'].idxmax() == 0):
            forecast_set = lr.predict(X_Predictions)
            chose = " with Linear Regression"
        elif (alg_vs_score['Accuracy'].idxmax() == 1):
            forecast_set = rg.predict(X_Predictions)
            chose = " with Ridge Regression"
        elif (alg_vs_score['Accuracy'].idxmax() == 2):
            forecast_set = rf.predict(X_Predictions)
            chose = " with Random Forest"
        elif (alg_vs_score['Accuracy'].idxmax() == 3):
            forecast_set = svr.predict(X_Predictions)
            chose = " with SVR"

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
        with fig_col2:
            st.title("Prediction" + chose)
            plot = plt.gcf()
            st.pyplot(plot)

    with st.container():
        fig_col3, fig_col4 = st.columns(2)
        with fig_col3:
            st.title("Detailed Data View")
            st.dataframe(data.head(n=100))
        with fig_col4:
            st.title("Market Sentiment")

    "#586e75"

def run():
    st.set_page_config(
        page_title="Real-Time Stock Forecast Dashboard",
        page_icon="✅",
        layout="wide",
        initial_sidebar_state="expanded"
    )


    with st.sidebar:
        st.title("Real-time Stock Forecasting Dashboard")
        st.write("Welcome. This program gives a 2 month forecast of a stock using a time series"
             " of it's historical close data from 2018 to the current date. \n "
             "It compares several machine learning models to make the most accurate forecast.\n")
        with st.form(key='my_form'):
            ticker = st.text_input(label="Enter stock ticker to continue: ")
            submit_button = st.form_submit_button(label='Submit')
    getData(ticker)



if __name__ == "__main__":
    run()

#twitter sentiment / fibonacci lines on charts
