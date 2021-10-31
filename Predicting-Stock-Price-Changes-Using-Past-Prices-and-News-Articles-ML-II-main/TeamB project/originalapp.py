import streamlit as st
import yfinance as yf
import datetime
from datetime import date
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import requests
import joblib
import re

st.sidebar.subheader('Choose the dates to show the stock according to:')
start_date = st.sidebar.date_input("First 10 raw data", datetime.date(2016, 1, 1))
end_date = st.sidebar.date_input("Next day stock value increase", datetime.date(2021, 6, 9))

def weekdaychecker(tradeday):
    checker = tradeday.weekday()
    if checker == 5:
        tradeday += datetime.timedelta(days=2)
    elif checker == 6:
        tradeday += datetime.timedelta(days=1)
    return tradeday

def clean_text(text, remove_stopwords = True):
    contractions = {"ain't": "am not","aren't": "are not","can't": "cannot","can't've": "cannot have","'cause": "because","could've": "could have",
    "couldn't": "could not","couldn't've": "could not have","didn't": "did not","doesn't": "does not","don't": "do not","hadn't": "had not","hadn't've": "had not have",
    "hasn't": "has not","haven't": "have not","he'd": "he would","he'd've": "he would have","he'll": "he will","he's": "he is","how'd": "how did",
    "how'll": "how will","how's": "how is","i'd": "i would","i'll": "i will","i'm": "i am","i've": "i have","isn't": "is not","it'd": "it would",
    "it'll": "it will","it's": "it is","let's": "let us","ma'am": "madam","mayn't": "may not","might've": "might have","mightn't": "might not","must've": "must have",
    "mustn't": "must not","needn't": "need not","oughtn't": "ought not","shan't": "shall not","sha'n't": "shall not","she'd": "she would","she'll": "she will",
    "she's": "she is","should've": "should have","shouldn't": "should not","that'd": "that would","that's": "that is","there'd": "there had",
    "there's": "there is","they'd": "they would","they'll": "they will","they're": "they are","they've": "they have","wasn't": "was not","we'd": "we would","we'll": "we will",
    "we're": "we are","we've": "we have","weren't": "were not","what'll": "what will","what're": "what are","what's": "what is","what've": "what have",
    "where'd": "where did","where's": "where is","who'll": "who will","who's": "who is","won't": "will not","wouldn't": "would not","you'd": "you would",
    "you'll": "you will","you're": "you are"}
    # Convert words to lower case
    text = text.lower()
    # Replace contractions with their longer forms
    if True:
        text = text.split()
        new_text = []
        for word in text:
            if word in contractions:
                new_text.append(contractions[word])
            else:
                new_text.append(word)
        text = " ".join(new_text)

    # Format words and remove unwanted characters
    text = re.sub(r'&amp;', '', text)
    text = re.sub(r'0,0', '00', text)
    text = re.sub(r'[_"\-;%()|.,+&=*%.,!?:#@\[\]]', ' ', text)
    text = re.sub(r'\'', ' ', text)
    text = re.sub(r'\$', ' $ ', text)
    text = re.sub(r'u s ', ' united states ', text)
    text = re.sub(r'u n ', ' united nations ', text)
    text = re.sub(r'u k ', ' united kingdom ', text)
    text = re.sub(r'j k ', ' jk ', text)
    text = re.sub(r' s ', ' ', text)
    text = re.sub(r' yr ', ' year ', text)
    text = re.sub(r' l g b t ', ' lgbt ', text)
    text = re.sub(r'0km ', '0 km ', text)
    # Optionally, remove stop words
    if remove_stopwords:
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)
    return text

def lemmatize_data(data, lemmatizer):
    cleaned_dataset = []
    for i in range(len(data)):
        clean_text = data[i].lower()
        clean_text = clean_text.split()
        clean_text = [lemmatizer.lemmatize(word) for word in clean_text if word not in stopwords.words('english')]
        cleaned_dataset.append(' '.join(clean_text))
    return ''.join(cleaned_dataset)


def main():

    st.title("APPLE Stocks Prediction")

    st.header("Select the stock and check its next day predicted value")

    choose_stock = st.sidebar.selectbox("Choose the Stock!",["NONE","Apple"])

    if(choose_stock == "Apple"):

        # get abfrl real time stock price
        symbol = 'AAPL'
        startdate = start_date
        enddate = end_date

        df1 = yf.download(symbol, start=startdate, end=enddate)

        st.header("Apple Stock First 10 Days Data:")
        # Insert Check-Box to show the snippet of the data.
        if st.checkbox('Show Raw Data'):
            st.dataframe(df1.head(10))

        df1['Date'] = df1.index
        close_col = df1.filter(['Close'])
        close_col_val = close_col.values
        train_len = math.ceil(len(close_col_val)*.75)
        mm_scale = MinMaxScaler(feature_range=(0,1))
        mm_scale_data = mm_scale.fit_transform(close_col_val)

        model = load_model("Model.h5")
        test_data_val = mm_scale_data[train_len-30: , :]
        x_test = []
        y_test = close_col_val[train_len : , :]
        for i in range(30, len(test_data_val)):
        	x_test.append(test_data_val[i-30:i,0])
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
        preds = model.predict(x_test)
        preds = mm_scale.inverse_transform(preds) #Undo Scaling

        training_data = close_col[:train_len]
        validation_data = close_col[train_len:]
        validation_data['Preds'] = preds

        fig1 = plt.figure(figsize=(16,8))
        plt.title('LSTM Network Predicted Model')
        plt.xlabel('Time/Date', fontsize=18)
        plt.ylabel('Stock Close Price', fontsize=18)
        plt.plot(validation_data[['Close','Preds']])
        plt.legend(['Validation_values','Predictions_values'], loc='lower right')
        st.pyplot(fig1)

        new_close_col = df1.filter(['Close'])
        new_close_col_val = new_close_col[-30:].values
        new_close_col_val_scale = mm_scale.transform(new_close_col_val)

        X_test = []
        X_test.append(new_close_col_val_scale)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        new_preds = model.predict(X_test)
        new_preds =mm_scale.inverse_transform(new_preds)

        NextDay_Date = end_date + datetime.timedelta(days=1)
        NextDay_Date = weekdaychecker(NextDay_Date)
        st.subheader("Close Price Prediction for the next trading day : " + str(NextDay_Date))
        st.write(new_preds[0][0], "USD")

        st.header('Top Five Latest Stock News')
        date = []
        text = []
        ticker = 'AAPL'
        limit = '5'
        key = 'db73b9bdb8d6dc556ccd898886d218bb'
        api_url = f'https://fmpcloud.io/api/v3/stock_news?tickers={ticker}&limit={limit}&apikey={key}'
        data = requests.get(api_url).json()
        for i in reversed(range(int(limit))):
            text.append(data[i]['title'])
            date.append(data[i]['publishedDate'].split()[0])
        news = pd.DataFrame(list(zip(date, text)),columns=['Date','Stock News'])
        # news.Date = pd.to_datetime(news.Date)
        news.set_index('Date', inplace=True)
        st.table(news)

# driver code
if __name__ == '__main__':
    main()
