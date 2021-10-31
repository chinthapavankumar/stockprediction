# Real Time Stock Movement Prediction (Team A)

**Deployed webapp link** </br>
https://realstockpred.herokuapp.com/ </br>

**Project report link** </br>
https://drive.google.com/file/d/1zbG23QtUTtckIJCBFpYeSoFvScICHAl7/view

## Goal of the project
To predict price changes in the future for a given stock. Information that is leveraged to make these predictions includes prices from previous days and financial news headlines related to the company of interest.

## Steps Involved
1. Data Preparation and Labelling (pulled from [Fmp Cloud](https://fmpcloud.io/) for news data and [yfinance](https://pypi.org/project/yfinance/) for Stock price data.
2. Text preprocessing for news data and Scaling of stock price data for prediction.
3. Exploratory data analysis for text and news data for better understanding of the data.
4. Modelling LSTM for stock price data and Random forest for news data.
5. Save model and build [Streamlit](https://streamlit.io/) webapp and deployed on [Heroku](https://www.heroku.com/)

## Demo of the project
![](https://github.com/Technocolabs100/Predicting-Stock-Price-Changes-Using-Past-Prices-and-News-Articles-ML-II/blob/main/Team%20A%20Final%20Submission/streamlit-app-2021-07-11-10-07-14.gif)

&copy; ML Team A (10th June Batch), Technocolabs.
