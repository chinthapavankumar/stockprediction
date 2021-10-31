
# Stock Price Prediction of Apple Inc. Using LSTM

Closed Price Prediction of Apple Inc. Using LSTM Recurrent Neural Network.

# Deployed App Link: (https://newstockpredictionapp.herokuapp.com/)

## Datasets:

The stock prices dataset is extracted from Xignite API in CSV format. The dataset consists of Open, High, Low and Closing Prices of Apple Inc. stocks from 1st january 2016 to 17th June 2021.
 
The news headlines dataset is obtained from Nasdaq website.
  
## Data Pre-Processing:

The sentiment analyser is applied on the headlines data such that the polarity scores of positivity,negativity,neutrality and compound as a whole in the statements are given.

All values of the stock prices have been normalized between -1 and 1 using MinMaxScaler.


 ![Sentiment Polarity Scores](https://user-images.githubusercontent.com/82156741/125252653-3c933c80-e316-11eb-99b0-70c25a4d39b3.png)


![2021-07-12 (3)](https://user-images.githubusercontent.com/82156741/125259616-1b821a00-e31d-11eb-9e29-5d640745402f.png)
## Model:

Two sequential LSTM layers have been stacked together and one dense layer is used to build the RNN model using Keras deep learning library. Since this is a regression task, 'tanh' activation has been used in final layer.

  
# Version:

Python 3.8 and latest versions of all libraries including deep learning library Keras and Tensorflow.

  
## Training:

80% data is used for training. Adam optimizer is used for faster convergence. After training starts it will look like:


![2021-07-12 (9)](https://user-images.githubusercontent.com/82156741/125259728-394f7f00-e31d-11eb-9a0f-d17482ce51b3.png)


  
## Test:

Test accuracy metric is mean square error (MSE).

  
## Results:

The comparison of Test Closing Price and Predicted Closing Price looks like :

![](https://user-images.githubusercontent.com/82156741/125258960-79623200-e31c-11eb-8b2c-d8054a7babf2.png)

  
## Observation and Conclusion:

By using LSTM,thr closed stock prices are predicted with minimal loss.The training and testing RMSE are: 2.74 and 1.60 respectively which is pretty good to predict future values of stock.However, future values for any time period can be predicted using this model.

Finally, this work can greatly help the quantitative traders to take decisions.

  
