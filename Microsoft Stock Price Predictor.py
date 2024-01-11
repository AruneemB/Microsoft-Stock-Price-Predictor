#Imports to perform linear algebra (numpy) and process data (pandas)
import numpy as np
import pandas as pd

#Optional imports to visualize data to help in optimizing the model
import matplotlib.pyplot as plot
import seaborn as sns

#Imports to allow us to construct our machine learning model
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

#Reads data and stores it
microsoft_stock_data = pd.read_csv('../input/microsoft-stock-time-series-analysis/Microsoft_Stock.csv')

#Splits the data into two categories
stock_indicators = microsoft_stock_data[["Open", "High", "Low"]] #Indicators are independent variables
stock_close_prices = microsoft_stock_data["Close"] #Prices when the NYSE closes are the dependent variables

#Converts these datasets into NumPy arrays
stock_indicators = stock_indicators.to_numpy()
stock_close_prices = stock_close_prices.to_numpy()
stock_close_prices = stock_close_prices.reshape(-1, 1)

#Splits datasets into training and testing categories
stock_indicators_train, stock_indicators_test, stock_close_prices_train, stock_close_prices_test = train_test_split(stock_indicators, stock_close_prices, test_size = 0.2, random_state = 50)

#Creates our decision tree regressor model, trains on the training data, and saves predictions on test data
microsoft_stock_model = DecisionTreeRegressor()
microsoft_stock_model.fit(stock_indicators_train, stock_close_prices_train)
stock_close_prices_predictions = microsoft_stock_model.predict(stock_indicators_test)

#Segment of code returning the average percent error of the 'Microsoft Stock Price Predictor'
#Loops through the test values for prices when the NYSE closes and the predicted values
#Calculates and returns average percent error

total_pct_error = 0.0

for i in range(len(stock_close_prices_predictions)):
    total_stock_indicators_test = stock_indicators_test[i][0] + stock_indicators_test[i][1] + stock_indicators_test[i][2]
    avg_stock_indicators_test = total_stock_indicators_test/3
    pct_error = (abs(stock_close_prices_predictions[i] - avg_stock_indicators_test)/avg_stock_indicators_test) * 100
    total_pct_error += pct_error.round(5)

avg_pct_error = total_pct_error/(len(stock_close_prices_predictions))
print('Average percent error of the Microsoft Stock Price Predictor: ' + str(avg_pct_error.round(5)) + '%. \n')

#Segment of code returning individual statistics for each datapoint
#Predicted close price, actual close price, and percent error

for i in range(len(stock_close_prices_predictions)):
    print('Microsoft Stock Close Price ' + str(i + 1) + ':')
    print('Predicted stock close price: ' + str(stock_close_prices_predictions[i]) + '.')

    total_stock_indicators_test = stock_indicators_test[i][0] + stock_indicators_test[i][1] + stock_indicators_test[i][2]
    avg_stock_indicators_test = total_stock_indicators_test/3
    print('Actual stock close price: ' + str(avg_stock_indicators_test.round(5)) + '.')

    pct_error = (abs(stock_close_prices_predictions[i] - avg_stock_indicators_test)/avg_stock_indicators_test) * 100
    print('Percent error: ' + str(pct_error.round(5)) + '%. \n')