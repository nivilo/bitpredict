# bitpredict

## Summary
This project aims to make high frequency bitcoin price predictions from market microstructure data. The dataset is a series of one second snapshots of open buy and sell orders on the Bitfinex exchange, combined with a record of executed transactions. Data collection in MongoDB logging trades and order book every 10 seconds.

A number of engineered features are used to train a Logistic Regression Model with regularization, and a theoretical trading strategy is simulated on historical data 

## Target
The target for prediction is the midpoint price 60,120,300,600,900 seconds in the future. The midpoint price is the average of the best bid price and the best ask price.

## Model
see Ipython Notebook Model: https://github.com/KlausGlueckert/bitpredict/blob/master/build_model_classify.ipynb

## Features

#### Width
This is the difference between the best bid price and best ask price.

#### Power Imbalance
This is a measure of imbalance between buy and sell orders. For each order, a weight is calculated as the inverse distance to the current midpoint price, raised to a power. Total weighted sell order volume is then subtracted from total weighted buy order volume. Powers of 2, 4, and 8 are used to create three separate features. 

#### Power Adjusted Price
This is similar to Power Imbalance, but the weighted distance to the current midpoint price (not inverted) is used for a weighted average of prices. The percent change from the current midpoint price to the weighted average is then calculated. Powers of 2, 4, and 8 are used to create three separate features. 

#### Trade Count
This is the number of trades in the previous X seconds. Offsets of 30, 60, 120, and 180 are used to create four separate features.

#### Trade Average
This is the percent change from the current midpoint price to the average of trade prices in the previous X seconds. Offsets of 30, 60, 120, and 180 are used to create four separate features.

#### Aggressor
This is measure of whether buyers or sellers were more aggressive in the previous X seconds. A buy aggressor is calculated as a trade where the buy order was more recent than the sell order. A sell aggressor is the reverse. The total volume created by sell aggressors is subtracted from the total volume created by buy aggressors. Offsets of 30, 60, 120, and 180 are used to create four separate features.

#### Trend
This is the linear trend in trade prices over the previous X seconds. Offsets of 30, 60, 120, and 180 are used to create four separate features.
