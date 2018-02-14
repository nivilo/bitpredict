# bitpredict

## Summary
High frequency bitcoin price predictions from market microstructure data. The dataset is a series of one second snapshots of open buy and sell orders on the Bitfinex exchange, combined with a record of executed transactions. Data collection in MongoDB logging trades and order book every 10 seconds.

A number of engineered features are used to train a Logistic Regression Model with regularization, and a trading strategy is simulated on historical data 

## Target
The target for prediction is the midpoint price 60,120,300,600,900 seconds in the future. The midpoint price is the average of the best bid price and the best ask price.

## Model
see Ipython Notebook Model: https://github.com/KlausGlueckert/bitpredict/blob/master/build_model_classify.ipynb