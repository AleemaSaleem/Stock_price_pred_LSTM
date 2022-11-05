import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import datetime
for dirname, _, filenames in os.walk(r'C:\Users\Abdullah\AppData\Local\Programs\Python\Python310\rnn\Google_Stock_Price_Train (1)'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
dataset=pd.read_csv('C:/Users/Abdullah/AppData/Local/Programs/Python/Python310/rnn/Google_Stock_Price_Train(1).csv',index_col="Date",parse_dates=True)
dataset.head()
dataset.tail()
dataset.shape
dataset.isna().sum()
dataset.info()
dataset.describe()
dataset['Open'].plot(figsize=(16,6))
# convert column "a" of a DataFrame
dataset["Close"] = dataset["Close"].str.replace(',', '').astype(float)

dataset["Volume"] = dataset["Volume"].str.replace(',', '').astype(float)
# 7 day rolling mean
dataset.rolling(7).mean().head(20)
dataset['Open'].plot(figsize=(16,6))
dataset.rolling(window=30).mean()['Close'].plot()
dataset['Close: 30 Day Mean'] = dataset['Close'].rolling(window=30).mean()
dataset[['Close','Close: 30 Day Mean']].plot(figsize=(16,6))
# Optional specify a minimum number of periods
dataset['Close'].expanding(min_periods=1).mean().plot(figsize=(16,6))
training_set=dataset['Open']
training_set=pd.DataFrame(training_set)
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# Part 2 - Building the RNN

# Importing the Keras libraries and packages
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import LSTM
#from keras.layers import Dropout

#regressor = Sequential()
