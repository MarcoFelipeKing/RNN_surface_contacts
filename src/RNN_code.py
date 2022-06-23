# This script uses the sequence of contacts from data/movements_StJames.csv and trains an RNN to predict the next contact.
# The RNN is trained on 80% of the data, and tested on the remaining 20%.

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Activation
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# import data
df = pd.read_csv('data/movements_StJames.csv')

# subset df where ActivityID=1 to 80 
df_train = df[df['ActivityID'] <= 80]
df_test = df[df['ActivityID'] > 80]

# For every unique Surface in df_train and df_test create a new column with a numerical value
df_train['value'] = df_train['Surface'].astype('category').cat.codes
df_test['value'] = df_test['Surface'].astype('category').cat.codes

# keep only value
df_train = df_train[['value']]
df_test = df_test[['value']]

# 


# split into input and output

# set up the model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(1,5)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.add(Activation('linear'))

# compile the model
model.compile(loss='mse', optimizer='adam')

# fit the model
model.fit(df_train.values, df_train.values, epochs=100, batch_size=32, validation_split=0.2, verbose=1, shuffle=False)

# make predictions
predictions = model.predict(df_test.values)

# plot the predictions
plt.plot(df_test.values, label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.show()
