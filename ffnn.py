import numpy as np
import pandas as pd
import math 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import keras.backend as K
from sklearn.metrics import mean_squared_error
from keras.layers import Dense 
from keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


# Data of QQQ from 2019-01-02 to 2021-10-29
file_path = 'QQQ.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'])

# First Difference Time Series
#df['Price_Difference'] = df['Adj Close'] - df['Adj Close'].shift(1)
# Drop the NaN value resulting from the shift operation
#df = df.dropna()


# Extract numerical representation (Unix timestamp in seconds)
df['Date'] = df['Date'].astype(int)/10**9

# Convert to float 
df['Date'] = df['Date'].astype('float32')

# Specify columns to drop from the given Data
columns_to_drop = ['Date','Open', 'High', 'Low', 'Close', 'Volume']
#columns_to_drop = ['Date','Open', 'High', 'Low', 'Close', 'Volume','Adj Close']

# Drop the specified columns
df = df.drop(columns=columns_to_drop)

# Normalization of data
scaler = MinMaxScaler(feature_range=(0,1))
df = scaler.fit_transform(df)

# Determining the training and testing sizes 
train_size = int(len(df)*0.8)
test_size = len(df) - train_size
train, test = df[0:train_size,:], df[train_size:len(df),:]


def create_sequences(df, lag=9):
    input_seq = []
    output_seq = []

    for i in range(len(df)-lag-1):
        #print(i)
        window = df[i:(i+lag),0]
        input_seq.append(window)
        output_seq.append(df[i+lag,0])
    return np.array(input_seq),np.array(output_seq)

lag = 9 #number of time steps to look back (Determined from lags)

trainX, trainY = create_sequences(train, lag)
testX, testY = create_sequences(test, lag)

optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

# Define custom RMSE loss function
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

# Input dimensions are (N, lag)
print("Build deep model...")
# Creation and fitting of the model
model = Sequential()
model.add(Dense(8, input_dim = lag, activation = 'relu')) #2nd layer
model.add(Dense(4, activation = 'relu')) # 3rd layer
model.add(Dense(4, activation = 'relu')) # 4th layer
model.add(Dense(1))
model.compile(loss=root_mean_squared_error,optimizer = optimizer, metrics = ['acc'])
# print(model.summary())


# Define early stopping
threshold = 0.001  # Set your desired threshold
early_stopping = EarlyStopping(monitor='val_loss', patience=10, min_delta=threshold, restore_best_weights=True)

history = model.fit(trainX, trainY, validation_data=(testX,testY), verbose = 2, epochs= 100, callbacks=[early_stopping],batch_size = 8)

# make predictions 
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)


trainPredict = scaler.inverse_transform(trainPredict)
trainY_inverse = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY_inverse = scaler.inverse_transform([testY])



trainPredictPlot = np.empty_like(df)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[lag:len(trainPredict)+lag, :] = trainPredict

testPredictPlot = np.empty_like(df)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(lag*2)+1:len(df)-1, :] = testPredict

# Plot of Actual QQQ Prices as well as Training and Testing Predictions
plt.figure(figsize=(10,6))
plt.title("QQQ ETF Adjusted Closing Price", fontsize=12)
plt.plot(scaler.inverse_transform(df), label='Actual Prices')
plt.plot(trainPredictPlot, label='Train Predictions')
plt.plot(testPredictPlot, label='Test Predictions')
plt.xlabel('Number of Days',fontsize=12)
plt.ylabel('Adjusted Close Price',fontsize=12)
plt.legend()


# Early Stopping Plot (training history)
plt.figure(figsize=(10,6))
plt.title("Train/Validation Loss",fontsize=12)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.xlabel('Number of Epochs',fontsize=12)
plt.ylabel('Error',fontsize=12)
plt.legend()

plt.show()









