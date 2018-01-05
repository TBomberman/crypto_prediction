import os
import time
import warnings
import numpy as np
import pandas as pd
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Convolution1D, MaxPooling1D, Flatten,  Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, Normalizer

np.random.seed(7)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

df = pd.read_csv('Data/CandlesJan2015-Dec2017.csv', skipinitialspace=True, header=0)
df = df.sort_values(by=['Timestamp'], ascending=[True])
df.set_index('Timestamp', inplace=True)

look_back = 12
sc = StandardScaler()
print(df.columns)
df.loc[:, 'Close'] = sc.fit_transform(df.loc[:, 'Close'])
# sc1 = StandardScaler()
# df.loc[:, 'High'] = sc1.fit_transform(df.loc[:, 'High'])
# sc2 = StandardScaler()
# df.loc[:, 'Low'] = sc1.fit_transform(df.loc[:, 'Low'])
# sc2 = StandardScaler()
# df.loc[:, 'Open'] = sc1.fit_transform(df.loc[:, 'Open'])

# train_df = df.loc[df.index < pd.to_datetime('2016-01-01')]

timeseries = np.asarray(df.Close)
timeseries = np.atleast_2d(timeseries)
if timeseries.shape[0] == 1:
        timeseries = timeseries.T
# training input
X = np.atleast_3d(np.array([timeseries[start:start + look_back] for start in range(0, timeseries.shape[0] - look_back)]))
# training values
y = timeseries[look_back:]

# building the network
predictors = ['Close']#, 'DJI','Inflation']#, 'InterestRate']
#TRAIN_SIZE = train_x.shape[0]
#EMB_SIZE = look_back
model = Sequential()
model.add(LSTM(input_shape = (1,), input_dim=1, output_dim=6, return_sequences=True))
model.add(LSTM(input_shape = (1,), input_dim=1, output_dim=6, return_sequences=False))
model.add(Dense(1))
model.add(Activation('linear'))
model.summary()
model.compile(loss="mse", optimizer="rmsprop")

# train
model.fit(X, y, epochs=30,
          batch_size=8192, verbose=0, shuffle=False)

# predicting the outputs
df['Pred'] = df.loc[df.index[0], 'Close']
for i in range(len(df.index)):
    if i <= look_back:
        continue
    a = None
    for c in predictors:
        b = df.loc[df.index[i-look_back:i], c].as_matrix()
        if a is None:
            a = b
        else:
            a = np.append(a,b)
        a = a
    y = model.predict(a.reshape(1,look_back*len(predictors),1))
    df.loc[df.index[i], 'Pred']=y[0][0]

# save
# df.to_hdf('DeepLearning.h5', 'Pred_LSTM')

# plot
df.loc[:, 'Close'] = sc.inverse_transform(df.loc[:, 'Close'])
df.loc[:, 'Pred'] = sc.inverse_transform(df.loc[:, 'Pred'])
plt.plot(df.Close,'y')
plt.plot(df.Pred, 'g')
plt.show()