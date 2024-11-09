# -*- coding:utf-8 -*-
import time
from random import sample

import pandas as pd
from keras import optimizers
from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.layers.core import dense

# load data from csv
sampleX = pd.read_csv("./data/sample_x.csv", on_bad_lines='skip')
# on_bad_lines(new)/error_bad_lines(old), skip bad lines.
sampleY = pd.read_csv("./data/sample_y.csv", on_bad_lines='skip')
sampleX = sampleX.values
sampleY = sampleY.values

# divide data into train, test dataset.
trainX = sampleX[0:80,:]
trainY = sampleY[0:80,:]
testX = sampleX[80:,:]
testY = sampleY[80:,:]
nOutput = 3

# normalize dataset
scalerX = MinMaxScaler(feature_range=(-1, 1))
scalerY = MinMaxScaler(feature_range=(0, 1))
xTrain = scalerX.fit_transform(trainX)
xTest = scalerX.fit_transform(testX)
yTrain = scalerY.fit_transform(trainY)
yTest = scalerY.fit_transform(testY)

# config model
model = Sequential()
model.add(
    Dense(
        units = 5,
        activation = 'sigmoid',
        input_dim = trainX.shape[1],
        kernel_initializer = 'normal'
    )
)
model.add(
    Dense(
        units = nOutput,
        activation = 'relu',
        kernel_initializer = 'normal'
    )
)
model.summary()

# compile model
"""
adam = optimizers.Adam(
    lr = 0.005,
    decay = 0,
    beta_1 = 0.9,
    beta_2 = 0.999,
    epsilon = 1e-7,
    amsgrad = False
)
"""
model.compile(loss='MSE', optimizer='adam', metrics=['accuracy'])

# train model
history = model.fit(xTrain, yTrain,
                    epochs = 50000,
                    validation_split = 0.25,
                    verbose = 2,
                    shuffle = False)
lossY = history.history['loss']
valLossY = history.history['val_loss']

# predict
predictYt = model.predict(xTrain)
predictYt = scalerY.inverse_transform(predictYt)
predictYp = model.predict(xTest)
predictYp = scalerY.inverse_transform(predictYp)

# output
name = ["lift", "drag", "moment"]
export1 = pd.DataFrame(columns = name, data = predictYt)
export2 = pd.DataFrame(columns = name, data = predictYp)
export3 = pd.DataFrame(columns = "loss", data = lossY)
export4 = pd.DataFrame(columns = "valLoss", data = valLossY)
export1.to_csv("./result/predictYt.csv")
export2.to_csv("./result/predictYp.csv")
export3.to_csv("./result/lossY.csv")
export4.to_csv("./result/valLossY.csv")