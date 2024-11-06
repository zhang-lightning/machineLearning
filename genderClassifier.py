# -*- coding: utf-8 -*-
import os
from tensorflow import keras
from PIL import Image
from keras import layers
import matplotlib.pyplot as plt
import numpy as np

#%% GENERATE DATA
# load labels
labels = np.loadtxt("./data/lfw-deepfunneled-gender.txt",
                    delimiter="\t",
                    skiprows=1,
                    dtype=str)
males = labels[labels[:,1] == "male",0]
males_train = males[0:3500]
males_test = males[3500:4000]
females = labels[labels[:,1] != "male",0]
females_train = females[0:1000]
females_test = females[1000:1200]
train = np.append(males_train, females_train)
train_label = np.vstack((np.full((3500, 1), 1), 
                        np.full((1000, 1), 0)))
test = np.append(males_test, females_test)
test_label = np.vstack((np.full((500, 1), 1), 
                        np.full((200, 1), 0)))
del labels, males, females, males_test, males_train, females_test, females_train

train_data = np.zeros((100, 100, 3, 99999))
i = 0
# load training_data
for dir in train:
    filepath = "./data/lfw-deepfunneled/" + dir
    files = os.listdir(filepath)
    img = Image.open(filepath + "/" + files[0])
    img = img.crop((25, 25, 225, 225))
    img = img.resize((100, 100))
    train_data[:,:,:,i] = img
    i += 1
train_data = train_data[:, :, :, :i] / 255.0

# load testing_data
test_data = np.zeros((100, 100, 3, 99999))
i = 0
for dir in test:
    filepath = "./data/lfw-deepfunneled/" + dir
    files = os.listdir(filepath)
    img = Image.open(filepath + "/" + files[0])
    img = img.crop((25, 25, 225, 225))
    img = img.resize((100, 100))
    test_data[:, :, :, i] = img
    i += 1

test_data = test_data[:, :, :, :i] / 255.0

train_data = np.transpose(train_data, (3, 0, 1, 2))
test_data = np.transpose(test_data, (3, 0, 1, 2))

#%% MODEL CONFIG
ngender = 2

# transform label vectors to binary
train_label = keras.utils.to_categorical(train_label, ngender)
test_label = keras.utils.to_categorical(test_label, ngender)

print("\n----------------------------------------------\nDATA LOADED.")
print("Training images: ", len(train_data))
print("Testing images: ", len(test_data))
print("Shape of train_data: ", train_data.shape)
print("Shape of test_data:", test_data.shape)
print("\n----------------------------------------------\n")

# layers config
model = keras.Sequential(
    [
    keras.Input(shape=(100, 100, 3)),
    layers.Conv2D(4, kernel_size=(3, 3), activation="relu"),
    layers.AveragePooling2D(pool_size=(4, 4)),
    layers.Conv2D(8, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(8, kernel_size=(3, 3), activation="relu"),
    layers.Flatten(),
    layers.Dense(ngender, activation="softmax"),
    ]
)

# export summary
model.summary()

# compile model
model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

#%% TRAINING
# send data to model
# As document said, Do not specify the batch_size if your data is in the form of datasets, 
# generators, or keras.utils.Sequence instances (since they generate batches).
model.fit(
    train_data,
    train_label,
    epochs = 20,
    validation_split=0.1
)

# evaluate model using test_data and test_label
score = model.evaluate(test_data, test_label, verbose=2)

# evaluation output
print('-----------------------\nEvaluating the trained model.')
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# export model
#model.save("./data/gender_classifier_model.h5")

predict_result = model.predict(np.expand_dims(test_data[0,:,:,:], axis=0))
print(predict_result)
if np.argmax(predict_result) == 1:
    predictedValue = "male"
    val = predict_result[:,1] * 100
else:
    predictedValue = "female"
    val = predict_result[:,0] * 100

# evaluate show
plt.figure(figsize=(3, 3))
plt.subplot(1,2,1)
plt.imshow(test_data[0, :, :, :])
plt.grid(False)
plt.xticks([])
plt.yticks([])
plt.title("true_value = male\npredicted = "+predictedValue+"\np = "+str(val)+"%")
plt.show()