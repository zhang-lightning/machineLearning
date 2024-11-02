# -*- coding: utf-8 -*-
import os
from tensorflow import keras
from PIL import Image
from keras import layers
import matplotlib.pyplot as plt
import numpy as np

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
train_label = np.vstack((np.full((3500, 1), "male", dtype="<U12"), 
                        np.full((1000, 1), "female", dtype="<U12")))
test = np.append(males_test, females_test)
test_label = np.vstack((np.full((500, 1), "male", dtype="<U12"), 
                        np.full((200, 1), "female", dtype="<U12")))
del labels, males, females, males_test, males_train, females_test, females_train

train_data = np.zeros((100, 100, 3, 99999))
i = 0
# load training_data
for dir in train:
    filepath = "./data/lfw-deepfunneled/" + dir
    files = os.listdir(filepath)
    for file in files:
        img = Image.open(filepath + "/" + file)
        img = img.crop((25, 25, 225, 225))
        img = img.resize((100, 100))
        train_data[:,:,:,i] = img
        i += 1
train_data = train_data[:, :, :, :i]

# load testing_data
test_data = np.zeros((100, 100, 3, 99999))
i = 0
for dir in test:
    filepath = "./data/lfw-deepfunneled/" + dir
    files = os.listdir(filepath)
    for file in files:
        img = Image.open(filepath + "/" + file)
        img = img.crop((25, 25, 225, 225))
        img = img.resize((100, 100))
        test_data[:, :, :, i] = img
        i += 1
test_data = test_data[:, :, :, :i]
print(len(train_label), len(test_label))