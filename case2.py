# -*- coding: utf-8 -*-
import os
import random
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import  keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

# load data
trainDIRNorm = "./data/xray/train/NORMAL"
trainDIRPneu = "./data/xray/train/PNEUMONIA"
testDIRNorm = "./data/xray/test/NORMAL"
testDIRPneu = "./data/xray/test/PNEUMONIA"

def picProcessor(filepath):
    files = os.listdir(filepath)
    for pic in files:
        if pic != "":
            img = Image.open(filepath + "/" + pic)
            img = img.convert('L')
            img = img.resize((300, 300))
            if not os.path.exists('./data/xray_modified'):
                os.mkdir('./data/xray_modified')
            img.save('./data/xray_modified/' + pic)

