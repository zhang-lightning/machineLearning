# -*- coding: utf-8 -*-
"""
Visualize lightning intensity variation versus time. 

@author: Zeshen
"""
import os
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

DIR = "./data/"

files = os.listdir(DIR+"original/")
filedirs = []
for file in files:
    if file.startswith("2011") and file.endswith(".txt"):
        filedirs.append(DIR+"original/"+file)

data = pd.DataFrame()
for filedir in filedirs:
    data._append(pd.read_table(filedir,encoding='GBK'))