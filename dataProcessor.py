# -*- coding: utf-8 -*-
"""
Extract lightning lightning intensity data in 2011, and 
save as a xlsx file.

@author: Zeshen
"""
import os
import numpy as np
import pandas as pd
import datetime as dt

files = os.listdir("./data/")
names = ["time", "lap1"]
for file in files:
    if file[-3:]=='csv':
        path = "./data/" + file
        data = pd.read_csv(path, sep=",")
        
    else:
        continue
