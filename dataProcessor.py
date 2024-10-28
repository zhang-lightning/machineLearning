# -*- coding: utf-8 -*-
"""
Extract lightning lightning intensity data in 2011, and 
save as a xlsx file.

@author: Zeshen
"""
import os
import numpy as np
import datetime as dt
import xlsxwriter as xlsx

DIR = "./data/"  # data directory

fileList = os.listdir(DIR + "original/")  # list all files in given directory
nfiles = len(fileList)
filenames = []
# extract data in 2011
for filename in fileList:
    if filename.startswith("2011") and filename.endswith(".txt"):
        filenames.append(filename)

# read files
dates = []
intensity = []

for filename in filenames:
    with open(DIR+"original/"+filename) as file:
        for line in file:
            lineW = line.split()
            dates.append(lineW[1] + ' ' + lineW[2])
            intensity.append(float(lineW[5][3:]))

# dates = [dt.datetime.fromisoformat(date) for date in dates]

workbook = xlsx.Workbook(DIR + "data_2011.csv")
worksheet = workbook.add_worksheet()

worksheet.set_column('A:B', 30)
worksheet.write('A1', 'Time')
worksheet.write('B1', 'Lightning Intensity')

nLine = len(intensity)
for i in range(nLine):
    rowNumber = i + 1
    worksheet.write_string(rowNumber, 0, dates[i])
    worksheet.write(rowNumber, 1, intensity[i])

workbook.close()
