# -*- coding: utf-8 -*-
# openpyxl is needed!
import glob
import pandas as pd

path = "./data/"
pattern = "*.csv"

# get file list
filelist = glob.glob(path + pattern)

# define column names
colNames = ["time", "lap1", "lap2", "lap3", "lap4", "lap5", "lap6",
            "magNS", "magWE", "slowE", "fastE"]

# read each files and merge into xlsx file
outputfile = path + "data_merged.xlsx"
with pd.ExcelWriter(outputfile) as writer:
    for i, file in enumerate(filelist):
        df = pd.read_csv(file, header=None, names = colNames, parse_dates=["time"])
        sheetName = f"data{i+1}"
        df.to_excel(writer, sheet_name=sheetName, index=False)