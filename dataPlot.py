# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import mu_0  # 真空磁导率

plt.rcParams['font.sans-serif'] = ['SimHei'] # 设置中文字体（SimHei 黑体）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

distance = 300  # 距离放电点的距离（米）
sampling_interval = 2e-9  # 每个点的采样间隔（秒）

file_path = "./data/data_merged.xlsx"  # Excel 文件路径
excel_data = pd.ExcelFile(file_path)  # 加载 Excel 文件

plt.figure(figsize=(10, 6))
for sheet_name in excel_data.sheet_names:
    df = pd.read_excel(excel_data, sheet_name=sheet_name)
    magNS = df["magNS"].values  # 磁场数据 (单位：特斯拉)
    time = np.arange(len(magNS)) * sampling_interval  # 生成时间序列
    current = (magNS * 2 * np.pi * distance) / mu_0  # 计算电流
    plt.plot(time * 1e9, current, label=sheet_name)  # 将时间轴转换为纳秒
plt.xlabel("时间 (ns)")
plt.ylabel("电流 (kA)")
plt.title("由磁场数据反演电流")
plt.legend()
plt.grid(True)
plt.show()
