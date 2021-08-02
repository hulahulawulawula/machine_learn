# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 15:50:40 2021

@author: 钟牙
"""
#神经网络
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier



bank = pd.read_csv(r"C:\Users\钟牙\Desktop\bank-additional-full_2.csv",index_col=0)

hidden_layer_sizes= (200, )
model = MLPClassifier(hidden_layer_sizes)
x = bank.iloc[:,:20]
y = bank.iloc[:,-1]

# model.fit(x,y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6)


model.fit(x_train, y_train)

print(model.score(x_train, y_train))
print(model.score(x_test, y_test))



y_pred = model.predict(x_test)

z = pd.read_csv(r"C:\Users\钟牙\Desktop\ceshi.csv")
z_pred = model.predict(z)

