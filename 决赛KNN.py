# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 13:15:35 2021

@author: L
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

path=r"C:\Users\L\Desktop\bank-additional-full_2(1).csv"
data=pd.read_csv(path,encoding='gbk',index_col=0)
data.columns
data_v=data.values


model = KNeighborsClassifier()
x=data.iloc[:,0:20]
y=data.iloc[:,20]

model.fit(x,y)
model.score(x,y)

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.4)
model=KNeighborsClassifier()
model.fit(train_x,train_y)
print(model.score(train_x,train_y))
print(model.score(test_x,test_y))  

model = KNeighborsClassifier()
model.fit(x,y)
y_pred=model.predict(test_x)
z=np.array([50,4,1,6,1,0,2,1,6,1,161,1,999,0,1,1.1,93.994,-36.4,4.857,5191])
z_1=z.reshape(-1,20)
z_pred=model.predict(z_1)
print(z_pred)

model = KNeighborsClassifier()
model.fit(x,y)
x_1=data.iloc[0,0:20]
y_1=data.iloc[0,20]
print(y_1)