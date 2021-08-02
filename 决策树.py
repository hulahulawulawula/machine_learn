# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 10:59:14 2021

@author: 10138
"""
import numpy as np
import sklearn.tree as tree
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

bank = pd.read_csv(r"C:\Users\10138\Desktop\bank-additional-full_2.csv",index_col=(0))

model = tree.DecisionTreeClassifier(criterion='entropy')
x = bank.iloc[:,:20]
y = bank.iloc[:,-1]

model.fit(x,y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6)

model = tree.DecisionTreeClassifier()
model.fit(x_train, y_train)

model.feature_importances_


print(model.score(x_train, y_train))
print(model.score(x_test, y_test))


y_pred = model.predict(x_test)

z = np.array([50,4,1,6,1,0,2,1,6,1,161,1,999,0,1,1.1,93.994,-36.4,4.857,5191])
z_1 = z.reshape(-1,20)
z_pred = model.predict(z_1)

