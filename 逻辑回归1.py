# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 16:15:56 2021

@author: 杨雅琦
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

bk = pd.read_csv(r'C:\Users\杨雅琦\Desktop\银行营销数据\bank-additional-full_2.csv',index_col = 0)
X = bk.iloc[:, :20]
y = bk.iloc[:, 20]

model = LogisticRegression()
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)
model.fit(X, y)

print(model.score(x_train, y_train))
print(model.score(x_test, y_test))

plt.figure()
plt.scatter(X.values.reshape(-1), y.values.reshape(-1))
