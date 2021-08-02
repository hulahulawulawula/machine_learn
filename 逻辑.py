# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 12:54:26 2021

@author: 10138
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

bk = pd.read_csv(r"C:\Users\10138\Desktop\bank-additional-full_2.csv",index_col = 0)
X = bk.iloc[:, :20]
y = bk.iloc[:, 20]

model = LogisticRegression()
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)
model.fit(X, y)

print(model.score(x_train, y_train))
print(model.score(x_test, y_test))

y_pred = model.predict(x_test)

z = np.array([50,4,1,6,1,0,2,1,6,1,161,1,999,0,1,1.1,93.994,-36.4,4.857,5191])
z_1 = z.reshape(-1,20)
z_pred = model.predict(z_1)
