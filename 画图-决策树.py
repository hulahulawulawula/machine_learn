# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 19:54:55 2021

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
X = bank.iloc[:,:20]
y = bank.iloc[:,-1]

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=8)

model.fit(X_train,y_train)

pred = model.predict(X_test)

print(accuracy_score(y_test,pred))

path = model.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
print(ccp_alphas)

models = []#这段是错的 不要运行
for ccp_alpha in ccp_alphas:
model = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
model.fit(X_train, y_train)
models.append(model)
print("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
models[-1].tree_.node_count, ccp_alphas[-1]))


model2 = tree.DecisionTreeClassifier(criterion='entropy',random_state=2, ccp_alpha=0.002)

model2.fit(X_train,y_train)

pred2=model2.predict(X_test)

print(accuracy_score(y_test, pred2))

tree.plot_tree(model2,filled=True)

plt.rcParams['savefig.dpi'] = 800

plt.rcParams['figure.dpi'] = 2000
