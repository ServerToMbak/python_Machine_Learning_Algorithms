# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 10:50:17 2022

@author: server
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

veriler = pd.read_csv("satislar.csv")
print(veriler)

aylar = veriler[['Aylar']]
print(aylar)

satıslar = veriler[['Satislar']]
print(satıslar)

x_train, x_test, y_train, y_test = train_test_split(aylar, satıslar,test_size=0.33, random_state=0)
'''
sc=StandardScaler()
X_train = sc.fit_transform(x_train)
X_test  = sc.fit_transform(x_test)

Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)
'''
#buildng model(linear regression)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train, y_train)

tahmin = lr.predict(x_test)

x_train = x_train.sort_index()
y_train = y_train.sort_index() 

plt.plot(x_train, y_train)

plt.plot(x_test, lr.predict(x_test))

plt.title( "aylara göre satış")
plt.xlabel("Aylar")
plt.ylabel("satışlar")