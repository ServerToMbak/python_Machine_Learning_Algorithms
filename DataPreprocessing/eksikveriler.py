# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 16:58:49 2022

@author: server
"""
import pandas as pd
import matplotlib.pyplot as plt
import  numpy as np

veriler=pd.read_csv("eksikveriler.csv")
boy=veriler[['boy']]
print(boy)

boykilo=veriler[['boy','kilo']]
print(boykilo)
x=10


#eksik veriler 
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

Yas=veriler.iloc[:,1:4].values
print(Yas)
imputer= imputer.fit(Yas[:,1:4])
Yas[:,1:4]=imputer.transform(Yas[:,1:4])    
print(Yas) 