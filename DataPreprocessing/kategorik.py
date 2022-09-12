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

ulke=veriler.iloc[:,0:1].values
print(ulke)

from sklearn import preprocessing 

le=preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(veriler.iloc[:,0])

print(ulke)

ohe = preprocessing.OneHotEncoder()
ulke=ohe.fit_transform(ulke).toarray()
print(ulke)
sonuc=pd.DataFrame(data=ulke, index=range(22), columns=['fr','tr','us'])
print(sonuc)

sonuc2= pd.DataFrame(data=Yas, index =range(22), columns=['boy','kilo','yas'])
print(sonuc2)

cinsiyet = veriler.iloc[:,4].values
print(cinsiyet)

sonuc3= pd.DataFrame(data=cinsiyet, index=range(22),columns=['cinsiyet'])
print(sonuc3)

s=pd.concat([sonuc,sonuc2],axis=1)
print(s)
 
s2=pd.concat([s,sonuc3],axis=1)
print(s2)