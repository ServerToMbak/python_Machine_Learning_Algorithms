# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 16:58:49 2022

@author: server
"""
#1.libraries
import pandas as pd
import matplotlib.pyplot as plt
import  numpy as np
#2.Data Preprocessing
#2.1.Data Downloading
veriler=pd.read_csv("veriler.csv")
    
Yas=veriler.iloc[:,1:4].values
print(Yas)





#encoder Nominal -->Numeric
ulke=veriler.iloc[:,0:1].values
print(ulke)

from sklearn import preprocessing 

le=preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(veriler.iloc[:,0])

print(ulke)

ohe = preprocessing.OneHotEncoder()
ulke=ohe.fit_transform(ulke).toarray()
print(ulke)





c=veriler.iloc[:,-1:].values
print(c)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
c[:,-1]= le.fit_transform(veriler.iloc[:,-1]) 
print(c)

ohe=preprocessing.OneHotEncoder()
c=ohe.fit_transform(c).toarray()
print(c)


#numpy arrays transformatian to DataFrame
sonuc=pd.DataFrame(data=ulke, index=range(22), columns=['fr','tr','us'])
print(sonuc)

sonuc2= pd.DataFrame(data=Yas, index =range(22), columns=['boy','kilo','yas'])
print(sonuc2)

cinsiyet = veriler.iloc[:,4].values
print(cinsiyet)

sonuc3= pd.DataFrame(data=c[:,:1], index=range(22),columns=['cinsiyet'])
print(sonuc3)







#dataframe'S combinating
s=pd.concat([sonuc,sonuc2],axis=1)
print(s)
 
s2=pd.concat([s,sonuc3],axis=1)
print(s2)

#training and testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(s2,sonuc3,test_size=0.33,random_state=0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train=sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)


#I did change the dependend variable and played it little bit more to understand

boy= s2.iloc[:,3:4].values
print(boy)

sol=s2.iloc[:,:3]
sağ=s2.iloc[:,4:]

veri =pd.concat([sol,sağ], axis=1)

x_train, x_test, y_train, y_test=train_test_split(veri,boy,test_size=0.33,random_state=0)

regressor2=LinearRegression()
regressor2.fit(x_train,y_train)
y_pred = regressor2.predict(x_test)