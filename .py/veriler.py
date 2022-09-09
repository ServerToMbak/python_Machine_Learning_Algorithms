# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 16:58:49 2022

@author: server
"""
import pandas as pd
import matplotlib.pyplot as plt
import  numpy as np

veriler=pd.read_csv("veriler.csv")
boy=veriler[['boy']]
print(boy)

boykilo=veriler[['boy','kilo']]
print(boykilo)
x=10
 