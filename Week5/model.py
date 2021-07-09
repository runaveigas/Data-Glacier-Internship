# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 19:15:34 2021

@author: RUNA
"""


import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression


df = pd.read_csv('salary.csv')
df['experience'].fillna(0,inplace = True)

X = df.iloc[:,:3]
y = df.iloc[:,-1]

regressor = LinearRegression()
regressor.fit(X,y)

pickle.dump(regressor,open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2,5,6]]))

