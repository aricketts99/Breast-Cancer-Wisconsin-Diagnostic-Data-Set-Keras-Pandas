# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 18:32:27 2020

@author: Andrew
"""


import tensorflow
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd

#Read in data and create a pandas dataframe with it
data = pd.read_csv('Breast-Cancer-Wisconsin-(Diagnostic)-Data-Set.csv')

df = pd.DataFrame(data)

#Data cleaning
df.drop('id',axis=1,inplace=True)
df.dropna(inplace=True)
df.replace(['M','B'],[0,1],inplace=True)

#Model, just playing around with random activations
model = Sequential()
model.add(Dense(12, input_dim=30, activation='relu'))
model.add(Dense(20, activation='tanh'))
model.add(Dense(10, activation='softmax'))
model.add(Dense(5,activation='linear'))
model.add(Dense(1, activation='sigmoid'))

#Seperating into x and f(x) columns then into train and test data
X = df.loc[:, df.columns != 'diagnosis']
Y = df.loc[:, df.columns == 'diagnosis']

X_train, X_test = X[:400], X[401:]
Y_train, Y_test = Y[:400], Y[401:]


#Compiling, fiting and evaluating (again just random playing around)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X_train,Y_train,epochs=150,batch_size=10)
model.evaluate(X_test, Y_test, batch_size=10)