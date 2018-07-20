#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 23:54:24 2018

@author: mp
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.model_selection import train_test_split,GridSearchCV
###reading thecsv file
df=pd.read_csv('/home/mp/MobilePricesClassification/train.csv')
rows=df.shape[0]
col=df.shape[1]
yprice_range=df['price_range']
del df['price_range']

### using feature scaling  and then using classifier
###Preprocessing
vals=df.loc[:,:]
scalar=MinMaxScaler()
rescaled_vals=scalar.fit_transform(vals)
 
###Train Test Split,X=features,Y=Labels

X_train,X_test,Y_train,Y_test=train_test_split(rescaled_vals,yprice_range,test_size=0.2,random_state=40)

###using classiifer(SVM),score was low with NB & DT 
svc=svm.SVC()
params={'kernel':('linear','rbf'),'C':[1,10]}
clf=GridSearchCV(svc,params)
clf.fit(X_train,Y_train)
pred=clf.predict(X_test)
print "Score:", clf.score(X_test,Y_test)

###plotting 
plt.scatter(Y_test,pred,c=['r','b'])
plt.xlabel("True_test_values")
plt.ylabel("Predicted_test_values")
plt.show()

### O/P file claculation
dft=pd.read_csv('/home/mp/MobilePricesClassification/test.csv')
test_vals=dft.loc[:,:]
scalar=MinMaxScaler()
rescaled_tvals=scalar.fit_transform(test_vals)
pred= clf.predict(rescaled_tvals)
np.savetxt('/home/mp/MobilePricesClassification/predicted_vals_range.csv',pred,delimiter=',')





