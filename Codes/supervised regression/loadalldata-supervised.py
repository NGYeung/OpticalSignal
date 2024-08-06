# -*- coding: utf-8 -*-
"""
@author: Lingxiao Zhou
"""
from scipy.io import loadmat 
import numpy as np

from sklearn import svm
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as pyplot

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn import preprocessing

import os
os.chdir('C:/Users/Lingxiao Zhou/Desktop/EECS 545 ML/final project/Data/using data')

# data split
def split_data(data_list, y_list, ratio=0.30):
    '''
    split data by given ratio
    '''
    X_train, X_test, y_train, y_test = train_test_split(data_list, y_list, test_size=ratio, random_state=42)
    print ('--------------------------------data shape-----------------------------------')
    print(len(X_train), len(y_train))
    print(len(X_test), len(y_test))
    return X_train, X_test, y_train, y_test

#%%
m=512*512
allimage=np.array(np.zeros((1,m)))
norm_i=np.zeros(1)

for i in range(1,41):
    alldata = loadmat(str(i)+'.mat')
    image=(np.array(alldata['ImData'])[0:m,:]).T
    feature= (np.array(alldata['ImData'])[m:m+4,:]).T
    norm_I=feature[:,1]
    
    allimage=np.vstack((allimage, image))
    norm_i=np.append(norm_i, norm_I)
    
allimage=np.delete(allimage, 0, axis=0)
norm_i=np.delete(norm_i, 0)

X_train, X_test, y_train, y_test=split_data(allimage, norm_i, ratio=0.30)

#%%
sample = np.reshape(X_train[2,:],(512,512))
pyplot.imshow(sample)

#%%
#pca
d=800
pca = PCA(n_components=0.95,svd_solver = 'full')
pca.fit(X_train)
pcaX_train = pca.transform(X_train)     
pcaX_test= pca.transform(X_test) 

#%%
#linear regression
lin_reg = LinearRegression().fit(pcaX_train, y_train)

y_pred=lin_reg.predict(pcaX_test)

loss=mean_squared_error(y_pred/y_test, y_test/y_test)
print("linear regression")
print(loss)

#%%
#random forest
rr_reg=RandomForestRegressor(max_depth=20, random_state=1).fit(pcaX_train, y_train)

y_pred=rr_reg.predict(pcaX_test)

loss=mean_squared_error(y_pred/y_test, y_test/y_test)
print("random forest")
print(loss)

#%%
#Gradient Boosting
gbr_reg = GradientBoostingRegressor(random_state=1,n_estimators=1000).fit(pcaX_train, y_train)

y_pred=gbr_reg.predict(pcaX_test)

loss=mean_squared_error(y_pred/y_test, y_test/y_test)
print("Gradient Boosting")
print(loss)

#%%
#SVR
svr_reg = SVR(C=10.0, epsilon=0.002).fit(pcaX_train, y_train)

y_pred=svr_reg.predict(pcaX_test)

loss=mean_squared_error(y_pred/y_test, y_test/y_test)
print("SVR")
print(loss)

#%%
#KernelRidge
kr_reg = KernelRidge(alpha=10).fit(pcaX_train, y_train)

y_pred=kr_reg.predict(pcaX_test)

loss=mean_squared_error(y_pred/y_test, y_test/y_test)
print("Kernel Ridge")
print(loss)

#%%
#mlp

mlp_reg = MLPRegressor(random_state=1, 
                   max_iter=100000000,
                   batch_size=900,
                   hidden_layer_sizes=(512,256,64,32),
                   activation='relu',
                   alpha=0.1
                   ).fit(pcaX_train, y_train)

y_pred=mlp_reg.predict(pcaX_test)

loss=mean_squared_error(y_pred/y_test, y_test/y_test)
print("MLP")
print(loss)
#%%
##################################################################
                     # preprocessing
##################################################################

scaler = preprocessing.RobustScaler().fit(pcaX_train)
pre_X_train=scaler.transform(pcaX_train)
pre_X_test=scaler.transform(pcaX_test)  

#%%
#linear regression
lin_reg = LinearRegression().fit(pre_X_train, y_train)

y_pred=lin_reg.predict(pre_X_test)

loss=mean_squared_error(y_pred/y_test, y_test/y_test)
print("linear regression")
print(loss)

#%%
#random forest
rr_reg=RandomForestRegressor(max_depth=50, random_state=1).fit(pre_X_train, y_train)

y_pred=rr_reg.predict(pre_X_test)

loss=mean_squared_error(y_pred/y_test, y_test/y_test)
print("random forest")
print(loss)

#%%
#Gradient Boosting
gbr_reg = GradientBoostingRegressor(random_state=1,n_estimators=1000).fit(pre_X_train, y_train)

y_pred=gbr_reg.predict(pre_X_test)

loss=mean_squared_error(y_pred/y_test, y_test/y_test)
print("Gradient Boosting")
print(loss)

#%%
#SVR
svr_reg = SVR(C=10.0, epsilon=0.002).fit(pre_X_train, y_train)

y_pred=svr_reg.predict(pre_X_test)

loss=mean_squared_error(y_pred/y_test, y_test/y_test)
print("SVR")
print(loss)

#%%
#Kernel Ridge
kr_reg = KernelRidge(alpha=0.01).fit(pre_X_train, y_train)

y_pred=kr_reg.predict(pre_X_test)

loss=mean_squared_error(y_pred/y_test, y_test/y_test)
print("Kernel Ridge")
print(loss)

#%%
#mlp

mlp_reg = MLPRegressor(random_state=1, 
                   max_iter=100000000,
                   batch_size=900,
                   hidden_layer_sizes=(512,256,64,32),
                   activation='relu',
                   alpha=0.5
                   ).fit(pre_X_train, y_train)

y_pred=mlp_reg.predict(pre_X_test)

loss=mean_squared_error(y_pred/y_test, y_test/y_test)
print("MLP")
print(loss)















