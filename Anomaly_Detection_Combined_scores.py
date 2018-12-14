# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 19:20:20 2018

@author: lidu_
"""

import numpy as np
import pandas as  pd
import matplotlib.pylab as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn import covariance
from sklearn import preprocessing
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.decomposition import PCA

def get_scores(y_pred, y):
    cnf_matix = confusion_matrix(y, y_pred)
    precision = cnf_matix[1][1] / (cnf_matix[0][1] + cnf_matix[1][1])
    recall =  cnf_matix[1][1] / (cnf_matix[1][1] + cnf_matix[1][0])
    return(precision, recall)

x = np.zeros([100,(1680)])
y = np.zeros([100,(1680)])

for i in range(x.shape[0]):
    x_str = 'A4Benchmark-TS' + str(1+i) + '.csv'
    x[i,:] = np.loadtxt(x_str, delimiter=',', skiprows=1, usecols=5)
    y[i,:] = np.loadtxt(x_str, delimiter=',', skiprows=1, usecols=2)

#obtain the list to remove the sets that don't contain enough anomalies
to_remove = list()
for i in range(100):
    if sum(y[i,:]) <= 3:
        to_remove.append(i)

#keep only those datasets with anomalies > 3 - 77 sets
x_class = np.zeros([(100-len(to_remove)),1680])
y_class = np.zeros([(100-len(to_remove)),1680])
j = 0
for i in range(x.shape[0]):
    if i in to_remove:
        continue
    else:
        x_class[j,:] = x[i,:]
        y_class[j,:] = y[i,:]
        j += 1    
        
prec_rec_train = [0.0,0.0]
prec_rec_test = [0.0,0.0]
model = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
for i in range(x_class.shape[0]):
    temp_mod = None
    prec_rec = [0.0,0.0]
    prec_rec_tem_train = [0.0,0.0]
    X_train, X_test, y_train, y_test = train_test_split(x_class[i,:], y_class[i,:], test_size=0.4, random_state=0, stratify = y_class[i,:])
    X_train = X_train.reshape(-1,1)
    X_test = X_test.reshape(-1,1)
    model1 = LogisticRegression(class_weight='balanced')
    model1 = model1.fit(X_train, y_train)
    y_pred_train = model1.predict(X_train)
    y_pred_test = model1.predict(X_test)
    temp_scores = get_scores(y_pred_test, y_test)
    if (temp_scores[0]+temp_scores[1]) > (prec_rec[0]+prec_rec[1]):
        temp_mod = 1
        prec_rec[0] = temp_scores[0]
        prec_rec[1] = temp_scores[1]
        prec_rec_tem_train = get_scores(y_pred_train,y_train)
    
    model2 = SVC(C=1.0,gamma=1.0, class_weight= 'balanced')
    model2 = model2.fit(X_train, y_train)
    y_pred_train = model2.predict(X_train)
    y_pred_test = model2.predict(X_test)
    temp_scores = get_scores(y_pred_test, y_test)
    if (temp_scores[0]+temp_scores[1]) > (prec_rec[0]+prec_rec[1]):
        temp_mod = 2
        prec_rec[0] = temp_scores[0]
        prec_rec[1] = temp_scores[1]
        prec_rec_tem_train = get_scores(y_pred_train,y_train)
    
    model3 = RandomForestClassifier(n_estimators=100, random_state=0, class_weight= 'balanced')
    model3 = model3.fit(X_train, y_train)
    y_pred_train = model3.predict(X_train)
    y_pred_test = model3.predict(X_test)
    temp_scores = get_scores(y_pred_test, y_test)
    if (temp_scores[0]+temp_scores[1]) > (prec_rec[0]+prec_rec[1]):
        temp_mod = 3
        prec_rec[0] = temp_scores[0]
        prec_rec[1] = temp_scores[1]    
        prec_rec_tem_train = get_scores(y_pred_train,y_train)
        
    x_new = x_class[i,:].reshape(-1,1)
    model4 = svm.OneClassSVM(nu=.1, kernel='rbf', gamma=.1)
    model4.fit(x_new)
    y_pred = model4.predict(x_new)
    y_pred = np.array([1 if x == -1 else 0 for x in y_pred])
    temp_scores = get_scores(y_pred, y_class[i,:])
    if (temp_scores[0]+temp_scores[1]) > (prec_rec[0]+prec_rec[1]):
        temp_mod = 4
        prec_rec[0] = temp_scores[0]
        prec_rec[1] = temp_scores[1]
        
    model5 = svm.OneClassSVM(nu=.1, kernel='rbf', gamma=.1)
    y_norm_idxs = (y_class[i,:] == 0).squeeze()
    x_norm = x_class[i,:][y_norm_idxs]   # Labeled examples
    x_norm = x_norm.reshape(-1,1)
    model5.fit(x_norm)
    y_pred = model5.predict(x_new)
    y_pred = np.array([1 if x == -1 else 0 for x in y_pred])    
    temp_scores = get_scores(y_pred, y_class[i,:])
    if (temp_scores[0]+temp_scores[1]) > (prec_rec[0]+prec_rec[1]):
        temp_mod = 5
        prec_rec[0] = temp_scores[0]
        prec_rec[1] = temp_scores[1]    

    model6 = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    y_pred = model6.fit_predict(x_new)
    y_pred = np.array([1 if x == -1 else 0 for x in y_pred])
    temp_scores = get_scores(y_pred, y_class[i,:])
    if (temp_scores[0]+temp_scores[1]) > (prec_rec[0]+prec_rec[1]):
        temp_mod = 6
        prec_rec[0] = temp_scores[0]
        prec_rec[1] = temp_scores[1] 
        
    model7 = covariance.EllipticEnvelope(assume_centered=False, contamination = .1, random_state=0)
    model7.fit(x_new)
    y_pred = model7.predict(x_new)
    y_pred = np.array([1 if j == -1 else 0 for j in y_pred])
    temp_scores = get_scores(y_pred, y_class[i,:])
    if (temp_scores[0]+temp_scores[1]) > (prec_rec[0]+prec_rec[1]):
        temp_mod = 6
        prec_rec[0] = temp_scores[0]
        prec_rec[1] = temp_scores[1] 
        
    model[temp_mod-1] += 1
    prec_rec_train[0] += prec_rec_tem_train[0]
    prec_rec_train[1] += prec_rec_tem_train[1]
    prec_rec_test[0] += prec_rec[0]
    prec_rec_test[1] += prec_rec[1]