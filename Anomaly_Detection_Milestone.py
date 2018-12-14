# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 22:07:13 2018

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
#7
#Source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
ts1 = pd.read_csv('A4Benchmark-TS1.csv')
#ts1.dtypes
ts1.timestamps = pd.to_datetime(ts1.timestamps, unit='s')
tv = ts1[['timestamps','value']]

x = np.loadtxt('A4Benchmark-TS1.csv', delimiter=',', skiprows=1, usecols=5)
y = np.loadtxt('A4Benchmark-TS1.csv', delimiter=',', skiprows=1, usecols=2)

plt.plot(tv.value)
plt.plot(x)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
#Logistic regression

model = LogisticRegression(class_weight='balanced')
x_new = x.reshape(-1,1)
model = model.fit(x_new, y)
model.score(x_new,y)
y_pred = model.predict(x_new)
cnf_matix = confusion_matrix(y, y_pred)
plt.figure()
#plot_confusion_matrix(cnf_matix, classes=['anomaly','non-anomaly'],title='Confusion matrix, without normalization')
plot_confusion_matrix(cnf_matix, classes=['non-anomaly','anomaly'], normalize=True, title='Normalized confusion matrix')

#Cross validation
model = LogisticRegression(class_weight='balanced')
X_train = X_train.reshape(-1,1)
X_test = X_test.reshape(-1,1)
model = model.fit(X_train, y_train)
model.score(X_test,y_test)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
cnf_matix = confusion_matrix(y_train, y_pred_train)
print(cnf_matix)
precision = cnf_matix[1][1] / (cnf_matix[0][1] + cnf_matix[1][1])
recall =  cnf_matix[1][1] / (cnf_matix[1][1] + cnf_matix[1][0])
print(precision, recall)
cnf_matix = confusion_matrix(y_test, y_pred_test)
precision = cnf_matix[1][1] / (cnf_matix[0][1] + cnf_matix[1][1])
recall =  cnf_matix[1][1] / (cnf_matix[1][1] + cnf_matix[1][0])
print(precision, recall)
print(cnf_matix)
plt.figure()
#plot_confusion_matrix(cnf_matix, classes=['anomaly','non-anomaly'],title='Confusion matrix, without normalization')
plot_confusion_matrix(cnf_matix, classes=['non-anomaly','anomaly'], normalize=True, title='Normalized confusion matrix')

#SVM
model_svm = SVC(C=1.0,gamma=1.0, class_weight= 'balanced')
model_svm = model_svm.fit(x_new,y)    
model_svm.score(x_new,y)
y_pred = model_svm.predict(x_new)
print(confusion_matrix(y, y_pred))
cnf_matix = confusion_matrix(y, y_pred)
plt.figure()
plot_confusion_matrix(cnf_matix, classes=['non-anomaly','anomaly'], normalize=True, title='Normalized confusion matrix')

#Cross validation
model_svm = SVC(C=1.0,gamma=1.0, class_weight= 'balanced')
model_svm = model_svm.fit(X_train, y_train)
model_svm.score(X_test,y_test)
y_pred_train = model_svm.predict(X_train)
y_pred_test = model_svm.predict(X_test)
cnf_matix = confusion_matrix(y_train, y_pred_train)
print(cnf_matix)
precision = cnf_matix[1][1] / (cnf_matix[0][1] + cnf_matix[1][1])
recall =  cnf_matix[1][1] / (cnf_matix[1][1] + cnf_matix[1][0])
print(precision, recall)
cnf_matix = confusion_matrix(y_test, y_pred_test)
precision = cnf_matix[1][1] / (cnf_matix[0][1] + cnf_matix[1][1])
recall =  cnf_matix[1][1] / (cnf_matix[1][1] + cnf_matix[1][0])
print(precision, recall)
print(cnf_matix)
plt.figure()
plot_confusion_matrix(cnf_matix, classes=['non-anomaly','anomaly'], normalize=True, title='Normalized confusion matrix')

#Random Forest
model_rfc = RandomForestClassifier(n_estimators=100, random_state=0, class_weight= 'balanced')
model_rfc = model_rfc.fit(x_new,y)
model_rfc.score(x_new,y)
y_pred_train = model_rfc.predict(X_train)
y_pred_test = model_rfc.predict(X_test)
cnf_matix = confusion_matrix(y_train, y_pred_train)
print(cnf_matix)
precision = cnf_matix[1][1] / (cnf_matix[0][1] + cnf_matix[1][1])
recall =  cnf_matix[1][1] / (cnf_matix[1][1] + cnf_matix[1][0])
print(precision, recall)
cnf_matix = confusion_matrix(y_test, y_pred_test)
precision = cnf_matix[1][1] / (cnf_matix[0][1] + cnf_matix[1][1])
recall =  cnf_matix[1][1] / (cnf_matix[1][1] + cnf_matix[1][0])
print(precision, recall)
print(cnf_matix)
plt.figure()
plot_confusion_matrix(cnf_matix, classes=['non-anomaly','anomaly'], normalize=True, title='Normalized confusion matrix')

#Cross validation
model_rfc = RandomForestClassifier(n_estimators=100, random_state=0, class_weight= 'balanced')
model_rfc = model_rfc.fit(X_train, y_train)
y_pred_train = model_rfc.predict(X_train)
y_pred_test = model_rfc.predict(X_test)
cnf_matix = confusion_matrix(y_train, y_pred_train)
print(cnf_matix)
precision = cnf_matix[1][1] / (cnf_matix[0][1] + cnf_matix[1][1])
recall =  cnf_matix[1][1] / (cnf_matix[1][1] + cnf_matix[1][0])
print(precision, recall)
cnf_matix = confusion_matrix(y_test, y_pred_test)
precision = cnf_matix[1][1] / (cnf_matix[0][1] + cnf_matix[1][1])
recall =  cnf_matix[1][1] / (cnf_matix[1][1] + cnf_matix[1][0])
print(precision, recall)
print(cnf_matix)
plt.figure()
plot_confusion_matrix(cnf_matix, classes=['non-anomaly','anomaly'], normalize=True, title='Normalized confusion matrix')

#SVM one class
clf = svm.OneClassSVM(nu=.1, kernel='rbf', gamma=.1)
clf.fit(x_new)
y_pred = clf.predict(x_new)
y_pred = np.array([1 if x == -1 else 0 for x in y_pred])
cnf_matix = confusion_matrix(y, y_pred)
precision = cnf_matix[1][1] / (cnf_matix[0][1] + cnf_matix[1][1])
recall =  cnf_matix[1][1] / (cnf_matix[1][1] + cnf_matix[1][0])
print(precision, recall)

# Split into anomaly and normal examples
y_norm_idxs = (y == 0).squeeze()
x_norm = x[y_norm_idxs]   # Labeled examples
x_norm = x_norm.reshape(-1,1)
x_an = x[~y_norm_idxs]         # anormal
x_an = x_an.reshape(-1,1)
#SVM on class without anormal
clf = svm.OneClassSVM(nu=.1, kernel='rbf', gamma=.1)
clf.fit(x_norm)
y_pred = clf.predict(x_new)
y_pred = np.array([1 if x == -1 else 0 for x in y_pred])
cnf_matix = confusion_matrix(y, y_pred)
precision = cnf_matix[1][1] / (cnf_matix[0][1] + cnf_matix[1][1])
recall =  cnf_matix[1][1] / (cnf_matix[1][1] + cnf_matix[1][0])
print(precision, recall)

#Local outlier facto
clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
y_pred = clf.fit_predict(x_new)
y_pred = np.array([1 if x == -1 else 0 for x in y_pred])
cnf_matix = confusion_matrix(y, y_pred)
precision = cnf_matix[1][1] / (cnf_matix[0][1] + cnf_matix[1][1])
recall =  cnf_matix[1][1] / (cnf_matix[1][1] + cnf_matix[1][0])
print(precision, recall)
print(confusion_matrix(y, y_pred))



