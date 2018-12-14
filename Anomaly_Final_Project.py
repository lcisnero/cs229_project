# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 23:08:58 2018

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

def train_model(model, to_remove, x_class, y_class):   
    y_pred_train = np.zeros([(100-len(to_remove)),1008])
    y_pred_test = np.zeros([(100-len(to_remove)),672])
    y_train_list = np.zeros([(100-len(to_remove)),1008])
    y_test_list = np.zeros([(100-len(to_remove)),672])
    for i in range(x_class.shape[0]):
        X_train, X_test, y_train, y_test = train_test_split(x_class[i,:], y_class[i,:], test_size=0.4, random_state=0, stratify = y_class[i,:])
        #model = LogisticRegression(class_weight='balanced')
        X_train = X_train.reshape(-1,1)
        X_test = X_test.reshape(-1,1)
        model = model.fit(X_train, y_train)
        y_pred_train[i,:] = model.predict(X_train)
        y_pred_test[i,:] = model.predict(X_test)
        y_train_list[i,:] = y_train
        y_test_list[i,:] = y_test

    #reshape
    y_pred_train_reshape = np.reshape(y_pred_train,-1)
    y_pred_test_reshape = np.reshape(y_pred_test,-1)
    y_train_reshape = np.reshape(y_train_list,-1)
    y_test_reshape = np.reshape(y_test_list,-1)
    
    return (y_pred_train_reshape, y_pred_test_reshape, y_train_reshape, y_test_reshape)
    
#scores
def get_scores(y_pred_train_reshape, y_pred_test_reshape, y_train_reshape, y_test_reshape):
    cnf_matix = confusion_matrix(y_train_reshape, y_pred_train_reshape)
    print("PRECISION AND RECALL FOR TRAINING SET")
    precision = cnf_matix[1][1] / (cnf_matix[0][1] + cnf_matix[1][1])
    recall =  cnf_matix[1][1] / (cnf_matix[1][1] + cnf_matix[1][0])
    print(precision, recall)
    cnf_matix = confusion_matrix(y_test_reshape, y_pred_test_reshape)
    precision = cnf_matix[1][1] / (cnf_matix[0][1] + cnf_matix[1][1])
    recall =  cnf_matix[1][1] / (cnf_matix[1][1] + cnf_matix[1][0])
    print("PRECISION AND RECALL FOR TEST SET")
    print(precision, recall)

def fit_model(model, x, y):
    y_pred = np.zeros_like(y)
    for i in range(x.shape[0]):
        x_new = x[i,:].reshape(-1,1)
        model.fit(x_new)
        y_pred[i,:] = model.predict(x_new)
        y_pred[i,:] = np.array([1 if j == -1 else 0 for j in y_pred[i,:]])
    y_pred_reshape = np.reshape(y_pred, -1)
    y_reshape = np.reshape(y, -1)
    cnf_matix = confusion_matrix(y_reshape, y_pred_reshape)
    precision = cnf_matix[1][1] / (cnf_matix[0][1] + cnf_matix[1][1])
    recall =  cnf_matix[1][1] / (cnf_matix[1][1] + cnf_matix[1][0])
    print(precision, recall)
    
def fit_novelty_model(model, x, y):
    y_pred = np.zeros_like(y)
    for i in range(x.shape[0]):
        y_norm_idxs = (y[i,:] == 0).squeeze()
        x_norm = x[i,y_norm_idxs]   # Labeled examples
        x_norm = x_norm.reshape(-1,1)        
        x_new = x[i,:].reshape(-1,1)
        model.fit(x_norm)
        y_pred[i,:] = model.predict(x_new)
        y_pred[i,:] = np.array([1 if j == -1 else 0 for j in y_pred[i,:]])
    y_pred_reshape = np.reshape(y_pred, -1)
    y_reshape = np.reshape(y, -1)
    cnf_matix = confusion_matrix(y_reshape, y_pred_reshape)
    precision = cnf_matix[1][1] / (cnf_matix[0][1] + cnf_matix[1][1])
    recall =  cnf_matix[1][1] / (cnf_matix[1][1] + cnf_matix[1][0])
    print(precision, recall)
    
def fit_model_loc(model, x, y):
    y_pred = np.zeros_like(y)
    for i in range(x.shape[0]):
        x_new = x[i,:].reshape(-1,1)
        y_pred[i,:] = model.fit_predict(x_new)
        y_pred[i,:] = np.array([1 if j == -1 else 0 for j in y_pred[i,:]])
    y_pred_reshape = np.reshape(y_pred, -1)
    y_reshape = np.reshape(y, -1)
    cnf_matix = confusion_matrix(y_reshape, y_pred_reshape)
    precision = cnf_matix[1][1] / (cnf_matix[0][1] + cnf_matix[1][1])
    recall =  cnf_matix[1][1] / (cnf_matix[1][1] + cnf_matix[1][0])
    print(precision, recall)



##### MAIN #####
    
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

model = LogisticRegression(class_weight='balanced')
y_pred_train_reshape, y_pred_test_reshape, y_train_reshape, y_test_reshape = train_model(model, to_remove, x_class, y_class)
get_scores(y_pred_train_reshape, y_pred_test_reshape, y_train_reshape, y_test_reshape)
cnf_matix = confusion_matrix(y_test_reshape, y_pred_test_reshape)
plot_confusion_matrix(cnf_matix, classes=['non-anomaly','anomaly'], normalize=True, title='Normalized confusion matrix for Logistic Regression')

model_svm = SVC(C=1.0,gamma=1.0, class_weight= 'balanced')
y_pred_train_reshape, y_pred_test_reshape, y_train_reshape, y_test_reshape = train_model(model_svm, to_remove, x_class, y_class)
get_scores(y_pred_train_reshape, y_pred_test_reshape, y_train_reshape, y_test_reshape)

model_rfc = RandomForestClassifier(n_estimators=100, random_state=0, class_weight= 'balanced')
y_pred_train_reshape, y_pred_test_reshape, y_train_reshape, y_test_reshape = train_model(model_rfc, to_remove, x_class, y_class)
get_scores(y_pred_train_reshape, y_pred_test_reshape, y_train_reshape, y_test_reshape)

#now lets do unsupervised learning: for unsupervised learning we don't need to subsample nor dividide between training
#one class svm anomalies
clf = svm.OneClassSVM(nu=.1, kernel='rbf', gamma=.1)
fit_model(clf,x_class,y_class)
fit_model(clf,x,y)

# Split into anomaly and normal examples
clf = svm.OneClassSVM(nu=.1, kernel='rbf', gamma=.1)
fit_novelty_model(clf,x_class,y_class)

clf = svm.OneClassSVM(nu=.1, kernel='rbf', gamma=.1)
fit_novelty_model(clf,x,y)

clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
fit_model_loc(clf,x_class,y_class)
clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
fit_model_loc(clf,x,y)

clf = covariance.EllipticEnvelope(assume_centered=False, contamination = .1, random_state=0)
fit_model(clf,x_class,y_class)
fit_model(clf,x,y)

#lets group - standarization


x_scaled = preprocessing.scale(x_class)
distorsions = []
for k in range(2, x_scaled.shape[0]):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(x_scaled)
    distorsions.append(kmeans.inertia_)

fig = plt.figure(figsize=(15, 5))
plt.plot(range(2, x_scaled.shape[0]), distorsions)
plt.grid(True)
plt.title('Elbow curve')

#51 =arg_min(distorsion/1000000 + 500*log(k))
print(distorsions)

kmeans = KMeans(n_clusters=51)
y_assig = kmeans.fit(x_scaled)

print(Counter(y_assig.labels_))

x_pca = PCA(n_components=2).fit_transform(x_scaled)
plt.scatter(x_pca[:,0],x_pca[:,1], s=10)

#can we improve accuracy?

kmeans = KMeans(n_clusters=74)
y_assig = kmeans.fit(x_scaled)
print(Counter(y_assig.labels_))

#lets take only 5 datasets, the most similars
#y_assig.labels_==65

y_norm_idxs = (y_assig.labels_==55).squeeze()
x_clus = x_scaled[y_norm_idxs,:]   # Labeled examples
y_clus = y_class[y_norm_idxs,:] 
to_remove = [0]*(x.shape[0]-x_clus.shape[0])

##############
model = LogisticRegression(class_weight='balanced')
y_pred_train_reshape, y_pred_test_reshape, y_train_reshape, y_test_reshape = train_model(model, to_remove, x_clus, y_clus)
get_scores(y_pred_train_reshape, y_pred_test_reshape, y_train_reshape, y_test_reshape)


model_svm = SVC(C=1.0,gamma=1.0, class_weight= 'balanced')
y_pred_train_reshape, y_pred_test_reshape, y_train_reshape, y_test_reshape = train_model(model_svm, to_remove, x_clus, y_clus)
get_scores(y_pred_train_reshape, y_pred_test_reshape, y_train_reshape, y_test_reshape)
cnf_matix = confusion_matrix(y_test_reshape, y_pred_test_reshape)
plot_confusion_matrix(cnf_matix, classes=['non-anomaly','anomaly'], normalize=True, title='Normalized confusion matrix')

model_rfc = RandomForestClassifier(n_estimators=100, random_state=0, class_weight= 'balanced')
y_pred_train_reshape, y_pred_test_reshape, y_train_reshape, y_test_reshape = train_model(model_rfc, to_remove, x_clus, y_clus)
get_scores(y_pred_train_reshape, y_pred_test_reshape, y_train_reshape, y_test_reshape)

#now lets do unsupervised learning: for unsupervised learning we don't need to subsample nor dividide between training
#one class svm anomalies
clf = svm.OneClassSVM(nu=.1, kernel='rbf', gamma=.1)
fit_model(clf,x_clus,y_clus)

# Split into anomaly and normal examples
clf = svm.OneClassSVM(nu=.1, kernel='rbf', gamma=.1)
fit_novelty_model(clf,x_clus,y_clus)

clf = svm.OneClassSVM(nu=.1, kernel='rbf', gamma=.1)
fit_novelty_model(clf,x_clus,y_clus)

clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
fit_model_loc(clf,x_clus,y_clus)

clf = covariance.EllipticEnvelope(assume_centered=False, contamination = .1, random_state=0)
fit_model(clf,x_clus,y_clus)

merge_x = x_clus.reshape(-1)
merge_y = y_clus.reshape(-1)

X_train, X_test, y_train, y_test = train_test_split(merge_x, merge_y, test_size=0.4, random_state=0, stratify = merge_y)
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
plot_confusion_matrix(cnf_matix, classes=['non-anomaly','anomaly'], normalize=True, title='Normalized confusion matrix')

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

x_new = merge_x.reshape(-1,1)
clf = svm.OneClassSVM(nu=.1, kernel='rbf', gamma=.1)
clf.fit(x_new)
y_pred = clf.predict(x_new)
y_pred = np.array([1 if x == -1 else 0 for x in y_pred])
cnf_matix = confusion_matrix(merge_y, y_pred)
precision = cnf_matix[1][1] / (cnf_matix[0][1] + cnf_matix[1][1])
recall =  cnf_matix[1][1] / (cnf_matix[1][1] + cnf_matix[1][0])
print(precision, recall)
# Split into anomaly and normal examples
y_norm_idxs = (merge_y == 0).squeeze()
x_norm = merge_x[y_norm_idxs]   # Labeled examples
x_norm = x_norm.reshape(-1,1)
#SVM on class without anormal
clf = svm.OneClassSVM(nu=.1, kernel='rbf', gamma=.1)
clf.fit(x_norm)
y_pred = clf.predict(x_new)
y_pred = np.array([1 if x == -1 else 0 for x in y_pred])
cnf_matix = confusion_matrix(merge_y, y_pred)
precision = cnf_matix[1][1] / (cnf_matix[0][1] + cnf_matix[1][1])
recall =  cnf_matix[1][1] / (cnf_matix[1][1] + cnf_matix[1][0])
print(precision, recall)

clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
y_pred = clf.fit_predict(x_new)
y_pred = np.array([1 if x == -1 else 0 for x in y_pred])
cnf_matix = confusion_matrix(merge_y, y_pred)
precision = cnf_matix[1][1] / (cnf_matix[0][1] + cnf_matix[1][1])
recall =  cnf_matix[1][1] / (cnf_matix[1][1] + cnf_matix[1][0])
print(precision, recall)

clf = covariance.EllipticEnvelope(assume_centered=False, contamination = .1, random_state=0)
clf.fit(x_new)
y_pred = clf.predict(x_new)
y_pred = np.array([1 if j == -1 else 0 for j in y_pred])
cnf_matix = confusion_matrix(merge_y, y_pred)
precision = cnf_matix[1][1] / (cnf_matix[0][1] + cnf_matix[1][1])
recall =  cnf_matix[1][1] / (cnf_matix[1][1] + cnf_matix[1][0])
print(precision, recall)

#corralation
import seaborn as sns
df = pd.DataFrame(x.T)
corr = df.corr()
corr2 = corr.mask(corr < .2, 0)
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), square=True)

#is there a differemt classified per server?
