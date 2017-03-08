#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 15:49:41 2017

@author: diego

Python implementation of the example in ML Azure 
https://gallery.cortanaintelligence.com/Experiment/Telco-Customer-Churn-5
"""

import pandas as pd
import matplotlib.pyplot as plt
from imblearn.combine import SMOTEENN 
from collections import Counter

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc,accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

f = "CRM Dataset Shared - Copy.tsv"
data = pd.read_csv(f,sep="\t")

f_labels = "CRM Churn Labels Shared - Copy.tsv"
labels = pd.read_csv(f_labels,sep="\t",header=None,names=['chunk'])
labels = labels['chunk'].astype('category')

col_names = data.columns
#exclude the following columns
idx_noselect = ["Var8","Var15","Var20","Var31","Var32","Var39",
            "Var42","Var48","Var52","Var55","Var79","Var141","Var167","Var175","Var185","Var6"]

idx_left = col_names[0:189]
idx_left= list(set(idx_left) - set(idx_noselect))
idx_right = list(col_names[190:230])
idx_left_numeric = data[idx_left].dtypes == 'float64'
idx_left_numeric = list(idx_left_numeric)
                       
#Fill NaN
data_left = data[idx_left] + 1
data_left = data_left.fillna(0)
    
#Binning numeric columns
for v in idx_left:
    try:
        data_left[v] = pd.cut(data_left[v],50,labels=False)
    except:
        print v
        
data_right = data[idx_right]
data_right = data_right.fillna(0)


#Encode categorical variables
encoders = dict()
for v in idx_right:
    try:
        le = preprocessing.LabelEncoder()
        data_right[v] = data_right[v].astype('category')
        le.fit(data_right[v])
        data_right[v] = le.transform(data_right[v])
        encoders[v] = le
    except:
        print v
        
data_preprocessed = pd.concat([data_left,data_right],axis=1)

#binarize labels
labels = labels.as_matrix()
y = preprocessing.label_binarize(labels, classes=[-1, 1]).ravel()

#Balance classes
sm = SMOTEENN()
X_resampled, y_resampled = sm.fit_sample(data_preprocessed, y)
print('Original dataset shape {}'.format(Counter(y)))
print('Resampled dataset shape {}'.format(Counter(y_resampled)))

#Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

#Train the model
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train, y_train)

#Predict the test dataset
y_score = clf.predict_proba(X_test)

#Evaluate the model
clf.score(X_test, y_test)  #accuracy
fpr,tpr, threshold = roc_curve(y_test,y_score[:,1])

print 'Accuracy:',accuracy_score(y_test,clf.predict(X_test))
print 'Recall:',recall_score(y_test,clf.predict(X_test))
# Compute ROC curve and ROC area for each class

roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

