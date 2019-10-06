#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 18:21:21 2019

@author: SIHao
"""
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
digits_train = pd.read_csv('pendigits.tra',header=None)
digits_test = pd.read_csv('pendigits.tes',header=None)
data = digits_train.iloc[:,0:16]
label = digits_train.iloc[:,-1]
knn.fit(data,label)
test_data = digits_test.iloc[:,0:16]
test_label = digits_test.iloc[:,-1]
counter=0

for i in range(0,len(test_data)):
    predictlable=knn.predict([test_data.iloc[i]])
    if predictlable == test_label[i]:
        counter+=1
correct=counter/len(test_data) 

np.random.seed(0)
digits_class = label.unique() #digits_class=[0,1,2,3,4,5,6,7,8,9]
n_class = digits_class.size  # n_class=10
y = pd.Categorical(label).codes
y_one_hot = label_binarize(test_label, np.arange(n_class))
alpha = np.logspace(-2, 2, 20)
y_score = knn.predict_proba(test_data)
metrics.roc_auc_score(y_one_hot, y_score, average='micro')
fpr, tpr, thresholds = metrics.roc_curve(y_one_hot.ravel(),y_score.ravel())

auc = metrics.auc(fpr, tpr)
mpl.rcParams['font.sans-serif'] = u'SimHei'
mpl.rcParams['axes.unicode_minus'] = False
plt.plot(fpr, tpr, c = 'r', lw = 2, alpha = 0.7, label = u'AUC=%.3f' % auc)
plt.plot((0, 1), (0, 1), c = '#808080', lw = 1, ls = '--', alpha = 0.7)
plt.xlim((-0.01, 1.02))
plt.ylim((-0.01, 1.02))
plt.xticks(np.arange(0, 1.1, 0.1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.xlabel('False Positive Rate', fontsize=13)
plt.ylabel('True Positive Rate', fontsize=13)
plt.grid(b=True, ls=':')
plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
plt.title(u'ROC and AUC', fontsize=17)
plt.show()

print('The success rate is %.5f' %(correct))
