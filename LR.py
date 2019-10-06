#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:02:33 2019

@author: SIHao
"""
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

digits_train = pd.read_csv('pendigits.tra',header=None)
digits_test = pd.read_csv('pendigits.tes',header=None)
#Training set
data = digits_train.iloc[:,0:16]
label = digits_train.iloc[:,-1]
#Test set
test_data = digits_test.iloc[:,0:16]
test_label = digits_test.iloc[:,-1]

lr = LogisticRegression(C = 100.0,
                      penalty = 'l2',
                      multi_class='multinomial',
                      max_iter=1000,
                      solver= 'newton-cg')
lr.fit(data,label)
'''
parameters = { "penalty":['l2'],
                "max_iter":[500, 1000, 2000],
                "multi_class":['multinomial'],
                "solver":['newton-cg','lbfgs','saga']
            }
lr_gs=GridSearchCV(LogisticRegression(),param_grid=parameters,cv= 10)
lr_gs.fit(data,label)
score=lr_gs.score(test_data,test_label)
pred=lr_gs.predict(test_data)
'''
print("Training accuracy rate：%.5f" %lr.score(data, label))
print("Logistic Regression accuracy rate：%.5f" %lr.score(test_data, test_label))

   

np.random.seed(0)
digits_class = label.unique()
n_class = digits_class.size
y = pd.Categorical(label).codes
y_one_hot = label_binarize(test_label, np.arange(n_class))
alpha = np.logspace(-2, 2, 20)
y_score = lr.predict_proba(test_data)
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
plt.title(u'Logistic Regression ROC and AUC', fontsize=17)
plt.show()