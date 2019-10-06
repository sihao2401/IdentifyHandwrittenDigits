import pandas as pd
from sklearn import svm
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.neighbors import KNeighborsClassifier

data_train = pd.read_csv('pendigits.tra',header=None)
x_train, y_train = np.split(data_train, (16,), axis=1)
data_test = pd.read_csv('pendigits.tes',header=None)
x_test, y_test = np.split(data_test, (16,), axis=1)


clf = svm.SVC(kernel='linear',decision_function_shape='ovo', probability=True)
clf.fit(x_train, y_train.values.ravel())
print("Training dataset accuracy:")
print (clf.score(x_train, y_train))  # accuracy
y_hat = clf.predict(x_train)
print("Test dataset accuracy:")
print (clf.score(x_test, y_test))
y_hat = clf.predict(x_test)

np.random.seed(0)
digits_class = np.unique(y_train)
n_class = digits_class.size
y = pd.Categorical(y_train).codes
y_one_hot = label_binarize(y_test, np.arange(n_class))
alpha = np.logspace(-2, 2, 20)
y_score = clf.predict_proba(x_test)
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
plt.title(u'SVM ROC and AUC', fontsize=17)
plt.show()
