import numpy as np

from sklearn import svm
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from utils import *

np.random.seed(0)

test_data, test_labels, train_data, train_labels = split_data()

svm_linear = svm.SVC(C=1000, kernel='linear')
svm_linear.fit(train_data, train_labels)

print ("LINEAR SVM PERFORMANCE")
print ("confusion matrix:")
print ("-- format --\n",np.array([["true_pos", "false_neg"], ["false_pos", "true_neg"]]))
print ("-- values --\n",confusion_matrix(test_labels, svm_linear.predict(test_data), [1,0]))
print ("precision:",precision_score(test_labels, svm_linear.predict(test_data), [1,0]))
print ("recall:",recall_score(test_labels, svm_linear.predict(test_data), [1,0]))

del svm_linear

