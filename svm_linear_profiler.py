import numpy as np

from sklearn import svm
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from utils import *

from memory_profiler import profile

np.random.seed(0)

@profile(precision=4)
def run():
    test_data, test_labels, train_data, train_labels = split_data()
    svm_linear = svm.SVC(C=1000, kernel='linear')
    svm_linear.fit(train_data, train_labels)

if __name__=='__main__':
    run()

