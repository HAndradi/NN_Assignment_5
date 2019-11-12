import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import SGD, Adagrad, RMSprop, Adam

extract_data = lambda raw_data, rows : [raw_data[col].values[:rows] for col in raw_data.columns[1:]]

def create_test_train_datasets(pos_data, neg_data, num_of_test_data):
    n = int(num_of_test_data/2)
    if min(len(pos_data),len(neg_data)) < num_of_test_data:
        print ("Not enough positive/negative data points")
    else:
        np.random.shuffle(pos_data)
        np.random.shuffle(neg_data)
        test_data = np.vstack((pos_data[:n],neg_data[:n]))
        train_data = np.vstack((pos_data[n:],neg_data[n:]))
        test_labels = np.hstack((np.ones(n),np.zeros(n)))
        train_labels = np.hstack((np.ones(len(pos_data)-n),np.zeros(len(neg_data)-n)))
    return test_data, test_labels, train_data, train_labels 

raw_data_skin = pd.read_csv('2016skin.csv', sep=';', decimal='.')
raw_data_material = pd.read_csv('2016material.csv', sep=';', decimal='.')
raw_data_material_fake = pd.read_csv('2016material-fake.csv', sep=';', decimal='.')

data_skin = np.array(extract_data(raw_data_skin,1021))
data_not_skin = np.array(extract_data(raw_data_material,1021) + extract_data(raw_data_material_fake,1021))


test_data, test_labels, train_data, train_labels = create_test_train_datasets(data_skin, data_not_skin, 100)

print ("confusion matrix format:")
print (np.array([["true_pos", "false_neg"], ["false_pos", "true_neg"]]), end="\n\n")

svm_linear = svm.SVC(C=1000, kernel='linear')
svm_linear.fit(train_data, train_labels)
print ("LINEAR SVM PERFORMANCE")
print ("confusion matrix:")
print (confusion_matrix(test_labels, svm_linear.predict(test_data), [1,0]))
print ("precision:")
print (precision_score(test_labels, svm_linear.predict(test_data), [1,0]))
print ("recall:")
print (recall_score(test_labels, svm_linear.predict(test_data), [1,0]), end="\n\n")


svm_rbf = svm.SVC(C=1000, kernel='rbf', gamma='scale')
svm_rbf.fit(train_data, train_labels)
print ("RBF SVM PERFORMANCE")
print ("confusion matrix:")
print (confusion_matrix(test_labels, svm_rbf.predict(test_data), [1,0]))
print ("precision:")
print (precision_score(test_labels, svm_rbf.predict(test_data), [1,0]))
print ("recall:")
print (recall_score(test_labels, svm_rbf.predict(test_data), [1,0]), end="\n\n")

input_data = Input(shape=(1021,))
hidden_layers = Dense(2000, activation='relu')(input_data)
hidden_layers = Dense(1000, activation='relu')(hidden_layers)
hidden_layers = Dense(500, activation='relu')(hidden_layers)
output_data = Dense(2, activation='linear')(hidden_layers)
nn = Model(input_data, output_data)
#sgd = SGD(lr=0.01, clipvalue=0.5)
sgd = Adam(lr=0.1, clipvalue=5)
nn.compile(optimizer=sgd, loss='mse')
adjusted_train_labels = np.array([[1,0] if label==1 else [0,1] for label in train_labels])
nn.fit(train_data, adjusted_train_labels, epochs=1000, batch_size=100, shuffle=True)
#nn.fit(train_data, adjusted_train_labels, epochs=100, batch_size=100, shuffle=True, verbose=0)
predicted_labels = nn.predict(test_data)
adjusted_predicted_labels = np.array([1 if label[0]>label[1] else 0 for label in predicted_labels])
print ("NN PERFORMANCE")
print ("confusion matrix:")
print (confusion_matrix(test_labels, adjusted_predicted_labels, [1,0]))
print ("precision:")
print (precision_score(test_labels, adjusted_predicted_labels, [1,0]))
print ("recall:")
print (recall_score(test_labels, adjusted_predicted_labels, [1,0]), end="\n\n")
