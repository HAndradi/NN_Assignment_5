import numpy as np

from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight

from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from utils import *

np.random.seed(0)

test_data, test_labels, train_data, train_labels = split_data()

# Define model
input_data = Input(shape=(1021,))
x = Dense(1021, activation='relu')(input_data)
output_data = Dense(1, activation='sigmoid')(x)
model = Model(input_data, output_data)
# Compile model
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.000001), metrics=['accuracy'])
# Evaluate model
weights = compute_class_weight('balanced',np.array([0,1]),train_labels)
model.fit(train_data, train_labels, batch_size=5, epochs=10, class_weight={0:weights[0],1:weights[1]}, verbose=1)
predicted_labels = model.predict(test_data)
predicted_labels = np.array([i > 0.5 for i in predicted_labels])

print ("NN PERFORMANCE")
print ("confusion matrix:")
print ("-- format --\n",np.array([["true_pos", "false_neg"], ["false_pos", "true_neg"]]))
print ("-- values --\n",confusion_matrix(test_labels, predicted_labels, [1,0]))
print ("precision:",precision_score(test_labels, predicted_labels, [1,0]))
print ("recall:",recall_score(test_labels, predicted_labels, [1,0]))

del model

