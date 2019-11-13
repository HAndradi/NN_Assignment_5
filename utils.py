import numpy as np
import pandas as pd

def extract_data(raw_data, rows): 
    return [raw_data[col].values[:rows] for col in raw_data.columns[1:]]

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

def split_data():
    raw_data_skin = pd.read_csv('2016skin.csv', sep=';', decimal='.')
    raw_data_material = pd.read_csv('2016material.csv', sep=';', decimal='.')
    raw_data_material_fake = pd.read_csv('2016material-fake.csv', sep=';', decimal='.')

    data_skin = np.array(extract_data(raw_data_skin,1021))
    data_not_skin = np.array(extract_data(raw_data_material,1021) + extract_data(raw_data_material_fake,1021))

    return create_test_train_datasets(data_skin, data_not_skin, 100)

