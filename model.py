#!/usr/bin/env python

import math as m
import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA

# Keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.normalization import BatchNormalization
from keras.constraints import maxnorm

import get_data

model_name = "NN50-50"

print("Importing data")
tr_data = get_data.import_data('numerai_training_data.csv')

n_col = tr_data.shape[1]
data = tr_data.ix[:,0:(n_col-1)].values
target = tr_data.ix[:,(n_col-1)].values

t_data = get_data.import_data('numerai_tournament_data.csv')
t_id = t_data.ix[:, 0].values
test_data = t_data.ix[:,1:n_col].values

print("Standardising data")
scaler = StandardScaler().fit(data)
data_std = scaler.transform(data)

print("Running PCA on data")
k = 20
pca = sklearnPCA(n_components=k).fit(data_std)
data_std_pca = pca.transform(data_std)

print("Training neural network")
input_dim = data_std_pca.shape[1]

np.random.seed(8)
np.random.shuffle(data_std_pca)
np.random.seed(8)
np.random.shuffle(target)

NN_model = Sequential()
NN_model.add(Dense(50, init='glorot_uniform', input_dim=input_dim))
NN_model.add(PReLU())
NN_model.add(Dropout(0.5))

NN_model.add(Dense(50, init='glorot_uniform'))
NN_model.add(PReLU())
NN_model.add(Dropout(0.5))

NN_model.add(Dense(1, activation='sigmoid'))
NN_model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['binary_crossentropy'])

NN_model.fit(data_std_pca, target, nb_epoch=50, batch_size=128, shuffle=True, validation_split=0.2, verbose=True)

print("Transforming test data")
test_data_std = scaler.transform(test_data)
test_data_std_pca = pca.transform(test_data_std)

print("Making predictions")
predictions = NN_model.predict_proba(test_data_std_pca)

print("Saving prediction output")
output_file = pd.DataFrame(t_id)
output_file.columns = ['t_id']
output_file['probability'] = predictions
output_file.to_csv(('predictions_%s.csv' % model_name), columns = ('t_id', 'probability'), index = None)

