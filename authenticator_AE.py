# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 21:12:55 2019

@author: Hendrik Serruys
"""

import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from nav_datahandler import nav_datahandler


#Get datahandler
#delay = 1800
delay = 7200
dh = nav_datahandler(delay)
df_original = dh.get_df_original()
df_features = dh.get_df_features()
feature_names = df_features.columns.values
feature_weights = dh.get_feature_weights()
trainX, valX, test1X, test2X = dh.get_tvt()



#Build autoencoder (also provide time in df as feature!)
input_size = dh.get_input_size()
hidden_size = input_size - 1
code_size = hidden_size - 1
input_layer = Input(shape=(input_size,))
hidden1 = Dense(hidden_size, activation='tanh')(input_layer)
enc = Dense(code_size, activation='tanh')(hidden1)
hidden2 = Dense(hidden_size, activation='tanh')(enc)
dec = Dense(input_size, activation='tanh')(hidden2)
autoencoder = Model(input_layer, dec)
# =============================================================================
# encoder = Model(input_layer, enc)
# encoder_output = Input(shape=(1,code_size))
# dec_layer = autoencoder.layers[-1]
# decoder = Model(encoder_output, dec_layer(encoder_output))
# =============================================================================
opt = Adam(lr=0.001)
autoencoder.compile(opt, loss='mean_squared_error')

#Train
history = autoencoder.fit(trainX, trainX,
                 epochs = 100,
                 batch_size = 32,
                 shuffle = True,
                 validation_data = (valX, valX))

#Plot Training info
# =============================================================================
# # summarize history for accuracy
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# =============================================================================
# summarize history for loss
figsize = (12,6)
plt.figure(figsize=figsize)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.yscale('log')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# Store model
model_fn = "models/ae2.h5"
autoencoder.save(model_fn)




