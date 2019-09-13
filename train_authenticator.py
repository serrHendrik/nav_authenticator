# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 21:12:55 2019

@author: serru
"""

import numpy as np
import pandas as pd
from keras.models import Model, model_from_json
from keras.layers import Input, Dense
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
#from satellite_nav_generator import satellite_nav_generator



# =============================================================================
# two_hours = 2*3600
# nb_sats = 10
# sat = satellite_nav_generator(two_hours)
# df = sat.data
# 
# for s in range(nb_sats-1):
#     sat = satellite_nav_generator(two_hours)
#     df = df.append(sat.data, ignore_index = True)
#
# plt.figure(figsize=(12,6))
# plt.plot(df.dX)
# plt.plot(df.dY)
# plt.plot(df.dZ)
# plt.plot(df.dT)
# plt.show()
# 
# plt.scatter(df.time, df.dX)
# plt.show()
# =============================================================================
    

#filename = "nav_data/10-09-2019_toe1800.csv"
filename = "nav_data/10-09-2019_toe7200.csv"
df = pd.read_csv(filename)



feat = ["sp_X", "sp_Y", "sp_Z", "sv_X", "sv_Y", "sv_Z", "svCb"]
df_feat = df[feat]
#add squared features
df_feat = df_feat.assign(sp_X2 = np.square(df_feat["sp_X"].values))
df_feat = df_feat.assign(sp_Y2 = np.square(df_feat["sp_Y"].values))
df_feat = df_feat.assign(sp_Z2 = np.square(df_feat["sp_Z"].values))
#add overall distance
df_feat = df_feat.assign(sp_D = np.sqrt(df_feat["sp_X2"] + df_feat["sp_Y2"] + df_feat["sp_Z2"]))



#Standardize data -> scikit learn
scaler = MinMaxScaler(feature_range=(0.1,0.9))
scaler.fit(df_feat)
df_scaled = pd.DataFrame(scaler.transform(df_feat), columns = df_feat.columns)

#Add time of week feature
toe_scaled = df["toe"].values / 7200.0
df_scaled = df_scaled.assign(toe = toe_scaled)

#Build autoencoder (also provide time in df as feature!)
input_size = df_scaled.columns.values.shape[0]
hidden_size = 10
code_size = 8
input_layer = Input(shape=(1,input_size))
hidden1 = Dense(hidden_size, activation='relu')(input_layer)
enc = Dense(code_size, activation='relu')(hidden1)
hidden2 = Dense(hidden_size, activation='relu')(enc)
dec = Dense(input_size, activation='relu')(hidden2)
autoencoder = Model(input_layer, dec)

# =============================================================================
# encoder = Model(input_layer, enc)
# encoder_output = Input(shape=(1,code_size))
# dec_layer = autoencoder.layers[-1]
# decoder = Model(encoder_output, dec_layer(encoder_output))
# =============================================================================

autoencoder.compile('adam', loss='mean_squared_error')

#Prepare data
val_start = np.where(df["svId"] == 30,)[0][0]
test_start = np.where(df["svId"] == 33,)[0][0]
trainX = df_scaled.values[0:val_start, :]
trainX = trainX.reshape(trainX.shape[0], 1, trainX.shape[1])
valX = df_scaled.values[val_start:test_start, :]
valX = valX.reshape(valX.shape[0], 1, valX.shape[1])
testX = df_scaled.values[test_start:, :]
testX = testX.reshape(testX.shape[0], 1, testX.shape[1])

#Train
history = autoencoder.fit(trainX, trainX,
                 epochs = 10,
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
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#Validation threshold
val_predict = autoencoder.predict(valX)
err_val = np.square(val_predict - valX)
for i in range(input_size):
    plt.plot(err_val[:,0,i])
plt.title("Error of all variables over time [Valdation set]")
plt.show()

err_val_mean = np.mean(err_val, axis=2)
val_std = np.std(err_val_mean)
threshold = np.mean(err_val_mean) + 3*val_std
plt.plot(err_val_mean)
plt.hlines(threshold, 0, valX.shape[0])
plt.title("Mean error over time [Validation set]")
plt.show()


# =============================================================================
# val_predict_dX = val_predict[:,0,0]
# val_dX = valX[:,0,0]
# 
# err = np.abs(val_predict_dX - val_dX)
# plt.plot(err)
# plt.title("Error of prediction on Validation set")
# plt.show()
# 
# plt.plot(val_dX)
# plt.plot(val_predict_dX)
# plt.title("Reproduction of valdation dX")
# plt.show()
# =============================================================================



#Test
test_predict = autoencoder.predict(testX)
err_test = np.square(test_predict - testX)
for i in range(input_size):
    plt.plot(err_test[:,0,i])
plt.title("Error of all variables over time [Test set]")
plt.show()

err_test_mean = np.mean(err_test, axis=2)
plt.plot(err_test_mean)
plt.hlines(threshold, 0, testX.shape[0])
plt.title("Mean error over time [Test set]")
plt.show()

# =============================================================================
# var_nb = 5
# test_predict_var = test_predict[:,0,var_nb]
# test_var = testX[:,0,var_nb]
# err_test = np.abs(test_predict_var - test_var)
# plt.plot(err_test)
# plt.title("Error of prediction on Test set")
# plt.show()
# 
# plt.plot(test_var)
# plt.plot(test_predict_var)
# plt.title("Reproduction of Test set")
# plt.show()
# =============================================================================



# Store model
model_fn = "models/ae1.h5"
autoencoder.save(model_fn)




