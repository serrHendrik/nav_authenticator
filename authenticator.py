# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 21:12:55 2019

@author: serru
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


#filename = "nav_data/10-09-2019_toe1800.csv"
filename = "nav_data/10-09-2019_toe7200.csv"
df = pd.read_csv(filename)

feat = ["sp_X", "sp_Y", "sp_Z", "sv_X", "sv_Y", "sv_Z", "svCb"]
df_feat = df[feat]
#add squared features
#df_feat = df_feat.assign(sp_X2 = np.square(df_feat["sp_X"].values))
#df_feat = df_feat.assign(sp_Y2 = np.square(df_feat["sp_Y"].values))
#df_feat = df_feat.assign(sp_Z2 = np.square(df_feat["sp_Z"].values))
#add overall distance
df_feat = df_feat.assign(sp_D = np.sqrt(df_feat["sp_X2"] + df_feat["sp_Y2"] + df_feat["sp_Z2"]))

#Standardize data -> scikit learn
scaler = MinMaxScaler(feature_range=(0.1,0.9))
scaler.fit(df_feat)
df_scaled = pd.DataFrame(scaler.transform(df_feat), columns = df_feat.columns)

#Add time of week feature
toe_scaled = df["toe"].values / 7200.0
df_scaled = df_scaled.assign(toe = toe_scaled)

#Prepare data
val_start = np.where(df["svId"] == 30,)[0][0]
test_start = np.where(df["svId"] == 33,)[0][0]
trainX = df_scaled.values[0:val_start, :]
#trainX = trainX.reshape(trainX.shape[0], 1, trainX.shape[1])
valX = df_scaled.values[val_start:test_start, :]
#valX = valX.reshape(valX.shape[0], 1, valX.shape[1])
testX = df_scaled.values[test_start:, :]
#testX = testX.reshape(testX.shape[0], 1, testX.shape[1])


# network parameters
input_size = df_scaled.columns.values.shape[0]
input_shape = (input_size,)
intermediate_dim = 6
batch_size = 32
latent_dim = 4
epochs = 2

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(intermediate_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
plot_model(encoder, to_file='vae_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(input_size, activation='sigmoid')(x) #Try act LINEAR for mse loss!

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='vae_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')



#data = (testX, testX)
reconstruction_loss = mse(inputs, outputs)
reconstruction_loss *= input_size
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.summary()
plot_model(vae,
           to_file='vae.png',
           show_shapes=True)

history = vae.fit(trainX,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(valX, None))
#vae.save_weights('vae_mlp_mnist.h5')


z_mean, _, _ = encoder.predict(valX, batch_size=16)
plt.figure(figsize=(12, 10))
plt.scatter(z_mean[:, 0], z_mean[:, 1], c=valX[:,-2]) #choose c the distance and produce extreme distances.
plt.colorbar()
plt.xlabel("z[0]")
plt.ylabel("z[1]")
#plt.savefig(filename)
plt.show()

valX_pred = decoder.predict(z_mean)

# =============================================================================
# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()
# 
# #Validation threshold
# val_predict = vae.predict(valX)
# err_val = np.square(val_predict - valX)
# for i in range(input_size):
#     plt.plot(err_val[:,0,i])
# plt.title("Error of all variables over time [Valdation set]")
# plt.show()
# 
# err_val_mean = np.mean(err_val, axis=2)
# val_std = np.std(err_val_mean)
# threshold = np.mean(err_val_mean) + 3*val_std
# plt.plot(err_val_mean)
# plt.hlines(threshold, 0, valX.shape[0])
# plt.title("Mean error over time [Validation set]")
# plt.show()
# 
# 
# #Test
# test_predict = vae.predict(testX)
# err_test = np.square(test_predict - testX)
# for i in range(input_size):
#     plt.plot(err_test[:,0,i])
# plt.title("Error of all variables over time [Test set]")
# plt.show()
# 
# err_test_mean = np.mean(err_test, axis=2)
# plt.plot(err_test_mean)
# plt.hlines(threshold, 0, testX.shape[0])
# plt.title("Mean error over time [Test set]")
# plt.show()
# =============================================================================




