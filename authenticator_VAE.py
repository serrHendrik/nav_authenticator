# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 21:12:55 2019

@author: Hendrik Serruys


"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from keras.optimizers import Adam

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from nav_datahandler import nav_datahandler
from test_authenticator import validate_learner, test_anomaly

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


#Get datahandler
dataset_id = 3
dh = nav_datahandler(dataset_id)
df_original = dh.get_df_original()
df_features = dh.get_df_features()
feature_names = df_features.columns.values
trainX, valX, test1X, test2X = dh.get_tvt()


# network parameters
input_size = dh.get_input_size()
input_shape = (input_size,)
intermediate_dim = input_size - 1
intermediate_dim2 = input_size - 2
batch_size = 32
latent_dim = input_size - 2
epochs = 200

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(intermediate_dim, activation='tanh')(inputs) #relu
x = Dense(intermediate_dim2, activation='tanh')(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
#plot_model(encoder, to_file='vae_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim2, activation='tanh')(latent_inputs) #relu
x = Dense(intermediate_dim, activation='tanh')(x)
outputs = Dense(input_size, activation='tanh')(x) #Try act LINEAR for mse loss!

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
#plot_model(decoder, to_file='vae_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')



#data = (testX, testX)
reconstruction_loss = mse(inputs, outputs)
reconstruction_loss *= input_size
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.005 #-0.5
vae_loss = K.mean(reconstruction_loss + kl_loss) #+kl_loss
vae.add_loss(vae_loss)
opt = Adam(lr=0.001)
vae.compile(optimizer=opt)
vae.summary()
#plot_model(vae, to_file='vae.png', show_shapes=True)

history = vae.fit(trainX,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(valX, None))

# Plot loss
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

# Save model
model_fn = "models/vae1.h5"
vae.save_weights(model_fn)

# =============================================================================
# z_mean, _, _ = encoder.predict(valX, batch_size=16)
# plt.figure(figsize=(12, 10))
# plt.scatter(z_mean[:, 0], z_mean[:, 1], c=valX[:,-1]) #choose c the distance and produce extreme distances.
# plt.colorbar()
# plt.xlabel("z[0]")
# plt.ylabel("z[1]")
# #plt.savefig(filename)
# plt.show()
# 
# valX_pred = decoder.predict(z_mean)
# =============================================================================


#Validate learner
threshold = validate_learner(dh, vae)

#Test anomaly
test_anomaly(dh, vae, threshold)


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




