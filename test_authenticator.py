# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 16:00:39 2019

@author: nsr
"""
from keras.models import load_model


#load model
model_fn = "models/ae1.h5"
autoencoder = load_model(model_fn)

#set threshold
thr = 0.019542888570611996

val_start = np.where(df["svId"] == 30,)[0][0]