# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 22:18:18 2019

@author: Hendrik Serruys
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from nav_datahandler import nav_datahandler
from test_authenticator import validate_learner, test_anomaly, visualise_code_space, check_variable_test1, check_variable_test2

#Get datahandler
#delay = 1800
delay = 7200
dh = nav_datahandler(delay)

#load model
#ae_fn = "models/ae2.h5"
#autoencoder = load_model(ae_fn)

# Threshold ae2
#thr = 0.0075835196024732435

#threshold = validate_learner(autoencoder)
#test_anomaly(autoencoder, threshold)



### VAE
#threshold = validate_learner(dh, vae)
#check_variable_test1(dh, vae, 3)

test_anomaly(dh, vae, threshold)
#check_variable_test2(dh, vae, 6)
visualise_code_space(encoder, dh.test2X, dh.scaler)







