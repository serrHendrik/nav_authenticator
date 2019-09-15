# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 16:00:39 2019

@author: Hendrik Serruys

"""
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from nav_datahandler import nav_datahandler





def validate_learner(dh, autoencoder):
    #df_sampled = dh.get_df_sampled()
    #df_feat_unscaled = dh.get_df_feat_unscaled()
    
    #delay = 7200
    #dh = nav_datahandler(delay)
    
    df_features = dh.get_df_features()
    feature_names = df_features.columns.values
    figsize = (12,6)
    input_size = dh.get_input_size()
    feature_weights = dh.get_feature_weights()
    trainX, valX, test1X, test2X = dh.get_tvt()
    
    ## TRAINING RESULTS
    #Check Training
    train_predict = autoencoder.predict(trainX)
    err_train = np.square(train_predict - trainX)
    plt.figure(figsize=figsize)
    for i in range(input_size):
        plt.plot(err_train[:,i], label = feature_names[i])
    plot_name = "Reconstruction error of all variables over time [Training set]"
    plt.xlabel("Time")
    plt.ylabel("Squared reconstruction error")
    plt.title(plot_name)
    plt.legend()
    plt.savefig(plot_name, dpi=300)
    plt.show()
    
    
    ## VALIDATION RESULTS
    #Validation threshold
    val_predict = autoencoder.predict(valX)
    err_val = np.square(val_predict - valX)
    plt.figure(figsize=figsize)
    for i in range(input_size):
        plt.plot(err_val[:,i], label = feature_names[i])
    plot_name = "Reconstruction error of all variables over time [Valdation set]"
    plt.xlabel("Time")
    plt.ylabel("Squared reconstruction error")
    plt.title(plot_name)
    plt.legend()
    plt.savefig(plot_name, dpi=300)
    plt.show()
    
    err_val_mean = np.average(err_val, axis=1, weights=feature_weights)
    val_std = np.std(err_val_mean)
    threshold = 5*np.mean(err_val_mean) + 5*val_std
    
    plt.figure(figsize=figsize)
    plt.plot(err_val_mean, 'k.')
    plt.axhline(y=threshold, color='r', linestyle='-')
    plt.xlabel("Time")
    plt.ylabel("Weighted MSE")
    plot_name = "Weighted average reconstruction error over time [Validation set]"
    plt.title(plot_name)
    plt.savefig(plot_name, dpi=300)
    plt.show()
    
    
# =============================================================================
#     var_nb = 10
#     var_name = feature_names[var_nb]
#     val_predict_var = val_predict[:,var_nb]
#     val_var = valX[:,var_nb]
#     plt.figure(figsize=figsize)
#     plt.plot(val_var, label = "GT")
#     plt.plot(val_predict_var, label = "Prediction")
#     plot_name = "Reproduction of one Validation set variable - " + var_name + " [Validation set]"
#     plt.title(plot_name)
#     plt.legend()
#     plt.savefig(plot_name, dpi=300)
#     plt.show()
# =============================================================================
    

    #TEST 1 FOR FALSE POSITIVES
    test_predict = autoencoder.predict(test1X)
    err_test = np.square(test_predict - test1X)
    plt.figure(figsize=figsize)
    for i in range(input_size):
        plt.plot(err_test[:,i], label = feature_names[i])
    plt.xlabel("Time")
    plt.ylabel("Squared reconstruction error")
    plot_name = "Reconstruction error of all variables over time [Test set 1]"
    plt.title(plot_name)
    plt.legend()
    plt.savefig(plot_name, dpi=300)
    plt.show()
    
    err_test_mean = np.average(err_test, axis=1, weights=feature_weights)
    plt.figure(figsize=figsize)
    plt.plot(err_test_mean, 'k.')
    plt.axhline(y=threshold, color='r', linestyle='-')
    plt.xlabel("Time")
    plt.ylabel("Weighted MSE")
    plot_name = "Weighted average reconstruction error over time [Test set 1]"
    plt.title(plot_name)
    plt.savefig(plot_name, dpi=300)
    plt.show()
    
    return threshold

def check_variable_test1(dh, autoencoder, var_nb):
    
    df_features = dh.get_df_features()
    feature_names = df_features.columns.values
    figsize = (12,6)

    trainX, valX, test1X, test2X = dh.get_tvt()
    test_predict = autoencoder.predict(test1X)
    var_name = feature_names[var_nb]
    test_predict_var = test_predict[:,var_nb]
    test_var = test1X[:,var_nb]
    plt.figure(figsize=figsize)
    plt.plot(test_var, 'm*', label = "Ground truth", color=(0.196, 0.8, 0.196))
    plt.plot(test_predict_var, 'k.', label = "VAE sample", color=(0.1,0.1,0.1))
    plt.xlabel("Time")
    plt.ylabel(var_name + " (scaled)")
    plot_name = "Reconstruction of one variable - " + var_name + " [Test set 1]"
    plt.title(plot_name)
    plt.legend()
    plt.savefig(plot_name, dpi=300)
    plt.show()
    

def test_anomaly(dh, autoencoder, threshold):
    
    #delay = 7200
    #dh = nav_datahandler(delay)
    
    df_features = dh.get_df_features()
    feature_names = df_features.columns.values
    figsize = (12,6)
    input_size = dh.get_input_size()
    feature_weights = dh.get_feature_weights()
    trainX, valX, test1X, test2X = dh.get_tvt()
    
    ### ANOMALY TEST
    #Plot error for variables individually
    test_predict = autoencoder.predict(test2X)
    err_test = np.square(test_predict - test2X)
    plt.figure(figsize=figsize)
    for i in range(input_size):
        plt.plot(err_test[:,i], label = feature_names[i])
    plt.xlabel("Time")
    plt.ylabel("Squared reconstruction error")
    plot_name = "Reconstruction error of all variables over time [Test set 2]"
    plt.title(plot_name)
    plt.legend()
    plt.savefig(plot_name, dpi=300)
    plt.show()
    
    #plot the mean error over all variables
    err_test_mean = np.average(err_test, axis=1, weights=feature_weights)
    plt.figure(figsize=figsize)
    plt.plot(err_test_mean, 'k')
    plt.axhline(y=threshold, color='r', linestyle='-')
    plt.xlabel("Time")
    plt.ylabel("Weighted MSE")
    plot_name = "Weighted average reconstruction error over time [Test set 2]"
    plt.title(plot_name)
    plt.savefig(plot_name, dpi=300)
    plt.show()

def check_variable_test2(dh, autoencoder, var_nb):
    df_features = dh.get_df_features()
    feature_names = df_features.columns.values
    figsize = (12,6)
    
    trainX, valX, test1X, test2X = dh.get_tvt()
    test_predict = autoencoder.predict(test2X)
    var_name = feature_names[var_nb]
    test_predict_var = test_predict[:,var_nb]
    test_var = test2X[:,var_nb]
    plt.figure(figsize=figsize)
    plt.plot(test_var, 'm*', label = "Ground truth", color=(0.196, 0.8, 0.196))
    plt.plot(test_predict_var, 'k.', label = "VAE sample", color=(0.1,0.1,0.1))
    plt.xlabel("Time")
    plt.ylabel(var_name + " (scaled)")
    plot_name = "Reconstruction of one variable - " + var_name + " [Test set 2]"
    plt.title(plot_name)
    plt.legend()
    plt.savefig(plot_name, dpi=300)
    plt.show()
    

def visualise_code_space(encoder, testX, scaler):
    z_mean, z_log_var, z = encoder.predict(testX, batch_size=16)
    testXrev = scaler.inverse_transform(testX)
    plt.figure(figsize=(12, 10))
    dim1 = 0
    dim2 = 2
    plt.scatter(z[:, dim1], z[:, dim2], c=testXrev[:,6]) #choose c the distance and produce extreme distances.
    plt.colorbar()
    plt.xlabel("z["+str(dim1)+"]")
    plt.ylabel("z["+str(dim2)+"]")
    plot_name = "Representation of latent code space [Test set 2]"
    plt.title(plot_name)
    plt.savefig(plot_name, dpi=300)
    plt.show()
    
    

