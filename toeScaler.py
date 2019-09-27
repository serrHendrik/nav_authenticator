# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:52:21 2019

@author: Hendrik Serruys
"""
import numpy as np


class toeScaler:
    def __init__(self):
        # Coef. obtained in R (see toe_effect.R)
        
        self.coef_sp_X = np.array([1.510558, 4.776131e-02, 2.854745e-07])
        self.coef_sp_Y = np.array([-1.901359e-01,  5.106983e-02,  1.136013e-07])
        self.coef_sp_Z = np.array([-2.033188,  5.667171e-02, -4.700973e-07])
        
        self.coef_sv_X = np.array([-1.544922e-04,  4.044515e-06, -3.144953e-11])
        self.coef_sv_Y = np.array([-7.135311e-05,  3.969120e-06, -2.894467e-11])
        self.coef_sv_Z = np.array([2.374321e-04, 5.885553e-06, 6.085669e-11])
        
        self.coef_svCb = np.array([3.743611e-11,  9.022669e-14, -2.151141e-18])
        
    def scale(self, df, toe_ref):
        toe_df = df.iloc[0]["toe"]
        stds_df = self.get_all_stds(toe_df)
        stds_ref = self.get_all_stds(toe_ref)
        scale_vector = np.divide(stds_ref, stds_df)
        
        df_new = df.copy(deep=True)
        df_new["sp_X"] = df_new["sp_X"].values * scale_vector[0]
        df_new["sp_Y"] = df_new["sp_Y"].values * scale_vector[1]
        df_new["sp_Z"] = df_new["sp_Z"].values * scale_vector[2]
        df_new["sv_X"] = df_new["sv_X"].values * scale_vector[3]
        df_new["sv_Y"] = df_new["sv_Y"].values * scale_vector[4]
        df_new["sv_Z"] = df_new["sv_Z"].values * scale_vector[5]
        df_new["svCb"] = df_new["svCb"].values * scale_vector[6]
        
        return df_new
        
        
        
    def get_all_stds(self, toe):
        std_sp_X = self.get_std(self.coef_sp_X, toe)
        std_sp_Y = self.get_std(self.coef_sp_Y, toe)
        std_sp_Z = self.get_std(self.coef_sp_Z, toe)
        
        std_sv_X = self.get_std(self.coef_sv_X, toe)
        std_sv_Y = self.get_std(self.coef_sv_Y, toe)
        std_sv_Z = self.get_std(self.coef_sv_Z, toe)
        
        std_svCb = self.get_std(self.coef_svCb, toe)
        
        result = np.array([std_sp_X, std_sp_Y, std_sp_Z, std_sv_X, std_sv_Y, std_sv_Z, std_svCb])
        return  result
        
    def get_std(self, coef_vector, toe):
        return np.sum(np.multiply(coef_vector, np.array([1, toe, toe**2])))
