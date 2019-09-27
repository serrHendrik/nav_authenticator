# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 09:26:51 2019

@author: Hendrik Serruys
"""

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import pandas as pd
from toeScaler import toeScaler

class nav_datahandler:
    def __init__(self, dataset_id):
        if dataset_id == 1:
            self.filename = "nav_data/10-09-2019_toe1800.csv"
        elif dataset_id == 2:
            self.filename = "nav_data/10-09-2019_toe7200.csv"
            #self.filename = "nav_data/week2_v4_accCheck_toe7200.csv"
        elif dataset_id == 3:
            self.filename1 = "nav_data/week2_v4_accCheck_toe600.csv"
            self.filename2 = "nav_data/week2_v4_accCheck_toe3600.csv"
            self.filename3 = "nav_data/week2_v4_accCheck_toe7200.csv"
            
        else:
            self.filename = ""
            print("Wrong delay given to datahandler.")
            
        
        self.load_data2()
        #self.remove_sats()
        self.sample_step = 30
        self.sample_data(self.sample_step)
        self.init_split_indices()
        self.inject_error_type7(15)
        self.extract_features()
        self.split_tvt()
            
            
            
    def load_data(self):
        self.df_original = pd.read_csv(self.filename)
    
    def load_data2(self):
        self.ts = toeScaler()
        self.ref_toe_value = 3600
        
        self.df1 = pd.read_csv(self.filename1)
        self.df2 = pd.read_csv(self.filename2)
        self.df3 = pd.read_csv(self.filename3)
        
        self.df1 = self.clean_data2(self.df1, 1000)
        self.df2 = self.clean_data2(self.df2, 1000)
        self.df3 = self.clean_data2(self.df3, 1000)
        
        
        self.df1_scaled = self.ts.scale(self.df1, self.ref_toe_value)
        self.df2_scaled = self.ts.scale(self.df2, self.ref_toe_value)
        self.df3_scaled = self.ts.scale(self.df3, self.ref_toe_value)
        
        self.df_original = self.df1_scaled
        self.df_original = self.df_original.append(self.df2_scaled, ignore_index = True)
        self.df_original = self.df_original.append(self.df3_scaled, ignore_index = True)
        
# =============================================================================
#     def clean_data(self):
#         ind_to_remove = np.array([])
#         ind_to_remove = np.append(ind_to_remove, np.where(np.abs(self.df_original["sp_X"]) > 1000))
#         self.df_original = self.df_original.drop(ind_to_remove)
# =============================================================================
    
    def clean_data2(self, df, cutoff_sp_X):
        df_new = df.loc[np.abs(df.sp_X) < cutoff_sp_X]
        cutoff_clock = 3 * np.std(df_new["svCb"].values)
        df_new = df_new.loc[np.abs(df_new["svCb"]) < cutoff_clock]
        
        return df_new
         
    def remove_sats(self):
        #remove 2 and 9 based on abnormal clock bias. Not good for training
        sats_to_remove = np.array([9])
        ind_to_remove = np.array([])
        for s in sats_to_remove:
            ind_to_remove = np.append(ind_to_remove,np.where(self.df_original["svId"] == s,)[0])
        
        self.df_original = self.df_original.drop(ind_to_remove)
 
    def sample_data(self, step):
        ind = np.arange(0,self.df_original.shape[0],step)
        self.df_sampled = self.df_original.iloc[ind,:]
        
    def sample_data2(self, step, df):
        ind = np.arange(0,df.shape[0],step)
        df_sampled = df.iloc[ind,:]
        return df_sampled
        
        
    def init_split_indices(self):
        self.val_sat = 4
        self.test1_sat = 1
        self.test2_sat = 3
        
        self.val_indices = np.where(self.df_sampled["svId"] == self.val_sat,)[0]
        #self.val_indices = np.append(self.val_indices, np.where(self.df_sampled["svId"] == 5,)[0])
        self.test1_indices = np.where(self.df_sampled["svId"] == self.test1_sat,)[0]
        self.test2_indices = np.where(self.df_sampled["svId"] == self.test2_sat,)[0]
        train_indices = np.arange(self.df_sampled.shape[0])
        not_train_ind = np.concatenate((self.val_indices,self.test1_indices,self.test2_indices))
        self.train_indices = np.delete(train_indices,not_train_ind)

    def inject_error_type1(self, nb_errors):
        # Error in Y causing large difference sp_Y
        rand_ind = np.random.choice(self.test2_indices, nb_errors, replace=False)
        col = np.where(self.df_sampled.columns.values == "sp_Y")[0][0]
        self.df_sampled.iloc[rand_ind, col] = 650
        
    def inject_error_type2(self, nb_errors):
        # Error where all values reflect a perfect server prediction (all deltas = 0)
        rand_ind = np.random.choice(self.test2_indices, nb_errors, replace=False)
        self.df_sampled.iloc[rand_ind, :] *= 0.5
        
    def inject_error_type3(self, nb_errors):
         # Error where distance error is fully along Z axis
        rand_ind = np.random.choice(self.test2_indices, nb_errors, replace=False)
        col1 = np.where(self.df_sampled.columns.values == "sp_X")[0][0]
        col2 = np.where(self.df_sampled.columns.values == "sp_Y")[0][0]
        col3 = np.where(self.df_sampled.columns.values == "sp_Z")[0][0]
        self.df_sampled.iloc[rand_ind, col1] = 350
        self.df_sampled.iloc[rand_ind, col2] = 350
        self.df_sampled.iloc[rand_ind, col3] = 350
    
    def inject_error_type4(self, nb_errors):
        # Cumulative error (fast drifting sat in direction of velocity)
        colpX = np.where(self.df_sampled.columns.values == "sp_X")[0][0]
        colpY = np.where(self.df_sampled.columns.values == "sp_Y")[0][0]
        colpZ = np.where(self.df_sampled.columns.values == "sp_Z")[0][0]
        colvX = np.where(self.df_sampled.columns.values == "sv_X")[0][0]
        colvY = np.where(self.df_sampled.columns.values == "sv_Y")[0][0]
        colvZ = np.where(self.df_sampled.columns.values == "sv_Z")[0][0]
        start_point = self.test2_indices[0] + 200
        stop_point = self.test2_indices[0] + self.test2_indices.shape[0]
        pX = self.df_sampled.iloc[start_point, colpX]
        pY = self.df_sampled.iloc[start_point, colpY]
        pZ = self.df_sampled.iloc[start_point, colpZ]
        velX = self.df_sampled.iloc[start_point, colvX]
        velY = self.df_sampled.iloc[start_point, colvY]
        velZ = self.df_sampled.iloc[start_point, colvZ]
        vel_scale = 5.0
        spoofedX = np.repeat(velX*vel_scale,stop_point-start_point)
        spoofedX = np.cumsum(spoofedX) + pX
        spoofedY = np.repeat(velY*vel_scale,stop_point-start_point)
        spoofedY = np.cumsum(spoofedY) + pY
        spoofedZ = np.repeat(velZ*vel_scale,stop_point-start_point)
        spoofedZ = np.cumsum(spoofedZ) + pZ
        self.df_sampled.iloc[start_point:stop_point, colpX] = spoofedX
        self.df_sampled.iloc[start_point:stop_point, colpY] = spoofedY
        self.df_sampled.iloc[start_point:stop_point, colpZ] = spoofedZ
        self.df_sampled.iloc[start_point:stop_point, colvX] = velX
        self.df_sampled.iloc[start_point:stop_point, colvY] = velY
        self.df_sampled.iloc[start_point:stop_point, colvZ] = velZ       
        
    def inject_error_type5(self, nb_errors):
        #Error in clock bias
        rand_ind = np.random.choice(self.test2_indices, nb_errors, replace=False)
        col = np.where(self.df_sampled.columns.values == "svCb")[0][0]
        self.df_sampled.iloc[rand_ind, col] = -3e-9 
        
    def inject_error_type6(self, nb_errors):
        # Drifting clock bias
        colCb = np.where(self.df_sampled.columns.values == "svCb")[0][0]
        
        start_offset = 200
        start_point = self.test2_indices[0] + start_offset
        stop_point = self.test2_indices[0] + self.test2_indices.shape[0]
        initialCb = self.df_sampled.iloc[start_point, colCb]
        drift_vel = 4e-12
        spoofedCb = np.repeat(drift_vel,stop_point-start_point)
        spoofedCb = np.cumsum(spoofedCb) + initialCb
        self.df_sampled.iloc[start_point:stop_point, colCb] = spoofedCb
        
    def inject_error_type7(self, nb_errors):
        #error type 7
        false_toe = 3600
        
        df1_temp = self.df1.copy(deep=False)
        df2_temp = self.df2.copy(deep=False)
        df3_temp = self.df3.copy(deep=False)
        
        #df1_temp.toe = np.repeat(false_toe,df1_temp.toe.shape[0])
        #df2_temp.toe = np.repeat(false_toe,df2_temp.toe.shape[0])
        #df3_temp.toe = np.repeat(false_toe,df3_temp.toe.shape[0])
        
        #df1_temp = self.ts.scale(df1_temp, self.ref_toe_value)
        #df2_temp = self.ts.scale(df2_temp, self.ref_toe_value)
        #df3_temp = self.ts.scale(df3_temp, self.ref_toe_value)
        
        df_temp = df1_temp
        df_temp = df_temp.append(df2_temp, ignore_index = True)
        df_temp = df_temp.append(df3_temp, ignore_index = True)
        
        df_temp = self.sample_data2(self.sample_step, df_temp)
        
        self.df_sampled.loc[self.df_sampled.svId == self.test2_sat] = df_temp.loc[df_temp.svId == self.test2_sat]
        
        
        
    def extract_features(self):
        base_feat = ["sp_X", "sp_Y", "sp_Z", "sv_X", "sv_Y", "sv_Z", "svCb"] #"toe", 
        df_feat = self.df_sampled[base_feat]
        
        
        #add squared features
        df_feat = df_feat.assign(sp_X2 = np.square(df_feat["sp_X"].values))
        df_feat = df_feat.assign(sp_Y2 = np.square(df_feat["sp_Y"].values))
        df_feat = df_feat.assign(sp_Z2 = np.square(df_feat["sp_Z"].values))
        df_feat = df_feat.assign(sv_X2 = np.square(df_feat["sv_X"].values))
        df_feat = df_feat.assign(sv_Y2 = np.square(df_feat["sv_Y"].values))
        df_feat = df_feat.assign(sv_Z2 = np.square(df_feat["sv_Z"].values))
        df_feat = df_feat.assign(svCb2 = np.square(df_feat["svCb"].values))
        
        #add overall distance
        df_feat = df_feat.assign(sp_D = np.sqrt(np.square(df_feat["sp_X"].values) + np.square(df_feat["sp_Y"].values) + np.square(df_feat["sp_Z"].values)))
        
        #add velocity vector
        df_feat = df_feat.assign(sv_V = np.sqrt(np.square(df_feat["sv_X"].values) + np.square(df_feat["sv_Y"].values) + np.square(df_feat["sv_Z"].values)))
        
        #store dataframe before scaling
        self.df_feat_unscaled = df_feat.copy(deep=True)
        
        #Standardize data -> scikit learn
        self.scaler = MinMaxScaler(feature_range=(-0.8,0.8))
        self.scaler.fit(df_feat.iloc[self.train_indices, :])
        self.df_feat = pd.DataFrame(self.scaler.transform(df_feat), columns = df_feat.columns)
        
    
    def split_tvt(self):
        self.trainX = self.df_feat.values[self.train_indices, :]
        self.valX = self.df_feat.values[self.val_indices, :]
        self.test1X = self.df_feat.values[self.test1_indices, :]
        self.test2X = self.df_feat.values[self.test2_indices, :]
    
    def calculate_feature_weights(self):
        std_scaler = StandardScaler()
        std_scaler.fit(self.trainX)
        variance = std_scaler.var_
        variance_inverse = 1 / variance
        self.feature_weights = variance_inverse / sum(variance_inverse)
    
    def get_feature_weights(self):
        self.calculate_feature_weights()
        return self.feature_weights
    
    def get_df_original(self):
        return self.df_original
    
    def get_df_sampled(self):
        return self.df_sampled
    
    def get_df_feat_unscaled(self):
        return self.df_feat_unscaled
    
    def get_df_features(self):
        return self.df_feat
    
    def get_input_size(self):
        return self.df_feat.columns.values.shape[0]
    
    def get_tvt(self):
        return self.trainX, self.valX, self.test1X, self.test2X
    

        
        
        
        
        
