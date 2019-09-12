# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 20:30:47 2019

@author: Hendrik Serruys
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



class satellite_nav_generator:
    def __init__(self, duration_in_sec):
        self.time = self.get_time(duration_in_sec)
        self.dX = self.generate_series(duration_in_sec)
        self.dY = self.generate_series(duration_in_sec)
        self.dZ = self.generate_series(duration_in_sec)
        self.dT = self.generate_series(duration_in_sec)
        #combine into table
        table = {'time': self.time, 
                 'dX': self.dX, 
                 'dY': self.dY, 
                 'dZ': self.dZ, 
                 'dT': self.dT}
        self.data = pd.DataFrame(table)
        
    def get_time(self, duration_in_sec):
        return np.array(range(duration_in_sec))
        
    def generate_series(self, duration_in_sec):
        #a in [-0.001, 0.001] and b in [-0.0000001, 0.0000001]
        #note that we should use absolute values (distances)
        a = np.random.uniform(-0.001, 0.001)
        b = np.random.uniform(-0.0000001, 0.0000001)
        series = np.array([a*i + b*(i**2) for i in range(duration_in_sec)])
        #add noise
        series += np.random.normal(0,0.05,duration_in_sec)
        #convert to absolute values
        series = np.abs(series)
        
        
        #plt.plot(series)
        #plt.show()
        return series
        



