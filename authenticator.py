# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 21:12:55 2019

@author: serru
"""

import numpy as np
import matplotlib.pyplot as plt
from satellite_nav_generator import satellite_nav_generator

two_hours = 2*3600
nb_sats = 10

#initial sat
sat = satellite_nav_generator(two_hours)
df = sat.data

for s in range(nb_sats-1):
    sat = satellite_nav_generator(two_hours)
    df = df.append(sat.data, ignore_index = True)
    
plt.figure(figsize=(12,6))
plt.plot(df.dX)
plt.plot(df.dY)
plt.plot(df.dZ)
plt.plot(df.dT)
plt.show()

#Standardize data -> scikit learn

#train autoencoder (also provide time in df as feature!)



