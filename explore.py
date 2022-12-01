import pandas as pd
import numpy as np

import env

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')


#################### CREATING CLUSTERS
def create_clusters_Q3(X_train, k, cluster_vars):
    '''This function uses pool_count, garage_car_count, and lot_sqft 
    to create a cluster for question 3'''
    
    k=4
    cluster_vars = ['pool_count', 'garage_car_count', 'lot_sqft']
    cluster_name = 'X3'
    
    # create kmean object
    kmeans = KMeans(n_clusters=k, random_state = 89)
    # fit to train and assign cluster ids to observations
    kmeans.fit(X_train[cluster_vars])

    return kmeans


