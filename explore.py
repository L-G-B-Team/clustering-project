import pandas as pd
import numpy as np
import wrangle as w

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

def generate_elbow(df:pd.DataFrame,k_min:int = 1,k_max:int = 30)->None:
    inertia = {i:KMeans(i).fit(df).inertia_ for i in range(k_min,k_max)}
    fig,axs = plt.subplots(1,2)
    sns.lineplot(data=inertia,ax=axs[0])
    axs[0].set_title('Inertia')
    axs[0].set_xlabel('No. of Clusters')
    axs[0].set_ylabel('Inertia')
    pct_change = [((inertia[i]-inertia[i+1])/inertia[i])*100 for i in range(k_min,k_max-1)]
    sns.lineplot(data=pct_change,ax=axs[1])
    axs[1].set_xlabel('No. of Clusters')
    axs[1].set_ylabel('% of Change')
    axs[1].set_title('% Change')
    fig.tight_layout()
    plt.show()