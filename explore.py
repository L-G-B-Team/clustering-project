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
def viz_for_Q3(train_df):
    
    #unscaled data
    X3 = train_df[['garage_car_count', 'pool_count', 'lot_sqft']]
    kmeans = KMeans(n_clusters = 4).fit(X3)
    train_df['cluster3'] = kmeans.predict(X3)
    
    #scaled data
    train_scaled3 = w.scale(train_df, ['garage_car_count', 'pool_count', 'lot_sqft'])
    X3_scaled = train_scaled3[['scaled_garage_car_count', 'scaled_pool_count', 'scaled_lot_sqft']]
    kmeans = KMeans(n_clusters = 4).fit(X3_scaled)
    train_scaled3['cluster3_scaled'] = kmeans.predict(X3_scaled)
    train_scaled3['log_error'] = train_df['log_error']
    
    #viz 
    fig, axes= plt.subplots(1,2, figsize =(15, 10), sharey = True)
    fig.subtitle = ('Unscaled vs. Scaled')

    #Unscaled
    sns.stripplot(ax=axes[0], data = train_df, x = 'cluster3', y = 'log_error')
    axes[0].set_title('Unscaled')

    #Scaled
    sns.stripplot(ax = axes[1],data = train_scaled3, x = 'cluster3_scaled', y = 'log_error')
    axes[1].set_title('Scaled')
    
    plt.show()

def cluster_kde(df:pd.DataFrame,cluster:str,target:str)->sns.FacetGrid:
    facet = sns.FacetGrid(data=df,col_wrap=4,col=cluster)
    facet = facet.map_dataframe(sns.boxplot,x=target)
    return facet