import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Markdown as md
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, RobustScaler

import wrangle as w

warnings.filterwarnings('ignore')


def p_to_md(p: float, alpha: float = .05, **kwargs) -> md:
    '''
    returns the result of a p test as a `Markdown` object
    ## Parameters
    p: `float` of the p value from performed Hypothesis test
    alpha: `float` of alpha value for test, defaults to 0.05
    kwargs: any additional return values of statistical test
    ## Returns
    formatted `Markdown` object containing results of hypothesis test.

    '''
    ret_str = ''
    p_flag = p < alpha
    for k, v in kwargs.items():
        ret_str += f'## {k} = {v}\n\n'
    ret_str += f'## Because $\\alpha$ {">" if p_flag else "<"} p,' + \
        f'we {"failed to " if ~(p_flag) else ""} reject $H_0$'
    return md(ret_str)


def t_to_md(p: float, t: float, alpha: float = .05, **kwargs):
    '''takes a p-value, alpha, and any T-test arguments and
    creates a Markdown object with the information.
    ## Parameters
    p: float of the p value from run T-Test
    t: float of the t-value from run T-Test
    alpha: desired alpha value, defaults to 0.05
    ## Returns
    `IPython.display.Markdown` object with results of the statistical test
    '''
    ret_str = ''
    t_flag = t > 0
    p_flag = p < alpha
    ret_str += f'## t = {t} \n\n'
    for k, v in kwargs.items():
        ret_str += f'## {k} = {v}\n\n'
    ret_str += f' ## p = {p} \n\n'
    ret_str += (f'## Because t {">" if t_flag else "<"} 0 '
                f'and $\\alpha$ {">" if p_flag else "<"} p, '
                f'we {"failed to " if ~(t_flag & p_flag) else ""} '
                ' reject $H_0$')
    return md(ret_str)


def anova_test(df: pd.DataFrame, col: str):
    # TODO Naomi: change to new stats test and fill in docstring
    ## Naomi feedback, will use another stat test, but not delete this one. Docstring filled.
    '''
    Nicely displays the results of and runs anova stat test
    ## Parameters
    X-train dataframe containing 
    ## Returns

    '''
    group_list = [df[df[col] == x].log_error.to_numpy() for x in range(4)]
    t, p = stats.kruskal(
        group_list[0], group_list[1], group_list[2], group_list[3])
    # cluster_3 = df[df.cluster3== 3]
    return t_to_md(t, p)


def create_clusters_pool_garage_lot_sqft(x_train: pd.DataFrame, k: int):
    # TODO Naomi check this works and approve docstring
    # I took out cluster_vars from the parameters and renamed the function
    
    ### Naomi feedback, Does not return clusters as I know what to do with them. Returns n_clusters and random_state
    '''
    Marks K-Means on pool count, garage car count, and lot square feet
    ## Parameters
    x_train: `DataFrame` containing training data features
    k: `int` indicating number of centroids
    ## Returns
    Fitted `KMeans` object
    '''

    k = 4
    cluster_vars = ['pool_count', 'garage_car_count', 'lot_sqft']
    cluster_name = 'x3'

    # create kmean object
    kmeans = KMeans(n_clusters=k, random_state=89)
    # fit to train and assign cluster ids to observations
    kmeans.fit(x_train[cluster_vars])

    return kmeans


def generate_elbow(df: pd.DataFrame, k_min: int = 1, k_max: int = 30) -> None:
    '''
    Plots KMeans elbow of a given potential cluster as well as the
    percent change for the graph
    ## Parameters
    df: `DataFrame` containing features to perform KMeans clustering on
    k_min: `int` specifying minimum number of centroids
    k_max: `int` specifying maximum number of centroids
    ## Returns
    None (plots graph to Jupyter notebook)
    '''
    with plt.style.context('seaborn-whitegrid'):
        inertia = {i: KMeans(i, random_state=420).fit(
            df).inertia_ for i in range(k_min, k_max)}
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        sns.lineplot(data=inertia, ax=axs[0])
        axs[0].set_title('Inertia')
        axs[0].set_xlabel('No. of Clusters')
        axs[0].set_ylabel('Inertia')
        pct_change = [((inertia[i]-inertia[i+1])/inertia[i])
                    * 100 for i in range(k_min, k_max-1)]
        sns.lineplot(data=pct_change, ax=axs[1])
        axs[1].set_xlabel('No. of Clusters')
        axs[1].set_ylabel('% of Change')
        axs[1].set_title('% Change')
        fig.tight_layout()
        plt.show()


def elbow_for_Q3(train_scaled3):
    
    X3_scaled = train_scaled3[['scaled_garage_car_count', 'scaled_pool_count', 'scaled_lot_sqft']]
    
    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(9, 6))
        pd.Series({k: KMeans(k).fit(X3_scaled).inertia_ for k in range(2, 12)}).plot(marker='x')
        plt.xticks(range(2, 12))
        plt.xlabel('k')
        plt.ylabel('inertia')
        plt.title('Change in inertia as k increases')
        
        plt.show()
     
    
def viz_for_Q3(train_df: pd.DataFrame) -> None:
    # TODO Naomi double check the docstring I wrote for this
    
    ### Naomi feedback: checked and updated 
    '''
    Generates visualizations of both scaled and unscaled clusters
    of garage car count, pool count, and lot square feet
    ## Parameters
    train_df: `DataFrame` containing training data.
    ## Returns
    updated train_df with new column cluser3 and visusal showing
    difference between scaled and unscaled data
    '''
    # unscaled data
    x3_unscaled = train_df[['garage_car_count', 'pool_count', 'lot_sqft']]
    kmeans = KMeans(n_clusters=4).fit(x3_unscaled)
    train_df['cluster3'] = kmeans.predict(x3_unscaled)

    # scaled data
    train_scaled3 = w.scale(
        train_df, ['garage_car_count', 'pool_count', 'lot_sqft'])
    x3_scaled = train_scaled3[[
        'scaled_garage_car_count', 'scaled_pool_count', 'scaled_lot_sqft']]
    kmeans = KMeans(n_clusters=4).fit(x3_scaled)
    train_scaled3['cluster3_scaled'] = kmeans.predict(x3_scaled)
    train_scaled3['log_error'] = train_df['log_error']

    # viz
    fig, axes = plt.subplots(1, 2, figsize=(15, 10), sharey=True)
    fig.subtitle = ('Unscaled vs. Scaled')

    # Unscaled
    sns.stripplot(ax=axes[0], data=train_df, x='cluster3', y='log_error')
    axes[0].set_title('Unscaled')

    # Scaled
    sns.stripplot(ax=axes[1], data=train_scaled3,
                  x='cluster3_scaled', y='log_error')
    axes[1].set_title('Scaled')

    plt.show()


def cluster_creator(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Creates `DataFrame` of a given list of features pairwise as well as plots
    visualizations
    ## Parameters
    df: `DataFrame` containing features to cluster
    ## Returns
    `DataFrame` of clusters for data
    '''
    all_features = df[['bath_count', 'bed_count',
                       'calc_sqft', 'latitude',
                       'longitude', 'tax_value', 'fireplace_count']].columns
    cluster_test = df.copy()
    for feature in all_features:
        for feature2 in all_features:
            if feature != feature2:
                features_cluster = cluster_test[[feature, feature2]]

                # Fit a new model to my scaled data
                kmeans_scale = KMeans(n_clusters=4)

                kmeans_scale.fit(features_cluster)
                cluster_name = feature + '_' + feature2 + '_cluster'
                cluster_test[cluster_name] = kmeans_scale.predict(
                    features_cluster)
                plt.title(f'{feature} and {feature2} cluster')
                sns.scatterplot(y='log_error', x=cluster_name,
                                palette='colorblind', data=cluster_test)
                plt.show()
    return cluster_test


def tax_sqft_plot(df: pd.DataFrame) -> None:
    '''
    Helper function which plots the tax value vs. the calculated square
    footage of the zillow data set
    ## Parameters
    df: `DataFrame` containing training data
    ## Returns
    None (plots values to Jupyter notebook)
    '''
    sns.set_palette('magma')
    fig, axs = plt.subplots(1, 2,figsize=(12,5),sharex=True,sharey=True)
    df.log_error = df.log_error.astype('float')
    sns.scatterplot(data=df, x='calc_sqft', y='tax_value',
                    hue='log_error', ax=axs[0],
                    palette='magma')
    calc = df[np.abs(df.log_error) >= 1]
    sns.scatterplot(data=calc, x='calc_sqft', y='tax_value',
                    hue='log_error', ax=axs[1],
                    palette='magma')
    fig.suptitle('Tax Value vs. Calculated Sqft.')
    axs[0].set_title('All Data')
    axs[1].set_title('Abs. Val of Log Error >= 1')
    plt.show()


def tax_sqft_cluster_plot(train: pd.DataFrame) -> None:
    '''
    Generates Histogram visualizations of tax value/calculated
    square footage cluster.
    ## Parameters
    train: `DataFrame` containing training data as well as the
    `tax_sqft_cluster` group for each value
    ## Returns
    None (plots visualizations to Jupyter notebook)
    '''
    tax_sqft = train[['calc_sqft', 'tax_value']]
    tax_sqft['calc_sqft'] = train['calc_sqft']
    tax_sqft['tax_value'] = train['tax_value']
    kmeans = KMeans(5, random_state=420)
    kmeans.fit(tax_sqft)
    tax_sqft['tax_sqft_cluster'] = kmeans.predict(tax_sqft)
    tax_sqft['log_error'] = train.log_error
    sns.set_palette('magma')
    g = sns.FacetGrid(data=tax_sqft, col='tax_sqft_cluster',
                      col_wrap=3, sharey=True).set(yscale='log')
    g.map_dataframe(sns.histplot, x='log_error')
    plt.show()
