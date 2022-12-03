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
    returns the result of a p test as a `IPython.display.Markdown`
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
    ret_str +=\
        f'## Because t {">" if t_flag else "<"} 0 and $\\alpha$ {">" if p_flag else "<"} p,' + \
        f'we {"failed to " if ~(t_flag & p_flag) else ""} reject $H_0$'
    return md(ret_str)


def anova_test(df: pd.DataFrame, col: str):
    '''
    TODO 
    ## Parameters

    ## Returns

    '''
    group_list = [df[df[col] == x].log_error.to_numpy() for x in range(4)]
    t, p = stats.kruskal(
        group_list[0], group_list[1], group_list[2], group_list[3])
    #cluster_3 = df[df.cluster3== 3]
    return e.t_to_md(t, p)

# CREATING CLUSTERS


def create_clusters_Q3(X_train, k, cluster_vars):
    '''This function uses pool_count, garage_car_count, and lot_sqft 
    to create a cluster for question 3'''

    k = 4
    cluster_vars = ['pool_count', 'garage_car_count', 'lot_sqft']
    cluster_name = 'X3'

    # create kmean object
    kmeans = KMeans(n_clusters=k, random_state=89)
    # fit to train and assign cluster ids to observations
    kmeans.fit(X_train[cluster_vars])

    return kmeans


def generate_elbow(df: pd.DataFrame, k_min: int = 1, k_max: int = 30) -> None:
    '''
    TODO Woody
    ## Parameters

    ## Returns

    '''
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


def viz_for_Q3(train_df: pd.DataFrame) -> None:
    '''
    TODO Naomi
    ## Parameters
    train_df: `DataFrame` containing training data.
    ## Returns
    None
    '''
    # unscaled data
    X3 = train_df[['garage_car_count', 'pool_count', 'lot_sqft']]
    kmeans = KMeans(n_clusters=4).fit(X3)
    train_df['cluster3'] = kmeans.predict(X3)

    # scaled data
    train_scaled3 = w.scale(
        train_df, ['garage_car_count', 'pool_count', 'lot_sqft'])
    X3_scaled = train_scaled3[[
        'scaled_garage_car_count', 'scaled_pool_count', 'scaled_lot_sqft']]
    kmeans = KMeans(n_clusters=4).fit(X3_scaled)
    train_scaled3['cluster3_scaled'] = kmeans.predict(X3_scaled)
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


def cluster_kde(df: pd.DataFrame, cluster: str, target: str) -> sns.FacetGrid:
    facet = sns.FacetGrid(data=df, col_wrap=4, col=cluster)
    facet = facet.map_dataframe(sns.boxplot, x=target)
    return facet


def cluster_fun(df: pd.DataFrame) -> pd.DataFrame:
    all_features = df[['bath_count', 'bed_count',
                       'calc_sqft', 'latitude', 'longitude', 'tax_value', 'fireplace_count']].columns
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
    TODO Woody
    ## Parameters

    ## Returns

    '''
    sns.set_palette('magma')
    fig, axs = plt.subplots(1, 2)
    df.log_error = df.log_error.astype('float')
    sns.scatterplot(data=df, x='calc_sqft', y='tax_value', hue='log_error', ax=axs[0], palette='magma').\
        set(yscale='log')
    calc = df[np.abs(df.log_error) >= 1]
    sns.scatterplot(data=calc, x='calc_sqft', y='tax_value', hue='log_error', ax=axs[1], palette='magma').\
        set(yscale='log')
    fig.suptitle('Tax Value vs. Calculated Sqft.')
    axs[0].set_title('All Data')
    axs[1].set_title('Abs. Val of Log Error >= 1')
    plt.show()


def tax_sqft_cluster_plot(train: pd.DataFrame) -> None:
    '''
    TODO Woody
    ## Parameters

    ## Returns

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
                      col_wrap=2, sharey=True, palette='mako').set(yscale='log')
    g.map_dataframe(sns.histplot, x='log_error')
    plt.show()
