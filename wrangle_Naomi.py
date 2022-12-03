from env import get_db_url
import wrangle as w
import explore as e

import pandas as pd
import numpy as np
import scipy.stats as stats

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm


def get_zillow_data():
    sql_query = '''
    SELECT *
    FROM predictions_2017
    LEFT JOIN properties_2017 USING (parcelid)
    LEFT JOIN airconditioningtype USING (airconditioningtypeid)
    LEFT JOIN architecturalstyletype USING (architecturalstyletypeid)
    LEFT JOIN buildingclasstype USING (buildingclasstypeid)
    LEFT JOIN heatingorsystemtype USING (heatingorsystemtypeid)
    LEFT JOIN propertylandusetype USING (propertylandusetypeid)
    LEFT JOIN storytype USING (storytypeid)
    LEFT JOIN typeconstructiontype USING (typeconstructiontypeid)
    WHERE YEAR(transactiondate) = 2017
    AND latitude IS NOT NULL
    AND longitude IS NOT NULL;
    '''
    df = pd.read_sql(sql_query, get_db_url('zillow'))
    df = df.drop(columns='id')
    dups = df[df.duplicated(subset='parcelid', keep='last')].index
    df.drop(dups, inplace=True)

    return df



def split_data(df):
    train_validate, test_df = train_test_split(df, test_size=.2,
                                               random_state=1989)
    train_df, validate_df = train_test_split(train_validate, test_size=.3,
                                             random_state=1989)
    return train_df, validate_df, test_df


def nulls_by_row(df):
    num_missing = df.isnull().sum(axis=1)
    prnt_miss = num_missing / df.shape[1] * 100
    rows_missing = pd.DataFrame(
        {'num_cols_missing': num_missing,
         'percent_cols_missing': prnt_miss})\
        .reset_index()\
        .groupby(['num_cols_missing', 'percent_cols_missing']).\
        count().reset_index().rename(
        columns={
            'customer_id': 'count'})
    return rows_missing


def nulls_by_col(df):
    num_missing = df.isnull().sum()
    percnt_miss = num_missing / df.shape[0] * 100
    cols_missing = pd.DataFrame(
        {
            'num_rows_missing': num_missing,
            'percent_rows_missing': percnt_miss
        })
    return cols_missing


def summarize(df):
    print('DataFrame head: \n')
    print(df.head())
    print('----------')
    print('DataFrame info: \n')
    print(df.info())
    print('----------')
    print('Dataframe Description: \n')
    print(df.describe())
    print('----------')
    print('Null value assessments: \n')
    print('nulls by column: ', nulls_by_col(df))
    print('--')
    print('nulls by row: ', nulls_by_row(df))
    numerical_cols = [col for col in df.columns if df[col].dtype != 'O']
    categorical_cols = [col for col in df.columns if col not in numerical_cols]
    print('--------')
    print('value_counts: \n')
    for col in df.columns:
        print('Column Name: ', col)
        if col in categorical_cols:
            print(df[col].value_counts())
        else:
            print(df[col].value_counts(bins=10, sort=False))
        print('--')
    print('---------')
    print('Report Finished')


def get_upper_outliers(s, k=1.5):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    upper_bound = q3 + k*iqr

    return s.apply(lambda x: max([x - upper_bound, 0]))

# USE AS IMPORTS


def viz_for_Q3(train_df):
    train = train_df

    # unscaled data
    X3 = train[['garage_car_count', 'pool_count', 'lot_sqft']]
    kmeans = KMeans(n_clusters=4, random_state=89).fit(X3)
    train['cluster3'] = kmeans.predict(X3)

    train_scale = train.copy()
    # scaled data
    train_scaled3 = w.scale(train_scale, ['garage_car_count', 'pool_count', 'lot_sqft'])
    X3_scaled = train_scaled3[['scaled_garage_car_count', 'scaled_pool_count', 'scaled_lot_sqft']]
    kmeans = KMeans(n_clusters=4, random_state=89).fit(X3_scaled)
    train_scaled3['cluster3_scaled'] = kmeans.predict(X3_scaled)
    train_scaled3['log_error'] = train['log_error']

    # viz
    fig, axes = plt.subplots(1, 2, figsize=(15, 10), sharey=True)
    fig.subtitle = ('Unscaled vs. Scaled')

    # Unscaled
    sns.stripplot(ax=axes[0], data=train, x='cluster3', y='log_error', cmap= 'flare')
    axes[0].set_title('Unscaled')

    # Scaled
    sns.stripplot(ax=axes[1], data=train_scaled3, x='cluster3_scaled', y='log_error', cmap = 'flare')
    axes[1].set_title('Scaled')

    plt.show()


def anova_test(df, col):
    group_list = [df[df[col] == x].log_error.to_numpy() for x in range(4)]
    t, p = stats.kruskal(
        group_list[0], group_list[1], group_list[2], group_list[3])
    return e.t_to_md(t, p)


def scaled_3(train_df):
    train_scaled3 = w.scale(train_df, ['garage_car_count', 'pool_count', 'lot_sqft'])
    X3_scaled = train_scaled3[['scaled_garage_car_count', 'scaled_pool_count', 'scaled_lot_sqft']]
    kmeans = KMeans(n_clusters=4, random_state=89).fit(X3_scaled)
    train_scaled3['cluster3_scaled'] = kmeans.predict(X3_scaled)
    train_scaled3['log_error'] = train_df['log_error']
    
    return train_scaled3







