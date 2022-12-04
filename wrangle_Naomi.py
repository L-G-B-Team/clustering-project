import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from matplotlib import cm
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

import explore as e
import wrangle as w
from env import get_db_url


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


def nulls_by_row(df: pd.DataFrame) -> pd.DataFrame:
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


def nulls_by_col(df: pd.DataFrame) -> pd.DataFrame:
    num_missing = df.isnull().sum()
    percnt_miss = num_missing / df.shape[0] * 100
    cols_missing = pd.DataFrame(
        {
            'num_rows_missing': num_missing,
            'percent_rows_missing': percnt_miss
        })
    return cols_missing


def summarize(df: pd.DataFrame) -> None:
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


def get_upper_outliers(s: pd.DataFrame, k: float = 1.5):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    upper_bound = q3 + k*iqr

    return s.apply(lambda x: max([x - upper_bound, 0]))

# USE AS IMPORTS

# TODO Functions moved to explore.py. Delete these functions one by one
# and replace with e.Xxx(XXX)


def viz_for_Q3(train_df):
    return e.viz_for_Q3(train_df)


def anova_test(df, col):
    return e.anova_test(df, col)


def scaled_3(train_df):
    return e.scaled_3(train_df)
