'''wrangle contains helper functions to assist in data acquisition and preparation
in zillow.ipynb'''
import os
import re
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from env import get_db_url

rename_dict = {
    'parcelid':'parcel_id', 'basementsqft':'basement_sqft',
    'bathroomcnt':'bath_count', 'bedroomcnt':'bed_count',
       'calculatedbathnbr':'calc_bath_and_bed','finishedfloor1squarefeet':'finished_floor1_sqft',
       'calculatedfinishedsquarefeet':'calc_sqft', 'finishedsquarefeet12':'finished_sqft12',
       'finishedsquarefeet13':'finished_sqft13', 'finishedsquarefeet15':'finished_sqft15',
        'finishedsquarefeet50':'finished_sqft50',
       'finishedsquarefeet6':'finished_sqft6', 'fireplacecnt':'fireplace_count',
        'fullbathcnt':'full_bath_count',
       'garagecarcnt':'garage_car_count', 'garagetotalsqft':'garage_sqft',
       'hashottuborspa':'has_hot_tub',
        'lotsizesquarefeet':'lot_sqft', 'poolcnt':'pool_count', 'poolsizesum':'sum_pool_size',
        'propertycountylandusecode':'property_county_use_code',
        'propertyzoningdesc':'property_zoning_desc',
       'rawcensustractandblock':'raw_census_tract_block', 'regionidcity':'region_id_city',
        'regionidcounty':'region_id_county',
       'regionidneighborhood':'region_id_neighbor', 'regionidzip':'region_id_zip',
        'roomcnt':'room_count', 'threequarterbathnbr':'three_quarter_bath',
       'unitcnt':'unit_count', 'yardbuildingsqft17':'yard_building_sqft17',
        'yardbuildingsqft26':'yard_building_sqft26', 'yearbuilt':'year_built',
       'numberofstories':'no_stories', 'fireplaceflag':'fireplace_flag',
        'structuretaxvaluedollarcnt':'structure_tax_value',
       'taxvaluedollarcnt':'tax_value', 'assessmentyear':'assessment_year',
        'landtaxvaluedollarcnt':'land_value',
       'taxamount':'tax_amount', 'taxdelinquencyflag':'tax_delinquency_flag',
       'taxdelinquencyyear':'tax_delinquency_year',
       'censustractandblock':'census_tract_and_block', 'logerror':'log_error',
       'transactiondate':'transaction_date',
       'airconditioningdesc':'air_conditioning_desc',
       'architecturalstyledesc':'architectural_style_desc',
       'buildingclassdesc':'building_class_desc',
       'heatingorsystemdesc':'heating_system_desc', 'propertylandusedesc':'property_land_use_desc',
        'storydesc':'story_desc',
       'typeconstructiondesc':'type_construction_desc'
}
def get_zillow_from_sql() -> pd.DataFrame:
    '''
    reads MySQL data from `zillow` database and returns `pandas.DataFrame` with raw data from query
    # Parameters
    None
    # Returns
    parsed DataFrame containing raw data from `zillow` database.
    '''
    query = '''
        SELECT *
    FROM properties_2017
    JOIN predictions_2017 USING(`parcelid`)
    LEFT JOIN airconditioningtype USING(airconditioningtypeid)
    LEFT JOIN architecturalstyletype USING(architecturalstyletypeid)
    LEFT JOIN buildingclasstype USING(buildingclasstypeid)
    LEFT JOIN heatingorsystemtype USING(heatingorsystemtypeid)
    LEFT JOIN propertylandusetype USING(propertylandusetypeid)
    LEFT JOIN storytype USING(storytypeid)
    LEFT JOIN typeconstructiontype USING(typeconstructiontypeid)
    WHERE transactiondate < "2018"
    AND latitude IS NOT NULL
    AND longitude IS NOT NULL;'''
    return pd.read_sql(query, get_db_url('zillow'))


def df_from_csv(path: str) -> Union[pd.DataFrame, None]:
    '''
    returns zillow DataFrame from .csv if it exists at `path`, otherwise returns None
    # Parameters
    path: string with path to .csv file
    # Returns
    `pd.DataFrame` if file exists at `path`, otherwise returns `None`.
    '''
    if os.path.exists(path):
        return pd.read_csv(path,low_memory=False)

    return None


def wrangle_zillow(from_sql: bool = False, from_csv: bool = False,\
    prop_row:float = .75, prop_col:float = .5,
    outlier_k:float=1.5,bound:float = 0.5) -> pd.DataFrame:
    '''
    wrangles Zillow data from either a MySQL query or a `.csv` file, prepares the  (if necessary)\
        , and returns a `pandas.DataFrame` object
    containing the prepared Zillow data. If data is acquired from MySQL,
     return `DataFrame` is also encoded to `.csv` in both prepared and
    unprepared states
    ## Parameters
    refresh: if `True`, ignores any `.csv` files and pulls new data from the SQL database,
    default=False.
    ## Return
    parsed and prepared `pandas.DataFrame` with Zillow data from 2017.
    '''
    # aquire Zillow data from .csv if exists
    ret_df = None
    if not from_sql and not from_csv:
        ret_df = df_from_csv('data/prepared_zillow.csv')
        if ret_df is not None:
            return ret_df
    if not from_sql:
        ret_df = df_from_csv('data/zillow.csv')
    if ret_df is None:
        # acquire zillow data from MySQL and caches to data/zillow.csv∏
        ret_df = get_zillow_from_sql()
        ret_df.to_csv('data/zillow.csv', index_label=False)

    return prep_zillow(ret_df,prop_row=prop_row,prop_col=prop_col,outlier_k=outlier_k,bound=bound)

def prep_zillow(df:pd.DataFrame,prop_row:float, prop_col:float,\
    outlier_k:float,bound:float)->pd.DataFrame:
    '''
    prepares `DataFrame` for processing
    ## Parameters
    df: `pandas.DataFrame` with unfiltered values
    prop_row: `float` to define `pct_row` passed to `handle_null_rows`
    prop_col: `float` to define `pct_col` passed to `handle_null_cols`
    outlier_k: `float` defining outer bound, passed to `mark_outliers`
    bound: `float` defining bound, passed to `mark_bounds`
    ## Returns
    formatted and prepared `pandas.DataFrame`
    '''
    df = df.dropna(subset='logerror')
    df = df.sort_values(by='transactiondate')
    df = df.drop_duplicates(subset=['parcelid'],keep='last')
    cols_to_remove = ['id','id.1','propertycountylandusecode', 'propertyzoningdesc',]
    for c in df.columns:
        if re.match('.*typeid',c) is not None:
            cols_to_remove.append(str(c))
    df = df.drop(columns=cols_to_remove)
    df = df.rename(columns=rename_dict)
    df.fips = df.fips.astype('category')
    df.transaction_date = pd.to_datetime(df.transaction_date)
    df.property_land_use_desc = df.property_land_use_desc.astype('category')
    df.property_land_use_desc = df.property_land_use_desc.cat.remove_categories(\
        ['Duplex (2 Units, Any Combination)','Quadruplex (4 Units, Any Combination)',\
        'Triplex (3 Units, Any Combination)','Commercial/Office/Residential Mixed Used']).dropna()
    df.transaction_date = df.transaction_date.replace('-','').astype(int)
    df.pool_count = df.pool_count.fillna(0)
    df.fireplace_count = df.fireplace_count.fillna(0)
    df.garage_car_count = df.garage_car_count.fillna(0)
    df.lot_sqft = df.lot_sqft.fillna(0)
    df.calc_sqft = df.calc_sqft.fillna(0)
    df.tax_value = df.tax_value.fillna(0)
    na_bath = df[df.calc_bath_and_bed.isna()]
    df.loc[df.calc_bath_and_bed.isna(),'calc_bath_and_bed'] = na_bath.bed_count + na_bath.bath_count
    df = handle_missing_values(df,prop_row,prop_col).reset_index(drop=True)
    df = df.dropna(axis=1)
    return df
def handle_null_cols(df:pd.DataFrame,pct_col:float)-> pd.DataFrame:
    '''removes columns that do not have at least `pct_col` non-null values
    ## Parameters:
    df: `DataFrame` of data with nulls to be pruned
    pct_col: Float `0 <= pct_col <=1` indicating what percentage of non-null values
    required to keep column
    ## Returns
    pruned `DataFrame`
    '''
    pct_col = 1-pct_col
    na_sums = pd.DataFrame(df.isna().sum())
    na_sums = na_sums.reset_index().rename(columns={0:'n_nulls'})
    na_sums['percentage'] =  na_sums.n_nulls / df.shape[0]
    ret_indices = na_sums[na_sums.percentage <= pct_col]['index'].to_list()
    return df[ret_indices]

def handle_null_rows(df:pd.DataFrame, pct_row:float)->pd.DataFrame:
    '''removes rows that do not have at least `pct_row` non-null values
    ## Parameters:
    df: `DataFrame` of data with nulls to be pruned
    pct_row: Float `0 <= pct_row <=1` indicating what percentage of non-null values
    required to keep row
    ## Returns
    pruned `DataFrame`
    '''
    pct_row = 1-pct_row
    return df[df.isna().sum(axis=1)/df.shape[1] <= pct_row]


def handle_missing_values(df:pd.DataFrame, pct_row:float,pct_col:float)->pd.DataFrame:
    '''
    runs `handle_null_cols` and `handle_null_rows` (in that order) on `df`
    ## Parameters
    df: `DataFrame` containing null values to be pruned
    pct_row: `float` to be passed to `handle_null_rows`
    pct_cols: `float` to be passed to `handle_null_cols
    ## Returns
    `DataFrame` pruned of rows and columns with excessive null values
    '''
    df = handle_null_cols(df,pct_col)
    return handle_null_rows(df,pct_row)

def mark_outliers(df:pd.DataFrame,s:str,k:float)->pd.DataFrame:
    '''
    ## Parameters
    df: `DataFrame` containing values to mark as outliers
    s: `string` where `s in df.columns` indicating which column to operate on
    k: `float` indicating how many Standard Deviations outside of the IQR a value must be
    to be marked as an outlier
    ## Returns
    `DataFrame` with outlier status marked as `s + '_outlier'`
    '''
    q1,q3 = df[s].quantile([.25,.75])
    iqr = q3-q1
    lower = (q1 - k * iqr)
    upper = (q3 + k * iqr)
    normals = df[(df[s] >= lower) & (df[s] <=upper)]
    df[s+'_outliers'] = ''
    df.loc[normals.index,'outliers'] = 'in_range'
    df.loc[df[s]<lower,'outliers'] = 'lower'
    df.loc[df[s]>upper,'outliers'] = 'upper'
    return df
def mark_bounds(df:pd.DataFrame, bound:float)->pd.DataFrame:
    '''
    adds a marker whether df objecs are below, above, or inside of the bound
    ## Parameters
    df: `DataFrame` containing zillow data
    bound: `float` indicating the lower and upper bound
    ## Returns
    `DataFrame` with values marked appropriately
    '''
    df['bound_group'] = ''
    df.loc[df.log_error < (0-bound),'bound_group'] = 'below'
    df.loc[df.log_error > bound,'bound_group'] = 'above'
    df.loc[df.bound_group == '','bound_group'] = 'in'
    df.bound_group = df.bound_group.astype('category')
    return df
def tvt_split(dframe: pd.DataFrame, stratify: Union[str, None] = None,
              tv_split: float = .2, validate_split: float = .3, \
                sample: Union[float, None] = None) -> \
                Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''tvt_split takes a pandas DataFrame, a string specifying the variable to stratify over,
    as well as 2 floats where 0< f < 1 and
    returns a train, validate, and test split of the DataFame,
    split by tv_split initially and validate_split thereafter. '''
    if stratify is not None:
        stratify = dframe[stratify]
    train_validate, test = train_test_split(
        dframe, test_size=tv_split, random_state=123, stratify=stratify)
    train, validate = train_test_split(
        train_validate, test_size=validate_split, random_state=123, stratify=stratify)
    if sample is not None:
        train = train.sample(frac=sample)
        validate = validate.sample(frac=sample)
        test = test.sample(frac=sample)

    return train, validate, test

def get_scaled_copy(dframe: pd.DataFrame, x: List[str], scaled_data: np.ndarray) -> pd.DataFrame:
    '''copies `df` and returns a DataFrame with `scaled_data`
    ## Parameters
    df: `DataFrame` to be copied and scaled
    x: features in `df` to be scaled
    scaled_data: `np.ndarray` with scaled values
    ## Returns
    a copy of `df` with features replaced with `scaled_data`
    '''
    ret_df = dframe.copy()
    ret_df.loc[:, x] = scaled_data
    return ret_df
def scale(df: pd.DataFrame,x:List[str])->pd.DataFrame:
    '''
    scales `df[[x]]` using a MinMaxScaler
    ## Parameters
    df: `Dataframe` of values to be scaled
    x: `List[str]` of x values to be scaled
    ## Returns
    `DataFrame` consisting of scaled values, labeled as `scaled_ + x`
    '''
    scaler = MinMaxScaler()
    scaler.fit(df[x])
    ret_x = ['scaled_' + s for s in x]
    return pd.DataFrame(scaler.transform(df[x]),columns=ret_x, index=df.index)

def scale_data(train: pd.DataFrame, validate: pd.DataFrame, test: pd.DataFrame,
               x: List[str]) ->\
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    scales `train`,`validate`, and `test` data using a `method`
    ## Parameters
    train: `pandas.DataFrame` of training data
    validate: `pandas.DataFrame` of validate data
    test: `pandas.DataFrame` of test data
    x: list of str representing feature columns in the data
    method: `callable` of scaling function (defaults to `sklearn.preprocessing.MinMaxScaler`)
    ## Returns
    a tuple of scaled copies of train, validate, and test.
    '''
    xtrain = train[x]
    xvalid = validate[x]
    xtest = test[x]
    scaler = MinMaxScaler()
    scaler.fit(xtrain)
    scale_train = scaler.transform(xtrain)
    scale_valid = scaler.transform(xvalid)
    scale_test = scaler.transform(xtest)
    ret_train = get_scaled_copy(train, x, scale_train)
    ret_valid = get_scaled_copy(validate, x, scale_valid)
    ret_test = get_scaled_copy(test, x, scale_test)
    return ret_train, ret_valid, ret_test


