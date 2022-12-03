'''model contains helper functions to assist in Modeling portion of final_report.ipynb'''
from typing import Union, Tuple, Dict, List, Callable

import numpy as np
import pandas as pd
from IPython.display import Markdown as md
from sklearn.cluster import KMeans
from sklearn.linear_model import LassoLars, LinearRegression, TweedieRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from wrangle import scale
import evaluate as ev
from custom_dtypes import LinearRegressionType, ModelDataType


def select_baseline(ytrain: pd.Series) -> md:
    '''tests mean and median of training data as a baseline metric.
    ## Parameters
    ytrain: `pandas.Series` containing the target variable
    ## Returns
    Formatted `Markdown` with information on best-performing baseline.
    '''
    med_base = ytrain.median()
    mean_base = ytrain.mean()
    mean_eval = ev.regression_errors(ytrain, mean_base, 'Mean Baseline')
    med_eval = ev.regression_errors(ytrain, med_base, 'Median Baseline')
    ret_md = pd.concat([mean_eval, med_eval]).to_markdown()
    ret_md += '\n### Because mean outperformed median on all metrics, \
        we will use mean as our baseline'
    return md(ret_md)


def linear_regression(x: pd.DataFrame, y: pd.DataFrame,
                      linreg: Union[LinearRegression, None] = None) -> None:
    '''runs linear regression on x and y
    ## Parameters
    x: DataFrame of features

    y: DataFrame of target

    linreg: Optional `LinearRegression` object,
    used if model has already been trained, default: None
    ## Returns
    ypred: numpy.array of predictions

    linreg: linear regression model trained on data.
    '''
    if linreg is None:
        linreg = LinearRegression(normalize=True)
        linreg.fit(x, y)
    ypred = linreg.predict(x)
    return ypred, linreg


def lasso_lars(x: pd.DataFrame, y: pd.DataFrame,
               llars: Union[None, LassoLars] = None)\
        -> Tuple[np.array, LassoLars]:
    '''runs LASSO+LARS on x and y
    ## Parameters
    x: Dataframe of features

    y: DataFrame of target

    llars: Optional LASSO + LARS object, 
    used if model has already been trained, default: None
    ## Returns
    ypred: numpy.array of predictions

    linreg: `LassoLars` model trained on data.
    '''
    if llars is None:
        llars = LassoLars(alpha=3.0)
        llars.fit(x, y)
    ypred = llars.predict(x)
    return ypred, llars


def lgm(x: pd.DataFrame, y: pd.DataFrame,
        tweedie: Union[TweedieRegressor, None] = None)\
        -> Tuple[np.array, TweedieRegressor]:
    '''runs Generalized Linear Model (GLM) on x and y
    ## Parameters
    x: `DataFrame` of features

    y: 'DataFrame' of target

    tweedie: `TweedieRegressor` object, used if model has already been trained, default: None
    ## Returns
    ypred: numpy.array of predictions

    tweedie: `TweedieRegressor` model trained on data.
    '''
    if tweedie is None:
        tweedie = TweedieRegressor(power=0, alpha=3.0)
        tweedie.fit(x, y)
    ypred = tweedie.predict(x)
    return ypred, tweedie


def rmse_eval(ytrue: Dict[str, np.array], **kwargs) -> pd.DataFrame:
    '''
    performs Root Mean Squared evaluation on parameters
    ## Parameters
    ytrue: a dictionary of `numpy.array` containing the true Y values on which to evaluate
    kwargs: named dictionary of `numpy.array` objects
        with predicted y values where for each key:value
    pain in ytrue there is a corresponding key:value pair in `kwargs[REGRESSION FUNCTION NAME]`
    ## Returns
    a `pandas.DataFrame` of Root Mean Squared Evaluation for each dataset in kwargs.
    '''
    ret_df = pd.DataFrame()
    for key, value in kwargs.items():
        for k_key, v_value in value.items():
            ret_df.loc[key, k_key] = np.round(
                np.sqrt(mean_squared_error(ytrue[k_key], v_value)), 2)
    return ret_df


def scale_and_cluster(df: pd.DataFrame, features: List[str],
                      cluster_cols: List[str],
                      cluster_name: str, target: str,
                      scaler: Union[MinMaxScaler, None] = None,
                      kmeans: Union[KMeans, None] = None,
                      k: Union[int, None] = None) -> Tuple[pd.DataFrame,
                                                           MinMaxScaler,
                                                           KMeans]:
    # TODO Woody Docstring
    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(df[features])
    return_df = pd.DataFrame(scaler.transform(
        df[features]), columns=df[features].columns, index=df.index)
    if kmeans is None:
        if k is None:
            raise Exception("KMeans not provided, but k not specified")
        kmeans = KMeans(k, random_state=420).fit(df[cluster_cols])
    return_df[cluster_name] = kmeans.predict(df[cluster_cols])
    return_df[target] = df[target]
    return return_df, scaler, kmeans


def generate_regressor(df: pd.DataFrame, features: List[str],
                       target: str,
                       cluster_col: str,
                       regressor: Callable,
                       **kwargs) -> Dict[int, Callable]:
    # TODO Woody docstring
    return_dict = {}
    for cluster in np.unique(df[cluster_col]):
        x_train = df[df[cluster_col] == cluster][features]
        y_train = df[df[cluster_col] == cluster][[target]]
        regressor = regressor.fit(x_train, y_train)
        return_dict[cluster] = regressor

    return return_dict


def apply_to_clusters(df: pd.DataFrame, features: str, target: str,
                      cluster_col: str,
                      regressors: Dict[int, LinearRegressionType],
                      **kwargs) -> pd.DataFrame:
    # TODO Woody Docstring

    predictions_df = pd.DataFrame()
    predictions_df['y_true'] = df.log_error
    predictions_df['y_pred'] = 1.0
    cluster_group = df.groupby(cluster_col)
    for i, group in cluster_group:
        predictions_df.iloc[group.index,
                            1] = regressors[i].predict(group[features])

    return predictions_df


def process_model(df: pd.DataFrame, features: List[str], target: str, scaler: MinMaxScaler = MinMaxScaler(), kmeans: Union[KMeans, None] = None, k: Union[int, None] = None,
                  regressors: Union[List[LinearRegressionType], None] = None) -> Tuple[pd.DataFrame, MinMaxScaler, LinearRegressionType]:
    # TODO Woody Docstring
    pass
