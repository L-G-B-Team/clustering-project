'''evaluate contains helper functions to assist in evaluation of models'''
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error

from custom_dtypes import ModelDataType,lmplot_kwargs


def get_residuals(y_true:ModelDataType,y_pred:Union[ModelDataType,float])->pd.DataFrame:
    '''
    gets the residual and residual squared values for predicted y values
    ## Parameters
    y_true: `DataType` containing true y values
    y_pred: either a `DataType` or `float` with predicted y values
    ## Returns
    `pandas.DataFrame` containing true y_true, the residual values, and the residual squared values
    '''
    ret_frame = pd.DataFrame()
    ret_frame['actual'] = y_true
    ret_frame['residual'] = y_true - y_pred
    ret_frame['residual_squared'] = ret_frame.residual ** 2
    return ret_frame
def plot_residuals(y_true:pd.Series,y_pred:pd.Series)->None:
    '''
    plots the residuals of y_pred vs. y_true
    ## Parameters
    y_true: `DataType` of true y values
    y_pred: `DataType` of predicted y values
    ## Returns
    None
    '''
    res = get_residuals(y_true,y_pred)
    sns.scatterplot(data=res,x='actual',y='residual',color=lmplot_kwargs['scatter']['color'])
    plt.axhline(0,color=lmplot_kwargs['line']['color'])
    plt.show()
def sum_of_squared_errors(y_true:ModelDataType,y_pred:Union[ModelDataType,float])->float:
    '''
    returns the Sum of Squared Errors for predicted y values
    ## Parameters
    y_true: `DataType` of true y values.
    y_pred: either a `DataType` or `float` (in case evaluating baseline) of predicted y
    ## Returns
    Sum of Squared Errors for input datasets
    '''
    ret_frame = get_residuals(y_true,y_pred)
    sse = np.sum(ret_frame.residual_squared)
    return sse
def explained_sum_of_sqrd(y_true:ModelDataType,y_pred:ModelDataType)->float:
    '''
    returns Explained Sum of Squared Errors for predicted y values
    ## Parameters
    y_true: `DataType` containing true y values
    y_pred: `DataType` of predicted y values
    ## Returns
    a `float` representing the Explained Sum of Squared Errors in y_pred
    '''
    return np.sum((y_pred - y_true.mean())**2)
def total_sum_of_squares(y_true:ModelDataType,y_pred:ModelDataType)->float:
    '''
    returns the Total Sum of Squares error for predicted y values
    ## Parameters
    ytrue: `DataType` containing true y values
    ypred: `DataType` of predicted y values
    ## Returns
    `float` representing the Total Sum of Squares in y_pred
    '''
    return explained_sum_of_sqrd(y_true,y_pred) + sum_of_squared_errors(y_true,y_pred)
def regression_errors(y_true:ModelDataType,y_pred:Union[ModelDataType,float],title:str)\
    ->pd.DataFrame:
    '''
    performs Sum of Squared Errors (SSE), Explained Sum of Squares (ESS),
    Total Sum of Squares (TSS), Mean Squared Error and Root Mean Squared Error (MSE)
    on predicted y values.
    ## Parameters
    y_true: `DataType` of true values for y
    y_pred: Either `DataType` or `float` (in case of evaluating baseline) of predicted y values
    ## Returns
    a formatted `pandas.DataFrame` containing the SSE, ESS (if not evaluating basleine),
    TSS (if not evaluating baseline), MSE, and RMSE values for y_pred.

    '''
    ret_dict = {}
    ret_dict['SSE'] = sum_of_squared_errors(y_true,y_pred)
    if not isinstance(y_pred,float):
        ret_dict['ESS'] = explained_sum_of_sqrd(y_true,y_pred)
        ret_dict['TSS'] = ret_dict['SSE'] + ret_dict['ESS']
        ret_dict['MSE'] = mean_squared_error(y_true,y_pred)
    else:
        ret_dict['MSE'] = mean_squared_error(y_true,[y_pred for i in range(y_true.count())])
    ret_dict['RMSE']= np.sqrt(ret_dict['MSE'])
    ret_frame =  pd.DataFrame(ret_dict,index=[title])
    return ret_frame
