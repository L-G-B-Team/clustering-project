'''model contains helper functions to assist in Modeling portion of final_report.ipynb'''
from typing import Union,Tuple,Dict

import numpy as np
import pandas as pd
from IPython.display import Markdown as md
from sklearn.linear_model import LassoLars, LinearRegression, TweedieRegressor
from sklearn.metrics import mean_squared_error

import evaluate as ev
from custom_dtypes import LinearRegressionType, ModelDataType


def select_baseline(ytrain:pd.Series)->md:
    '''tests mean and median of training data as a baseline metric.
    ## Parameters
    ytrain: `pandas.Series` containing the target variable
    ## Returns
    Formatted `Markdown` with information on best-performing baseline.
    '''
    med_base = ytrain.median()
    mean_base = ytrain.mean()
    mean_eval = ev.regression_errors(ytrain,mean_base,'Mean Baseline')
    med_eval = ev.regression_errors(ytrain,med_base,'Median Baseline')
    ret_md = pd.concat([mean_eval,med_eval]).to_markdown()
    ret_md += '\n### Because mean outperformed median on all metrics, \
        we will use mean as our baseline'
    return md(ret_md)
def linear_regression(x:pd.DataFrame,y:pd.DataFrame,\
    linreg:Union[LinearRegression,None]=None)->None:
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
        linreg.fit(x,y)
    ypred = linreg.predict(x)
    return ypred,linreg
def lasso_lars(x:pd.DataFrame,y:pd.DataFrame,llars:Union[None,LassoLars] = None)\
    ->Tuple[np.array,LassoLars]:
    '''runs LASSO+LARS on x and y
    ## Parameters
    x: Dataframe of features

    y: DataFrame of target

    llars: Optional LASSO + LARS object, used if model has already been trained, default: None
    ## Returns
    ypred: numpy.array of predictions

    linreg: `LassoLars` model trained on data.
    '''
    if llars is None:
        llars = LassoLars(alpha=3.0)
        llars.fit(x,y)
    ypred = llars.predict(x)
    return ypred,llars
def lgm(x:pd.DataFrame,y:pd.DataFrame, tweedie:Union[TweedieRegressor,None] = None)\
    ->Tuple[np.array,TweedieRegressor]:
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
        tweedie = TweedieRegressor(power=0,alpha=3.0)
        tweedie.fit(x,y)
    ypred = tweedie.predict(x)
    return ypred, tweedie
def rmse_eval(ytrue:Dict[str,np.array],**kwargs)->pd.DataFrame:
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
    for key,value in kwargs.items():
        for k_key,v_value in value.items():
            ret_df.loc[key,k_key] = np.round(np.sqrt(mean_squared_error(ytrue[k_key],v_value)),2)
    return ret_df
def evaluate_models(xtrain:ModelDataType,ytrain:ModelDataType,\
    xvalid:ModelDataType,yvalid:ModelDataType)->pd.DataFrame:
    '''
    Performs LASSO+LARS, Linear Regression, and Generalized Linear Model regression
        on train and validate data sets provided
    ## Parameters
    xtrain: `pandas.DataFrame` containing x values from training dataset
        on which to perform regression fit and prediction
    ytrain: `pandas.DataFrame` containing y values on which to perform regression prediction
    xvalid: `pandas.DataFrame` containing x values from validate data set
        on which to perform regression predictions
    yvalid: `pandas.DataFrame` containing y values from
        validate data set on which to perform regression predictions
    ## Returns
    `IPython.display.Markdown` object containing table of Reverse Mean Squared Error
        evaluation of train and validate.
    data sets with each of the models.

    `ModelType` which performed the best against both data sets (LASSO+LARS)
    '''
    linreg_pred = dict()
    llars_pred = dict()
    glm_pred = dict()
    linreg_pred['train'], linreg = linear_regression(xtrain,ytrain)
    linreg_pred['validate'], _ = linear_regression(xvalid,yvalid,linreg)
    llars_pred['train'],llars = lasso_lars(xtrain,ytrain)
    llars_pred['validate'],_ = lasso_lars(xvalid,yvalid,llars)
    glm_pred['train'], tweedie = lgm(xtrain,ytrain)
    glm_pred['validate'],_ = lgm(xvalid,yvalid,tweedie)
    ytrue = {'train':ytrain,'validate':yvalid}
    baseline_pred = dict()
    baseline_pred['train'] = np.full_like(\
        np.arange(xtrain.shape[0],dtype=int),ytrain.tax_value.mean())
    baseline_pred['validate'] = np.full_like(\
        np.arange(xvalid.shape[0],dtype=int),ytrain.tax_value.mean())
    evaluation_matrix = rmse_eval(ytrue,baseline=baseline_pred,linear_regression=linreg_pred,\
        lasso_lars=llars_pred,glm=glm_pred)
    evaluation_matrix.index= ['Baseline','Linear Regression','LASSO LARS','General Linear Model']
    evaluation_matrix.columns = ['Train RMSE','Validate RMSE']
    return md('| Methodology' + evaluation_matrix.to_markdown()[1:]),llars
def run_test(model:LinearRegressionType,xtest:ModelDataType,ytest:ModelDataType)->md:
    '''
    Runs best performing regression model on test data set.
    ## Parameters
    model: `ModelType` of best performing model on train and validate data sets
    xtest: Features of test data set

    ytest: Target of test data set
    ## Returns
    displays a chart of residual plot of test results

    `IPython.display.Markdown` object containing the Root Mean Squared Error
        of predictions on test data set.
    '''
    ypred = model.predict(xtest)
    ev.plot_residuals(ytest.tax_value,ypred)
    return md('### Mean Squared Error: ' + \
        str(np.round(np.sqrt(mean_squared_error(ytest,ypred)),2)))
        