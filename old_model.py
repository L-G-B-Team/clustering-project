from sklearn.metrics import mean_squared_error
from custom_dtypes import LinearRegressionType, ModelDataType
import pandas as pd
import numpy as np
import model as m
import evaluate as ev
from IPython.display import Markdown as md


def evaluate_models(xtrain: ModelDataType, ytrain: ModelDataType,
                    xvalid: ModelDataType,
                    yvalid: ModelDataType) -> pd.DataFrame:
    '''
    Performs LASSO+LARS, Linear Regression, and
    Generalized Linear Model regression
    on train and validate data sets provided
    ## Parameters
    xtrain: `pandas.DataFrame` containing x values from training dataset
        on which to perform regression fit and prediction
    ytrain: `pandas.DataFrame` containing y values
    on which to perform regression prediction
    xvalid: `pandas.DataFrame` containing x values from validate data set
        on which to perform regression predictions
    yvalid: `pandas.DataFrame` containing y values from
        validate data set on which to perform regression predictions
    ## Returns
    `Markdown` object containing table ofReverse Mean Squared Error
        evaluation of train and validate.
    data sets with each of the models.

    `ModelType` which performed the best against both data sets (LASSO+LARS)
    '''
    linreg_pred = dict()
    llars_pred = dict()
    glm_pred = dict()
    linreg_pred['train'], linreg = m.linear_regression(xtrain, ytrain)
    linreg_pred['validate'], _ = m.linear_regression(xvalid, yvalid, linreg)
    llars_pred['train'], llars = m.lasso_lars(xtrain, ytrain)
    llars_pred['validate'], _ = m.lasso_lars(xvalid, yvalid, llars)
    glm_pred['train'], tweedie = m.lgm(xtrain, ytrain)
    glm_pred['validate'], _ = m.lgm(xvalid, yvalid, tweedie)
    ytrue = {'train': ytrain, 'validate': yvalid}
    baseline_pred = dict()
    baseline_pred['train'] = np.full_like(
        np.arange(xtrain.shape[0], dtype=int), ytrain.tax_value.mean())
    baseline_pred['validate'] = np.full_like(
        np.arange(xvalid.shape[0], dtype=int), ytrain.tax_value.mean())
    evaluation_matrix = m.rmse_eval(ytrue, baseline=baseline_pred,
                                    linear_regression=linreg_pred,
                                    lasso_lars=llars_pred, glm=glm_pred)
    evaluation_matrix.index = [
        'Baseline', 'Linear Regression', 'LASSO LARS', 'General Linear Model']
    evaluation_matrix.columns = ['Train RMSE', 'Validate RMSE']
    return md('| Methodology' + evaluation_matrix.to_markdown()[1:]), llars


def run_test(model: LinearRegressionType,
             xtest: ModelDataType, ytest: ModelDataType) -> md:
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
    ev.plot_residuals(ytest.tax_value, ypred)
    return md('### Mean Squared Error: ' +
              str(np.round(np.sqrt(mean_squared_error(ytest, ypred)), 2)))
