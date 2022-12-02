'''contains the data type aliases used in typing of each function for docstrings'''
import typing as t

import pandas as pd
from numpy.typing import ArrayLike
from sklearn.linear_model import LassoLars, LinearRegression, TweedieRegressor

ModelDataType = t.Union[ArrayLike,pd.DataFrame,pd.Series]
LinearRegressionType = t.Union[LinearRegression,LassoLars,TweedieRegressor]
PandasDataType = t.Union[pd.Series,pd.DataFrame]
lmplot_kwargs = {'scatter':{'color':'#40b7ad'},'line':{'color':'#2e1e3b'}}
