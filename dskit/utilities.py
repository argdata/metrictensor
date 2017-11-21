## Import common python libraries

import math
import numpy as np
import pandas as pd

from copy import deepcopy

# Import scikit-learn
from sklearn.preprocessing import (StandardScaler, RobustScaler, MinMaxScaler,
                                   LabelEncoder, OneHotEncoder)
from sklearn.model_selection import ParameterGrid

##  Number of subplots for lay out
def lay_out(naxes):
    # determine number of rows and columns for figure
    f = lambda x: int(math.ceil(float(x)/2))
    g = lambda x: 1 if naxes==1 else 2
    
    nrows = f(naxes)
    ncols = g(naxes)
    
    return [nrows, ncols]


## Generate a list of models base on a set of hyper-parameters
def model_grid_setup(estimator, param_grid):
    models = []

    for params in list(ParameterGrid(param_grid)):
        model = estimator.set_params(**params)

        model.fit(X_train, y_train)
        models.append(deepcopy(model))

    return models

## Label encoder
def label_encoder(columns, X_train, X_test):
    le = LabelEncoder()

    # Iterating over all the common columns in train and test
    for col in columns:
        # label ecoding only categorical variables
        le.fit(np.hstack([X_train[col].values, X_test[col].values]))

        # Implemented in the following fashion to avoid
        # SettingWithCopyException warning

        #val = pd.DataFrame(le.transform(X_train[col]),
        #                   index=X_train[col].index, columns=[col+'_le'])

        #X_train = X_train.reset_index().join(val, on='index').set_index('index')

        val = pd.DataFrame(le.transform(X_train[col]),
                           index=X_train[col].index, columns=[col+'_le'])

        X_train = pd.concat([X_train, val], axis=1, join='inner')

        val = pd.DataFrame(le.transform(X_test[col]),
                           index=X_test[col].index, columns=[col+'_le'])

        X_test = pd.concat([X_test, val], axis=1, join='inner')

    return [X_train, X_test]
