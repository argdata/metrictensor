## Import common python libraries

import math
import numpy as np
import pandas as pd

# Import panda library
import pandas.core.common as com
from pandas.tools import plotting
from pandas.tools.plotting import scatter_matrix
from pandas.core.index import Index

# Import scikit-learn
from sklearn.preprocessing import (StandardScaler, RobustScaler, MinMaxScaler,
                                   LabelEncoder, OneHotEncoder)
from sklearn.model_selection import ParameterGrid

## Subplot lay out
def lay_out(naxes):
    # determine number of rows and columns for figure
    f = lambda x: int(math.ceil(float(x)/2))
    g = lambda x: 1 if naxes==1 else 2
    
    nrows = f(naxes)
    ncols = g(naxes)
    
    return [nrows, ncols]

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
