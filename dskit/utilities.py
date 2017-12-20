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

        models.append(deepcopy(model))

    return models

## Label encoder
def label_encoder(X_train, X_test, columns):
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


# Source:
#       - https://github.com/scipy/scipy/blob/v0.14.0/scipy/stats/stats.py#L3809
#       - https://stackoverflow.com/questions/40044375/
#         how-to-calculate-the-kolmogorov-smirnov-statistic-between-two-weighted-samples
def ks_weighted_2samp(data1, data2, wei1, wei2, alpha = 0.05):
    """
    Computes the Kolmogorov-Smirnov statistic on 2 samples.
    This is a two-sided test for the null hypothesis that 2 independent samples
    are drawn from the same continuous distribution.

    Parameters
    ----------
    data1, data2 : sequence of 1-D ndarrays
        two arrays of sample observations assumed to be drawn from a continuous
        distribution, sample sizes can be different

    wei1, wei2 : sequence of 1-D ndarrays
        two arrays with corresponding sample weights

    alpha : float
        confidence level

    Returns
    -------
    D : float
        KS statistic
    p-value : float
        two-tailed p-value
    """

    data1, data2 = map(np.asarray, (data1, data2))

    hist1, bin_edges1 = np.histogram(data1, weights=wei1)
    n1 = sum(hist1)
    hist2, bin_edges2 = np.histogram(data2, weights=wei2)
    n2 = sum(hist2)

    ix1 = np.argsort(data1)
    ix2 = np.argsort(data2)

    data1 = data1[ix1]
    data2 = data2[ix2]

    wei1 = wei1[ix1]
    wei2 = wei2[ix2]

    data_all = np.concatenate([data1,data2])

    cwei1 = np.hstack([0.,np.cumsum(wei1)*1./sum(wei1)])
    cwei2 = np.hstack([0.,np.cumsum(wei2)*1./sum(wei2)])

    cdf1we = cwei1[[np.searchsorted(data1,data_all,side='right')]]
    cdf2we = cwei2[[np.searchsorted(data2,data_all,side='right')]]

    d = np.max(np.absolute(cdf1we - cdf2we))

    # Note: d absolute not signed distance
    en = np.sqrt(n1*n2/float(n1+n2))

    try:
        prob = distributions.kstwobign.sf((en + 0.12 + 0.11 / en) * d)
    except:
        prob = 1.0

    c_alpha = (-0.5*np.log(alpha/2.))**(0.5)
    k_alpha = c_alpha/en

    print "\n=============================="
    print "Summary Report:"
    print "KS(data) value: ", d
    print "KS(null) value: ", k_alpha

    if d > k_alpha:
        print "KS test: ", True, " (null-hypothesis rejected)"
    else:
        print "KS test: ", False, " (null-hypothesis not rejected)"

    return [d, prob]
