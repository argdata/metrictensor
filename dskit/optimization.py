## Import common python libraries

import numpy as np
import matplotlib.pyplot as plt
import random

import pandas

# Import scikit-learn
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

# Fix random seed for reproducibility
seed = 7
random.seed(a=seed)


## Standard nested k-fold cross validation
def nested_grid_search_cv(model, X, y, outer_cv, inner_cv,
                          param_grid, scoring="accuracy",
                          n_jobs=1):
    """
    Nested k-fold crossvalidation.

    Parameters
    ----------
    Classifier : array, shape = [n_samples]
            true class, intergers in [0, n_classes - 1)
    X : array,  shape = [n_samples, n_classes]
    y : array,  shape = [n_samples, n_classes]
    outer_cv:   shape = [n_samples, n_classes]
    inner_cv:   shape = [n_samples, n_classes]
    param_grid: shape = [n_samples, n_classes]
    scoring:    shape = [n_samples, n_classes]

    Returns
    -------
    Grid classifier: classifier re-fitted to full dataset
    """

    outer_scores = []

    # Set aside a hold-out test dataset for model evaluation
    for k, (training_samples, test_samples) in enumerate(outer_cv.split(X, y)):

        # x training and test datasets
        if isinstance(X, pandas.core.frame.DataFrame):
            x_train = X.iloc[training_samples]
            x_test = X.iloc[test_samples]
        else:  # in case of spare matrices
            x_train = X[training_samples]
            x_test = X[test_samples]

        # y training and test datasets
        if isinstance(y, pandas.core.frame.Series):
            y_train = y.iloc[training_samples]
            y_test = y.iloc[test_samples]
        else: # in case of numpy arrays
            y_train = y[training_samples]
            y_test = y[test_samples]

        # Set up grid search configuration
        grid = GridSearchCV(estimator=model, param_grid=param_grid,
                            cv=inner_cv, scoring=scoring, n_jobs=n_jobs)

        # Build classifier on best parameters using outer training set
        # Fit model to entire training dataset (i.e tuning & validation dataset)
        print "fold-%s model fitting ..." % (k+1)

        # Train on the training set
        grid.fit(x_train, y_train)

        # Evaluate
        score = grid.score(x_test, y_test)

        outer_scores.append(score)
        print "\tModel validation score", score

    # Print final model evaluation (i.e. mean cross-validation scores)
    print "Final model evaluation (mean cross-val scores):\n", np.array(outer_scores).mean()

    # Note: the scoring is being done without the weights associated with X
    # Fit model to entire training dataset (i.e tuning & validation dataset)
    print "Performing fit over entire training data\n"
    grid.fit(X, y)

    return grid


## Feature ranking
def feature_ranking_plot(X, importances, std, indices, title):

    # Customize the major grid
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.grid(which='major', linestyle='-', linewidth='0.2', color='gray')
    ax.set_facecolor('white')

    for i in xrange(X.shape[1]):
        print("%d. %s (%f)" % (i + 1, X.columns[indices[i]], importances[indices[i]]))

    # Plot the feature importances of the model
    plt.title(title, fontsize=14)

    plt.bar(range(X.shape[1]), importances[indices],
            color="r", yerr=std, align="center")

    plt.xticks(range(X.shape[1]), X.columns[indices], rotation='vertical')

    plt.xlim([-1, X.shape[1]])

    plt.tight_layout()

    return plt.show()


## Extract feature selection
def extract_feature_selected(clf, X_train, X_test, y_train, y_test):

    clf.fit(X_train, y_train)

    select_indices = clf.named_steps['SELECT'].transform(
           np.arange(len(X_train.columns)).reshape(1, -1))

    feature_names = X_train.columns[select_indices]

    return feature_names

## Feature selection
def features_selection_model_performance(clf, X, y, parameter_set):

    # Customize the major grid
    fig, ax = plt.subplots()
    ax.grid(which='major', linestyle='-', linewidth='0.2', color='gray')
    ax.set_facecolor('white')

    score_means = list()
    score_stds = list()

    params = {'SELECT__k': 'top k features',
              'SELECT__threshold': 'feature threshold',
              'SELECT__n_features_to_select': 'n features to select',
              'SELECT__percentile': 'percentile',
              'SELECT__cv': 'k-fold',
              'SELECT__selection_threshold':'selection threshold'}

    label = [keyname for keyname in clf.get_params().keys() if keyname in params.keys()][0]

    for k in parameter_set:

        param = {label: k}
        clf.set_params(**param)

        # Compute cross-validation score using 1 CPU
        this_scores = cross_val_score(clf, X, y, cv=3, n_jobs=1)
        score_means.append(this_scores.mean())
        score_stds.append(this_scores.std())

    plt.errorbar(parameter_set, score_means, np.array(score_stds))

    model = clf.steps[1][0]

    title = 'Performance of the {}-{} varying for features selected'.format(model,
                                                                            clf.get_params().keys()[1])

    plt.title(title, fontsize=14)
    plt.xlabel(params[label])
    plt.ylabel('Prediction rate')

    print extract_feature_selected(clf, X, y).values[0]

    return plt.show()

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