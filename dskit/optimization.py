## Import common python libraries

import numpy as np
import random

import pandas

# Import scikit-learn
from sklearn.model_selection import GridSearchCV


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
    grid = GridSearchCV()

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
        grid.set_params(estimator=model, param_grid=param_grid,
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
