## Import common python libraries

import numpy as np
import matplotlib.pyplot as plt
import random

# Import scikit-learn
from sklearn.model_selection import cross_val_score

# Import Jupyter
from IPython.display import display

# Fix random seed for reproducibility
seed = 7
random.seed(a=seed)


## Extract feature selection
def extract_feature_selected(clf, X_train, y_train):

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

    return display(plt.show())


## Feature ranking
def plot_feature_ranking(X, importances, std, indices, title):

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

    return display(plt.show())
