## Import common python libraries

import sys
import time
import math
import heapq
import os.path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import random
import operator
import collections

# Import panda library
import pandas.core.common as com
from pandas.tools import plotting
from pandas.tools.plotting import scatter_matrix
from pandas.core.index import Index

# Import scipy
import scipy as sp
from scipy.stats import ks_2samp

# Import itertools
import itertools
from itertools import cycle

# Import collections
from collections import defaultdict, Counter

# Import Jupyter
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import display

# Import scikit-learn
import sklearn

from sklearn.preprocessing import (StandardScaler, RobustScaler, MinMaxScaler,
                                   LabelEncoder, OneHotEncoder)

from sklearn import feature_selection

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif

from sklearn.calibration import calibration_curve, CalibratedClassifierCV

from sklearn.pipeline import make_pipeline, Pipeline

from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve

from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.ensemble import VotingClassifier, BaggingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RandomizedLasso
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyClassifier

from sklearn.metrics import (confusion_matrix, roc_auc_score, roc_curve, 
                             auc, average_precision_score, precision_score, 
                             brier_score_loss, recall_score, f1_score, log_loss, 
                             classification_report, precision_recall_curve,
                             accuracy_score)

from sklearn.externals import joblib

# Import imblearn
import imblearn
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Check the versions of libraries/packages
print("Python version " + sys.version)
print("Numpy version " + np.__version__)
print("Pandas version " + pd.__version__)
print("Matplotlib version " + matplotlib.__version__)
print("Seaborn version " + sns.__version__)
print("Scipy version " + sp.__version__)
print("Scikit-learn version " + sklearn.__version__)
print("Imblance version " + imblearn.__version__)

# Fix random seed for reproducibility
seed = 7
random.seed(a=seed)

# Specifying which nodes should be run interactively
#InteractiveShell.ast_node_interactivity = "all"


## Subplot lay out
def lay_out(naxes):
    # determine number of rows and columns for figure
    f = lambda x: int(math.ceil(float(x)/2))
    g = lambda x: 1 if naxes==1 else 2
    
    nrows = f(naxes)
    ncols = g(naxes)
    
    return (nrows, ncols)

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


## Plot categorical features for signal and background
def plot_categorical_features(signal, background, columns=None, 
                              top_n=None, normed=False, style=None,
                              legend_title=None, legend_label=None, **kwargs):
    """
    Draw histogram of the DataFrame's series comparing the distribution
    in `signal` to `background`.
    
    Parameters
    ----------
    signal : pandas dataframe
        signal
    background : pandas dataframe, background
    columns : string or list
        If passed, will be used to limit data to a subset of columns
    top_n : int, default None
        top n levels of categorical feature
    normed : boolean, default False
        True will normalize bar plot to unity
    style : string, default 'stacked'
        Bar plot style options are 'stacked', 'grouped', and None (layered)
    legend_title : string, default None
    legend_label : tuple
        Tuple of signal and background labels
    kwargs : other plotting keyword arguments
        To be passed to bar function
        
    Returns
    -------
    plot : matplotlib plot
    """
        
    """
    Describe possible kwargs values
    
    Keys
    ----------
    
    alpha : float
        alpha tranparency of bar plot color
    title : string
        legend title
    grid : boolean, default True
        Whether to show axis grid lines
    ax : matplotlib axes object, default None
    xlabelsize : int, default None
        If specified changes the x-axis label size
    xrot : float, default None
        rotation of x axis labels
    ylabelsize : int, default None
        If specified changes the y-axis label size
    yrot : float, default None
        rotation of y axis labels
    sharex : bool, if True, the X axis will be shared amongst all subplots
    sharey : bool, if True, the Y axis will be shared amongst all subplots
    squeeze : (optional) bool, default True
        If True, extra dimensions are squeezed out from the returned Axes object:
    figsize : tuple (w, h)
        The size of the figure to create in inches by default
    layout : (optional) a tuple (rows, columns) for the layout of the histograms
    align : string, default 'center'
        Aligns the x-axis tick labels using 'center' or 'edge'
    """
    
    args = {'alpha': 1.0, 'title': 20, 'legend_title_size': 20, 'grid': True, 
            'xlabelsize': 20, 'xrot': None, 'ylabelsize': 20, 'yrot': None, 
            'sharex': False, 'sharey': False, 'squeeze': False, 'figsize': (18, 20),
            'layout': None, 'wspace': 0.25, 'hspace': 0.4, 'width': None}
        
    # override default args values
    xrot = None
    for key in args:
        if key in kwargs:
            args[key] = kwargs[key]
            if key == 'xrot':
                xrot = kwargs[key]
            kwargs.pop(key)

    # check if column is string or list
    if columns is not None:
        if not isinstance(columns, (list, np.ndarray, Index)):
            columns = [columns]
    
    # define signal and background samples
    signal = signal[columns]
    background = background[columns]
    
    # determine number of subplots to generate
    naxes = signal.shape[1]

    # set layout of the subplots
    if args['layout'] is None:
        args['layout'] = lay_out(naxes)
    
    # set figzie of figure
    if args['layout']==(1, 1):
        args['figsize'] = (10, 6)
    else:
        args['figsize'] = (18, 6+7*(args['layout'][0]-1))

    # create blank canvas
    fig, axes = plt.subplots(nrows=args['layout'][0], ncols=args['layout'][1],
                             squeeze=args['squeeze'],figsize=args['figsize'])

    # contains all subplot objects
    xs = axes.flat
    
    # generate each subplot
    for i, col in enumerate(com._try_sort(signal.columns)):
        # Check if column is a categorical variable
        val = signal[col].values[0]
        if isinstance(val, str) or isinstance(val, int):
            if args['xrot'] is None:
                args['xrot'] = 90
        else:
            raise ValueError('Categorical features are not strings or integers!')
        
        # calculate frequncy of each level of the categorical feature
        sg = signal[col].value_counts()
        bk = background[col].value_counts()

        # set ith subplot
        ax = xs[i]
            
        # set levels of category features
        labels = list(set(sg.index.tolist()).union(set(bk.index.tolist())))

        # for categorical features with small integer values
        # keep horizontal
        if isinstance(val, int) and max(labels) < 100:
            args['xrot'] = 0
            
        # force signal and background arrays to be of the same size
        SG = pd.Series(np.zeros(len(labels)), index=labels, dtype=np.float)
        BK = SG.copy()

        SG.update(sg)
        BK.update(bk)

        # select top n levels from categorical feature
        if top_n is not None:
            top_levels = sorted(zip(*heapq.nlargest(top_n, enumerate(SG+BK),
                                key=operator.itemgetter(1)))[0])

            SG = SG.take(top_levels)
            BK = BK.take(top_levels)

            positions = range(0, SG.shape[0])
            labels = SG.index.values
        else:
            positions = np.arange(float(len(labels)))

        # normalize bar plot
        if normed is True:
            SG = SG/SG.sum()
            BK = BK/BK.sum()
            legend_title = 'Normalized'

        # stacked/segmented, grouped, layered bar plot
        if style == 'stacked':
            bottom = BK
            width = 0.9 if args['width'] is None else args['width']
            shift = 0
        elif style == 'grouped':
            bottom = 0
            width = 0.45 if args['width'] is None else args['width']
            shift = width
        else:
            kwargs['alpha'] = 0.7
            bottom = 0
            width = 0.9 if args['width'] is None else args['width']
            shift = 0

        
        # display bar plot of categorical feature
        ax.bar(np.array(positions), BK, width=width, bottom=0.0, **kwargs)
        ax.bar(np.array(positions)-shift, SG, width=width, bottom=bottom, **kwargs)

        # set x-axis tick labels
        ax.xaxis.set_ticklabels(labels)
        
        # set x-axis tick positions
        ax.xaxis.set_ticks(positions)

        # set subplot title and subplot color
        ax.set_title(col)
        ax.set_facecolor('white')
        
        # set subplot title font size
        ax.title.set_size(args['title'])
        
        # set x-axis tick label font size
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(args['ylabelsize'])
        
        # set y-axis tick label font size
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(args['ylabelsize'])
        
        # rotate x-axis tick label
        for label in ax.get_xmajorticklabels():
            label.set_rotation(args['xrot'])
            label.set_horizontalalignment('center')
            
        # customize the major grid
        ax.grid(which='major', linestyle='-', linewidth='0.2', color='gray')
        
        # set legend
        lg = ax.legend(legend_label, title=legend_title,
                       loc='best', prop={'size': args['legend_title_size']})
        lg.get_title().set_fontsize(args['legend_title_size'])
    
        # reset args['xrot'] to original value
        args['xrot'] = xrot
        
    # adjust spacing between subplots
    fig.subplots_adjust(wspace=args['wspace'], hspace=args['hspace'])
    
    return plt.show()


## Plot numerical features for signal and background
def plot_numerical_features(signal, background, columns=None, bins=50,
                            normed=False, style='stacked', discrete=False,
                            legend_title=None, legend_label=None, **kwargs):
    """
    Draw histogram of the DataFrame's series comparing the distribution
    in `signal` to `background`.
    
    Parameters
    ----------
    signal : pandas dataframe
        signal
    background : pandas dataframe, background
    columns : string or list
        If passed, will be used to limit data to a subset of columns
    bins : int, default 50
    normed : boolean, default False
        True will normalize bar plot to unity
    style : string, default 'stacked'
        Histogram plot style options are 'stacked', 'grouped', and None (layered)
    discrete : boolean, default False
        True if numerical feature is discrete False otherwise
    legend_title : string, default None
    legend_label : tuple
        Tuple of signal and background labels
    kwargs : other plotting keyword arguments
        To be passed to bar function
        
    Returns
    -------
    plot : matplotlib plot
    """
        
    """
    Describe possible kwargs values
    
    Keys
    ----------
    
    alpha : float
        alpha tranparency of bar plot color
    title : string
        legend title
    grid : boolean, default True
        Whether to show axis grid lines
    ax : matplotlib axes object, default None
    xlabelsize : int, default None
        If specified changes the x-axis label size
    xrot : float, default None
        rotation of x axis labels
    ylabelsize : int, default None
        If specified changes the y-axis label size
    yrot : float, default None
        rotation of y axis labels
    sharex : bool, if True, the X axis will be shared amongst all subplots
    sharey : bool, if True, the Y axis will be shared amongst all subplots
    squeeze : (optional) bool, default True
        If True, extra dimensions are squeezed out from the returned Axes object:
    figsize : tuple (w, h)
        The size of the figure to create in inches by default
    layout : (optional) a tuple (rows, columns) for the layout of the histograms
    align : string, default 'center'
        Aligns the x-axis tick labels using 'left', 'mid', or 'right'
    """
    
    args = {'alpha': 1.0, 'title': 20, 'legend_title_size': 20, 'grid': True, 
            'xlabelsize': 20, 'xrot': None, 'ylabelsize': 20, 'yrot': None, 
            'sharex': False, 'sharey': False, 'squeeze': False,
            'figsize': (10, 6), 'layout': None,
            'wspace': 0.2, 'hspace': 0.4}
        
    # override default args values
    for key in args:
        if key in kwargs:
            args[key] = kwargs[key]
            if key == 'xrot':
                xrot = kwargs[key]
            kwargs.pop(key)

    # check if column is string or list
    if columns is not None:
        if not isinstance(columns, (list, np.ndarray, Index)):
            columns = [columns]
    
    # define signal and background samples
    signal = signal[columns]
    background = background[columns]
    
    # determine number of subplots to generate
    naxes = signal.shape[1]

    # set layout of the subplots
    if args['layout'] is None:
        args['layout'] = lay_out(naxes)
    
    # set figzie of figure
    if args['layout']==(1, 1):
        args['figsize'] = (10, 6)
    else:
        args['figsize'] = (18, 6+7*(args['layout'][0]-1))

    # create blank canvas
    fig, axes = plt.subplots(nrows=args['layout'][0], ncols=args['layout'][1],
                             squeeze=args['squeeze'], figsize=args['figsize'])

    # contains all subplot objects
    xs = axes.flat
    
    # generate each subplot
    for i, col in enumerate(com._try_sort(signal.columns)):
        # Check if column is a numerical variable
        val = signal[col].values[0]
        if isinstance(val, str):
            raise ValueError('Numerical features is not a float!')

        # set ith subplot
        ax = xs[i]

        # background and signal sample values
        bk_val = background[col].values
        sg_val = signal[col].values
        
        # minimum and maximum sample values
        low = min(sg_val.min(), bk_val.min())
        high = max(sg_val.max(), bk_val.max())

        # determine step size of discrete numerical feature
        if discrete==True:
            uniq_lst = sorted(set(sg_val.tolist()+bk_val.tolist()))
            step = abs(uniq_lst[1]-uniq_lst[0])
            high = high + step


        # normalization
        weights = None
        if normed is True:
            legend_title = 'Normalized'
            bk_weights = np.ones_like(bk_val)/float(len(bk_val))
            sg_weights = np.ones_like(sg_val)/float(len(sg_val))
            weights = [bk_weights, sg_weights]
        
        # display histogram plot of numerical feature
        if style == 'stacked':
            ax.hist([bk_val, sg_val], weights=weights, bins=bins,
                     range=(low, high), histtype='barstacked', **kwargs)
        elif style == 'grouped':
            ax.hist([bk_val, sg_val], weights=weights, bins=bins,
                     range=(low, high), histtype='bar', **kwargs)
        else:
            kwargs['alpha'] = 0.7
            ax.hist(bk_val, bins=bins, range=(low, high), normed=normed,
                    histtype='stepfilled', **kwargs)
            ax.hist(sg_val, bins=bins, range=(low, high), normed=normed,
                    histtype='stepfilled', **kwargs)
        
        # set subplot title and subplot color
        ax.set_title(col)
        ax.set_facecolor('white')
        
        # set subplot title font size
        ax.title.set_size(args['title'])
        
        # set x-axis tick label font size
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(args['ylabelsize'])
        
        # set y-axis tick label font size
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(args['ylabelsize'])
        
        # rotate x-axis tick label
        for label in ax.get_xmajorticklabels():
            label.set_rotation(args['xrot'])
            label.set_horizontalalignment('center')
            
        # customize the major grid
        ax.grid(which='major', linestyle='-', linewidth='0.2', color='gray')
        
        # set legend
        lg = ax.legend(legend_label, title=legend_title,
                       loc='best', prop={'size': args['legend_title_size']})
        lg.get_title().set_fontsize(args['legend_title_size'])
        
    # adjust spacing between subplots
    fig.subplots_adjust(wspace=args['wspace'], hspace=args['hspace'])
    
    return plt.show()


## Define linear correlation matrix

# http://www.statisticssolutions.com/correlation-pearson-kendall-spearman/
# Question: Which correlation method to apply

def plot_correlation_matrix(data, **kwds):
    """
    To calculate pairwise correlation between features.
    
    Extra arguments are passed on to DataFrame.corr()
    """

    if (data['clicks'] > 0).all(axis=0):
        label = "only clicks"
        data = data.drop(labels="clicks", axis=1, inplace=False)
    elif (data['clicks'] < 1).all(axis=0):
        label = "only non-clicks"
        data = data.drop(labels="clicks", axis=1, inplace=False)
    else:
        label = "all"
    
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    labels = data.corr(**kwds).columns.values
    
    fig, ax1 = plt.subplots(ncols=1, figsize=(8, 7))
    ax1.set_title("Correlations: " + label)
    
    args = {"annot":True, "ax": ax1, "vmin": 0, "vmax":1, "annot_kws": {"size": 8},
            "cmap": plt.get_cmap("Blues", 20)}

    # Correlations are calculated with .corr() method
    sns.heatmap(data.corr(method="spearman"), **args)
    
    for ax in (ax1,):
        # shift location of ticks to center of the bins
        ax.set_xticks(np.arange(len(labels))+0.5, minor=False)
        ax.set_yticks(np.arange(len(labels))+0.5, minor=False)

        ax.set_xticklabels(labels, minor=False, ha="right", rotation=70)
        ax.set_yticklabels(np.flipud(labels), minor=False)
    
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)

    plt.tight_layout()

    return plt.show()


## Compute ROC curve and area under the curve
def plot_roc_curve(models, X_train, X_test, y_train, y_test,
                   n_folds=0, sample_weight_flag=False):
    """
    Multi class version of Logarithmic Loss metric.
    https://www.kaggle.com/wiki/MultiClassLogLoss

    Parameters
    ----------
    models : dictionary, shape = [n_models]
    X_train : DataFrame, shape = [n_samples, n_classes]
    X_train : DataFrame, shape = [n_samples, n_classes]
    y_train : DataFrame, shape = [n_classes]
    y_test : DataFrame, shape = [n_classes]
    n_folds : int, default 0
    sample_weight_flag: boolean, default False

    Returns
    -------
    roc : matplotlib plot
    """

    # contains rates for ML classifiers
    fpr = dict()
    tpr = dict()
    thresholds = dict()
    rocauc = dict()
    logloss = dict()

    # color choices: https://css-tricks.com/snippets/css/named-colors-and-hex-equivalents/
    colors = cycle(['DarkCyan', 'darkorange', 'cornflowerblue',
                    'darkseagreen', 'thistle', 'slateblue', 'darkslategrey',
                    'cadetblue', 'chocolate', 'darkred', 'goldenrod',
                    'darkgoldenrod'])

    # check to see if models is a dictionary
    if not isinstance(models, dict) or not isinstance(models, collections.OrderedDict):
        # check to see if model is a pipeline object or not
        if isinstance(models, sklearn.pipeline.Pipeline):
            data_type = type(models._final_estimator)
        else:
            data_type = type(models)

        name = filter(str.isalnum, str(data_type).split(".")[-1])
        models = {name: models}

    # Customize the major grid
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.grid(which='major', linestyle='-', linewidth='0.2', color='gray')
    ax.set_facecolor('white')

    # Loop through classifiers
    for (name, model) in models.items():
        print "\n\x1b[1;31m"+name+" model ...\x1b[0m"

        # check to see if model file exist
        if not os.path.isfile('models/'+name+'.pkl'):
            model.fit(X_train, y_train)
            joblib.dump(model, 'models/'+name+'.pkl')
        else:
            model = joblib.load('models/'+name+'.pkl')

        y_pred = model.predict(X_test)

        # Statistics summary report
        print classification_report(y_test, y_pred,
                                    target_names=['signal', 'background'])
        print("\tScore (i.e. accuracy) of test dataset: {:.5f}"
              .format(model.score(X_test, y_test)))

        if n_folds!=0:
            if sample_weight_flag:
                weights = {name.lower()+'__sample_weight': sample_weight_dev}
            else:
                weights = None

            scores = cross_val_score(model, X_test, y_train, scoring='rocauc',
                                     cv=n_folds, n_jobs=1, fit_params=weights)

            print "\tCross-validated AUC ROC score: %0.5f (+/- %0.5f)"%(scores.mean(),
                                                                        scores.std())

        if hasattr(model, "predict_proba"):
            # probability estimates of the positive class
            # (as needed in the roc_curve function)
            y_score = model.predict_proba(X_test)[:, 1]
        else:  # use decision function
            y_score = model.decision_function(X_test)

        total = len(y_test)

        # Calculate roc curve values
        fpr[name], tpr[name], thresholds[name] = roc_curve(y_test, y_score)

        # Non-cross-validated AUROC
        rocauc[name] = roc_auc_score(y_test, y_score)
        logloss[name] = log_loss(y_test, y_score)

        print "\tAUC ROC score for %s: %.4f" % (name, rocauc[name])
        print "\tLog Loss score for %s: %.4f" % (name, logloss[name])

    for (name, model), color in zip(models.items(), colors):

        signal_efficiecy = tpr[name] # true positive rate (tpr)
        background_efficiecy = fpr[name] # false positive rate (fpr)
        
        # NOTE: background rejection rate = 1 - background efficiency (i.e specicity)
        background_rejection_rate = 1 - background_efficiecy

        # Plot all ROC curves
        plt.plot(signal_efficiecy, background_rejection_rate, color=color, lw=2,
                 label='%s (AUC = %0.3f, Log Loss = %0.3f)' % (name, rocauc[name],
                                                               logloss[name]))

    title = "Receiver operating characteristic ({} samples)".format(total)

    plt.title(title, fontsize=14)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel('Signal Efficiency (True Positive Rate)')
    plt.ylabel('Background Rejection Rate (1- False Positive Rate)')

    leg = plt.legend(loc="lower left", frameon=True, fancybox=False, fontsize=11)
    leg.get_frame().set_edgecolor('w')

    frame = leg.get_frame()
    frame.set_facecolor('White')

    return plt.show() 


## Define precision-recall curve

def plot_precision_recall_curve(models, X_train, X_test, y_train, y_test):
    """
    Plot a basic precision/recall curve.

    Parameters
    ----------
    models : dictionary, shape = [n_models]
    X_train : DataFrame, shape = [n_samples, n_classes]
    X_train : DataFrame, shape = [n_samples, n_classes]
    y_train : DataFrame, shape = [n_classes]
    y_test : DataFrame, shape = [n_classes]

    Returns
    -------
    plot : matplotlib plot
    """

    # check to see if models is a dictionary
    if not isinstance(models, dict) or not isinstance(models, collections.OrderedDict):
        # check to see if model is a pipeline object or not
        if isinstance(models, sklearn.pipeline.Pipeline):
            data_type = type(models._final_estimator)
        else:
            data_type = type(models)

        name = filter(str.isalnum, str(data_type).split(".")[-1])
        models = {name: models}

    # Customize the major grid
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.grid(which='major', linestyle='-', linewidth='0.2', color='gray')
    ax.set_facecolor('white')

    # Calculate the random luck for PR
    # (above the constant line is a classifier that is well modeled)
    signal_count = sum(y_train[y_train > 0])
    background_count = sum(np.ones(len(y_train[y_train < 1])))
    ratio = float(signal_count)/float(signal_count + background_count)

    # store average precision calculation
    avg_scores = []

    # Loop through classifiers
    for (name, model) in models.items():

        if not os.path.isfile('models/'+name+'.pkl'):
            model.fit(X_train, y_train)
            joblib.dump(model, 'models/'+name+'.pkl')

        else:
            model = joblib.load('models/'+name+'.pkl')

        if hasattr(model, "predict_proba"):
            # probability estimates of the positive class
            # (as needed in the roc_curve function)
            y_score = model.predict_proba(X_test)[:, 1]
        else:  # use decision function
            y_score = model.decision_function(X_test)

        total = len(y_test)

        # Compute precision recall curve
        precision, recall, thresholds = precision_recall_curve(y_test, y_score,
                                                               pos_label=y_test.max())

        # Area under the precision-recall curve (AUCPR)
        average_precision = average_precision_score(y_test, y_score)

        avg_scores.append(average_precision)

        plt.plot(recall, precision, lw=1,
                 label='%s (AUC = %0.3f)' % (name, np.mean(avg_scores, axis=0)))

    plt.plot([ratio, ratio], '--', color=(0.1, 0.1, 0.1),
             label='Luck (AUC = %0.3f)' % ratio)


    plt.title('Precision-Recall curve', fontsize=14)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])

    plt.xlabel('Recall')
    plt.ylabel('Precision')

    leg = plt.legend(loc="best", frameon=True, fancybox=False, fontsize=12)
    leg.get_frame().set_edgecolor('w')

    frame = leg.get_frame()
    frame.set_facecolor('White')

    return plt.show()


# Learning curve
def plot_learning_curve(model, X_train, y_train,
                        ylim=None, cv=None, n_jobs=-1,
                        train_sizes=np.linspace(0.1, 1.0, 10, endpoint=True)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    model : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    X_train : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y_train : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).

    train_sizes = np.linspace(0.1, 1.0, 10, endpoint=True) produces
        8 evenly spaced points in the range 0 to 10
    """

    # check to see if model is a pipeline object or not
    if isinstance(model, sklearn.pipeline.Pipeline):
        data_type = type(model._final_estimator)
    else:
        data_type = type(model)

    # plot title
    name = filter(str.isalnum, str(data_type).split(".")[-1])
    title = "Learning Curves (%s)" % name

    # create blank canvas
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.grid(which='major', linestyle='-', linewidth='0.2', color='gray')
    ax.set_facecolor('white')

    train_sizes_abs, train_scores, test_scores = \
    learning_curve(model, X_train, y_train,
                   train_sizes=np.linspace(0.1, 1.0, 10),
                   cv=cv, scoring=None, exploit_incremental_learning=False,
                   n_jobs=n_jobs, pre_dispatch="all", verbose=0)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # plot the std deviation as a transparent range at each training set size
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    # plot the average training and test score lines at each training set size
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")


    plt.title(title, fontsize=14)

    # sizes the window for readability and displays the plot
    # shows error from 0 to 1.1
    if ylim is not None:
        plt.ylim(*ylim)
        #plt.ylim(-.1, 1.1)

    plt.xlabel("Training set size")
    plt.ylabel("Score")

    leg = plt.legend(loc="best", frameon=True, fancybox=False, fontsize=12)
    leg.get_frame().set_edgecolor('w')

    frame = leg.get_frame()
    frame.set_facecolor('White')

    # box-like grid
    #plt.grid(figsize=(8, 6))

    #plt.gca().invert_yaxis()

    return plt.show()


## Define validation plots
def plot_validation_curve(models, X_train, X_test, y_train, y_test):

    # check to see if models is a dictionary
    if not isinstance(models, list):
        models = [models]

    # check to see if model is a pipeline object or not
    if isinstance(models[0], sklearn.pipeline.Pipeline):
        data_type = type(models[0]._final_estimator)
    else:
        data_type = type(models[0])

    # plot title
    name = filter(str.isalnum, str(data_type).split(".")[-1])
    title = "Validation Curves (%s)" % name

    # create blank canvas
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.grid(which='major', linestyle='-', linewidth='0.2', color='gray')
    ax.set_facecolor('white')

    for n, model in enumerate(models):
        test_score = np.empty(len(model.estimators_))
        train_score = np.empty(len(model.estimators_))
        
        for j, pred in enumerate(model.staged_decision_function(X_test)):
            test_score[j] = 1-roc_auc_score(y_test, pred)
        
        for k, pred in enumerate(model.staged_decision_function(X_train)):
            train_score[k] = 1 - roc_auc_score(y_train, pred)
        
        best_iter = np.argmin(test_score)
        learn = model.get_params()['learning_rate']
        depth = model.get_params()['max_depth']

        test_line = plt.plot(test_score,
                             label='learn=%.1f depth=%i (%.2f)'%(learn, depth,
                                                                 test_score[best_iter]))
        
        colour = test_line[-1].get_color()

        plt.plot(train_score, '--', color=colour)

        plt.axvline(x=best_iter, color=colour)

    plt.title(title, fontsize=14)

    plt.xlabel('Number of boosting iterations')
    plt.ylabel('1 - AUC')

    plt.legend(loc='best', frameon=False, fancybox=True, fontsize=12)

    return plt.show()

## Defined overfitting plot
def plot_overfitting(model, X_train, X_test, y_train, y_test, bins=50):
    """
    Multi class version of Logarithmic Loss metric

    Parameters
    ----------
    y_true : array, shape = [n_samples]
            true class, intergers in [0, n_classes - 1)
    y_pred : array, shape = [n_samples, n_classes]

    Returns
    -------
    loss : float
    """

    # check to see if model is a pipeline object or not
    if isinstance(model, sklearn.pipeline.Pipeline):
        data_type = type(model._final_estimator)
    else:
        data_type = type(model)

    name = filter(str.isalnum, str(data_type).split(".")[-1])

    # check to see if model file exist
    if not os.path.isfile('models/'+name+'.pkl'):
        model.fit(X_train, y_train)
        joblib.dump(model, 'models/'+name+'.pkl')
    else:
        model = joblib.load('models/'+name+'.pkl')

    # use subplot to extract axis to add ks and p-value to plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Customize the major grid
    ax.grid(which='major', linestyle='-', linewidth='0.2', color='gray')
    ax.set_facecolor('white')

    if not hasattr(model, 'predict_proba'): # use decision function
        d = model.decision_function(sp.sparse.vstack([X_train, X_test]))
        bin_edges_low_high = np.linspace(min(d), max(d), bins + 1)
    else: # use prediction function
        bin_edges_low_high = np.linspace(0., 1., bins + 1)

    label_name = ""
    y_scores = []
    for X, y in [(X_train, y_train), (X_test, y_test)]:

        if hasattr(model, 'predict_proba'):
            label_name = 'Prediction Probability'
            y_scores.append(model.predict_proba(X[y > 0])[:, 1])
            y_scores.append(model.predict_proba(X[y < 1])[:, 1])
        else:
            label_name = 'Decision Function'
            y_scores.append(model.decision_function(X[y > 0]))
            y_scores.append(model.decision_function(X[y < 1]))

    width = np.diff(bin_edges_low_high)

    # Signal training histogram
    hist_sig_train, bin_edges = np.histogram(y_scores[0], bins=bin_edges_low_high)

    hist_sig_train = hist_sig_train/sum(hist_sig_train)

    plt.bar(bin_edges[:-1], hist_sig_train, width=width, color='r', alpha=0.5,
            label='signal (train)')

    # Background training histogram
    hist_bkg_train, bin_edges = np.histogram(y_scores[1], bins=bin_edges_low_high)

    hist_bkg_train = hist_bkg_train/sum(hist_bkg_train)

    plt.bar(bin_edges[:-1], hist_bkg_train, width=width,
            color='steelblue', alpha=0.5, label='background (train)')

    # Signal test histogram
    hist_sig_test, bin_edges = np.histogram(y_scores[2], bins=bin_edges_low_high)

    hist_sig_test = hist_sig_test/sum(hist_sig_test)
    scale = len(y_scores[2]) / sum(hist_sig_test)
    err = np.sqrt(hist_sig_test * scale) / scale
    center = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.errorbar(center, hist_sig_test, yerr=err, fmt='o', c='r', label='signal (test)')

    # Background test histogram
    hist_bkg_test, bin_edges = np.histogram(y_scores[3], bins=bin_edges_low_high)

    hist_bkg_test = hist_bkg_test/sum(hist_bkg_test)
    scale = len(y_scores[3]) / sum(hist_bkg_test)
    err = np.sqrt(hist_bkg_test * scale) / scale

    plt.errorbar(center, hist_bkg_test, yerr=err, fmt='o', c='steelblue', #range=low_high,
                 label='background (test)')

    # Estimate ks-test and p-values as an indicator of overtraining of fit model
    s_ks, s_pv = ks_2samp(hist_sig_test, hist_sig_train)
    b_ks, b_pv = ks_2samp(hist_bkg_test, hist_bkg_train)

    #s_ks, s_pv = ks_weighted_2samp(y_scores[0], y_scores[2],
    #                               signal_sample_weight_train, signal_sample_weight_test)
    #b_ks, b_pv = ks_weighted_2samp(y_scores[1], y_scores[3],
    #                               background_sample_weight_train,
    #                               background_sample_weight_test)

    name = filter(str.isalnum, str(type(model)).split(".")[-1])

    ax.set_title("%s: sig (bkg)\nks: %0.3f (%0.3f)\np-value: %0.3f (%0.3f)"
                 % (name, s_ks, b_ks, s_pv, b_pv), fontsize=14)

    plt.xlabel(label_name)
    plt.ylabel('Arbitrary units')

    leg = plt.legend(loc='best', frameon=False, fancybox=False, fontsize=12)
    leg.get_frame().set_edgecolor('w')

    frame = leg.get_frame()
    frame.set_facecolor('White')

    return plt.show()


## Calibration curve (reliability curve)
def plot_calibration_curve(model, X_train, X_test, y_train, y_test,
                           fig_index, n_bins=10):
    """
    Plot calibration curve for est w/o and with calibration.
    """

    # check to see if model is a pipeline object or not
    if isinstance(model, sklearn.pipeline.Pipeline):
        data_type = type(model._final_estimator)
    else:
        data_type = type(model)

    # model name
    name = filter(str.isalnum, str(data_type).split(".")[-1])

    # Calibrated with isotonic calibration
    isotonic = CalibratedClassifierCV(model, cv=2, method='isotonic')

    # Calibrated with sigmoid calibration (i.e. Platt Scaling)
    sigmoid = CalibratedClassifierCV(model, cv=2, method='sigmoid')

    # collect all models
    models = collections.OrderedDict()

    # define models dictionary
    models[name] = model
    models[name+'_Isotonic'] = isotonic
    models[name+'_Sigmoid'] = sigmoid

    plt.figure(fig_index, figsize=(8, 6))

    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "--", label="Perfectly calibrated")

    for (name, model) in models.items():

        # check to see if model file exist
        if not os.path.isfile('models/'+name+'.pkl'):
            model.fit(X_train, y_train)
            joblib.dump(model, 'models/'+name+'.pkl')
        else:
            model = joblib.load('models/'+name+'.pkl')

        y_pred = model.predict(X_test)

        if hasattr(model, 'predict_proba'): # use prediction probability
            y_score = model.predict_proba(X_test)[:, 1]
        else:  # use decision function
            y_score = model.decision_function(X_test)
            y_score = \
                (y_score - y_score.min()) / (y_score.max() - y_score.min())

        clf_score = brier_score_loss(y_test, y_score, pos_label=y_test.max())

        print "\n\x1b[1;31m%s:\x1b[0m" % name
        print "\tBrier: %1.3f" % (clf_score)
        print "\tPrecision: %1.3f" % precision_score(y_test, y_pred)
        print "\tRecall: %1.3f" % recall_score(y_test, y_pred)
        print "\tF1: %1.3f\n" % f1_score(y_test, y_pred)

        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, y_score, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s (%1.3f)" % (name, clf_score))

        ax2.hist(y_score, range=(0, 1), bins=n_bins, label=name,
                 histtype="step", lw=2)

    # First plot
    ax1.set_title('Probability Calibration Curves (Reliability Curves)', fontsize=14)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])

    ax1.legend(loc="lower right", frameon=True, fancybox=True, fontsize=12)

    # Customize the major grid
    ax1.grid(which='major', linestyle='-', linewidth='0.2', color='gray')
    ax1.set_facecolor('white')

    # Second plot
    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")

    ax2.legend(loc="best", frameon=True, fancybox=True, fontsize=12)

    # Customize the major grid
    ax2.grid(which='major', linestyle='-', linewidth='0.2', color='gray')
    ax2.set_facecolor('white')

    ##plt.tight_layout()

    return plt.show()


## Confusion matrix plot
def plot_confusion_matrix(model, X_train, X_test, y_train, y_test,
                          columns, normalize=False, cmap=plt.cm.Blues):
    """
    Multi class version of Logarithmic Loss metric.
    https://www.kaggle.com/wiki/MultiClassLogLoss

    Parameters
    ----------
    y_true : array, shape = [n_samples]
            true class, intergers in [0, n_classes - 1)
    y_pred : array, shape = [n_samples, n_classes]

    Returns
    -------
    loss : float

    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    # check to see if model is a pipeline object or not
    if isinstance(models, sklearn.pipeline.Pipeline):
        data_type = type(models._final_estimator)
    else:
        data_type = type(models)

    name = filter(str.isalnum, str(data_type).split(".")[-1])

    # check to see if model file exist
    if not os.path.isfile('models/'+name+'.pkl'):
        model.fit(X_train, y_train)
        joblib.dump(model, 'models/'+name+'.pkl')
    else:
        model = joblib.load('models/'+name+'.pkl')

    y_score = model.predict(X_test)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_score)

    if normalize:
        title = "Normalized Confusion Matrix"
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        title = "Confusion Matrix"

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title, fontsize=14)

    tick_marks = np.arange(len(columns))

    plt.xticks(tick_marks, columns, rotation=45)
    plt.yticks(tick_marks, columns)

    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    plt.figure(figsize=(20,10))
    plt.colorbar()

    plt.tight_layout()

    plt.grid(False, which='both')

    return plt.show()


## Standard nested k-fold cross validation
def nested_grid_search_cv(Classifier, X, y, outer_cv, inner_cv,
                          param_grid, scoring="accuracy"):
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

        # Training datasets
        x_training = X[training_samples]
        y_training = y.iloc[training_samples]

        # Testing datasets
        x_testing  = X[test_samples]
        y_testing  = y.iloc[test_samples]

        # Set up grid search configuration
        cv = GridSearchCV(estimator=Classifier, param_grid=param_grid,
                          cv=inner_cv, scoring=scoring, n_jobs=-1)

        # Build classifier on best parameters using outer training set
        # Fit model to entire training dataset (i.e tuning & validation dataset)
        print "%s-fold model fitting ..."%(k+1)

        # Train on the training set
        cv.fit(x_training, y_training)

        # Evaluate
        score = cv.score(x_testing, y_testing)
        outer_scores.append(score)
        print "\tModel validation score", score

    # Print final model evaluation (i.e. mean cross-validation scores)
    print "Final model evaluation (mean cross-val scores):\n", np.array(outer_scores).mean()

    # Note: the scoring is being done without the weights associated with X
    # Fit model to entire training dataset (i.e tuning & validation dataset)
    cv.fit(X, y)
    print "Final fit completed"

    return cv


## Feature ranking
def feature_ranking_plot(X, importances, std, indices, title):

    # Customize the major grid
    fig, ax = plt.subplots()
    ax.grid(which='major', linestyle='-', linewidth='0.2', color='gray')
    ax.set_facecolor('white')

    print(title)
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
def extract_feature_selected(clf, X, y):

    # Split data into a development and evaluation set
    X_dev, X_eval, y_dev, y_eval = train_test_split(df_X, df_y, test_size=.33,
                                                    random_state=seed)
    # Split development set into a train and test set
    X_train, X_test, y_train, y_test = train_test_split(X_dev, y_dev, test_size=0.33,
                                                        random_state=seed+31415)

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

    this_scores = list()
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

    print  extract_feature_selected(clf, X, y).values[0]

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

    return d, prob


## Plot signal and background distributions for some variables

# The first two arguments select what is 'signal'
# and what is 'background'. This means you can
# use it for more general comparisons of two
# subsets as well.

# NOTE: deprecated
def signal_background(signal, background, column=None, grid=True,
                      xlabelsize=None, xrot=None, ylabelsize=None,
                      yrot=None, ax=None, sharex=False,
                      sharey=False, figsize=None,
                      layout=None, bins=10, normed=0, **kwargs):
    """
    Draw histogram of the DataFrame's series comparing the distribution
    in `signal` to `background`.

    Parameters
    ----------
    signal : Signal DataFrame
    background : Background DataFrame
    column : string or sequence
        If passed, will be used to limit data to a subset of columns
    grid : boolean, default True
        Whether to show axis grid lines
    xlabelsize : int, default None
        If specified changes the x-axis label size
    xrot : float, default None
        rotation of x axis labels
    ylabelsize : int, default None
        If specified changes the y-axis label size
    yrot : float, default None
        rotation of y axis labels
    ax : matplotlib axes object, default None
    sharex : bool, if True, the X axis will be shared amongst all subplots.
    sharey : bool, if True, the Y axis will be shared amongst all subplots.
    figsize : tuple (w, h)
        The size of the figure to create in inches by default
    layout : (optional) a tuple (rows, columns) for the layout of the histograms
    bins : integer, default 10
        Number of histogram bins to be used
    kwargs : other plotting keyword arguments
        To be passed to hist function

    Returns
    -------
    signal_background : matplotlib plot
    """

    if 'alpha' not in kwargs:
        kwargs['alpha'] = 0.5

    if column is not None:
        if not isinstance(column, (list, np.ndarray, Index)):
            column = [column]

    signal = signal[column]
    background = background[column]

    naxes = signal.shape[1]

    fig, axes = plotting._tools._subplots(naxes=naxes, ax=ax, squeeze=False,
                                          sharex=sharex, sharey=sharey,
                                          figsize=(18, 12))

    xs = plotting._tools._flatten(axes)

    for i, col in enumerate(com._try_sort(signal.columns)):
        ax = xs[i]

        # Check if column is a categorical variable
        if isinstance(signal[col].values[0], str):
            sg = signal[col].value_counts()
            bk = background[col].value_counts()

            labels = list(set(sg.index.tolist()).union(bk.index.tolist()))
            positions = np.arange(float(len(labels)))

            # Force signal and background arrays to be of the same size
            SG = pd.Series(np.zeros(len(labels)), index=labels, dtype=np.float)
            BK = SG.copy()

            SG.update(sg)
            BK.update(bk)

            if normed==1:
                SG = SG/SG.sum(axis=1)
                BK = BK/BK.sum(axis=1)

            ax.bar(positions, BK, width=0.8, **kwargs)
            ax.xaxis.set_ticks(positions)

            ax.bar(positions, SG, width=0.8, **kwargs)
            ax.xaxis.set_ticks(positions)

            ax.xaxis.set_ticklabels(labels, rotation='vertical', fontsize=5)

            # Else it is considered as a numerical variable
        else:
            low = min(signal[col].min(), background[col].min())
            high = max(signal[col].max(), background[col].max())

            ax.hist(background[col].values,
                    bins=bins, range=(low,high), histtype='stepfilled',
                    normed=1,**kwargs)
            ax.hist(signal[col].values,
                    bins=bins, range=(low,high), histtype='stepfilled',
                    normed=1, **kwargs)

        ax.set_title(col)
        ax.legend(['non-clicks normed', 'clicks normed'], loc='best')

        # Customize the major grid
        ax.grid(which='major', linestyle='-', linewidth='0.2', color='gray')
        ax.set_facecolor('white')

    plotting._tools._set_ticks_props(axes, xlabelsize=xlabelsize, xrot=xrot,
                                     ylabelsize=ylabelsize, yrot=yrot)
    fig.subplots_adjust(wspace=0.4, hspace=0.8)

    return display(plt.show())

#### PYSPARK

## Plot signal and background distributions of features

def signal_background_plot(data, target = None, features=None, bins=10,
                           normed=0, legend=None, **kwds):
    """Draw histogram of the DataFrame's series comparing the distribution
    in `signal` to `background`

    Parameters
    ----------
    data : pyspark dataframe, shape = [num_sample] [num_features]
    target : string, default None
    features : list, default None
      If passed, will be used to limit data to a subset of columns
    label : string, default None
      Plot title
    bins: integer, default 10
      Number of histogram bins to be used
    normed: integer, default 0
      If 1 is passed the histograms are normalized to unity
    kwargs : other plotting keyword arguments
      To be passed to hist function

    Returns
    -------
    signal_background_plot : matplotlib plot
    """


    """Describe args arguments

    Keys
    ----------
    alpha: float, default 0.5
        Indidcates level of hue
    grid : boolean, default True
        Whether to show axis grid lines
    xlabelsize : int, default None
        If specified changes the x-axis label size
    xrot : float, default None
        rotation of x axis labels
    ylabelsize : int, default None
        If specified changes the y-axis label size
    yrot : float, default None
        rotation of y axis labels
    ax : matplotlib axes object, default None
    sharex : bool, if True, the X axis will be shared amongst all subplots.
    sharey : bool, if True, the Y axis will be shared amongst all subplots.
    figsize : tuple (w, h)
        The size of the figure to create in inches by default
    layout: (optional) a tuple (rows, columns) for the layout of the histograms
    """

    args = {'alpha': 0.5, 'grid':True, 'xlabelsize': None, 'xrot': None,
            'ylabelsize': None, 'yrot': None, 'ax': None, 'sharex': False,
            'sharey': False, 'figsize': None, 'layout': None}

    if 'alpha' not in kwds:
        kwds['alpha'] = args['alpha']

    if features:
        signal = data.filter(data[target]>0.5).select(features)
        background = data.filter(data[target]<0.5).select(features)
        naxes = len(features)
    else:
        signal = data.filter(data[target]>0.5)
        background = data.filter(data[target]<0.5)
        naxes = len(data.columns)

    fig, axes = plotting._subplots(naxes=naxes, ax=args['ax'], squeeze=False,
                                   sharex=args['sharex'], sharey=args['sharey'],
                                   figsize=(12, 8), layout=None);

    xs = plotting._flatten(axes)

    for i, col in enumerate(signal.columns):
        ax = xs[i]

        # Check if column is a categorical variable
        if isinstance(signal.schema.fields[i].dataType, StringType):
            print 'failed'
            sg = signal.groupBy(signal[col]).count().toPandas()
            bk = background.groupBy(signal[col]).count().toPandas()

            sg = pd.Series(data=sg['count'].values, index=sg[col])
            bk = pd.Series(data=bk['count'].values, index=bk[col])

            labels = list(set(sg.index.tolist()).union(bk.index.tolist()))
            positions = np.arange(float(len(labels)))

            # Force signal and background arrays to be of the same size
            SG = pd.Series(np.zeros(len(labels)), index=labels, dtype=np.float)
            BK = SG.copy()

            SG.update(sg)
            BK.update(bk)

            if normed==1:
                SG = SG/SG.sum(axis=1)
                BK = BK/BK.sum(axis=1)

            ax.bar(positions, BK.values, width=0.8, **kwds)
            ax.xaxis.set_ticks(positions)

            ax.bar(positions, SG.values, width=0.8, **kwds)
            ax.xaxis.set_ticks(positions)

            ax.xaxis.set_ticklabels(labels, rotation='vertical', fontsize=5)

            # Else it is considered as a numerical variable
        else:
            low = min(signal.groupBy().min(col).collect()[0][0],
                      background.groupBy().min(col).collect()[0][0])
            high = max(signal.groupBy().max(col).collect()[0][0],
                       background.groupBy().max(col).collect()[0][0])

            bk = np.array(background.select(col).rdd.map(lambda row: row[0]).collect())
            sg = np.array(signal.select(col).rdd.map(lambda row: row[0]).collect())
            # bins=bins # range=(low, high) # range(0, 15)
            ax.hist(bk, bins=range(1, 9), histtype='stepfilled',
                    normed=normed, **kwds)
            ax.hist(sg, bins=range(1, 9), histtype='stepfilled',
                    normed=normed, **kwds)

        ax.set_title(col)
        ax.legend(labels=legend, loc='best')

        # Customize the major grid
        ax.grid(which='major', linestyle='-', linewidth='0.2', color='gray')
        ax.patch.set_facecolor('white')

    fig.subplots_adjust(wspace=0.4, hspace=0.8)

    return display(plt.show())
