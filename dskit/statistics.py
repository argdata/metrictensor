# Import common python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import seaborn as sns

# Import scipy
import scipy.stats

# Import itertools
import itertools

# Import Jupyter
from IPython.display import display


# chi2 matrix plot
def plot_chi2_matrix(data, columns, alpha=0.95, p_value=False):
    """
    To calculate pairwise chi2 statistic between features.

    Parameters
    ----------
    data : pandas dataframe
    columns : list
    alpha : float, default 0.95
        significance level
    p_value : boolean, default False
        If true report p-values else report the difference
        between chi2 of data and chi2 critical at alpha level

    Returns
    -------
    plot : matplotlib plot
    """

    cols = []

    args = {'annot': True, 'ax': None, 'annot_kws': {'size': 10},
            'cmap': plt.get_cmap('Blues', 20)}

    chi2_matrix = pd.DataFrame()
    chi2_alpha_matrix = pd.DataFrame()
    delta_chi2_matrix = pd.DataFrame()

    # produce upper matrix triangle including diagonal elements
    for c in list(itertools.product(columns, repeat=2)):
        if (c[1], c[0]) not in cols:
            cols.append(c)

    # create blank canvas
    fig, ax = plt.subplots(ncols=1, figsize=(16, 12))

    for col1, col2 in cols:
        # calculate contingency table
        contingency_table = pd.crosstab(index=data[col1].values,
                                        columns=data[col2].values)

        contingency_table.index.names = [col1]
        contingency_table.columns.name = col2

        # calculate chi2 scores and p-values for data
        chi2, p, dof, expected = scipy.stats.chi2_contingency(contingency_table,
                                                              correction=False)

        # fill upper matrix triangle
        if p_value is True:
            chi2_matrix.loc[col1, col2] = p if p > 0.000001 else 0
        else:
            chi2_matrix.loc[col1, col2] = chi2

        # fill lower matrix triangle
        if col1 != col2:
            chi2_matrix.loc[col2, col1] = chi2_matrix.loc[col1, col2]

        # calculate chi2 critical score for the given
        # degree of freedom of the contingency table
        # note: df = (nrows -1)*(ncols -1)
        df = np.prod(contingency_table.shape - np.array([1, 1]))

        # fill upper matrix triangle
        chi2_alpha_matrix.loc[col1, col2] = scipy.stats.chi2.ppf(alpha, df)

        # fill lower matrix triangle
        if col1 != col2:
            chi2_alpha_matrix.loc[col2, col1] = chi2_alpha_matrix.loc[col1, col2]

        # calculate delta chi2 between data and critical
        delta_chi2_matrix = (chi2_matrix - chi2_alpha_matrix)

    # set z-axis limits to reasonable values
    if p_value is True:
        vmin = 0
        vmax = 1
    else:
        vmin = min(delta_chi2_matrix.min().min(), 0)
        vmax = max(0, delta_chi2_matrix.max().max())

    # plot chi2 scores
    title = 'Delta Chi2'
    if p_value is True:
        title = title+' (p-values)'
        sns.heatmap(chi2_matrix, fmt='g',
                    vmin=vmin, vmax=vmax, **args)
    else:
        sns.heatmap(delta_chi2_matrix, fmt='g',
                    vmin=vmin, vmax=vmax, **args)

    # plot title
    plt.title(title, fontsize=14)

    # set x-axis and y-axis tick labels
    ax.set_xticklabels(columns, minor=False, ha="center")
    ax.set_yticklabels(columns, minor=False)

    # set x-axis tick label font size
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)

    # set y-axis tick label font size
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)

    # shift location of ticks to center of the bins
    ax.set_xticks(np.arange(len(columns))+0.5, minor=False)
    ax.set_yticks(np.arange(len(columns))+0.5, minor=False)

    # rotate ticks marks
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    return display(plt.show())


# f one way boxplot
def plot_f_oneway(data, catcol, numcol,alpha=0.95):
    """
    To calculate pairwise f statistic between features.
    
    Parameters
    ----------
    data : pandas dataframe
    columns : list
    alpha : float, default 0.95
        significance level
    p_value : boolean, default False
        If true report p-values else report the difference
        between chi2 of data and chi2 critical at alpha level
        
    the Fligner-Killeen:med X^2 test statistic.
    Returns
    -------
    plot : matplotlib plot
    """

    samples = []
        
    levels = data[catcol].unique()
    
    args = {'annot': True, 'ax': None, 'annot_kws': {'size': 10},
            'cmap': plt.get_cmap('Blues', 20)}

    for ls in levels:
        if isinstance(ls, str):
            qry = catcol+'=='+'"%s"' % ls
        else:
            qry = catcol+'=='+str(ls)

        samples.append(data.query(qry)[numcol])

    # calculate f score and p-value for data
    f_statistic, p_value1 = scipy.stats.f_oneway(*samples)
    fk_statistic, p_value2 = scipy.stats.fligner(*samples)

    # format p-values
    p_value1 = p_value1 if p_value1 > 0.000001 else 0
    p_value2 = p_value2 if p_value2 > 0.000001 else 0
    
    # calculate f critical for the
    # given degree of freedoms
    # note: dfn = K-1, dfd = N-K
    dfn = data[catcol].unique().shape[0]-1
    dfd = data.shape[0]-levels.shape[0]

    f_critical = scipy.stats.f.ppf(0.95, dfn, dfd)
        
    # plot f scores
    title = 'Boxplot (one-way ANOVA)'
    ax = data.boxplot(column=numcol, by=catcol,
                      figsize=(8, 6))

    # plot title
    plt.title(title, fontsize=14)
    plt.suptitle('')

    plt.xlabel(ax.xaxis.get_label_text(), fontsize=12)
    plt.ylabel(numcol, fontsize=12)

    # set x-axis tick label font size
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)

    # set y-axis tick label font size
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)

    f_score = '{0:.6g}'.format(f_statistic)
    f_critical = '{0:.6g}'.format(f_critical)
    p_value1 = '{0:.6g}'.format(p_value1)
    p_value2 = '{0:.6g}'.format(p_value2)
    
    label = 'f score (f_critical): %s (%s)' % (f_score, f_critical)
    f_patch = mpatches.Patch(label=label)
    
    label = 'f p-value (fk p-value): %s (%s)' % (p_value1, p_value2)
    p_values_patch = mpatches.Patch(label=label)

    plt.legend(handles=[f_patch, p_values_patch],
               loc='lower center', frameon=False,
               fancybox=True, fontsize=12)
    
    return display(plt.show())

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

    args = {"annot": True, "ax": ax1, "vmin": 0, "vmax": 1, "annot_kws": {"size": 8},
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

    return display(plt.show())
