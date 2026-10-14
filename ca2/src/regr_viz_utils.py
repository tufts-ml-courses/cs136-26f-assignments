import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import os

from FeatureTransformPolynomial import PolynomialFeatureTransform

# Allow user to specify location of csv files with DATA_DIR env var
# but by default, use folder structure distributed with source code
src_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get('DATA_DIR', 
    os.path.join(os.path.dirname(src_dir), 'traffic_data'))

def load_dataset(seed=123, val_size=0, data_dir=DATA_DIR):
    # Load and unpack training and test data
    train_csv_fpath = os.path.join(data_dir, 'traffic_train.csv')
    test_csv_fpath = os.path.join(data_dir, 'traffic_test.csv')
    if not os.path.exists(train_csv_fpath):
        raise FileNotFoundError("Please set DATA_DIR. Cannot find CSV files at path: ",
            train_csv_fpath)

    train_df = pd.read_csv(train_csv_fpath)
    test_df = pd.read_csv(test_csv_fpath)
    x_train_ND = train_df['x'].values[:,np.newaxis]
    t_train_N = train_df['y'].values

    random_state = np.random.RandomState(int(seed))
    shuffle_ids = random_state.permutation(t_train_N.size)
    x_train_ND = x_train_ND[shuffle_ids]
    t_train_N = t_train_N[shuffle_ids]

    x_test_ND = test_df['x'].values[:,np.newaxis]
    t_test_N = test_df['y'].values

    if val_size == 0:
        return x_train_ND, t_train_N, x_test_ND, t_test_N
    else:
        assert val_size > 0
        V = int(val_size)
        x_val_VD, t_val_V = x_train_ND[-V:], t_train_N[-V:]
        x_train_ND, t_train_N = x_train_ND[:-V], t_train_N[:-V]
        return x_train_ND, t_train_N, x_val_VD, t_val_V


def make_fig_for_estimator(
        Estimator,
        order_list, alpha_list, beta_list,
        x_train_ND, t_train_N,
        x_test_ND, t_test_N,
        test_scores_list=None,
        color='b',
        num_stddev=2,
        legend_label='MAP prediction',
        ):
    ''' Create figure showing estimator's predictions across orders

    Returns
    -------
    None

    Post Condition
    --------------
    Creates matplotlib figure
    '''
    fig, axgrid = prepare_x_vs_t_fig(order_list)
    xgrid_G1 = prepare_xgrid_G1(x_train_ND)

    # Loop over order of polynomial features
    # and associated axes of our plots
    for fig_col_id in range(len(order_list)):
        order = order_list[fig_col_id]
        alpha = alpha_list[fig_col_id]
        beta = beta_list[fig_col_id]
        cur_ax = axgrid[0, fig_col_id]

        feature_transformer = PolynomialFeatureTransform(
            order=order, input_dim=1)

        estimator = Estimator(
            feature_transformer, alpha=alpha, beta=beta)
        estimator.fit(x_train_ND, t_train_N)

        # Obtain predicted mean and stddev for estimator
        # at each x value in provided dense grid of size G
        mean_G = estimator.predict(xgrid_G1)
        var_G = estimator.predict_variance(xgrid_G1)

        plot_predicted_mean_with_filled_stddev_interval(
            cur_ax, # plot on figure's current axes
            xgrid_G1, mean_G, np.sqrt(var_G),
            num_stddev=num_stddev,
            color=color,
            legend_label=legend_label)
    finalize_x_vs_t_plot(
        axgrid, x_train_ND, t_train_N, x_test_ND, t_test_N,
        order_list, alpha_list, beta_list)


def prepare_x_vs_t_fig(
        order_list,
        ):
    ''' Prepare figure for visualizing predictions on top of train data

    Returns
    -------
    fig : figure handle object
    axgrid : axis grid object
    '''
    nrows = 1
    ncols = len(order_list)
    Wpanel = 4
    Hpanel = 3
    fig1, fig1_axgrid = plt.subplots(
        nrows=nrows, ncols=ncols,
        sharex=True, sharey=True, squeeze=False,
        figsize=(Wpanel * ncols, Hpanel * nrows))
    return fig1, fig1_axgrid


def prepare_xgrid_G1(
        x_train_ND,
        G=301,
        extrapolation_width_factor=0.5,
        ):
    '''
    
    Returns
    -------
    xgrid_G1 : 2D array, shape (G, 1)
        Grid of x points for making predictions 
    '''

    # To visualize prediction function learned from data,            
    # Create dense grid of G values between x.min() - R, x.max() + R
    # Basically, 2x as wide as the observed data values for 'x'
    xmin = x_train_ND[:,0].min()
    xmax = x_train_ND[:,0].max()
    R = extrapolation_width_factor * (xmax - xmin)
    xgrid_G = np.linspace(xmin - R, xmax + R, G)
    xgrid_G1 = np.reshape(xgrid_G, (G, 1))
    return xgrid_G1


def plot_predicted_mean_with_filled_stddev_interval(
        ax, xgrid_G1, t_mean_G, t_stddev_G,
        num_stddev=3,
        color='b',
        legend_label='MAP prediction',
        ):
    xgrid_G = np.squeeze(xgrid_G1)
    # Plot predicted mean and +/- num_stddev std dev interval
    ax.fill_between(
        xgrid_G,
        t_mean_G - num_stddev * t_stddev_G,
        t_mean_G + num_stddev * t_stddev_G,
        facecolor=color, alpha=0.2)
    ax.plot(
        xgrid_G, t_mean_G, 
        linestyle='-',
        color=color,
        label=legend_label)


def finalize_x_vs_t_plot(
        axgrid, 
        x_train_ND, t_train_N,
        x_test_ND, t_test_N,
        order_list=None,
        alpha_list=None,
        beta_list=None,
        score_list=None,
        transparency_level=0.2,
        heldout_label='test data'):
    # Make figure beautiful
    for ii, ax in enumerate(axgrid.flatten()):
        ax.plot(x_test_ND, t_test_N, 
            'r.', markersize=6, label=heldout_label, mew=0, markeredgecolor='none',
            alpha=transparency_level)

        ax.plot(x_train_ND, t_train_N,
            'k.', markersize=9, label='train data', mew=0, markeredgecolor='none')

        alpha = alpha_list[ii]
        beta = beta_list[ii]
        order = order_list[ii]
        if score_list is None:
            ax.set_title(
                "order = %d \n alpha=% .3g  beta=% .3g" % (
                order, alpha, beta))
        else:
            ax.set_title(
                "order=%d  score=% .3g\n alpha=% .3g  beta=% .3g" % (
                order, score_list[ii], alpha, beta))            
        ax.set_ylim([-2, 12])
        ax.set_xlim([-1.75, 1.6]) # little to left for legend
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_xticklabels([-24, -12, 0, 12, 24])
        ax.set_yticks([0, 3, 6, 9])
        ax.grid(axis='y')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        #ax.set_aspect('equal', 'box')
        ax.set_xlabel("hours since Mon midnight")
        if ii == 0:
            ax.set_ylabel("traffic count (1000s)")
            ax.legend(loc='upper right', fontsize=0.9*plt.rcParams['legend.fontsize'])
    plt.subplots_adjust(top=0.85, bottom=0.19)
