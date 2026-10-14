'''
Summary
-------
Plot predicted mean + high confidence interval for MAP estimator
across different orders of the polynomial features
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("notebook")

import regr_viz_utils
from FeatureTransformPolynomial import PolynomialFeatureTransform
from LinearRegressionMAPEstimator import LinearRegressionMAPEstimator
from LinearRegressionPosteriorPredictiveEstimator import LinearRegressionPosteriorPredictiveEstimator


if __name__ == '__main__':    
    # Set training set size
    N = 20
    x_train_ND, t_train_N, x_val_VD, t_val_V = regr_viz_utils.load_dataset(
        val_size=100)

    # Polynomial orders to try
    for order in [1, 4, 7]:

        # Set precisions to try for likelihood
        beta_list = [0.2, 0.6, 1.8]
        L = len(beta_list)

        # Set precisions of prior (alpha)
        alpha_list = 0.001 * np.ones(L)
        order_list = order * np.ones(L, dtype=np.int32)
        score_list = []

        # Prepare 1-row, 3-col plot to look at different beta values
        map_fig, map_axgrid = regr_viz_utils.prepare_x_vs_t_fig(order_list)
        xgrid_G1 = regr_viz_utils.prepare_xgrid_G1(x_train_ND)

        # Loop over each panel of our 3-column plot
        for fig_col_id in range(L):
            order = order_list[fig_col_id]
            alpha = alpha_list[fig_col_id]
            beta = beta_list[fig_col_id]

            feature_transformer = PolynomialFeatureTransform(
                order=order, input_dim=1)

            # Train MAP estimator using only first N examples
            map_estimator = LinearRegressionMAPEstimator(
                feature_transformer, alpha=alpha, beta=beta)
            map_estimator.fit(x_train_ND[:N], t_train_N[:N])

            # Compute score on train and test
            map_tr_score = map_estimator.score(x_train_ND[:N], t_train_N[:N])
            map_va_score = map_estimator.score(x_val_VD, t_val_V)
            print("order %2d alpha %6.3f beta %6.3f : %8.4f tr score  %8.4f va score" % (
                order, alpha, beta, map_tr_score, map_va_score))
            score_list.append(map_va_score)

            # Obtain predicted mean and stddev for MAP estimator
            # at each x value in provided dense grid of size G
            map_mean_G = map_estimator.predict(xgrid_G1)
            map_var_G = map_estimator.predict_variance(xgrid_G1)
            map_stddev_G = 2.0 # TODO FIXME, go from variance to stddev

            cur_map_ax = map_axgrid[0, fig_col_id]
            regr_viz_utils.plot_predicted_mean_with_filled_stddev_interval(
                cur_map_ax, # plot on MAP figure's current axes
                xgrid_G1, map_mean_G, map_stddev_G,
                num_stddev=2,
                color='g',
                legend_label='MAP +/- 2 stddev')

        regr_viz_utils.finalize_x_vs_t_plot(
            map_axgrid, x_train_ND[:N], t_train_N[:N], x_val_VD, t_val_V,
            order_list, alpha_list, beta_list, score_list,
            heldout_label='validation data')
        plt.savefig("fig1b_order%02d_viz_predictions.jpg" % order,
            bbox_inches='tight', pad_inches=0)
        plt.show()
