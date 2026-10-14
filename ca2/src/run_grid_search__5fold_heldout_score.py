'''
Summary
-------
1. Select best hyperparameters (alpha, beta) of linear regression via a grid search
-- Use the score function of MAPEstimator on heldout set (average across K=5 folds).
2. Plot the best score found vs. polynomial feature order.
-- Normalize scale of log probabilities by dividing by train size N
3. Report test set performance of best overall model (alpha, beta, order)
4. Report overall time required for model selection

'''
import numpy as np
import pandas as pd
import time
import copy

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")
sns.set_context("notebook")

import sklearn.model_selection

import regr_viz_utils
from FeatureTransformPolynomial import PolynomialFeatureTransform
from LinearRegressionMAPEstimator import LinearRegressionMAPEstimator

def main(block=False):
    x_train_ND, t_train_N, x_test_ND, t_test_N = regr_viz_utils.load_dataset()

    # 3 sizes of train set to explore
    Nsm = 20
    Nmed = 60
    Nbig = 187

    # Coarse list of possible alpha values
    # Finer list of possible beta (likelihoods matter more)
    hypers_to_search = dict(
        order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        alpha=np.logspace(-8, 1, 10).tolist(),
        beta=np.logspace(-1, 1, 3 + 24).tolist(),
        )
    n_to_search_total = (len(hypers_to_search['order'])
         * len(hypers_to_search['alpha']) * len(hypers_to_search['beta']))

    print("Searching %6d possible hyperparam configs" % n_to_search_total)
    print("Possible alpha values")
    print(', '.join(['%.3g' % a for a in hypers_to_search['alpha']]))
    print("Possible beta values")
    print(', '.join(['%.3g' % a for a in hypers_to_search['beta']]))

    order_list = hypers_to_search['order']
    score_vs_N_fig, score_vs_N_axgrid = plt.subplots(
        nrows=2, ncols=1,
        figsize=(4,5), sharex=True, sharey=True)
    default_score = -3.0
    all_va_score_per_order = list()
    all_test_score_per_order = list()

    # Iterate over different train set sizes
    # For each one, do a full grid search to find best alpha/beta/order
    for N, line_color in [(Nsm, '#a1d99b'), (Nmed, '#31a354'), (Nbig, 'k')]:
        print("\n === Grid search for (alpha, beta, order) on N=%d train set" % N)
        score_per_order = list()
        test_score_per_order = list()
        estimator_per_order = list()
        start_time_sec = time.time()
        for order in order_list:
            cur_hypers_to_search = copy.deepcopy(hypers_to_search)
            cur_hypers_to_search['order'] = [order]

            feature_transformer = PolynomialFeatureTransform(order=order, input_dim=1)
            default_estimator = LinearRegressionMAPEstimator(feature_transformer)

            kfold_splitter = sklearn.model_selection.KFold(
                n_splits=5, shuffle=True, random_state=101)

            # Create grid searcher object that will use estimator's score function
            # TODO make sure you understand what each keyword arg does
            kfold_grid_searcher = sklearn.model_selection.GridSearchCV(
                LinearRegressionMAPEstimator(feature_transformer),
                cur_hypers_to_search, # only search alpha/beta
                cv=kfold_splitter,
                scoring=None,
                refit=True,
                return_train_score=False)

            # TODO Perform grid search on first N train points
            # Hint: call a method already provided by kfold_grid_searcher
            # After that call, the searcher stores the best average heldout score
            # and the best estimator as attributes (see sklearn's GridSearchCV docs)

            # Select best scoring parameters
            best_score = default_score + order/10 + (N-20)/200 # TODO FIXME
            best_estimator = default_estimator # TODO FIXME

            # TODO use best estimator to get score on test data
            test_score = default_score + order/10 + (N-20)/200 # TODO FIXME

            estimator_per_order.append(best_estimator)
            score_per_order.append(best_score)
            test_score_per_order.append(test_score)            

        # record test_score_per_order for autograder
        all_va_score_per_order.append(score_per_order)
        all_test_score_per_order.append(test_score_per_order)

        if N == Nsm:
            # Create Fig 2a
            key_order_list = [1, 4, 10]
            key_est_list = [estimator_per_order[oo] for oo in key_order_list]
            regr_viz_utils.make_fig_for_estimator(
                LinearRegressionMAPEstimator,
                order_list=key_order_list,
                alpha_list=[est.alpha for est in key_est_list],
                beta_list=[est.beta for est in key_est_list],
                x_train_ND=x_train_ND[:N],
                t_train_N=t_train_N[:N],
                x_test_ND=x_test_ND,
                t_test_N=t_test_N,
                num_stddev=2,
                color='g',
                legend_label='MAP +/- 2 stddev',
                )
            fig2a_path = 'fig2a-viz_predictions_N%d.jpg' % N
            plt.savefig(fig2a_path,
                bbox_inches='tight', pad_inches=0)
            print("Saved figure: ", fig2a_path)

        # Add line to Fig 2b
        plt.figure(score_vs_N_fig.number)
        score_vs_N_axgrid[0].plot(order_list, score_per_order, 
            color=line_color,
            linestyle='-',
            marker='s',
            label='N=%d' % N)
        score_vs_N_axgrid[0].set_ylabel('log lik. on val. (5 fold)')
        # Add small vertical bar to indicate maximum
        vert_bar_xs = np.zeros(2)
        vert_bar_ys = np.asarray([-0.2, +0.2])
        best_id = np.argmax(score_per_order)
        score_vs_N_axgrid[0].plot(
            vert_bar_xs + order_list[best_id],
            vert_bar_ys + score_per_order[best_id],
            linestyle='-',
            color=line_color)
        
        score_vs_N_axgrid[1].plot(order_list, test_score_per_order, 
            color=line_color,
            linestyle='-',
            marker='.', markersize=7,
            label='N=%d' % N)
        score_vs_N_axgrid[1].set_ylabel('log lik. on test')
        
        ## Report best performance of the best estimator across orders
        best_estimator_overall = estimator_per_order[best_id]

        # Summarize search
        print("Best Overall MAP at N=%d" % (N))
        print("order = %d" % order_list[best_id])
        print("alpha = %.3g" % best_estimator_overall.alpha)
        print("beta = %.3g" % best_estimator_overall.beta)
        print("test score = % 9.7f" % (
            best_estimator_overall.score(x_test_ND, t_test_N)))
        print("required time = %.2f sec" % (time.time() - start_time_sec))

    ## Finalize figure 2b
    plt.figure(score_vs_N_fig.number)
    plt.xlabel('polynomial order')
    plt.xticks(order_list)
    plt.legend(loc='upper left')
    plt.ylim([-2.8, -0.7]); 
    plt.yticks(np.arange(-2.5, -.7001, 0.3));
    plt.xticks([0, 2, 4, 6, 8, 10, 12]);
    for ax in score_vs_N_axgrid:
        ax.grid(axis='y')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    plt.tight_layout();
    fig2b_path = 'fig2b-score_vs_order.jpg'
    plt.savefig(fig2b_path,
        bbox_inches='tight', pad_inches=0)
    print("Saved figure: ", fig2b_path)
    plt.show(block=block)

    return all_va_score_per_order, all_test_score_per_order

if __name__ == '__main__':
    main(block=True)