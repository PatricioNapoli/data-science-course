from math import sqrt

import pandas
from sklearn.metrics import mean_squared_error
from sklearn import tree
from sklearn.model_selection import cross_val_score

import numpy as np
import matplotlib.pyplot as plt
import seaborn

from dtreeviz.trees import dtreeviz


def run_cross_validation_on_trees(X, y, tree_depths, cv=5):
    cv_scores_list = []
    cv_scores_std = []
    cv_scores_mean = []
    accuracy_scores = []
    for depth in tree_depths:
        tree_model = tree.DecisionTreeRegressor(max_depth=depth)
        cv_scores = cross_val_score(tree_model, X, y, cv=cv)
        cv_scores_list.append(cv_scores)
        cv_scores_mean.append(cv_scores.mean())
        cv_scores_std.append(cv_scores.std())
        accuracy_scores.append(tree_model.fit(X, y).score(X, y))
    cv_scores_mean = np.array(cv_scores_mean)
    cv_scores_std = np.array(cv_scores_std)
    accuracy_scores = np.array(accuracy_scores)
    return cv_scores_mean, cv_scores_std, accuracy_scores


def plot_cross_validation_on_trees(depths, cv_scores_mean, cv_scores_std, accuracy_scores, title):
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.plot(depths, cv_scores_mean, '-o', label='promedio precisión cross-validation', alpha=0.9)
    ax.fill_between(depths, cv_scores_mean - 2 * cv_scores_std, cv_scores_mean + 2 * cv_scores_std, alpha=0.2)
    ylim = plt.ylim()
    ax.plot(depths, accuracy_scores, '-*', label='precisión entrenamiento', alpha=0.9)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Niveles', fontsize=14)
    ax.set_ylabel('Precisión', fontsize=14)
    ax.set_ylim(ylim)
    ax.set_xticks(depths)
    ax.legend()
    plt.savefig("plots/cross-validation-tree.png")
    plt.clf()


def run_cross_tree(X_train, y_train):
    sm_tree_depths = range(1, 20)
    sm_cv_scores_mean, sm_cv_scores_std, sm_accuracy_scores = run_cross_validation_on_trees(X_train, y_train,
                                                                                            sm_tree_depths)

    plot_cross_validation_on_trees(sm_tree_depths, sm_cv_scores_mean, sm_cv_scores_std, sm_accuracy_scores,
                                   'Precisión de cada nivel de árbol de decisión')

    idx_max = sm_cv_scores_mean.argmax()
    sm_best_tree_depth = sm_tree_depths[idx_max]
    sm_best_tree_cv_score = sm_cv_scores_mean[idx_max]
    sm_best_tree_cv_score_std = sm_cv_scores_std[idx_max]
    print('The depth-{} tree achieves the best mean cross-validation accuracy {} +/- {}% on training dataset'.format(
        sm_best_tree_depth, round(sm_best_tree_cv_score * 100, 5), round(sm_best_tree_cv_score_std * 100, 5)))


def run_single_tree(X_train, y_train, X_test, y_test, depth, plot_tree, feature_cols):
    model = tree.DecisionTreeRegressor(max_depth=depth).fit(X_train, y_train)
    accuracy_train = model.score(X_train, y_train)
    accuracy_test = model.score(X_test, y_test)

    y_predictions = model.predict(X_test)
    y_predictions_series = pandas.Series(y_predictions, index=y_test.index)

    def resid(row):
        return row["true"] - row["pred"]

    resid_df = pandas.concat([y_test, y_predictions_series], keys=["true", "pred"], axis=1)
    resid_df["residual"] = resid_df.apply(resid, axis=1)

    plt.scatter(resid_df["true"][:round(len(resid_df)/30)], resid_df["residual"][:round(len(resid_df)/30)], alpha=0.5)
    plt.savefig(f"plots/residplot-depth-{depth}.png")
    plt.clf()

    print('Single tree depth: ', depth)
    print('Accuracy, Training Set: ', round(accuracy_train * 100, 5), '%')
    print('Accuracy, Test Set: ', round(accuracy_test * 100, 5), '%')
    print('RMSE Tree, Test Set: ', sqrt(mean_squared_error(y_test, y_predictions)))

    if plot_tree:
        viz = dtreeviz(model,
                       X_train,
                       y_train,
                       fancy=False,
                       target_name='chance',
                       feature_names=feature_cols)

        viz.save("plots/plot.svg")
    return accuracy_train, accuracy_test
