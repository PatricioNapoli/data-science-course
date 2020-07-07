from math import sqrt

import pandas
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
import matplotlib.pyplot as plt


def run_svr(X_train, y_train, X_test, y_test, feature_cols):
    regr = SVR(C=1.0, epsilon=0.2)
    regr.fit(X_train, y_train)

    y_predictions = regr.predict(X_test)
    y_predictions_series = pandas.Series(y_predictions, index=y_test.index)

    print("SVR RMSE Score: ", sqrt(mean_squared_error(y_test, y_predictions)))

    def resid(row):
        if row["pred"] <= 0.0:
            return 0.0
        return row["true"] - row["pred"]

    resid_df = pandas.concat([y_test, y_predictions_series], keys=["true", "pred"], axis=1)
    resid_df["residual"] = resid_df.apply(resid, axis=1)

    print(resid_df.describe(include="all").transpose())

    plt.scatter(resid_df["true"][:round(len(resid_df)/30)], resid_df["residual"][:round(len(resid_df)/30)], alpha=0.5)
    plt.savefig(f"plots/residplot-svr.png")
    plt.clf()
