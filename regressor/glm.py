from math import sqrt

import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import pandas

import matplotlib.pyplot as plt


def run_glm(X_train, y_train, X_test, y_test, feature_cols):
    df = pandas.concat([X_train, y_train], axis=1)
    formula = f'target ~ {"+".join(feature_cols)}'
    model = smf.glm(formula=formula, data=df, family=sm.families.Gaussian())
    result = model.fit()
    print(result.summary())
    print("Coefficeients")
    print(result.params)
    print()
    print("p-Values")
    print(result.pvalues)
    print()
    print("Dependent variables")
    print(result.model.endog_names)
    y_predictions = result.predict(X_test)
    print('RMSE GLM, Test Set: ', sqrt(mean_squared_error(y_test, y_predictions)))

    def resid(row):
        if row["pred"] <= 0.0:
            return 0.0
        return row["true"] - row["pred"]

    resid_df = pandas.concat([y_test, y_predictions], keys=["true", "pred"], axis=1)
    resid_df["residual"] = resid_df.apply(resid, axis=1)

    print(resid_df.describe(include="all").transpose())

    plt.scatter(resid_df["true"][:round(len(resid_df)/30)], resid_df["residual"][:round(len(resid_df)/30)], alpha=0.5)
    plt.savefig(f"plots/residplot-glm.png")
    plt.clf()

