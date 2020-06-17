import pandas as pd
from dtreeviz.trees import dtreeviz

from sklearn import tree
from sklearn.model_selection import train_test_split

df = pd.read_csv("dataset-classified.csv", index_col="id")
del df["dailySessions"]

feature_cols = ["lastLevel", "totalSessions", "totalAge", "winRate"]
X = df[feature_cols]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

model = tree.DecisionTreeRegressor(max_depth=4)
model_fit = model.fit(X_train, y_train)

print(f"Score: {model_fit.score(X_train, y_train)}")

del df["target"]

viz = dtreeviz(model_fit,
               X_train,
               y_train,
               fancy=False,
               target_name='chance',
               feature_names=feature_cols,
               class_names=["nonpayer", "payer"])

viz.save("plot.svg")
