import pandas as pd

from sklearn.model_selection import train_test_split

from exploration import explore
from tree import run_cross_tree, run_single_tree
from glm import run_glm
from svr import run_svr

df = pd.read_csv("dataset-classified.csv", index_col="id")
feature_cols = ["lastLevel", "totalSessions", "totalAge", "winRate", "dailySessions"]
X_train, X_test, y_train, y_test = train_test_split(df[feature_cols], df["target"], test_size=0.3, random_state=1)

# Exploration
explore(df)

# Decision Tree
run_cross_tree(X_train, y_train)
run_single_tree(X_train, y_train, X_test, y_test, 4, True, feature_cols)
run_single_tree(X_train, y_train, X_test, y_test, 12, False, feature_cols)

# GLM
run_glm(X_train, y_train, X_test, y_test, feature_cols)

# GPR
run_svr(X_train, y_train, X_test, y_test, feature_cols)
