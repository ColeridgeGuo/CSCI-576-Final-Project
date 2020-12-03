import pandas as pd
from pandas import DataFrame as DF
import json
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

test_path = os.path.join("Data", "test_data")
train_path = os.path.join("Data", "train_data")
test_feat_paths = [p for p in os.listdir(test_path) if p.endswith('.json')]
train_feat_paths = [p for p in os.listdir(train_path) if p.endswith('.json')]

test_feat, train_feat = {}, {}
for pt in test_feat_paths:
    with open(os.path.join(test_path, pt), 'r') as f:
        feat = json.load(f)
        test_feat[feat["feature_name"]] = feat["values"]

for pt in train_feat_paths:
    with open(os.path.join(train_path, pt), 'r') as f:
        feat = json.load(f)
        train_feat[feat["feature_name"]] = feat["values"]

tr_dfs = [{cat: DF.from_dict(train_feat[ft][cat], orient='index', columns=[ft])
           for cat in train_feat[ft]} for ft in train_feat]
tr_dfs = [pd.concat(df) for df in tr_dfs]
tr_df = tr_dfs[0].join(tr_dfs[1:])
tr_df.sort_index(axis=1, inplace=True)
tr_df.to_json('train_output_df.json', indent=2)

te_dfs = [{cat: DF.from_dict(test_feat[ft][cat], orient='index', columns=[ft])
           for cat in test_feat[ft]} for ft in test_feat]
te_dfs = [pd.concat(df) for df in te_dfs]
te_df = te_dfs[0].join(te_dfs[1:])
te_df.sort_index(axis=1, inplace=True)
te_df.to_json('test_output_df.json', indent=2)

tr_idx1 = tr_df.index.get_level_values(0)
te_idx1 = te_df.index.get_level_values(0)

X_train, X_test, y_train, y_test = tr_df, te_df, tr_idx1, te_idx1


def decision_tree() -> list:
    dt_clf = DecisionTreeClassifier()
    dt_clf.fit(X_train, y_train)
    result = dt_clf.predict(X_test)
    return result


def svm():
    svm_clf = SVC()
    svm_clf.fit(X_train, y_train)
    result = svm_clf.predict(X_test)
    return result


dt_res = decision_tree()
svm_res = svm()
print(f'Decision Tree: {list(dt_res)}')
print(f'SVM:           {list(svm_res)}')
print(f'Y-test:        {list(y_test)}')
