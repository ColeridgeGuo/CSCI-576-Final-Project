import pandas as pd
from pandas import DataFrame as DF
import json
import os
from typing import List
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def list_files(test_train: str) -> List[str]:
    path = os.path.join(root_path, test_train)
    feat_paths = [p for p in os.listdir(path) if p.endswith('.json')]
    return feat_paths


def read_json(data_paths: list, test_train: str) -> dict:
    data = {}
    for pt in data_paths:
        with open(os.path.join(root_path, test_train, pt), 'r') as f:
            feat = json.load(f)
            data[feat["feature_name"]] = feat["values"]
    return data


def join_dfs(features: dict):
    dfs = [{cat: DF.from_dict(features[ft][cat], orient='index', columns=[ft])
            for cat in features[ft]} for ft in features]
    dfs = [pd.concat(df) for df in dfs]
    df = dfs[0].join(dfs[1:])
    df.sort_index(axis=1, inplace=True)
    return df


# list all json files of all features
root_path = "Data"
test_path, train_path = "test_data", "train_data"
test_feat_paths = list_files(test_path)
train_feat_paths = list_files(train_path)
# read all json files into dictionaries
test_feat = read_json(test_feat_paths, test_path)
train_feat = read_json(train_feat_paths, train_path)
# join dataframes into single dataframe
tr_df = join_dfs(train_feat)
te_df = join_dfs(test_feat)
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
