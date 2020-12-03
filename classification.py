import pandas as pd
from pandas import DataFrame as DF
import json
import os
from typing import List
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier


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
# train-test data
X_train, X_test, y_train, y_test = tr_df, te_df, tr_idx1, te_idx1

def fit_predict(clf) -> list:
    clf.fit(X_train, y_train)
    return clf.predict(X_test)

def grid_search_cv(model, param_grid, cv=4):
    gs = GridSearchCV(model(),
                      param_grid=param_grid,
                      scoring='accuracy', cv=cv, n_jobs=-1)
    gs.fit(X_train, y_train)
    return gs.best_params_
    
def format_output(input_list) -> List[str]:
    return [f'{item:<9}' for item in input_list]

def compare(l1) -> int:
    return sum(tup1 == tup2 for tup1, tup2 in zip(l1, y_test))


xgb_params = {'n_estimators': (2, 5, 10, 20, 30, 50),
              'learning_rate': (.01, .05, .1, .2, .3)}
xgb_params_best = grid_search_cv(XGBClassifier, xgb_params)
xgb_res = fit_predict(XGBClassifier(**xgb_params_best))
print(f'XGBoost:        {format_output(xgb_res)} - {compare(xgb_res)}')
print(f'Y-test:         {format_output(y_test)}')
