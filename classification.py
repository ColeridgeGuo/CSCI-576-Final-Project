import pandas as pd
from pandas import DataFrame as DF
import json
import os
from typing import List
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
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

X_train, X_test, y_train, y_test = tr_df, te_df, tr_idx1, te_idx1

def fit_predict(model) -> list:
    clf = model()
    clf.fit(X_train, y_train)
    return clf.predict(X_test)
    
def format_output(input_list) -> List[str]:
    return [f'{item:<9}' for item in input_list]

def compare(l1, l2) -> int:
    return sum(tup1 == tup2 for tup1, tup2 in zip(l1, l2))


dt_res = fit_predict(DecisionTreeClassifier)
svm_res = fit_predict(SVC)
gnb_res = fit_predict(GaussianNB)
mnb_res = fit_predict(MultinomialNB)
adb_res = fit_predict(AdaBoostClassifier)
rf_res = fit_predict(RandomForestClassifier)
xgb_res = fit_predict(XGBClassifier)

print(f'Decision Tree:  {format_output(dt_res)} - {compare(dt_res, y_test)}')
print(f'SVM:            {format_output(svm_res)} - {compare(svm_res, y_test)}')
print(f'Gaussian NB:    {format_output(gnb_res)} - {compare(gnb_res, y_test)}')
print(f'Multinomial NB: {format_output(mnb_res)} - {compare(mnb_res, y_test)}')
print(f'AdaBoost:       {format_output(adb_res)} - {compare(adb_res, y_test)}')
print(f'Random Forest:  {format_output(rf_res)} - {compare(rf_res, y_test)}')
print(f'XGBoost:        {format_output(xgb_res)} - {compare(xgb_res, y_test)}')
print()
print(f'Y-test:         {format_output(y_test)}')
