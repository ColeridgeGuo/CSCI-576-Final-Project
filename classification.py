import pandas as pd
from pandas import DataFrame as DF
import json
import os
import numpy as np

data_path = "output_data"
features_paths = [p for p in os.listdir(data_path) if p.endswith('.json')]

features = {}
for pt in features_paths:
    with open(os.path.join(data_path, pt), 'r') as f:
        feat = json.load(f)
        features[feat["feature_name"]] = feat["values"]

for ft in features:
    for cat in features[ft]:
        for vid in features[ft][cat]:
            features[ft][cat][vid] = float(features[ft][cat][vid])

dfs = [{cat: DF.from_dict(features[ft][cat], orient='index', columns=[ft])
        for cat in features[ft]} for ft in features]
dfs = [pd.concat(df) for df in dfs]
df = dfs[0].join(dfs[1:])
df.sort_index(axis=1, inplace=True)
df.to_json('output_df.json', indent=2)
