import numpy as np
import pandas as pd
import requests
import tasks

BASE = "http://openml.org/api/v1/json"
all_qualities = []
for task in tasks.tasks:
    data = requests.get(BASE + f"/task/{task}").json()
    dataset_id = data['task']['input'][0]['data_set']['data_set_id']
    print(task)
    data = requests.get(BASE + f"/data/qualities/{dataset_id}").json()
    qualities = data['data_qualities']['quality']
    converted_qualities = {i['name']: i['value'] for i in qualities if not (isinstance(i['value'], list) or np.isnan(float(i['value'])))}
    all_qualities.append(converted_qualities)
pd.DataFrame(all_qualities).to_csv("results/metafeatures.csv")
