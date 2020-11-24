import json
import re

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


class RunLoader:
    """
    OpenML platform introduce REST api for developers to directly interact with Server
    """
    @staticmethod
    def build_query(task_id, flow_id):
        """
        according to the given task_id and flow_id return a formed query

        Args:
            task_id 
            flow_id 

        Returns:
            dict
        """
        return {
            "_source": [
                "run_id",
                "date",
                "run_flow.name",
                "run_flow.flow_id",
                "evaluations.evaluation_measure",
                "evaluations.value",
                "run_flow.parameters.parameter",
                "run_flow.parameters.value",
                "run_task.source_data.data_id"
            ],
            "query": {
                "bool": {
                    "must": [
                        {
                            "term": {
                                "run_task.task_id": task_id
                            },
                        },
                        {
                            "term": {
                                "run_flow.flow_id": flow_id
                            },
                        },
                        {
                            "nested": {
                                "path": "evaluations",
                                "query": {
                                    "exists": {
                                        "field": "evaluations"
                                    }
                                }
                            }
                        }
                    ]
                }
            },
            "sort": {
                "date": "asc"
            }
        }

    @staticmethod
    def load_task_runs(task_id, flow_id, metric, size=5000):
        # 构造query
        query = RunLoader.build_query(task_id, flow_id)
        url = f"https://www.openml.org/es/run/run/_search?size={size}"
        result = json.loads(requests.request(method="post", url=url, json=query).content)
        
        if len(result["hits"]["hits"]) == 0:
            return None, []

        # Get parameter names
        columns = [i["parameter"] for i in result["hits"]["hits"][0]["_source"]["run_flow"]["parameters"]]
        converted = [RunLoader.convert_param_name(i) for i in columns]

        # Construct header
        header = ["data_id", "task_id", "run_id", metric, *converted]

        # Get all values
        values = [
            # Dataset id
            [int(result["hits"]["hits"][0]["_source"]["run_task"]["source_data"]["data_id"])] +

            # Task id
            [int(task_id)] +

            # Run id
            [int(sample["_source"]["run_id"])] +

            # Metric
            [
                [i["value"] for i in sample["_source"]["evaluations"] if i["evaluation_measure"] == metric]
                or [None]
            ][0] +

            # Parameter values
            [json.loads(i["value"]) for i in sample["_source"]["run_flow"]["parameters"]]

            for sample in result["hits"]["hits"]
        ]

        return header, values

    @staticmethod
    def load_tasks(tasks, flow_id, metric="predictive_accuracy", max_per_task=5000):
        all = []
        columns = None
        for task in tqdm(tasks):
            header, values = RunLoader.load_task_runs(task_id=task, flow_id=flow_id, metric=metric, size=max_per_task)
            if values:
                columns = header
                all += values
        return pd.DataFrame(all, columns=columns)

    @staticmethod
    def load_meta_features(groups=None):

        if groups is None:
            _, groups = RunLoader.get_cc18_benchmarking_suite()

        all_qualities = []

        for dataset_id in tqdm(groups):
            url = f"http://openml.org/api/v1/json/data/qualities/{dataset_id}"
            data = requests.get(url).json()
            qualities = data['data_qualities']['quality']
            converted_qualities = {i['name']: i['value'] for i in qualities if
                                   not (isinstance(i['value'], list) or np.isnan(float(i['value'])))}
            all_qualities.append(converted_qualities)

        return pd.DataFrame(all_qualities, index=groups)

    @staticmethod
    def convert_param_name(param_name):
        """
        Examples:
        sklearn.feature_selection.variance_threshold.VarianceThreshold(4)_threshold
        --> variancethreshold__threshold

        (...).AdaBoostClassifier(base_estimator=(...).DecisionTreeClassifier)(2)_random_state
        --> decisiontreeclassifier__random_state

        :param param_name: Parameter name to convert
        :return: (str) Converted parameter name

        """
        splits = re.compile(r"(?:\(.*\))+_").split(param_name)
        prefix = splits[0].split(".")[-1].lower()
        postfix = splits[1]
        result = f"{prefix}__{postfix}"
        return result

    @staticmethod
    def get_cc18_benchmarking_suite():
        tasks = [146825, 146800, 146822, 146824, 167119, 146817, 14954, 37, 219, 9964, 3573, 12, 9957, 14970, 9946, 31,
                 3021, 146195, 18, 29, 11, 53, 6, 23, 2079, 14969, 3918, 3902, 9976, 15, 16, 32, 125922, 167120, 167121,
                 167124, 167125, 146819, 167141, 9910, 14952, 146818, 167140, 146820, 9952, 3904, 14, 49, 2074, 3022,
                 3481, 43, 3903, 9971, 3, 28, 9978, 7592, 3549, 22, 9985, 9960, 3913, 9977, 3560, 10101, 45, 10093,
                 146821, 3917, 9981, 125920, 14965]

        datasets = [11, 12, 14, 15, 16, 18, 54, 3, 6, 32, 37, 38, 44, 46, 50, 28, 29, 22, 23, 182, 188, 300, 307, 458,
                    469, 554, 1049, 1050, 1067, 1068, 1053, 1590, 1485, 1486, 1487, 1475, 1478, 1480, 1461, 1462, 1468,
                    1501, 1510, 1494, 1497, 4534, 4538, 6332, 23381, 23517, 40994, 40996, 41027, 40668, 40670, 40701,
                    1489, 40923, 40927, 40975, 40978, 40979, 40981, 40982, 40983, 40984, 40966, 40499, 4134, 1063, 1464,
                    31, 151]

        return tasks, datasets

    @staticmethod
    def convert_runs_to_features(frame, metric="predictive_accuracy"):
        copied = frame.copy()
        datasets = copied.pop("data_id")

        if "task_id" in frame.columns:
            _ = copied.pop("task_id")

        if "run_id" in frame.columns:
            _ = copied.pop("run_id")

        y = copied.pop(metric)
        X = copied.copy()

        renames = {
            "decisiontreeclassifier__min_impurity_split": "decisiontreeclassifier__min_impurity_decrease",
            "randomforestclassifier__min_impurity_split": "randomforestclassifier__min_impurity_decrease",
            "onehotencoder__n_values": "onehotencoder__categories"
        }
        copied = copied.rename(renames, axis=1)
        X = X.rename(renames, axis=1)

        # Drop columns we can not use
        # num_unique = copied.nunique()
        for c in copied.columns:
            if any(i in c for i in [
                "random_state", "n_jobs", "verbose", "warm_start", "categorical_features", "dtype", "sparse",
                "missing_values", "class_weight"
            ]):
                del copied[c]
                del X[c]
                print("Removed", c, "(inactive parameter)")
            elif copied.dtypes[c] == object and RunLoader.is_json(copied[c][0]):
                if RunLoader.is_number(copied[c][0]):
                    is_number = np.array([RunLoader.is_number(i) for i in copied[c]])
                    numeric = np.where(is_number)[0]
                    nominal = np.where(~is_number)[0]
                    num_values = np.full_like(copied[c], np.nan)
                    nom_values = np.full_like(copied[c], np.nan)
                    num_values[numeric] = np.array(copied[c])[numeric].astype(float)
                    nom_values[nominal] = np.array(copied[c])[nominal].astype(str)
                    copied[c + "__num"] = num_values
                    copied[c + "__nom"] = nom_values
                    copied[c + "__num"] = copied[c + "__num"].astype(float)
                    del copied[c]
                    print("Divided", c)

                    # Also do something for X (so numeric values are not treated as strings)
                    num_values[nominal] = nom_values[nominal]
                    X[c] = num_values
                else:
                    print("Removed", c, "(object)")
                    del copied[c]
            elif copied[c].nunique() <= 1:
                print("Removed", c, "(1 unique value)")
                del copied[c]

        X_conv = pd.get_dummies(copied).astype(float)
        return X, X_conv, y, datasets

    @staticmethod
    def is_json(s):
        try:
            json.loads(s)
        except ValueError:
            return False
        return True

    @staticmethod
    def is_number(s):
        try:
            float(s)
        except ValueError:
            return False
        return True
 