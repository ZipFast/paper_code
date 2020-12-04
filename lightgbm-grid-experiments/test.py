from random import shuffle
from lightgbm import LGBMClassifier
import openml
import itertools
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# We want to exclude tasks that have a dimensionality of more than 9 million
excluded_tasks = [
    # Id     # Dataset name             # Dimensionality #
    # ------ # ------------------------ # -------------- #
    9981,    # cnae-9                   # 925560         #
    9976,    # madelon                  # 1302600        #
    167120,  # Numerai28.6              # 2119040        #
    146195,  # Connect-4                # 2904951        #
    9977,    # nomao                    # 4101335        #
    3481,    # isolet                   # 4818546        #
    167125,  # Internet-Advertisements  # 5111961        #
    14970,   # har                      # 5788038        #
    9910,    # Bioresponse              # 6665527        #
    3573,    # Mnist_784                # 54950000       #
    146825,  # Fashion-MNIST            # 54950000       #
    167121,  # Devnagari-Script         # 94300000       #
    167124,  # CIFAR_10                 # 184380000      #

]

tasks = [i for i in openml.study.get_study(99).tasks if i not in excluded_tasks]
model = LGBMClassifier(verbose=-1)
grid = {
    "n_estimators": [100],
    "learning_rate": [0.001, 0.0055, 0.01, 0.055, 0.1],
    "num_leaves": [4, 8, 16, 32, 64, 128],
    "reg_alpha": [0, 0.1, 0.2],
    "reg_lambda": [0, 0.1, 0.2],
    "min_child_samples": [1, 20, 100],
    "max_depth": [4, 6, 8, 10, 12]
}

# Gather Experiments
print("[SEARCH] Preparing experiments")
experiments = []
keys, values = zip(*grid.items())
for v in itertools.product(*values):
    experiments.append(dict(zip(keys, v)))
shuffle(experiments)

for task_id in tasks:
    print("[SEARCH] Downloading task")
    task = openml.tasks.get_task(task_id)

    print("[SEARCH] Downloading data")
    X, y = task.get_X_and_y()

    print("[SEARCH] Downloading splits")
    splits = task.download_split().split

    print("[SEARCH] Starting experiments")
    for experiment in tqdm(experiments, ascii=True):
        model = LGBMClassifier(verbose=-1, **experiment)
        scores = []
        for folds in splits.values():
            for fold in folds.values():
                for repeat in fold.values():
                    train_index, test_index = repeat.train, repeat.test
                    model.fit(X[train_index], y[train_index])
                    y_pred = model.predict(X[test_index])
                    scores.append(accuracy_score(y[test_index], y_pred))

        # run = openml.runs.run_model_on_task(task, model)
        # score = np.mean(run.get_metric_fn(sklearn.metrics.accuracy_score))
        mean = np.mean(scores)
        print("")
        print(experiment)
        print(f'Data set: {task_id}; Accuracy: {mean}')
