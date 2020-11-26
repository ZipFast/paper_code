#%%
from run_loader import RunLoader
import json
import requests
from meta_learner import MetaLearner
from optimizer import Optimizer

flow_id = 6969

ml = MetaLearner()
ml.download_runs(flow_id)

X, X_conv, y, groups = ml.convert_runs_to_features()
# %%
X
# %%
X_conv
# %%
y
# %%
groups
# %%
grid = ml.suggest_grid()
grid
# %%
X[X.columns[0]].unique()
# %%
ml.download_meta_features()
# %%
ml.combine_features()
# %%
ml.train()
# %%
ml.plot_importance(max_num_features=15)

# %%
ml.plot_correlation(ml.X_hp, ml.groups)
# %%
ml.std_over_group_means(ml.X_hp, ml.groups)
# %%
input_meta_features = ml.meta_features.loc[16]
input_meta_features
# %%
triple = ml.suggest_triple(mf=input_meta_features)
triple

#%%
import openml
import numpy as np

task = openml.tasks.get_task(31)
X, y = task.get_X_and_y()
# Possible to change proposed grid
grid["randomforestclassifier__max_features"] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
grid["randomforestclassifier__random_state"] = [42]

# Setup optmizier
optimizer = Optimizer(pipeline=pipeline)
optimizer.setup(grid, X, y, cv=3)

# Feed triple given by warm-staring
optimizer.seed(triple)
# Start optimization loop
np.random.seed(43)
suggestion = optimizer.loop()

#%%
suggestion
#%%
