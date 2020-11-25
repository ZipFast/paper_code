#%%
from run_loader import RunLoader
import json
import requests
from meta_learner import MetaLearner

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
