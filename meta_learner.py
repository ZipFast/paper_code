import hashlib
import os
from lightgbm import LGBMRegressor, plot_importance
from run_loader import RunLoader
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
import scipy
from param_preprocessor import ParamPreprocessor

plt.style.use("seaborn")

class MetaLearner:
    def __init__(self, cache_dir="cache", metric="predictive_accuracy"):
        self.cache_dir = cache_dir
        self.metric = metric
        self.model = LGBMRegressor(n_estimators=500, num_leaves=16, learning_rate=0.05, min_child_samples=1, verbose=-1)
        self.preprocessor = ParamPreprocessor()

        self.X_hp = None 
        self.X_runs = None 
        self.X = None 
        self.y = None 
        self.groups = None 
        self.meta_features = None 
        self.runs = None 

        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)

    def download_runs(self, flow_id, tasks=None, metric=None, max_per_task=5000):
        # set metric 
        if metric is None:
            metric = self.metric 
        
        # define path 
        file = os.path.join(self.cache_dir, f"funs-{flow_id}.csv")

        #load from cache if it exists 
        if os.path.exists(file):
            with open(file, "r+") as f:
                frame = pd.read_csv(f, index_col=0)
                self.runs = frame 
                return frame  
        
        # Set tasks 
        if tasks is None:
            tasks, _ = RunLoader.get_cc18_benchmarking_suite()
        
        # load run-evaluations and save to cache 
        frame = RunLoader.load_tasks(tasks, flow_id, metric=metric, max_per_task=max_per_task)
        frame.to_csv(file)

        self.runs = frame 
        return frame

    def convert_runs_to_features(self, frame=None, metric=None):
        frame = self.runs if frame is None else frame 
        if metric is None:
            metric = self.metric
        X, X_conv, y, groups = RunLoader.convert_runs_to_features(frame, metric)
        self.X_runs = X
        self.X_hp = X_conv 
        self.y = y 
        self.groups = groups
        return X, X_conv, y, groups
    
    def download_meta_features(self, datasets=None):
        if datasets is None:
            _, datasets = RunLoader.get_cc18_benchmarking_suite()
        
        datasets = np.unique(datasets)
        hash = hashlib.md5(str.encode("".join([str(i) for i in datasets]))).hexdigest()[:8]
        file = os.path.join(self.cache_dir, f"metafeatures-{hash}.csv")

        if os.path.exists(file):
            with open(file, "r+") as f:
                frame = pd.read_csv(f, index_col=0)
                self.meta_features = frame 
                return frame
        
        frame = RunLoader.load_meta_features(datasets)
        frame.to_csv(file)
        self.meta_features = frame 
        return frame
    
    def suggest_grid(self):
        grid = {}
        for c in self.X_runs.columns:
            unique_values = self.X_runs[c].unique()

            if len(unique_values) > 30:
                selection = [RunLoader.is_number(s) for s in unique_values]
                unique_values = np.array(unique_values)[selection]
                min = np.round(np.min(unique_values))
                scale = np.round(np.max(unique_values) - min)
                scale = np.maximum(scale + 1 - scale, scale)
                grid[c] = scipy.stats.uniform(min, scale)
            else:
                try:
                    unique_values_ = np.sort(unique_values)
                except TypeError:
                    unique_values_ = unique_values
                unique_values = unique_values_
                grid[c] = unique_values
        return grid