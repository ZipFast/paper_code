import hashlib
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from param_preprocessor import ParamPreprocessor
from lightgbm import LGBMRegressor, plot_importance
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from run_loader import RunLoader

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

    def train(self, X=None, y=None, groups=None):
        X = self.X if X is None else X
        y = self.y if y is None else y
        groups = self.groups if groups is None else groups
        unique_groups = np.unique(groups)

        y_ = np.zeros_like(y)
        for g in unique_groups:
            indices = np.where(g == groups)[0]
            selection = np.array(y[indices])
            y_[indices] = StandardScaler().fit_transform(X=selection.reshape(-1, 1)).reshape(-1)

        self.model.fit(X, y_)

    def suggest_triple(self, mf):
        # We should keep the indices, so we can lookup the originals
        X_hp_ = self.X_hp.drop_duplicates()
        mfs = pd.DataFrame(np.repeat(mf.to_dict(), len(X_hp_)).tolist())
        X_ = pd.concat([X_hp_, mfs], axis=1)
        pred = MinMaxScaler().fit_transform(self.model.predict(X_).reshape(-1, 1)).reshape(-1)
        triple = [np.argmax((1 - pred) ** delta + pred) for delta in [0, 0.10, 0.20]]
        if triple[0] == triple[1]:
            print("Could not suggest useful features")
            triple = np.random.choice(self.X_runs, 3)
        else:
            triple = self.X_runs.iloc[triple].to_dict(orient="records")
        return triple

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

    def plot_importance(self, **kwargs):
        plot_importance(self.model, **kwargs)
        plt.show()

    @staticmethod
    def plot_correlation(X, y, method="pearson"):
        correlation = X.copy()
        correlation["y"] = y
        ax = plt.gca()
        plot = correlation.corr(method=method)["y"].sort_values().drop(["y"])
        plot = plot.plot.barh(ax=ax, color="#4C72B0")
        ax.set_title("Correlation")
        ax.set_xlabel(f"Correlation ({method})")
        ax.set_ylabel("Features")
        x_width_1 = plot.axes.viewLim.x1
        x_width_0 = plot.axes.viewLim.x0
        ax.set_xlim(x_width_0 - x_width_1 / 10, x_width_1 + x_width_1 / 10)
        for rect in plot.containers[0]:
            width = rect.get_width()
            x_width = plot.axes.viewLim.x1
            ha = "right" if width < 0 else "left"
            offset = -x_width / 100 if width < 0 else x_width / 100
            ax.text(x=width + offset, y=rect.xy[1] + 0.05, s=f"{width:.2}", color="black", ha=ha, va="bottom")
        plt.show()

    def std_over_group_means(self, X, groups):
        X["groups"] = groups
        ax = plt.gca()
        mean_per_group = X.groupby("groups").mean()
        variances = (mean_per_group / X.drop("groups", axis=1).mean()).std()
        variances = variances.sort_values()
        plot = variances.plot.barh(ax=ax, color="#4C72B0")
        ax.set_title("Standard deviation over normalized group means")
        ax.set_xlabel("$\sigma(\mu_{group} / \mu)$")
        ax.set_ylabel("Features")
        x_width = plot.axes.viewLim.x1
        ax.set_xlim(0, x_width + x_width / 10)
        for rect in plot.containers[0]:
            width = rect.get_width()

            ax.text(x=width + x_width / 100, y=rect.xy[1] + 0.05, s=f"{width:.3}", color="black", va="bottom")
        plt.show()

    def download_runs(self, flow_id, tasks=None, metric=None, max_per_task=5000):
        # Set metric
        if metric is None:
            metric = self.metric

        # Define path
        file = os.path.join(self.cache_dir, f"runs-{flow_id}.csv")

        # Load from cache if it exists
        if os.path.exists(file):
            with open(file, "r+") as f:
                frame = pd.read_csv(f, index_col=0)
                self.runs = frame
                return frame

        # Set tasks
        if tasks is None:
            tasks, _ = RunLoader.get_cc18_benchmarking_suite()

        # Load run-evaluations and save to cache
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

        # Set tasks
        if datasets is None:
            _, datasets = RunLoader.get_cc18_benchmarking_suite()

        # Make a short hash for the downloaded file
        datasets = np.unique(datasets)
        hash = hashlib.md5(str.encode("".join([str(i) for i in datasets]))).hexdigest()[:8]
        file = os.path.join(self.cache_dir, f"metafeatures-{hash}.csv")

        # Load file from cache
        if os.path.exists(file):
            with open(file, "r+") as f:
                frame = pd.read_csv(f, index_col=0)
                self.meta_features = frame
                return frame

        # Download qualities
        frame = RunLoader.load_meta_features(datasets)

        # Write to file
        frame.to_csv(file)

        self.meta_features = frame
        return frame

    def combine_features(self, X=None, groups=None, meta_features=None, save=True):
        X = self.X_hp if X is None else X
        groups = self.groups if groups is None else groups
        meta_features = self.meta_features if meta_features is None else meta_features

        mf = meta_features.T.to_dict()
        metas = pd.DataFrame([mf[g] for g in groups])
        combined = pd.concat([X, metas], axis=1)

        if save:
            self.X = combined
        return combined