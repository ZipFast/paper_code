import hashlib
import pdb

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn import clone
from sklearn.model_selection import cross_val_score
from tqdm import tqdm


class Optimizer:
    def __init__(self, pipeline):
        self.model = LGBMRegressor(
            num_leaves=8, min_child_samples=1, min_data_in_bin=1, verbose=-1, objective="quantile",
            alpha=0.85
        )
        self.observed_X = []
        self.observed_y = []
        self.observed_hashes = set()
        self.pipeline = pipeline
        self.X = None
        self.y = None
        self.cv = None
        self.metric = None
        self.grid = None
        self.n_draws = None

    def setup(self, grid, X, y, cv, metric="accuracy", n_draws=500):
        self.grid = grid
        self.n_draws = n_draws
        self.X = X
        self.y = y
        self.cv = cv
        self.metric = metric

    def seed(self, configurations):
        for config in tqdm(configurations):
            self.observe(config)

    def loop(self, n_iter=150):
        best_sample = None
        best_score = 0
        for n in tqdm(range(n_iter)):
            sample = self.suggest()
            score = self.observe(sample)
            if score > best_score:
                best_sample = sample
            print(f"{score:.6}/{np.max(self.observed_y):.6}")
        return best_sample

    def observe(self, params):

        # Create hash (using np.nan's instead of None's)
        params = {i: np.nan if j is None else j for i, j in params.items()}
        hash = hashlib.md5(str(pd.Series(params).values).encode()).hexdigest()[:8]

        # Convert params (using None's instead of np.nan's)
        params = {i: None if isinstance(j, np.float) and np.isnan(j) else j for i, j in params.items()}

        # Setup pipeline
        pipeline = clone(self.pipeline).set_params(**params)
        scores = cross_val_score(estimator=pipeline, X=self.X, y=self.y, cv=self.cv, scoring=self.metric)
        score = np.mean(scores)
        self.observed_X.append(params)
        self.observed_y.append(score)

        assert(hash not in self.observed_hashes)
        self.observed_hashes.add(hash)

        return score

    def clean(self, X):
        return pd.get_dummies(X).astype(float)

    def sample_random(self, amount=500):
        values = {}
        for c, v in self.grid.items():
            if hasattr(v, "rvs"):
                values[c] = v.rvs(amount)
            else:
                indices = np.random.randint(len(v), size=amount)
                values[c] = np.array(v)[indices]
        converted = pd.DataFrame(values)
        return converted

    def suggest(self):

        # Some quick checks
        assert(len(self.observed_y) == len(self.observed_X))
        assert(len(self.observed_X) == len(self.observed_hashes))

        # Get random samples
        samples = self.sample_random()

        # Remove already observed
        hashes = [hashlib.md5(str(i).encode()).hexdigest()[:8] for i in samples.values]
        selection = [hash not in self.observed_hashes for hash in hashes]
        samples = samples[selection]

        # Clean samples and observed samples together
        frame = pd.concat([pd.DataFrame(self.observed_X), samples], sort=True)
        frame = self.clean(frame)

        # Extract
        num_observed = len(self.observed_hashes)
        X_ = frame.iloc[:num_observed]
        samples_ = frame.iloc[num_observed:]

        # Train model
        self.model.fit(X_, self.observed_y)
        predicted = self.model.predict(samples_)

        # Get suggested and save hash
        suggested = samples.iloc[np.argmax(predicted)]

        return suggested.to_dict()
