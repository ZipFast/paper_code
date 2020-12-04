from run_loader import RunLoader
import json
import requests
from meta_learner import MetaLearner
from optimizer import Optimizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import openml
from preprocessing import CategoricalImputer
pipeline = make_pipeline(
    RandomForestClassifier(),
)

ml = MetaLearner()
ml.download_runs(6969)
ml.convert_runs_to_features()
grid = ml.suggest_grid()
ml.download_meta_features()
grid["randomforestclassifier__max_features"] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
grid["randomforestclassifier__random_state"] = [42]
ml.combine_features()
ml.train()
triple = ml.suggest_triple(mf=ml.meta_features.loc[31])
# print(triple)
nan = None
triple = [
    {'randomforestclassifier__bootstrap': True, 'randomforestclassifier__class_weight': nan, 'randomforestclassifier__criterion': 'entropy', 'randomforestclassifier__max_depth': nan, 'randomforestclassifier__max_features': 0.6593746771833334, 'randomforestclassifier__max_leaf_nodes': nan, 'randomforestclassifier__min_impurity_decrease': 1e-07, 'randomforestclassifier__min_samples_leaf': 2, 'randomforestclassifier__min_samples_split': 20, 'randomforestclassifier__min_weight_fraction_leaf': 0.0, 'randomforestclassifier__n_estimators': 100, 'randomforestclassifier__oob_score': False, 'conditionalimputer__axis': 0, 'conditionalimputer__copy': True, 'conditionalimputer__fill_empty': 0, 'conditionalimputer__missing_values': "NaN", 'conditionalimputer__strategy': 'median', 'conditionalimputer__strategy_nominal': 'most_frequent', 'onehotencoder__handle_unknown': 'ignore', 'onehotencoder__categories': 'auto', 'variancethreshold__threshold': 0.0},
    {'randomforestclassifier__bootstrap': True, 'randomforestclassifier__class_weight': nan, 'randomforestclassifier__criterion': 'entropy', 'randomforestclassifier__max_depth': nan, 'randomforestclassifier__max_features': 0.27070581347378087, 'randomforestclassifier__max_leaf_nodes': nan, 'randomforestclassifier__min_impurity_decrease': 1e-07, 'randomforestclassifier__min_samples_leaf': 5, 'randomforestclassifier__min_samples_split': 14, 'randomforestclassifier__min_weight_fraction_leaf': 0.0, 'randomforestclassifier__n_estimators': 100, 'randomforestclassifier__oob_score': False, 'conditionalimputer__axis': 0, 'conditionalimputer__copy': True, 'conditionalimputer__fill_empty': 0, 'conditionalimputer__missing_values': "NaN", 'conditionalimputer__strategy': 'median', 'conditionalimputer__strategy_nominal': 'most_frequent', 'onehotencoder__handle_unknown': 'ignore', 'onehotencoder__categories': 'auto', 'variancethreshold__threshold': 0.0},
    {'randomforestclassifier__bootstrap': False, 'randomforestclassifier__class_weight': nan, 'randomforestclassifier__criterion': 'entropy', 'randomforestclassifier__max_depth': nan, 'randomforestclassifier__max_features': 0.11987533889966198, 'randomforestclassifier__max_leaf_nodes': nan, 'randomforestclassifier__min_impurity_decrease': 1e-07, 'randomforestclassifier__min_samples_leaf': 2, 'randomforestclassifier__min_samples_split': 18, 'randomforestclassifier__min_weight_fraction_leaf': 0.0, 'randomforestclassifier__n_estimators': 100, 'randomforestclassifier__oob_score': False, 'conditionalimputer__axis': 0, 'conditionalimputer__copy': True, 'conditionalimputer__fill_empty': 0, 'conditionalimputer__missing_values': "NaN", 'conditionalimputer__strategy': 'mean', 'conditionalimputer__strategy_nominal': 'most_frequent', 'onehotencoder__handle_unknown': 'ignore', 'onehotencoder__categories': 'auto', 'variancethreshold__threshold': 0.0}
]

# grid = {
#     "randomforestclassifier__bootstrap": [True, False],
#     "randomforestclassifier__max_features": np.arange(0.01, 1.001, 0.01),
#     "randomforestclassifier__min_samples_leaf": [2, 5],
#     "randomforestclassifier__min_samples_split": np.arange(1, 20, 1),
#     "conditionalimputer__strategy": ["mean", "median"],
#     "onehotencoder__handle_unknown": ["ignore"],
#     "randomforestclassifier__n_estimators": [100],
#     "randomforestclassifier__min_samples_split": [2]
# }
task = openml.tasks.get_task(31)
X, y = task.get_X_and_y()

optimizer = Optimizer(pipeline=pipeline)
optimizer.setup(grid, X, y, cv=3)
optimizer.seed(triple)
suggestion = optimizer.loop()
print()