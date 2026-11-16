"""Leaf models other than AutoGluon: sklearn-style regressors with shared preprocessing."""
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

LEARNERS = ("autogluon", "catboost", "xgboost", "lightgbm", "random_forest", "linear")


def _estimator(name, params, random_state):
    params = dict(params or {})
    if name == "catboost":
        from catboost import CatBoostRegressor
        defaults = {"random_seed": random_state, "verbose": 0, "allow_writing_files": False}
        return CatBoostRegressor(**{**defaults, **params})
    if name == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(**{"random_state": random_state, "n_jobs": -1, **params})
    if name == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(**{"random_state": random_state, "n_jobs": -1, "verbose": -1, **params})
    if name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(**{"random_state": random_state, "n_jobs": -1, **params})
    if name == "linear":
        from sklearn.linear_model import Ridge
        return Ridge(**params)
    raise ValueError(f"unknown learner {name!r}; choose from {LEARNERS} or pass an estimator")


def _bools_to_float(X):
    bools = X.select_dtypes(include="bool").columns
    return X.astype({c: float for c in bools}) if len(bools) else X


def _preprocessor(scale):
    numeric = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric.append(("scale", StandardScaler()))
    categorical = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", max_categories=50, sparse_output=False)),
    ]
    columns = ColumnTransformer([
        ("num", Pipeline(numeric), make_column_selector(dtype_include="number")),
        ("cat", Pipeline(categorical), make_column_selector(dtype_exclude="number")),
    ])
    return Pipeline([("bools", FunctionTransformer(_bools_to_float)), ("columns", columns)])


def make_model(learner, params=None, random_state=42):
    """Unfitted regressor for a DataFrame: median imputation, one-hot encoding of
    non-numeric columns (scaling too for 'linear'), then the learner.
    `learner` is a name from LEARNERS (not 'autogluon') or any sklearn-style
    regressor, which is cloned."""
    if learner == "autogluon":
        raise ValueError("make_model does not build AutoGluon; use autogluon.tabular.TabularPredictor")
    if isinstance(learner, str):
        estimator, scale = _estimator(learner, params, random_state), learner == "linear"
    else:
        estimator, scale = clone(learner), False
    return Pipeline([("prep", _preprocessor(scale)), ("model", estimator)])
