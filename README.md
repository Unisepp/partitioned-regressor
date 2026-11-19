# partitioned-regressor

Splits the data into subsets with a CART tree and trains one model per subset (leaf): AutoGluon by default, or CatBoost, XGBoost, LightGBM, a random forest, a linear model, or any scikit-learn regressor. The tree's settings are chosen automatically on validation data; the test set is never used.

## Install

```bash
pip install "git+https://github.com/Unisepp/partitioned-regressor"
```

For development, from a local clone:

```bash
pip install -e path/to/partitioned-regressor
```

## Usage

```python
from partitioned_regressor import PartitionedRegressor

# random validation splits
model = PartitionedRegressor(label="target", path="models/my_model")

# time-ordered data: chronological validation splits
model = PartitionedRegressor(label="target", path="models/my_model",
                             split="chronological", time_col="timestamp")

model.fit(train_df)              # search, final training, saves to path
preds = model.predict(test_df)
model.evaluate(test_df)          # {"rmse": ...}
model.summary()                  # chosen partition
model.search_results_            # all candidates with validation scores
model.leaf_of(test_df)           # leaf id per row

model = PartitionedRegressor.load("models/my_model")
```

## Comparing with a global model

With `compare_global=True`, `fit()` also trains one global model (same learner and settings, no partitioning) on the full training set after the search. It plays no part in the search.

```python
model = PartitionedRegressor(label="target", compare_global=True)
model.fit(train_df)
model.evaluate(test_df)          # {"rmse": ..., "rmse_global": ..., "improvement_pct": ...}
model.predict_global(test_df)    # predictions of the global model
```

## Leaf models

```python
PartitionedRegressor(label="target", learner="catboost")
PartitionedRegressor(label="target", learner="xgboost", learner_params={"n_estimators": 500})
PartitionedRegressor(label="target", learner=MyRegressor())   # any scikit-learn-style regressor
```

| `learner` | Model | Notes |
|---|---|---|
| `"autogluon"` (default) | AutoGluon `TabularPredictor` | Uses `final_fit_kwargs` and `search_time_limit`. |
| `"catboost"` | `CatBoostRegressor` | |
| `"xgboost"` | `XGBRegressor` | |
| `"lightgbm"` | `LGBMRegressor` | |
| `"random_forest"` | scikit-learn `RandomForestRegressor` | |
| `"linear"` | scikit-learn `Ridge` | Features are also standardized. |

All non-AutoGluon learners use default hyperparameters (override with `learner_params`) and the same preprocessing: median imputation for numeric columns, one-hot encoding for the others. The candidate partitions are the same for every learner. They are scored with the selection learner (see below).

To build a global baseline yourself with the same learner and preprocessing:

```python
from partitioned_regressor import make_model

global_model = make_model("catboost").fit(train_df.drop(columns="target"), train_df["target"])
```

## Selection learner

With AutoGluon, scoring every candidate with AutoGluon is slow, so by default the candidates are scored with CatBoost and only the final models (and the optional global model) use AutoGluon:

```python
PartitionedRegressor(label="target")                                   # CatBoost chooses, AutoGluon trains
PartitionedRegressor(label="target", selection_learner="autogluon")    # AutoGluon chooses and trains
```

For all other learners, the learner itself scores the candidates.

## How `fit()` works

1. **Candidates.** For each `min_samples_leaf` value, a CART-only k-fold CV is run over leaf counts. The leaf cap starts at 15 and doubles (up to 60) while the lowest CV RMSE sits exactly on the cap. The 1-SE rule picks one leaf count per cap level (2-15, 2-30, ...). These CART curves always use random folds (`cart_cv="random"`): this stage only proposes candidates, and the time order is respected where the decision is made (scoring and selection). The default value (5% of the rows, clipped to 300-800) and the 3 other values with the lowest CV RMSE are kept, each with all its picks.
2. **Full-data check.** Each candidate's tree is fit on the full training set. Candidates that collapse to 1 leaf are dropped; the leaf count is set to what the tree reaches. Candidates that split the rows identically are merged. If only one partition is left, scoring is skipped.
3. **Scoring.** Each candidate is trained per leaf with the selection learner and scored on validation data: k-fold CV below 5,000 rows, one holdout split above.
4. **Selection.** The simplest candidate within one standard error of the best wins (fewer leaves, then larger `min_samples_leaf`).
5. **Final model.** The winner is refit on all training data and saved.

`min_samples_leaf` refers to the full training set and is scaled down when a tree is fit on fewer rows.

## Options

| Option | Default | Meaning |
|---|---|---|
| `label` | required | Target column. |
| `path` | `"partitioned_model"` | Folder for the final models. |
| `split` | `"random"` | `"random"` or `"chronological"` (needs `time_col`). |
| `time_col` | `None` | Timestamp column for chronological splits. |
| `drop_time_col` | `True` | Keep `time_col` out of the model features. |
| `search` | `True` | `False` uses the default `min_samples_leaf` without a search. |
| `min_samples_leaf_grid` | `(200, 300, 500, 800)` | Values tried, relative to the full training set. |
| `leaf_cap` | `15` | Starting leaf-count cap. |
| `max_leaf_cap` | `60` | Largest leaf-count cap. |
| `top_k` | `3` | `min_samples_leaf` values scored end-to-end besides the default. |
| `small_data_rows` | `5000` | Below this, k-fold CV in scoring; above, one holdout split. |
| `n_folds` | `3` | Folds for CV. |
| `val_fraction` | `0.2` | Holdout size. |
| `search_time_limit` | `None` | Seconds per AutoGluon model during the search (no limit by default). |
| `final_fit_kwargs` | `presets="medium", auto_stack=False` | Passed to AutoGluon's `fit`, also used in the search. |
| `learner` | `"autogluon"` | Leaf model, see above. |
| `learner_params` | `None` | Constructor arguments for non-AutoGluon learners. |
| `cart_cv` | `"random"` | Folds for the CART curves in step 1. `"split"` uses the same folds as `split` (chronological if set). |
| `selection_learner` | `"auto"` | Learner that scores the candidates. `"auto"`: CatBoost when `learner="autogluon"`, otherwise `learner` itself. Set `selection_learner="autogluon"` for the exact (slow) AutoGluon search. |
| `compare_global` | `False` | Also train a global model; `evaluate()` then reports both and the improvement. |
| `random_state` | `42` | Seed. |
| `verbosity` | `1` | `0` for silent, `2` for AutoGluon logs. |

