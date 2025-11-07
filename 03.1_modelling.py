#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm
from utils import OptunaOptimizer

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
import optuna

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight

#%%
# We ensure that the predictions correspond to the correct products
predictions_id = (
    pd.read_parquet("data/predictions/2025-06-01T19-26-44.parquet")
    ["product_id"].unique()
)

#%%
# We drop 'store_pickup' column because it has a single value for all observations.
data = (
    pd.read_parquet("data/processed_data/2025-06-01T19-09-22.parquet")
    .drop(columns=["store_pickup", "seller_nickname"])
    .set_index(["product_id", "seller_id"])
)

#%%
# Split data into train and test sets by product_id
product_ids = data.index.get_level_values("product_id").unique()

train_product_ids, test_product_ids = (
    product_ids[~product_ids.isin(predictions_id)],
    product_ids[product_ids.isin(predictions_id)],
)

train_product_ids, valid_product_ids = train_test_split(
    train_product_ids, test_size=0.3, random_state=42
)

train_data, valid_data, test_data = (
    data.loc[train_product_ids],
    data.loc[valid_product_ids],
    data.loc[test_product_ids],
)

train_data = train_data.sample(frac=1, random_state=42)
valid_data = valid_data.sample(frac=1, random_state=42)

#%%
# Separate features and target
X_train, y_train = train_data.drop(columns=["buy_box_winner"]), train_data["buy_box_winner"]
X_valid, y_valid = valid_data.drop(columns=["buy_box_winner"]), valid_data["buy_box_winner"]
X_test, y_test = test_data.drop(columns=["buy_box_winner"]), test_data["buy_box_winner"]

sample_weights = compute_sample_weight('balanced', y_train)

#%%
# Initialize optimizer
optimizer = OptunaOptimizer(X_train, y_train, X_valid, y_valid)

models_functions = {
    #"TabNet": {"optimize": optimizer.objective_tabnet, "model_class": TabNetClassifier},
    "LightGBM": {"optimize": optimizer.objective_lightgbm, "model_class": LGBMClassifier},
    "RandomForest": {"optimize": optimizer.objective_random_forest, "model_class": RandomForestClassifier},
    "XGBoost": {"optimize": optimizer.objective_xgboost, "model_class": XGBClassifier},
    "CatBoost": {"optimize": optimizer.objective_catboost, "model_class": CatBoostClassifier},
    "LogisticRegression": {"optimize": optimizer.objective_logistic_regression, "model_class": LogisticRegression},
}

#%%
predictions = []
for model_name, model_info in tqdm(models_functions.items(), total=len(models_functions)):
    # Run the optimization
    st = time.time()

    study = optuna.create_study(direction="maximize")
    study.optimize(model_info["optimize"], n_trials=50)

    # Train final model with best parameters on full training set
    X_train_full = pd.concat([X_train, X_valid])
    y_train_full = pd.concat([y_train, y_valid])

    best_params = study.best_trial.params

    if model_info["model_class"]().__class__.__name__ == "TabNetClassifier":
        tabnet_params = {k: v for k, v in best_params.items() if k not in [
            "learning_rate", "max_epochs", "patience", "batch_size", "virtual_batch_size"
        ]}
        tabnet_params["optimizer_params"] = dict(lr=best_params["learning_rate"])

        # Initialize model (constructor only gets architecture + optimizer-related params)
        final_model = TabNetClassifier(**tabnet_params, seed=42)

        # Compute sample weights for the full dataset (handle imbalance)
        classes = np.array([0, 1])
        cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_full)
        w_map = {c: w for c, w in zip(classes, cw)}
        sample_weights = np.vectorize(w_map.get)(
            y_train_full.values if hasattr(y_train_full, "values") else y_train_full
        )

        # Train final model with full data
        final_model.fit(
            X_train_full.values,
            y_train_full.values,
            weights=sample_weights,
            max_epochs=best_params.get("max_epochs", 200),
            patience=best_params.get("patience", 30),
            batch_size=best_params.get("batch_size", 512),
            virtual_batch_size=best_params.get("virtual_batch_size", 64),
        )

        # Predict probabilities on test data
        y_pred_prob = final_model.predict_proba(X_test.values)
    else:
        final_model = model_info["model_class"](**best_params, random_state=42)    
        final_model.fit(X_train_full, y_train_full)
        y_pred_prob = final_model.predict_proba(X_test)

    et = time.time()

    prediction_df = (
        pd.DataFrame(y_test)
        .copy()
        .reset_index()
        .assign(
            y_pred_prob=y_pred_prob[:, 1],
            model_name=f"{final_model.__class__.__name__}_FT",
            execution_time=et - st
        )
    )

    prediction_df["max_prob_flag"] = (prediction_df["y_pred_prob"] == prediction_df.groupby("product_id")["y_pred_prob"].transform("max")).astype(int)

    predictions.append(prediction_df)

predictions = pd.concat(predictions, ignore_index=True)

#%%
predictions.to_parquet(
    f'data/predictions/{time.strftime("%Y-%m-%dT%H-%M-%S")}_fine_tuned.parquet'
)

# %%
study = optuna.create_study(direction="maximize")
study.optimize(optimizer.objective_xgboost, n_trials=50)

# %%
fig = optuna.visualization.plot_optimization_history(study)