#%%
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm

from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight

#%%
# We drop 'store_pickup' column because it has a single value for all observations.
data = (
    pd.read_parquet("data/processed_data.parquet")
    .drop(columns=["store_pickup", "seller_nickname"])
    .set_index(["product_id", "seller_id"])
)

#%%
# Split data into train and test sets by product_id
product_ids = data.index.get_level_values("product_id").unique()

train_product_ids, test_product_ids = train_test_split(
    product_ids, test_size=0.3, random_state=42
)

train_data, test_data = data.loc[train_product_ids], data.loc[test_product_ids]

train_data = train_data.sample(frac=1, random_state=42)

#%%
# Separate features and target
X_train, y_train = train_data.drop(columns=["buy_box_winner"]), train_data["buy_box_winner"]

X_test, y_test = test_data.drop(columns=["buy_box_winner"]), test_data["buy_box_winner"]

#%%
sample_weights = compute_sample_weight('balanced', y_train)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = {0: class_weights[0], 1: class_weights[1]}
imbalance_proportion = class_weights[1] / class_weights[0]  # To match scale_pos_weight

models = [
    LGBMClassifier(random_state=42, verbose=-1, scale_pos_weight=imbalance_proportion),
    XGBClassifier(random_state=42, scale_pos_weight=imbalance_proportion),
    CatBoostClassifier(random_state=42, class_weights=[class_weights_dict[0], class_weights_dict[1]], verbose=100),
    RandomForestClassifier(random_state=42, class_weight=class_weights_dict),
    LogisticRegression(random_state=42, max_iter=1000, class_weight=class_weights_dict),
    TabNetClassifier(seed=42, verbose=0, device_name='cpu')  # Still lacks direct class weights
]

#%%
# Train models
predictions = []
for model in tqdm(models, total=len(models)):
    print(f"Training {model.__class__.__name__}...")

    # Train model and get predictions
    st = time.time()

    if model.__class__.__name__ == "TabNetClassifier":
        model.fit(
            X_train.values, y_train.values,
            weights=sample_weights,
        )
        y_pred_prob = model.predict_proba(X_test.values)
    else:    
        model.fit(X_train, y_train)
        y_pred_prob = model.predict_proba(X_test)

    et = time.time()

    prediction_df = (
        pd.DataFrame(y_test)
        .copy()
        .reset_index()
        .assign(
            y_pred_prob=y_pred_prob[:, 1],
            model_name=model.__class__.__name__,
            execution_time=et - st
        )
    )

    prediction_df["max_prob_flag"] = (prediction_df["y_pred_prob"] == prediction_df.groupby("product_id")["y_pred_prob"].transform("max")).astype(int)
    
    predictions.append(prediction_df)

predictions = (
    pd.concat(predictions)
    .reset_index(drop=True)
)

#%%
# A dummy model that always predicts the lowest price as the buy box winner
lowest_price_dummy = (
    data
        .query("product_id in @test_product_ids")
        .reset_index()
        .groupby("product_id")
        .apply(
            lambda df: df.assign(lowest_price=(df['pct_price_diff'] == df['pct_price_diff'].min()).astype(int)),
            include_groups=False
        )
        .reset_index()
        .assign(
            model_name="LowestPriceDummy",
            execution_time=0
        )
        .rename(columns={"lowest_price": "max_prob_flag"})
)[[
    "product_id", "seller_id", "buy_box_winner", 
    "model_name", "execution_time", "max_prob_flag"
]]

predictions = pd.concat([predictions, lowest_price_dummy])

#%%
# A dummy model that always predicts the seller that offers full with the lowest price as the buy box winner
## If there is no seller that offers fullfilment, the model will predict the seller with the lowest price.
lowest_price_full= (
    data
        .query("product_id in @test_product_ids")
        .reset_index()
)

product_ids = lowest_price_full["product_id"].unique()

lowest_price_fullfilment_dummy = []
for product_id in tqdm(product_ids, desc="Processing products", total=len(product_ids)):
    lowest_price_full_product = lowest_price_full\
        .query("product_id == @product_id")

    lowest_price_with_full = lowest_price_full_product\
        .query("fullfilment == 1")

    if lowest_price_with_full.empty:
        lowest_price_full_product = lowest_price_full_product\
            .assign(
                max_prob_flag=(lowest_price_full_product['pct_price_diff'] == lowest_price_full_product['pct_price_diff'].min()).astype(int)
            )
    else:
        lowest_price_full_product = lowest_price_full_product\
            .assign(
                max_prob_flag=(lowest_price_full_product['fullfilment'] == 1) & (lowest_price_full_product['pct_price_diff'] == lowest_price_full_product[lowest_price_full_product["fullfilment"] == 1]['pct_price_diff'].min()).astype(int)
            )
    
    lowest_price_fullfilment_dummy.append(lowest_price_full_product)

lowest_price_fullfilment_dummy = pd.concat(lowest_price_fullfilment_dummy)\
    .reset_index(drop=True)\
    .assign(
        model_name="LowestPriceFullfilmentDummy",
        execution_time=0
    )[[
        "product_id", "seller_id", "buy_box_winner", 
        "model_name", "execution_time", "max_prob_flag"
    ]]

predictions = pd.concat([predictions, lowest_price_fullfilment_dummy])

#%%
classification_report(
    lowest_price_fullfilment_dummy["buy_box_winner"], lowest_price_fullfilment_dummy["max_prob_flag"],
    output_dict=True
)

#%%
product_ids = data.index.get_level_values("product_id").unique()

train_product_ids, test_product_ids = train_test_split(
    product_ids, test_size=0.3, random_state=42
)

train_data, test_data = data.loc[train_product_ids], data.loc[test_product_ids]

#%%
X_train, y_train = train_data.drop(columns=["buy_box_winner"]), train_data["buy_box_winner"]

X_test, y_test = test_data.drop(columns=["buy_box_winner"]), test_data["buy_box_winner"]

imbalance_proportion = y_train.value_counts()[0] / y_train.value_counts()[1]

#%%
train_data = lgb.Dataset(X_train, label=y_train)

# Train the model
lgb_train = lgb.train(
    params={
        "objective": "binary",
        "metric": "auc",
        "scale_pos_weight": imbalance_proportion,
    },
    train_set=train_data
)

#%%
y_pred_prob = lgb_train.predict(X_test)

y_test = (
    pd.DataFrame(y_test)
        .reset_index()
        .assign(y_pred_prob=y_pred_prob)
        .groupby("product_id")
        .apply(lambda df: df.assign(max_prob_flag=(df['y_pred_prob'] == df['y_pred_prob'].max()).astype(int)))
        .reset_index(drop=True)
)

#%%
classification_report(y_test["buy_box_winner"], y_test["max_prob_flag"], output_dict=True)

#%%
# Model feature importance
feature_importance = lgb_train.feature_importance(importance_type="gain")

feature_importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": feature_importance
})\
    .sort_values(by="Importance", ascending=True)\
    .reset_index(drop=True)\
    .assign(
        Feature = lambda x: x["Feature"].apply(lambda f: f.replace("_", " ").title())

    )
    
# Plot
fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(feature_importance_df["Feature"], feature_importance_df["Importance"], color="royalblue", edgecolor="black")

# Improve readability
ax.set_xlabel("Importance Gain (×10⁶)", fontsize=12)
ax.set_ylabel("Feature", fontsize=12)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}"))  # Scale x-axis

ax.set_xlim([0, 1.75e6])

# Annotate top features
for bar in bars[-5:]:  # Annotate only the top 5 features
    ax.text(bar.get_width() + 20000, bar.get_y() + bar.get_height()/2, f"{bar.get_width():,.0f}",
            va='center', fontsize=10)

# Show gridlines lightly
ax.xaxis.grid(True, linestyle="--", alpha=0.5)

# Save and display
plt.tight_layout()
plt.savefig("results/feature_importance.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
