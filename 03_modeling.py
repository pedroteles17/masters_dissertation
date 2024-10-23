#%%
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

#%%
# We also drop 'store_pickup' column because it has a single value for all observations.
data = pd.read_parquet("data/processed_data.parquet")

data = data.drop(columns=["product_id", "seller_id", "seller_nickname", "store_pickup"])

#%%
X = data.drop(columns=["buy_box_winner"])
y = data["buy_box_winner"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

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
y_pred_prob = lgb_train.predict(X_val)

fpr, tpr, thresholds = roc_curve(y_val, y_pred_prob)
roc_auc = auc(fpr, tpr)

gmeans = np.sqrt(tpr * (1-fpr))

ix = np.argmax(gmeans)
print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))

#%%
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best Threshold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# %%
y_pred_prob = lgb_train.predict(X_test)

y_pred = y_pred_prob > thresholds[ix]

classification_report(y_test, y_pred)

# %%
