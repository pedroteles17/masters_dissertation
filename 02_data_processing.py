# %%
import pandas as pd
import time

# %%
sellers_data = pd.read_parquet("data/sellers_by_product/2024-10-22T17-34-45.parquet")

sellers_data["smallest_price"] = sellers_data.groupby("product_id")["price"].transform(
    "min"
)

# %%
# Sellers can have multiple listings for the same product, even with the same listing type.
## We need to remove duplicates to avoid biasing the analysis.
## We will keep the listing with the smallest price or the buy box winner listing.
processed_data = (
    sellers_data.copy()
    .query('condition == "new"')
    .query('seller_site_status == "active"')
    .sort_values(by=["competition_status", "price"])
    .drop_duplicates(
        subset=["product_id", "seller_id", "listing_type_id"], keep="first"
    )
    .reset_index(drop=True)
)

# %%
"""
# To avoid dealing with huge imalanced data, we will only consider the buy box winner and X listings with the smallest price per product.
buy_box_winners = processed_data.copy().query('competition_status == "buy_box_winner"')

competitors = (
    processed_data.copy()
    .query('competition_status != "buy_box_winner"')
    .sort_values(by=["product_id", "price"])
    .groupby("product_id")
    .head(49)
    .reset_index(drop=True)
)

processed_data = pd.concat([buy_box_winners, competitors])
"""
# %%
# Now we will process the data to extract the features we want to analyze.
processed_data = (
    processed_data.assign(
        price_diff=lambda x: x["price"] - x["smallest_price"],
        premium_listing=lambda x: x["listing_type_id"].apply(
            lambda id: 1 if id == "gold_pro" else 0
        ),
        official_store=lambda x: x["official_store_id"].apply(
            lambda id: 1 if pd.notnull(id) else 0
        ),
        meli_direct_seller=lambda x: x["official_store_id"].apply(
            lambda id: 1 if id == 2707 else 0
        ),
        free_shipping=lambda x: x["shipping_free_shipping"].apply(
            lambda free: 1 if free else 0
        ),
        store_pickup=lambda x: x["shipping_store_pick_up"].apply(
            lambda pickup: 1 if pickup else 0
        ),
        fullfilment=lambda x: x["shipping_logistic_type"].apply(
            lambda logistic: 1 if logistic == "fulfillment" else 0
        ),
        cross_docking=lambda x: x["shipping_logistic_type"].apply(
            lambda logistic: 1 if logistic == "cross_docking" else 0
        ),
        flex_shipping=lambda x: x["shipping_tags"].apply(
            lambda tags: 1 if "self_service_in" in tags else 0
        ),
        southeast_seller=lambda x: x["seller_state_x"].apply(
            lambda state: (
                1
                if state
                in ["São Paulo", "Minas Gerais", "Rio de Janeiro", "Espírito Santo"]
                else 0
            )
        ),
        seller_level=lambda x: x["seller_level_id"].apply(
            lambda id: int(id.split("_")[0]) if pd.notnull(id) else 0
        ),
        power_seller=lambda x: x["seller_power_seller_status"].apply(
            lambda level: (
                1
                if level == "silver"
                else 2 if level == "gold" else 3 if level == "platinum" else 0
            )
        ),
        buy_box_winner=lambda x: x["competition_status"].apply(
            lambda status: 1 if status == "buy_box_winner" else 0
        ),
    )
)[
    [
        "product_id",
        "seller_id",
        "seller_nickname",
        "price_diff",
        "premium_listing",
        "official_store",
        "meli_direct_seller",
        "free_shipping",
        "store_pickup",
        "fullfilment",
        "cross_docking",
        "flex_shipping",
        "southeast_seller",
        "seller_level",
        "power_seller",
        "seller_transactions_total",
        "buy_box_winner",
    ]
]

# %%
processed_data.to_parquet(f"data/processed_data.parquet")

#%%
