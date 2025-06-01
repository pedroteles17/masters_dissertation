# %%
import pandas as pd
import time

# %%
sellers_data = pd.read_parquet("data/sellers_by_product/2025-02-19T17-15-50.parquet")

sellers_data["smallest_price"] = sellers_data.groupby("product_id")["price"].transform(
    "min"
)

# %%
# To compete, sellers must offer new products, have an active account and have a seller level of at least 3_yellow.
## https://www.mercadolivre.com.br/ajuda/4676 (not sure if it is a permanent link)
processed_data = (
    sellers_data.copy()
    .query('condition == "new"')
    .query('seller_site_status == "active"')
    .query('seller_level_id in ["3_yellow", "4_light_green", "5_green"]')
)

# Sellers can have multiple listings for the same product, even with the same listing type.
## We need to remove duplicates to avoid biasing the analysis.
## We will keep the listing with the smallest price or the buy box winner listing.
processed_data = (
    sellers_data.copy()
    .sort_values(by=["competition_status", "price"])
    .drop_duplicates(
        subset=["product_id", "seller_id"], keep="first"
    )
    .reset_index(drop=True)
)

buy_boxes_with_no_winner = (
    processed_data
    .assign(
        buy_box_winner=lambda x: x["competition_status"].apply(
            lambda status: 1 if status == "buy_box_winner" else 0
        )
    )
    .groupby("product_id")
    .agg(
        buy_box_winner=("buy_box_winner", "sum")
    )
    .query("buy_box_winner == 0")
)

# We remove products that do not have a buy box winner.
## We cannot predict the buy box winner if there is no buy box winner.
processed_data = (
    processed_data
    .query("product_id not in @buy_boxes_with_no_winner.index")
)

# %%
# Now we will process the data to extract the features we want to analyze.
processed_data = (
    processed_data.assign(
        pct_price_diff=lambda x: (x["price"] - x["smallest_price"]) / x["smallest_price"],
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
        drop_off=lambda x: x["shipping_logistic_type"].apply(
            lambda logistic: 1 if logistic == "drop_off" else 0
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
        "pct_price_diff",
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

#%%
# To give more context to the model, we will add features that aggregate information about the competition.
competition_features = (
    processed_data
        .groupby("product_id")
        .agg(
            n_competitors=("seller_id", "count"),
            premium_listing_pct=("premium_listing", "mean"),
            official_store_pct=("official_store", "mean"),
            free_shipping_pct=("free_shipping", "mean"),
            fullfilment_pct=("fullfilment", "mean"),
            flex_shipping_pct=("flex_shipping", "mean"),
            seller_level_mean=("seller_level", "mean"),
            power_seller_mean=("power_seller", "mean"),
            seller_transactions_total_mean=("seller_transactions_total", "mean"),
        )
)

processed_data = processed_data.merge(
    competition_features, on="product_id", how="left"
)

# %%
processed_data.to_parquet(
    f'data/processed_data/{time.strftime("%Y-%m-%dT%H-%M-%S")}.parquet'
)

#%%
