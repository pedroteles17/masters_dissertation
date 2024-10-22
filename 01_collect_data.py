# %%
from utils import MercadoLivreScraper, MercadoLivreAuthenticator
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
import logging
import time

load_dotenv()
logging.basicConfig(level=logging.INFO)

# %%
# Update the access token
# MercadoLivreAuthenticator().get_access_from_refresh_token()

# %%
# Load the best sellers product list
products_barcode = pd.read_excel("data/best_sellers.xls", dtype=str)["barcode"].tolist()[:10]

scraper = MercadoLivreScraper(parse=True)

# %%
# Search the products in Mercado Livre
meli_products, full_product_data = [], []
for barcode in tqdm(
    products_barcode, total=len(products_barcode), desc="Searching products"
):
    try:
        product = scraper.search_by_term(barcode)
        full_product_data.append(pd.json_normalize(product["results"][0]))
        gtin = next(
            (
                attr["value_name"]
                for attr in product["results"][0]["attributes"]
                if attr["id"] == "GTIN"
            ),
            None,
        )
        gtin = gtin.split(",")[0] if gtin else None
        meli_products.append(
            {
                "barcode": barcode,
                "product_id": product["results"][0]["catalog_product_id"],
                "extracted_barcode": gtin,
            }
        )
    except Exception as e:
        logging.error(f"Error fetching product {barcode}: {e}")

full_product_data = (
    pd.concat(full_product_data)
    .reset_index(drop=True)
    .to_parquet(f'data/product_metadata/{time.strftime("%Y-%m-%dT%H-%M-%S")}.parquet')
)

meli_products = (
    pd.DataFrame(meli_products)
    .query("barcode == extracted_barcode")
    .dropna(subset=["product_id"])
    .drop(columns=["extracted_barcode"])
    .reset_index(drop=True)
)

# %%
buy_box_winners = []
for product_id in tqdm(
    meli_products["product_id"],
    total=len(meli_products["product_id"]),
    desc="Getting buy box winners",
):
    try:
        product_info = MercadoLivreScraper(parse=False).get_product_info(product_id)
        buy_box_winners.append(
            {
                "product_id": product_id,
                "seller_id": product_info["buy_box_winner"]["seller_id"],
                "listing_type_id": product_info["buy_box_winner"]["listing_type_id"],
                "competition_status": "buy_box_winner",
            }
        )
    except Exception as e:
        logging.error(f"Error fetching buy box winner for product {product_id}: {e}")

buy_box_winners = pd.DataFrame(buy_box_winners)

# %%
# Based on product_id, get the sellers
sellers_by_product = []
for product_id in tqdm(
    meli_products["product_id"],
    total=len(meli_products["product_id"]),
    desc="Getting sellers by product",
):
    try:
        sellers_by_product.append(scraper.get_all_sellers_by_product(product_id))
    except Exception as e:
        logging.error(f"Error fetching sellers for product {product_id}: {e}")

sellers_by_product = pd.concat(sellers_by_product).reset_index(drop=True)

# %%
# For each seller, get info
unique_sellers_ids = sellers_by_product["seller_id"].unique()

sellers_info = []
for seller_id in tqdm(
    unique_sellers_ids, total=len(unique_sellers_ids), desc="Getting sellers info"
):
    sellers_info.append(scraper.get_user_info(seller_id))

sellers_info = pd.concat(sellers_info).reset_index(drop=True)

# %%
# Merge the sellers info with the sellers by product and the buy box winners
sellers_info = sellers_info[
    [
        "id",
        "nickname",
        "country_id",
        "user_type",
        "site_id",
        "permalink",
        "transactions_period",
        "transactions_total",
        "city",
        "state",
        "level_id",
        "power_seller_status",
        "site_status",
    ]
]

sellers_info = sellers_info.rename(columns=lambda x: "seller_" + x)

sellers_by_product = (
    sellers_by_product.merge(sellers_info, on="seller_id", how="left")
    .merge(
        buy_box_winners, on=["product_id", "seller_id", "listing_type_id"], how="outer"
    )
    .merge(meli_products, on="product_id", how="left")
)

# %%
sellers_by_product.to_parquet(
    f'data/sellers_by_product/{time.strftime("%Y-%m-%dT%H-%M-%S")}.parquet', index=False
)

# %%
