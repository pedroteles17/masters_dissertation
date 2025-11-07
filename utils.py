import requests
import time
import logging
import tqdm
import pandas as pd
import os
import json
from dotenv import set_key
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import precision_score
from sklearn.utils.class_weight import compute_class_weight

class OptunaOptimizer:
    def __init__(self, X_train, y_train, X_valid, y_valid):
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid

    def objective_lightgbm(self, trial):
        # Compute weight for imbalance
        pos = np.sum(self.y_train == 1)
        neg = np.sum(self.y_train == 0)
        scale_pos_weight = neg / pos if pos > 0 else 1.0

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 3000),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": scale_pos_weight,
            "random_state": 42,
            "n_jobs": -1,
        }

        model = LGBMClassifier(**params, verbosity=-1)
        model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_valid, self.y_valid)],
            eval_metric="average_precision"
        )

        preds = model.predict(self.X_valid)
        score = precision_score(self.y_valid, preds, zero_division=0)
        return score
    
    def objective_catboost(self, trial):
        pos = np.sum(self.y_train == 1)
        neg = np.sum(self.y_train == 0)
        scale_pos_weight = neg / pos if pos > 0 else 1.0

        params = {
            "iterations": trial.suggest_int("iterations", 200, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "depth": trial.suggest_int("depth", 3, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 1.0),
            "scale_pos_weight": scale_pos_weight,
            "border_count": trial.suggest_int("border_count", 32, 255),
            "verbose": 0,
            "random_seed": 42,
        }

        model = CatBoostClassifier(**params)
        model.fit(self.X_train, self.y_train, eval_set=(self.X_valid, self.y_valid), verbose=0)

        preds = model.predict(self.X_valid)
        score = precision_score(self.y_valid, preds, zero_division=0)
        return score
    
    def objective_logistic_regression(self, trial):
        # Compute class weight for imbalance
        pos = np.sum(self.y_train == 1)
        neg = np.sum(self.y_train == 0)
        class_weight = {0: 1.0, 1: neg / pos if pos > 0 else 1.0}

        params = {
            "penalty": trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"]),
            "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
            "solver": trial.suggest_categorical("solver", ["saga", "liblinear"]),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0) if "elasticnet" in trial.params.get("penalty", "") else None,
            "max_iter": 2000,
            "random_state": 42,
            "class_weight": class_weight,
            "n_jobs": -1,
        }

        # Clean up unsupported combinations
        if params["penalty"] != "elasticnet":
            params.pop("l1_ratio", None)

        # liblinear doesn't support elasticnet
        if params["solver"] == "liblinear" and params["penalty"] == "elasticnet":
            params["penalty"] = "l1"  # fallback to l1

        model = LogisticRegression(**params)
        model.fit(self.X_train, self.y_train)

        preds = model.predict(self.X_valid)
        score = precision_score(self.y_valid, preds, zero_division=0)
        return score

    def objective_xgboost(self, trial):
        pos = np.sum(self.y_train == 1)
        neg = np.sum(self.y_train == 0)
        scale_pos_weight = neg / pos if pos > 0 else 1.0

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": scale_pos_weight,
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "logloss",
        }

        model = XGBClassifier(**params)
        model.fit(self.X_train, self.y_train,
                  eval_set=[(self.X_valid, self.y_valid)],
                  verbose=False)

        preds = model.predict(self.X_valid)
        score = precision_score(self.y_valid, preds, zero_division=0)
        return score
    
    def objective_random_forest(self, trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 40),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample"]),
            "random_state": 42,
            "n_jobs": -1,
        }

        model = RandomForestClassifier(**params)
        model.fit(self.X_train, self.y_train)

        preds = model.predict(self.X_valid)
        score = precision_score(self.y_valid, preds, zero_division=0)

        return score

    def objective_tabnet(self, trial):
        params = {
            "n_d": trial.suggest_int("n_d", 8, 64),
            "n_a": trial.suggest_int("n_a", 8, 64),
            "n_steps": trial.suggest_int("n_steps", 3, 8),
            "gamma": trial.suggest_float("gamma", 1.0, 2.5),
            "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True),
            "momentum": trial.suggest_float("momentum", 0.01, 0.4),
            "clip_value": trial.suggest_float("clip_value", 1.0, 2.0),
            "optimizer_params": dict(lr=trial.suggest_float("learning_rate", 1e-3, 5e-2, log=True)),
            "seed": 42,
            "verbose": 0,
        }

        model = TabNetClassifier(**params)

        # Class-weighted sample weights (helps with imbalance)
        classes = np.array([0, 1])
        cw = compute_class_weight(class_weight="balanced", classes=classes, y=self.y_train)
        w_map = {c: w for c, w in zip(classes, cw)}
        sample_weight = np.vectorize(w_map.get)(self.y_train.values if hasattr(self.y_train, "values") else self.y_train)

        model.fit(
            self.X_train.values, self.y_train.values,
            eval_set=[(self.X_valid.values, self.y_valid.values)],
            max_epochs=trial.suggest_int("max_epochs", 100, 300),
            patience=trial.suggest_int("patience", 20, 60),
            batch_size=trial.suggest_categorical("batch_size", [256, 512, 1024]),
            virtual_batch_size=trial.suggest_categorical("virtual_batch_size", [32, 64, 128]),
            weights=sample_weight,
        )

        preds = model.predict(self.X_valid.values)  # uses threshold=0.5
        return precision_score(self.y_valid, preds, zero_division=0)

class MakeRequest:
    def __init__(
        self,
        proxies: dict[str, str],
        headers: dict[str, str] = None,
        max_retries: int = 3,
    ):
        self.proxies = proxies
        self.headers = headers if headers else self.get_headers()
        self.max_retries = max_retries

    def get(self, url: str, **kwargs) -> requests.Response:
        """Make a request to some API.

        Args:
            url (str): URL to make the request.

        Returns:
            requests.Response: Response from the API.
        """
        for _ in range(self.max_retries):
            try:
                response = requests.get(
                    url, headers=self.headers, proxies=self.proxies, **kwargs
                )
                response.raise_for_status()
                return response
            except requests.exceptions.ConnectionError as e:
                logging.error(f"Connection error when searching {url}: {e}")
                time.sleep(2)  # wait for 2 seconds before next retry
            except requests.exceptions.RequestException as e:
                logging.error(f"Error when searching {url}: {e}")
                return None  # return None if a non-connection error occurs
        logging.error(f"Max retries reached for {url}")
        return None  # return None if all retries failed

    @staticmethod
    def get_headers() -> dict[str, str]:
        """Get headers for the request.

        Returns:
            dict[str, str]: Headers for the request.
        """
        return {
            "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7,es;q=0.6,it;q=0.5,fr;q=0.4",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Referer": "https://www.google.com/",
            "User-Agent": UserAgent().random,
        }


class MercadoLivreScraper:
    URL = "https://api.mercadolibre.com/"

    def __init__(self, access_token: str = None, parse: bool = True):
        self.access_token = (
            access_token if access_token else os.getenv("ML_ACCESS_TOKEN")
        )
        self.parse = parse

    def get_catalog_competition(self, item_id: str) -> dict[str, any]:
        url = f"{self.URL}items/{item_id}/price_to_win?MLB&version=v2"
        response = MakeRequest(None, self._get_headers()).get(url)
        if self.parse and response:
            return MercadoLivreParser(response.json()).parse_catalog_competition()
        return response.json() if response else None

    def get_user_info(self, user_id: str) -> dict[str, any]:
        url = f"{self.URL}users/{user_id}"
        response = MakeRequest(None, self._get_headers()).get(url)
        if self.parse and response:
            return MercadoLivreParser(response.json()).parse_user_info()
        return response.json() if response else None

    def get_product_info(self, product_id: str) -> dict[str, any]:
        url = f"{self.URL}products/{product_id}"
        response = MakeRequest(None, self._get_headers()).get(url)
        if self.parse and response:
            return MercadoLivreParser(response.json()).parse_product_info()
        return response.json() if response else None

    def get_all_sellers_by_product(self, product_id: str) -> list[dict[str, any]]:
        """
        Get all sellers by product by iterating over the offset until there are no more results.
        """
        all_sellers = []

        sellers = self.get_sellers_by_product(product_id, parse=False)
        if sellers and "results" in sellers:
            all_sellers.append(sellers)

        if sellers and "paging" in sellers:
            total_pages = sellers["paging"].get("total", 0)
            page_size = sellers["paging"].get("limit", 0)

            for i in range(len(sellers["results"]), total_pages, page_size):
                sellers = self.get_sellers_by_product(product_id, offset=i, parse=False)
                if not sellers or "results" not in sellers:
                    break
                all_sellers.append(sellers)

        parsed_sellers = []
        if self.parse:
            for seller in all_sellers:
                seller["product_id"] = product_id
                parsed = MercadoLivreParser(seller).parse_sellers_by_product()
                parsed_sellers.append(
                    parsed.dropna(axis=1, how="all")
                )  # Drop empty columns

            return pd.concat(
                [df for df in parsed_sellers if not df.empty], ignore_index=True
            )

        return all_sellers

    def get_sellers_by_product(
        self, product_id: str, offset=0, parse=False
    ) -> dict[str, any]:
        url = f"{self.URL}products/{product_id}/items/?offset={offset}"
        response = MakeRequest(None, self._get_headers()).get(url)
        if parse and response:
            data = {"product_id": product_id} | response.json()
            return MercadoLivreParser(data).parse_sellers_by_product()
        return {"product_id": product_id} | response.json() if response else None

    def search_by_identifier(self, identifier: str) -> dict[str, any]:
        url = f"{self.URL}products/search?status=active&site_id=MLB&product_identifier={identifier}"
        response = MakeRequest(None, self._get_headers()).get(url)
        return response.json() if response else None

    def search_by_query(self, query: str) -> dict[str, any]:
        url = f"{self.URL}sites/MLB/search?q={query}"
        response = MakeRequest(None, self._get_headers()).get(url)
        return response.json() if response else None

    def get_itens_by_category(self, category_id: str) -> dict[str, any]:
        url = f"{self.URL}sites/MLB/search?category={category_id}"
        response = MakeRequest(None, self._get_headers()).get(url)
        return response.json() if response else None

    def get_item_info(self, item_id) -> dict[str, any]:
        if isinstance(item_id, list):
            item_id = ",".join(item_id)

        url = f"{self.URL}items?ids={item_id}"
        response = MakeRequest(None, self._get_headers()).get(url)
        return response.json() if response else None

    def best_selling_by_category(self, category_id: str) -> dict[str, any]:
        url = f"{self.URL}highlights/MLB/category/{category_id}"
        response = MakeRequest(None, self._get_headers()).get(url)
        if self.parse and response:
            return MercadoLivreParser(response.json()).parse_best_selling_by_category()
        return response.json() if response else None

    def list_all_items_by_user(
        self, user_id: str, n_items=1000, parameters: dict[str, str] = None
    ) -> dict[str, any]:
        items = []
        scroll_id = None

        for _ in tqdm.tqdm(range(n_items // 100), total=n_items // 100):
            seller_items = self.list_items_by_user(
                user_id, scroll_id=scroll_id, parameters=parameters
            )
            if not seller_items:
                break

            items.extend(seller_items["results"])
            scroll_id = seller_items.get("scroll_id")

            if not scroll_id:
                break

        return items

    def list_items_by_user(
        self, user_id: str, parameters: dict, scroll_id=None
    ) -> dict[str, any]:
        url = f"{self.URL}users/{user_id}/items/search?limit=100"
        if parameters:
            url += "&" + "&".join([f"{k}={v}" for k, v in parameters.items()])
        if scroll_id:
            url += f"&scroll_id={scroll_id}"
        response = MakeRequest(None, self._get_headers()).get(url)
        return response.json() if response else None

    def list_items_by_seller(
        self, seller_id: str, category_id: str, sleep_seconds=0.5
    ) -> dict[str, any]:
        url = f"{self.URL}sites/MLB/search?seller_id={seller_id}&category={category_id}&offset=0"
        response = MakeRequest(None, self._get_headers()).get(url)

        total_pages = response.json()["paging"]["total"]
        page_size = response.json()["paging"]["limit"]

        products_info = []
        for i in tqdm.tqdm(
            range(0, total_pages, page_size), total=total_pages // page_size
        ):
            try:
                time.sleep(sleep_seconds)
                url = f"{self.URL}sites/MLB/search?seller_id={seller_id}&category={category_id}&offset={i}&orders=available_quantity_desc"
                response = MakeRequest(None, self._get_headers()).get(url)
                if response:
                    products_info.extend(response.json()["results"])
            except Exception as e:
                logging.error(f"Error fetching page {i}: {e}")

        return {
            "products_info": products_info,
            "total_pages": total_pages,
            "page_size": page_size,
        }

    def _get_headers(self) -> dict[str, str]:
        if self.access_token is None:
            raise ValueError("Access token is required")

        return {"Authorization": f"Bearer {self.access_token}"}


class MercadoLivreParser:
    def __init__(self, response_data: dict[str, any]):
        self.response_data = response_data

    def parse_catalog_competition(self) -> pd.DataFrame:
        catalog = self.response_data

        if catalog["status"] == "not_listed":
            return pd.json_normalize(catalog)

        catalog.update(
            {boost["id"]: boost["status"] for boost in catalog.pop("boosts", [])}
        )

        if "winner" in catalog:
            catalog["winner"].update(
                {
                    boost["id"]: boost["status"]
                    for boost in catalog["winner"].pop("boosts", [])
                }
            )

        return pd.json_normalize(catalog)

    def parse_user_info(self) -> pd.DataFrame:
        user_info = self.response_data

        user_info["transactions_period"] = user_info["seller_reputation"][
            "transactions"
        ]["period"]
        user_info["transactions_total"] = user_info["seller_reputation"][
            "transactions"
        ]["total"]

        user_info["seller_reputation"].pop("transactions", None)

        user_info = (
            user_info
            | user_info["address"]
            | user_info["seller_reputation"]
            | user_info["status"]
        )

        for key in ["address", "seller_reputation", "status"]:
            user_info.pop(key, None)

        return pd.DataFrame([user_info])

    def parse_best_selling_by_category(self) -> pd.DataFrame:
        return pd.DataFrame(self.response_data["content"])

    def parse_sellers_by_product(self) -> pd.DataFrame:
        product_id = self.response_data["product_id"]

        seller_info = []
        for seller in self.response_data["results"]:
            seller = seller | {
                f"shipping_{key}": value for key, value in seller["shipping"].items()
            }

            seller["seller_city"] = seller["seller_address"]["city"]["name"]
            seller["seller_state"] = seller["seller_address"]["state"]["name"]
            seller["seller_neighborhood"] = seller["seller_address"]["neighborhood"][
                "name"
            ]

            seller = {
                k: v
                for k, v in seller.items()
                if k not in ["shipping", "seller_address"]
            }

            seller_info.append(seller)

        seller_info = pd.DataFrame(seller_info)
        seller_info.insert(0, "product_id", product_id)

        return seller_info

    def parse_product_info(self) -> pd.DataFrame:
        product_info = self.response_data

        for feature in product_info["main_features"]:
            text = feature["text"].split(":")
            if len(text) > 1:
                product_info[text[0]] = text[1].strip().replace(".", "")
            else:
                product_info["extra_feature"] = text[0]

        for attribute in product_info["attributes"]:
            product_info[attribute["id"]] = attribute["value_name"]

        product_info["short_description"] = product_info["short_description"]["content"]

        # Drop first level unnecessary keys
        drop_keys = [
            "main_features",
            "attributes",
            "disclaimers",
            "settings",
            "pictures",
            "pickers",
            "description_pictures",
            "children_ids",
        ]

        for key in drop_keys:
            product_info.pop(key, None)

        # Deal with buy box
        product_info["buy_box_winner"].pop("sale_terms", None)

        product_info["buy_box_city"] = product_info["buy_box_winner"]["seller_address"][
            "city"
        ]["name"]
        product_info["buy_box_state"] = product_info["buy_box_winner"][
            "seller_address"
        ]["state"]["name"]

        return pd.json_normalize(product_info)


class MercadoLivreAuthenticator:
    HEADERS = {
        "accept": "application/json",
        "content-type": "application/x-www-form-urlencoded",
    }
    URL = "https://api.mercadolibre.com/oauth/token"

    def __init__(
        self,
        app_id: str = None,
        secret_key: str = None,
        authorization_code: str = None,
        redirect_uri: str = None,
        code_verifier: str = None,
        update_dotenv: bool = True,
    ):
        self.app_id = app_id if app_id else os.getenv("ML_CLIENT_ID")
        self.secret_key = secret_key if secret_key else os.getenv("ML_CLIENT_SECRET")
        self.authorization_code = (
            authorization_code if authorization_code else os.getenv("ML_AUTH_CODE")
        )
        self.redirect_uri = (
            redirect_uri if redirect_uri else os.getenv("ML_REDIRECT_URI")
        )
        self.code_verifier = (
            code_verifier if code_verifier else os.getenv("ML_CODE_VERIFIER")
        )
        self.update_dotenv = update_dotenv

    def get_access_from_refresh_token(self) -> dict[str, any]:
        data = {
            "grant_type": "refresh_token",
            "client_id": self.app_id,
            "client_secret": self.secret_key,
            "refresh_token": os.getenv("ML_REFRESH_TOKEN"),
        }
        response = requests.post(self.URL, headers=self.HEADERS, data=data)
        data = response.json()

        self._update_dotenv(data) if self.update_dotenv else None

        return data

    def get_access_token(self) -> str:
        data = {
            "grant_type": "authorization_code",
            "client_id": self.app_id,
            "client_secret": self.secret_key,
            "code": self.authorization_code,
            "redirect_uri": self.redirect_uri,
            "code_verifier": "KINGSBRIDGE",
        }
        response = requests.post(self.URL, headers=self.HEADERS, data=data)
        data = response.json()

        self._update_dotenv(data) if self.update_dotenv else None

        return data

    def _update_dotenv(self, data: dict[str, any]):
        if "access_token" in data:
            logging.info("Updating .env file with new access token")
            set_key(".env", "ML_ACCESS_TOKEN", data["access_token"])
        if "refresh_token" in data:
            logging.info("Updating .env file with new refresh token")
            set_key(".env", "ML_REFRESH_TOKEN", data["refresh_token"])
