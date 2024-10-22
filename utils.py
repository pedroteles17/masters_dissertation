import requests
import time
import logging
import tqdm
import pandas as pd
import os
import json
from dotenv import set_key

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
                parsed_sellers.append(parsed.dropna(axis=1, how='all')) # Drop empty columns
            
            return pd.concat([df for df in parsed_sellers if not df.empty], ignore_index=True)
        
        return all_sellers

    def get_sellers_by_product(self, product_id: str, offset = 0, parse = False) -> dict[str, any]:
        url = f"{self.URL}products/{product_id}/items/?offset={offset}"
        response = MakeRequest(None, self._get_headers()).get(url)
        if parse and response:
            data = {"product_id": product_id} | response.json()
            return MercadoLivreParser(data).parse_sellers_by_product()
        return {"product_id": product_id} | response.json() if response else None

    def search_by_term(self, term: str) -> dict[str, any]:
        url = f"{self.URL}sites/MLB/search?q={term}"
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

    def list_all_items_by_user(self, user_id: str, n_items = 1000, parameters: dict[str, str] = None) -> dict[str, any]:
        items = []
        scroll_id = None

        for _ in tqdm.tqdm(range(n_items//100), total=n_items//100):
            seller_items = self.list_items_by_user(user_id, scroll_id=scroll_id, parameters=parameters)
            if not seller_items:
                break

            items.extend(seller_items["results"])
            scroll_id = seller_items.get("scroll_id")

            if not scroll_id:
                break

        return items

    def list_items_by_user(self, user_id: str, parameters: dict, scroll_id = None) -> dict[str, any]:
        url = f"{self.URL}users/{user_id}/items/search?limit=100"
        if parameters:
            url += "&" + "&".join([f"{k}={v}" for k, v in parameters.items()])
        if scroll_id:
            url += f"&scroll_id={scroll_id}"
        response = MakeRequest(None, self._get_headers()).get(url)
        return response.json() if response else None
        
    def list_items_by_seller(self, seller_id: str, category_id: str, sleep_seconds = 0.5) -> dict[str, any]:
        url = f"{self.URL}sites/MLB/search?seller_id={seller_id}&category={category_id}&offset=0"
        response = MakeRequest(None, self._get_headers()).get(url)

        total_pages = response.json()["paging"]["total"]
        page_size = response.json()["paging"]["limit"]

        products_info = []
        for i in tqdm.tqdm(range(0, total_pages, page_size), total=total_pages//page_size):
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
            "page_size": page_size
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

        catalog.update({boost["id"]: boost["status"] for boost in catalog.pop("boosts", [])})

        if "winner" in catalog:
            catalog["winner"].update({boost["id"]: boost["status"] for boost in catalog["winner"].pop("boosts", [])})

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
            seller = seller | {f"shipping_{key}": value for key, value in seller["shipping"].items()}

            seller["seller_city"] = seller["seller_address"]["city"]["name"]
            seller["seller_state"] = seller["seller_address"]["state"]["name"]
            seller["seller_neighborhood"] = seller["seller_address"]["neighborhood"][
                "name"
            ]

            seller = {
                k: v
                for k, v in seller.items()
                if k
                not in ["shipping", "seller_address"]
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