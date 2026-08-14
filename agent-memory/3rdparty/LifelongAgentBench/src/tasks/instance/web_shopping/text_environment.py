from .utility import Utility
from typing import Optional

from src.typings import SampleIndex


class WebShoppingTextEnvironment:
    def __init__(
        self,
        product_data_path: str,
        product_attribute_path: str,
        human_attribute_path: str,
        server_root_url: str,
    ):
        self.server_root_url = server_root_url
        self.server = SimulatedServer(
            product_data_path,
            product_attribute_path,
            human_attribute_path,
            server_root_url,
        )
        self.browser = SimulatedBrowser(self.server)
        self.web_session_index: Optional[SampleIndex] = None
        self.reset()

    def reset(self, sample_index: Optional[SampleIndex] = None) -> None:
        if sample_index is None:
            # The original implementation generates a random session_index here
            self.web_session_index = "dummy"


class SimulatedServer:
    def __init__(
        self,
        product_data_path: str,
        product_attribute_path: str,
        human_attribute_path: str,
        server_root_url: str,
    ):
        self.server_root_url = server_root_url
        (
            self.processed_product_list,
            self.asin_to_product_dict,
            self.product_to_price_dict,
            _,
        ) = Utility.load_product(
            product_data_path, product_attribute_path, human_attribute_path
        )
        self.search_engine = None
        self.goal_list = Utility.get_goal(
            self.processed_product_list, self.product_to_price_dict
        )


class SimulatedBrowser:
    def __init__(self, server: SimulatedServer):
        self.server = server
