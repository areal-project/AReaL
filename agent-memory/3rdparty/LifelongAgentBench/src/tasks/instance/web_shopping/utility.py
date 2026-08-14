import json
import re
from decimal import Decimal
from pydantic import BaseModel


class Goal(BaseModel):
    asin: str
    category: str
    query: str
    name: str
    product_category: str
    instruction_text: str
    attribute_list: list[str]
    price_upper: float
    goal_option_list: list[str]
    weight: int


class Utility:
    PRICE_RANGE = [10.0 * i for i in range(1, 100)]

    @staticmethod
    def load_product(
        product_data_path: str,
        product_attribute_path: str,
        human_attribute_path: str,
    ):
        raw_product_list = json.load(open(product_data_path))
        product_attribute_dict = json.load(open(product_attribute_path))
        human_attribute_dict = json.load(open(human_attribute_path))
        # region Clean product keys
        for product in raw_product_list:
            product.pop("product_information", None)
            product.pop("brand", None)
            product.pop("brand_url", None)
            product.pop("list_price", None)
            product.pop("availability_quantity", None)
            product.pop("availability_status", None)
            product.pop("total_reviews", None)
            product.pop("total_answered_questions", None)
            product.pop("seller_id", None)
            product.pop("seller_name", None)
            product.pop("fulfilled_by_amazon", None)
            product.pop("fast_track_message", None)
            product.pop("aplus_present", None)
            product.pop("small_description_old", None)
        # endregion
        asin_set = set()
        processed_product_list = []

        for i, product in enumerate(raw_product_list):
            # region Filter data based on asin
            asin = product["asin"]
            if asin == "nan" or len(asin) > 10:
                continue
            if asin in asin_set:
                continue
            asin_set.add(asin)
            # endregion
            # region Format keys
            # region Do some replacement
            # The original implementation is very ugly
            product["query"] = product["query"].lower().strip()
            product["Title"] = product["name"]
            product["Description"] = product["full_description"]
            product["Reviews"] = []
            product["Rating"] = "N.A."
            product["BulletPoints"] = (
                product["small_description"]
                if isinstance(product["small_description"], list)
                else [product["small_description"]]
            )
            product["MainImage"] = product["images"][0]
            # endregion
            # region Handle pricing, price_tag
            original_pricing = product.get("pricing")
            assert isinstance(original_pricing, str) or original_pricing is None
            if original_pricing is None or not original_pricing:
                pricing = [100.0]
                price_tag = "$100.0"
            else:
                pricing = [
                    float(Decimal(re.sub(r"[^\d.]", "", price)))
                    for price in original_pricing.split("$")[1:]
                ]
                if len(pricing) == 1:
                    price_tag = f"${pricing[0]}"
                else:
                    price_tag = f"${pricing[0]} to ${pricing[1]}"
                    pricing = pricing[:2]
            product["pricing"] = pricing
            product["Price"] = price_tag
            # endregion
            # region Handle options, customization_options
            option_dict = dict()
            customization_options = product["customization_options"]
            option_to_image = dict()
            for option_name, option_contents in customization_options.items():
                if option_contents is None:
                    continue
                option_name = option_name.lower()
                option_value_list = []
                for option_content in option_contents:
                    option_value = (
                        option_content["value"].strip().replace("/", " | ").lower()
                    )
                    option_image = option_content.get("image", None)

                    option_value_list.append(option_value)
                    option_to_image[option_value] = option_image
                option_dict[option_name] = option_value_list
            product["options"] = option_dict
            product["option_to_image"] = option_to_image
            # endregion
            # region Handle product_attribute and human_instruction
            if (
                asin in product_attribute_dict
                and "attributes" in product_attribute_dict[asin]
            ):
                product["Attributes"] = product_attribute_dict[asin]["attributes"]
            else:
                product["Attributes"] = ["DUMMY_ATTR"]
            if asin in human_attribute_dict:
                product["instructions"] = human_attribute_dict[asin]
            # endregion
            # endregion
            processed_product_list.append(product)
        attribute_to_asin_dict = {}  # The original implementation is defaultdict(set)
        for product in processed_product_list:
            for attribute in product["Attributes"]:
                attribute_to_asin_dict.setdefault(attribute, set()).add(product["asin"])
        asin_to_product_dict = {
            product["asin"]: product for product in processed_product_list
        }
        product_to_price_dict = dict()
        for product in processed_product_list:
            asin: str = product["asin"]
            pricing = product["pricing"]
            if not pricing:
                raise RuntimeError("Should not reach here")
            elif len(pricing) == 1:
                price = pricing[0]
            else:
                # The original implementation is:
                # price = random.uniform(*pricing[:2])
                # But I do not want to introduce randomness here
                price = sum(pricing) / 2
            product_to_price_dict[asin] = price
        return (
            processed_product_list,
            asin_to_product_dict,
            product_to_price_dict,
            attribute_to_asin_dict,
        )

    @staticmethod
    def get_goal(processed_product_list, product_to_price_dict) -> list[Goal]:
        goal_list = []
        for outer_product in processed_product_list:
            asin: str = outer_product["asin"]
            if "instructions" not in outer_product:
                continue
            for inner_product in outer_product["instructions"]:
                attributes: list[str] = inner_product["instruction_attributes"]
                if len(attributes) == 0:
                    continue
                price = product_to_price_dict[asin]
                price_range = [p for p in Utility.PRICE_RANGE if p > price][:4]
                if len(price_range) >= 2:
                    # The original implementation is:
                    # _, price_upper = sorted(random.sample(price_range, 2))
                    price_upper = price_range[-1]
                    price_text = f", and price lower than {price_upper:.2f} dollars"
                else:
                    # No price limitation
                    price_upper = 1000000
                    price_text = ""

                # region Maintain goal_list
                goal_list.append(
                    Goal(
                        asin=asin,
                        category=outer_product["category"],
                        query=outer_product["query"],
                        name=outer_product["name"],
                        product_category=outer_product["product_category"],
                        instruction_text=inner_product["instruction"].strip(".")
                        + price_text,
                        attribute_list=attributes,
                        price_upper=price_upper,
                        goal_option_list=inner_product["instruction_options"],
                        weight=1,
                    )
                )
                # endregion
        return goal_list
