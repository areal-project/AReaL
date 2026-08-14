# memrl/envs/webshop_simple.py
"""
Simplified WebShop environment for memory transfer experiments.
Uses a mock environment instead of full WebShop to avoid heavy dependencies.
"""

import random
import json
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

# Sample product data for mock environment
SAMPLE_PRODUCTS = [
    {"id": 1, "name": "Laptop Stand", "price": 29.99, "category": "electronics", "attrs": ["adjustable", "aluminum"]},
    {"id": 2, "name": "Wireless Mouse", "price": 19.99, "category": "electronics", "attrs": ["bluetooth", "ergonomic"]},
    {"id": 3, "name": "USB-C Hub", "price": 39.99, "category": "electronics", "attrs": ["7-port", "fast-charging"]},
    {"id": 4, "name": "Mechanical Keyboard", "price": 79.99, "category": "electronics", "attrs": ["rgb", "cherry-mx"]},
    {"id": 5, "name": "Monitor Light Bar", "price": 49.99, "category": "electronics", "attrs": ["led", "dimmable"]},
    {"id": 6, "name": "Running Shoes", "price": 89.99, "category": "clothing", "attrs": ["breathable", "cushioned"]},
    {"id": 7, "name": "Cotton T-Shirt", "price": 14.99, "category": "clothing", "attrs": ["soft", "blue"]},
    {"id": 8, "name": "Yoga Mat", "price": 24.99, "category": "sports", "attrs": ["non-slip", "thick"]},
    {"id": 9, "name": "Water Bottle", "price": 12.99, "category": "sports", "attrs": ["insulated", "32oz"]},
    {"id": 10, "name": "Desk Lamp", "price": 34.99, "category": "home", "attrs": ["led", "touch-control"]},
]

SAMPLE_GOALS = [
    "Find a laptop stand that is adjustable and under $40",
    "Buy a wireless mouse with bluetooth for under $25",
    "Get a USB-C hub with fast charging",
    "Find a mechanical keyboard with RGB lighting",
    "Buy running shoes that are breathable under $100",
    "Get a cotton t-shirt in blue color",
    "Find a yoga mat that is non-slip",
    "Buy an insulated water bottle",
    "Find an LED desk lamp with touch control",
    "Get a monitor light bar that is dimmable",
]


@dataclass
class WebShopState:
    """Current state of the WebShop environment"""
    page: str = "search"  # search, results, product, done
    query: str = ""
    results: List[Dict] = None
    selected_product: Optional[Dict] = None
    cart: List[Dict] = None

    def __post_init__(self):
        if self.results is None:
            self.results = []
        if self.cart is None:
            self.cart = []


class SimpleWebShopEnv:
    """
    Simplified WebShop environment for memory transfer experiments.

    Actions:
    - search[query]: Search for products
    - click[product_name]: Select a product
    - click[Buy Now]: Add to cart and complete
    - click[Back to Search]: Return to search
    """

    def __init__(
        self,
        products: List[Dict] = None,
        goals: List[str] = None,
        seed: int = 42,
        **kwargs
    ):
        self.products = products or SAMPLE_PRODUCTS
        self.goals = goals or SAMPLE_GOALS
        self.seed = seed
        self.rng = random.Random(seed)

        self.state = WebShopState()
        self.goal = ""
        self.goal_idx = 0
        self.steps = 0
        self.max_steps = kwargs.get('max_steps', 15)

    def reset(self, session: int = None) -> Tuple[str, Dict]:
        """Reset environment with a new goal"""
        if session is not None:
            self.goal_idx = session % len(self.goals)
        else:
            self.goal_idx = self.rng.randint(0, len(self.goals) - 1)

        self.goal = self.goals[self.goal_idx]
        self.state = WebShopState()
        self.steps = 0

        obs = self._get_observation()
        info = {
            'goal': self.goal,
            'available_actions': self._get_available_actions()
        }
        return obs, info

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """Execute action and return (obs, reward, done, info)"""
        self.steps += 1
        reward = 0.0
        done = False

        action = action.lower().strip()

        if action.startswith('search['):
            query = action[7:-1] if action.endswith(']') else action[7:]
            self._do_search(query)

        elif action.startswith('click['):
            target = action[6:-1] if action.endswith(']') else action[6:]
            reward, done = self._do_click(target)

        elif action == 'done':
            done = True
            reward = self._calculate_reward()

        # Check max steps
        if self.steps >= self.max_steps:
            done = True

        obs = self._get_observation()
        info = {
            'goal': self.goal,
            'available_actions': self._get_available_actions(),
            'reward': reward
        }

        return obs, reward, done, info

    def _do_search(self, query: str):
        """Perform product search"""
        self.state.query = query
        self.state.page = "results"

        # Simple keyword matching
        query_words = query.lower().split()
        results = []
        for product in self.products:
            score = 0
            text = f"{product['name']} {product['category']} {' '.join(product['attrs'])}".lower()
            for word in query_words:
                if word in text:
                    score += 1
            if score > 0:
                results.append((score, product))

        # Sort by relevance
        results.sort(key=lambda x: -x[0])
        self.state.results = [p for _, p in results[:5]]

    def _do_click(self, target: str) -> Tuple[float, bool]:
        """Handle click action"""
        target_lower = target.lower()

        if target_lower == 'buy now' and self.state.selected_product:
            self.state.cart.append(self.state.selected_product)
            self.state.page = "done"
            reward = self._calculate_reward()
            return reward, True

        elif target_lower == 'back to search':
            self.state.page = "search"
            self.state.selected_product = None
            return 0.0, False

        else:
            # Try to find product by name
            for product in self.state.results:
                if target_lower in product['name'].lower():
                    self.state.selected_product = product
                    self.state.page = "product"
                    return 0.0, False

        return 0.0, False

    def _calculate_reward(self) -> float:
        """Calculate reward based on goal completion"""
        if not self.state.cart:
            return 0.0

        product = self.state.cart[-1]
        goal_lower = self.goal.lower()

        # Check if product matches goal
        score = 0.0

        # Name match
        if product['name'].lower() in goal_lower or any(w in goal_lower for w in product['name'].lower().split()):
            score += 0.3

        # Category match
        if product['category'].lower() in goal_lower:
            score += 0.2

        # Attribute match
        for attr in product['attrs']:
            if attr.lower() in goal_lower:
                score += 0.2

        # Price constraint
        if 'under' in goal_lower:
            try:
                # Extract price limit from goal
                import re
                match = re.search(r'under \$?(\d+)', goal_lower)
                if match:
                    limit = float(match.group(1))
                    if product['price'] <= limit:
                        score += 0.3
            except:
                pass

        return min(score, 1.0)

    def _get_observation(self) -> str:
        """Get current observation as text"""
        if self.state.page == "search":
            return f"[Search Page]\nGoal: {self.goal}\n\nEnter a search query to find products."

        elif self.state.page == "results":
            obs = f"[Search Results for '{self.state.query}']\nGoal: {self.goal}\n\nProducts:\n"
            for i, p in enumerate(self.state.results, 1):
                obs += f"{i}. {p['name']} - ${p['price']:.2f} ({', '.join(p['attrs'])})\n"
            if not self.state.results:
                obs += "No products found. Try a different search.\n"
            return obs

        elif self.state.page == "product":
            p = self.state.selected_product
            obs = f"[Product Page]\nGoal: {self.goal}\n\n"
            obs += f"Name: {p['name']}\n"
            obs += f"Price: ${p['price']:.2f}\n"
            obs += f"Category: {p['category']}\n"
            obs += f"Features: {', '.join(p['attrs'])}\n"
            obs += "\n[Buy Now] [Back to Search]"
            return obs

        elif self.state.page == "done":
            return f"[Order Complete]\nYou purchased: {self.state.cart[-1]['name']}"

        return "Unknown state"

    def _get_available_actions(self) -> List[str]:
        """Get list of available actions"""
        if self.state.page == "search":
            return ["search[query]"]
        elif self.state.page == "results":
            actions = ["search[query]", "click[Back to Search]"]
            for p in self.state.results:
                actions.append(f"click[{p['name']}]")
            return actions
        elif self.state.page == "product":
            return ["click[Buy Now]", "click[Back to Search]"]
        return []

    def render(self, mode='text') -> str:
        """Render current state"""
        return self._get_observation()

    def close(self):
        """Clean up"""
        pass


# Factory function for compatibility
def make_simple_webshop_env(seed: int = 42, **kwargs) -> SimpleWebShopEnv:
    """Create a simple WebShop environment"""
    return SimpleWebShopEnv(seed=seed, **kwargs)
