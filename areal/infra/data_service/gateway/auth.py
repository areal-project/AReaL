# SPDX-License-Identifier: Apache-2.0

"""Authentication and API key registry for the data service gateway."""

from __future__ import annotations

import hmac
import uuid

from fastapi import HTTPException, Request

from areal.utils import logging

logger = logging.getLogger("DataGatewayAuth")


class DatasetKeyRegistry:
    """Maps API keys to dataset IDs. Manages admin + dataset keys."""

    def __init__(self, admin_api_key: str):
        self._admin_key = admin_api_key
        self._key_to_dataset: dict[str, str] = {}  # api_key → dataset_id
        self._dataset_to_key: dict[str, str] = {}  # dataset_id → api_key

    def generate_key(self, dataset_id: str) -> str:
        """Generate a new API key for a dataset."""
        api_key = f"ds-{uuid.uuid4().hex[:16]}"
        self._key_to_dataset[api_key] = dataset_id
        self._dataset_to_key[dataset_id] = api_key
        logger.info("Generated API key for dataset %s", dataset_id)
        return api_key

    def resolve(self, api_key: str) -> str | None:
        """Resolve API key to dataset_id. Returns None if not found."""
        return self._key_to_dataset.get(api_key)

    def revoke(self, dataset_id: str) -> str | None:
        """Revoke API key for a dataset. Returns the revoked key."""
        api_key = self._dataset_to_key.pop(dataset_id, None)
        if api_key:
            self._key_to_dataset.pop(api_key, None)
        return api_key

    def is_admin(self, api_key: str) -> bool:
        return hmac.compare_digest(api_key, self._admin_key)

    def is_valid_dataset_key(self, api_key: str) -> bool:
        return api_key in self._key_to_dataset


def extract_bearer_token(request: Request) -> str:
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    raise HTTPException(
        status_code=401, detail="Missing or malformed Authorization header."
    )


def require_admin_key(request: Request, admin_api_key: str) -> str:
    token = extract_bearer_token(request)
    if not hmac.compare_digest(token, admin_api_key):
        raise HTTPException(status_code=403, detail="Admin API key required.")
    return token
