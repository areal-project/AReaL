from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from areal.infra.data_service.gateway.auth import (
    DatasetKeyRegistry,
    extract_bearer_token,
    require_admin_key,
)


class TestDatasetKeyRegistry:
    def test_generate_key_returns_string(self):
        registry = DatasetKeyRegistry(admin_api_key="admin-key")
        key = registry.generate_key("dataset-a")
        assert isinstance(key, str)
        assert key.startswith("ds-")

    def test_generate_key_unique(self):
        registry = DatasetKeyRegistry(admin_api_key="admin-key")
        key1 = registry.generate_key("dataset-a")
        key2 = registry.generate_key("dataset-b")
        assert key1 != key2

    def test_resolve_returns_dataset_id(self):
        registry = DatasetKeyRegistry(admin_api_key="admin-key")
        key = registry.generate_key("dataset-a")
        assert registry.resolve(key) == "dataset-a"

    def test_resolve_unknown_returns_none(self):
        registry = DatasetKeyRegistry(admin_api_key="admin-key")
        assert registry.resolve("ds-does-not-exist") is None

    def test_revoke_removes_key(self):
        registry = DatasetKeyRegistry(admin_api_key="admin-key")
        key = registry.generate_key("dataset-a")
        assert registry.revoke("dataset-a") == key
        assert registry.resolve(key) is None

    def test_revoke_unknown_returns_none(self):
        registry = DatasetKeyRegistry(admin_api_key="admin-key")
        assert registry.revoke("dataset-missing") is None

    def test_is_admin_correct_key(self):
        registry = DatasetKeyRegistry(admin_api_key="admin-key")
        assert registry.is_admin("admin-key") is True

    def test_is_admin_wrong_key(self):
        registry = DatasetKeyRegistry(admin_api_key="admin-key")
        assert registry.is_admin("not-admin") is False

    def test_is_admin_timing_safe(self):
        registry = DatasetKeyRegistry(admin_api_key="admin-key")
        assert registry.is_admin("admin-key") is True
        assert registry.is_admin("admin-keyx") is False
        assert registry.is_admin("admin-keY") is False

    def test_is_valid_dataset_key_after_generate(self):
        registry = DatasetKeyRegistry(admin_api_key="admin-key")
        key = registry.generate_key("dataset-a")
        assert registry.is_valid_dataset_key(key) is True

    def test_is_valid_dataset_key_unknown(self):
        registry = DatasetKeyRegistry(admin_api_key="admin-key")
        assert registry.is_valid_dataset_key("ds-unknown") is False

    def test_is_valid_dataset_key_after_revoke(self):
        registry = DatasetKeyRegistry(admin_api_key="admin-key")
        key = registry.generate_key("dataset-a")
        registry.revoke("dataset-a")
        assert registry.is_valid_dataset_key(key) is False

    def test_generate_revoke_generate_new_key(self):
        registry = DatasetKeyRegistry(admin_api_key="admin-key")
        first_key = registry.generate_key("dataset-a")
        registry.revoke("dataset-a")
        second_key = registry.generate_key("dataset-a")
        assert second_key != first_key
        assert registry.resolve(second_key) == "dataset-a"

    def test_multiple_datasets_independent(self):
        registry = DatasetKeyRegistry(admin_api_key="admin-key")
        key_a = registry.generate_key("dataset-a")
        key_b = registry.generate_key("dataset-b")
        assert registry.resolve(key_a) == "dataset-a"
        assert registry.resolve(key_b) == "dataset-b"
        registry.revoke("dataset-a")
        assert registry.resolve(key_a) is None
        assert registry.resolve(key_b) == "dataset-b"


class TestExtractBearerToken:
    def test_extract_bearer_token_with_valid_header(self):
        request = SimpleNamespace(headers={"authorization": "Bearer token-123"})
        assert extract_bearer_token(request) == "token-123"

    def test_extract_bearer_token_missing_header_raises_401(self):
        request = SimpleNamespace(headers={})
        with pytest.raises(HTTPException) as exc_info:
            extract_bearer_token(request)
        assert exc_info.value.status_code == 401

    def test_extract_bearer_token_basic_auth_raises_401(self):
        request = SimpleNamespace(headers={"authorization": "Basic token-123"})
        with pytest.raises(HTTPException) as exc_info:
            extract_bearer_token(request)
        assert exc_info.value.status_code == 401

    def test_extract_bearer_token_empty_header_raises_401(self):
        request = SimpleNamespace(headers={"authorization": ""})
        with pytest.raises(HTTPException) as exc_info:
            extract_bearer_token(request)
        assert exc_info.value.status_code == 401


class TestRequireAdminKey:
    def test_require_admin_key_accepts_valid_admin_token(self):
        request = SimpleNamespace(headers={"authorization": "Bearer admin-key"})
        assert require_admin_key(request, "admin-key") == "admin-key"

    def test_require_admin_key_rejects_non_admin_token(self):
        request = SimpleNamespace(headers={"authorization": "Bearer user-key"})
        with pytest.raises(HTTPException) as exc_info:
            require_admin_key(request, "admin-key")
        assert exc_info.value.status_code == 403
