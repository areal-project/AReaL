"""Tests for safe experiment configuration serialization."""

from areal.utils.config_utils import REDACTED_VALUE, redact_sensitive_config


def test_redact_sensitive_config_redacts_nested_credentials_only():
    """Credentials should be removed without hiding token-count configuration."""
    config = {
        "admin_api_key": "admin-secret",
        "arena": {"access_token": "arena-secret"},
        "workers": [{"password": "worker-secret", "max_tokens": 131072}],
        "tokenizer_path": "/models/tokenizer",
    }

    redacted = redact_sensitive_config(config)

    assert redacted == {
        "admin_api_key": REDACTED_VALUE,
        "arena": {"access_token": REDACTED_VALUE},
        "workers": [{"password": REDACTED_VALUE, "max_tokens": 131072}],
        "tokenizer_path": "/models/tokenizer",
    }
    assert config["admin_api_key"] == "admin-secret"


def test_redact_sensitive_config_preserves_empty_optional_credentials():
    """Empty optional credential fields should remain empty for readability."""
    config = {"wandb_api_key": "", "refresh_token": None}

    assert redact_sensitive_config(config) == config
