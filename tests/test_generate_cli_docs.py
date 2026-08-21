from dataclasses import dataclass, field, fields

import pytest

from docs.generate_cli_docs import format_default_value


@dataclass
class _DefaultValueConfig:
    required: int
    optional_list: list[str] = field(default_factory=list)
    optional_dict: dict[str, str] = field(default_factory=dict)


@pytest.mark.parametrize(
    ("field_name", "expected"),
    [
        ("required", "**Required**"),
        ("optional_list", "`[]`"),
        ("optional_dict", "`{}`"),
    ],
)
def test_format_default_value_distinguishes_factories_from_required_fields(
    field_name: str, expected: str
):
    """Dataclass factories should render their value instead of Required."""
    field_by_name = {
        field_obj.name: field_obj for field_obj in fields(_DefaultValueConfig)
    }

    assert format_default_value(field_by_name[field_name]) == expected
