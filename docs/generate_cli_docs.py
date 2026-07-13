#!/usr/bin/env python3
"""
Script to automatically generate CLI documentation from areal.api.cli_args dataclasses.
This creates markdown documentation compatible with jupyter-book.

The script automatically discovers all dataclasses in the cli_args module and generates
documentation with appropriate categorization and hyperlinks.
"""

import inspect
import sys
import types
from dataclasses import MISSING as DATACLASSES_MISSING
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Union, get_args, get_origin

import mdformat
from omegaconf import MISSING as OMEGACONF_MISSING

# Add the project root to the path so we can import areal
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import areal.api.cli_args as cli_args_module  # noqa: E402


def discover_dataclasses() -> dict[str, Any]:
    """Discover all dataclasses in the cli_args module."""
    dataclasses = {}
    for name in dir(cli_args_module):
        obj = getattr(cli_args_module, name)
        if inspect.isclass(obj) and is_dataclass(obj) and not name.startswith("_"):
            dataclasses[name] = obj
    return dataclasses


def categorize_dataclasses(
    dataclasses: dict[str, Any],
) -> dict[str, list[tuple[str, Any]]]:
    """Categorize dataclasses by their purpose/type."""
    categories = {
        "Core Experiment Configurations": [],
        "Training Configurations": [],
        "Inference Configurations": [],
        "Dataset": [],
        "System and Cluster Configurations": [],
        "Logging and Monitoring": [],
        "Others": [],
    }

    # Define categorization rules - only include the most important configs
    experiment_configs = [
        "BaseExperimentConfig",
        "SFTConfig",
        "GRPOConfig",
        "PPOConfig",
        "RWConfig",
    ]
    training_configs = [
        "TrainEngineConfig",
        "PPOActorConfig",
        "PPOCriticConfig",
        "OptimizerConfig",
        "MicroBatchSpec",
        "NormConfig",
        "FSDPEngineConfig",
        "FSDPWrapPolicy",
    ]
    inference_configs = [
        "InferenceEngineConfig",
        "SGLangConfig",
        "vLLMConfig",
        "GenerationHyperparameters",
    ]
    dataset_configs = ["TrainDatasetConfig", "ValidDatasetConfig"]
    system_configs = [
        "ClusterSpecConfig",
        "NameResolveConfig",
        "LauncherConfig",
        "SlurmLauncherConfig",
    ]
    logging_configs = [
        "StatsLoggerConfig",
        "WandBConfig",
        "SwanlabConfig",
        "TensorBoardConfig",
        "TrackioConfig",
        "SaverConfig",
        "EvaluatorConfig",
        "RecoverConfig",
    ]

    for name, cls in dataclasses.items():
        if name in experiment_configs:
            categories["Core Experiment Configurations"].append((name, cls))
        elif name in training_configs:
            categories["Training Configurations"].append((name, cls))
        elif name in inference_configs:
            categories["Inference Configurations"].append((name, cls))
        elif name in dataset_configs:
            categories["Dataset"].append((name, cls))
        elif name in system_configs:
            categories["System and Cluster Configurations"].append((name, cls))
        elif name in logging_configs:
            categories["Logging and Monitoring"].append((name, cls))
        else:
            # All other configs go to "Others" section
            categories["Others"].append((name, cls))

    # Remove empty categories
    return {k: v for k, v in categories.items() if v}


def get_anchor_name(class_name: str) -> str:
    """Convert a class name to an anchor name for hyperlinks."""
    # Convert CamelCase to kebab-case
    import re

    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1-\2", class_name)
    s2 = re.sub("([a-z0-9])([A-Z])", r"\1-\2", s1).lower()

    # Clean up common patterns
    s2 = s2.replace("-config", "").replace("-spec", "")

    return s2


def get_class_description(cls: Any) -> str:
    """Get description for a dataclass from its docstring."""
    if cls.__doc__ and not cls.__doc__.startswith(cls.__name__ + "("):
        # Clean up the docstring
        doc = cls.__doc__.strip()
        # Take only the first line or sentence
        first_line = doc.split("\n")[0].strip()
        if first_line:
            return first_line

    # Fallback for classes without docstrings
    return f"Configuration class: {cls.__name__}"


def get_type_description(field_type, all_dataclasses: dict[str, Any]) -> str:
    """Convert a type annotation to a readable string."""
    # Handle union types (Type | None)
    origin = get_origin(field_type)
    # Check if it's a Union type (includes both Union[Type, None] and Type | None syntax)
    if origin is Union or isinstance(field_type, types.UnionType):
        args = get_args(field_type)
        # Check if it's a union with None (optional type)
        if len(args) == 2 and type(None) in args:
            non_none_type = args[0] if args[1] is type(None) else args[1]
            return f"{get_type_description(non_none_type, all_dataclasses)} &#124; None"
        else:
            # Multiple non-None types in union
            return " &#124; ".join(
                get_type_description(arg, all_dataclasses) for arg in args
            )

    # Handle basic types
    if field_type is int:
        return "integer"
    elif field_type is float:
        return "float"
    elif field_type is str:
        return "string"
    elif field_type is bool:
        return "boolean"
    elif field_type is list or get_origin(field_type) is list:
        if get_args(field_type):
            inner_type = get_args(field_type)[0]
            return f"list of {get_type_description(inner_type, all_dataclasses)}"
        return "list"
    elif hasattr(field_type, "__name__") and field_type.__name__ in all_dataclasses:
        # Create hyperlinks for dataclass types
        class_name = field_type.__name__
        # Convert class name to anchor format (lowercase with hyphens)
        anchor_name = get_anchor_name(class_name)
        return f"[`{class_name}`](section-{anchor_name})"
    elif hasattr(field_type, "__name__"):
        return f"`{field_type.__name__}`"
    else:
        return str(field_type).replace("typing.", "")


def format_default_value(field_obj) -> str:
    """Format default values for display."""

    if field_obj.default is not inspect._empty:
        default_value = field_obj.default
        # Check for MISSING by string representation to avoid import issues
        if default_value is DATACLASSES_MISSING or default_value is OMEGACONF_MISSING:
            return "**Required**"
        elif default_value is None:
            return "`None`"
        elif isinstance(default_value, str):
            return f'`"{default_value}"`'
        elif isinstance(default_value, list) and len(default_value) == 0:
            return "`[]`"
        elif isinstance(default_value, bool):
            return f"`{default_value}`"
        else:
            return f"`{default_value}`"
    elif field_obj.default_factory is not inspect._empty:
        try:
            factory_result = field_obj.default_factory()
            if isinstance(factory_result, list) and len(factory_result) == 0:
                return "`[]`"
            elif isinstance(factory_result, dict) and len(factory_result) == 0:
                return "`{}`"
            else:
                return f"*{type(factory_result).__name__}*"
        except Exception:
            return f"*default {field_obj.default_factory.__name__}*"
    else:
        return "`None`"


def generate_config_section(
    config_class,
    all_dataclasses: dict[str, Any],
    title: str = "",
    description: str = "",
    anchor: str = "",
) -> str:
    """Generate documentation for a single configuration dataclass."""
    if not is_dataclass(config_class):
        return ""

    # Auto-generate title and description if not provided
    if not title:
        title = config_class.__name__.replace("Config", " Configuration").replace(
            "Spec", " Specification"
        )
        # Handle special cases
        if title.endswith(" Configuration Configuration"):
            title = title.replace(" Configuration Configuration", " Configuration")

    if not description:
        description = get_class_description(config_class)

    if not anchor:
        anchor = get_anchor_name(config_class.__name__)

    # Create anchor for table of contents linking
    doc = f"(section-{anchor})=\n## {title}\n\n"

    if description:
        doc += f"{description}\n\n"

    # Add additional docstring content if available (beyond the first line used in description)
    if config_class.__doc__ and not config_class.__doc__.startswith(
        config_class.__name__ + "("
    ):
        docstring = config_class.__doc__.strip()
        lines = docstring.split("\n")
        if len(lines) > 1:
            # Use remaining lines after first line
            remaining_doc = "\n".join(lines[1:]).strip()
            if remaining_doc:
                doc += f"{remaining_doc}\n\n"

    doc += "| Parameter | Type | Default | Description |\n"
    doc += "|-----------|------|---------|-------------|\n"

    for field in fields(config_class):
        field_name = field.name
        field_type = get_type_description(field.type, all_dataclasses)
        default_value = format_default_value(field)

        # Get help text from metadata
        help_text = field.metadata.get(
            "help",
            "-",
        )

        # Get choices if available
        choices = field.metadata.get("choices")
        if choices:
            help_text += f" **Choices:** {', '.join([f'`{c}`' for c in choices])}"

        doc += f"| `{field_name}` | {field_type} | {default_value} | {help_text} |\n"

    doc += "\n"
    return doc


def generate_cli_documentation(lang: str = "en"):
    """Generate the complete CLI documentation automatically."""
    # Discover all dataclasses
    all_dataclasses = discover_dataclasses()

    # Categorize them
    categories = categorize_dataclasses(all_dataclasses)

    # Header templates for different languages
    headers = {
        "en": """# Configurations

This page provides a comprehensive reference for all configuration parameters available in AReaL's command-line interface. These parameters are defined using dataclasses and can be specified in YAML configuration files or overridden via command line arguments.

## Usage

Configuration files are specified using the `--config` parameter:

```bash
python3 train.py --config path/to/config.yaml
```

You can override specific parameters from the command line:

```bash
python3 train.py --config path/to/config.yaml actor.lr=1e-4 seed=42
```

For detailed examples, see the experiment configurations in the `examples/` directory.

## Table of Contents

""",
        "zh": """# 配置参考

本页面提供 AReaL 命令行界面所有配置参数的完整参考。这些参数使用 dataclass 定义，可在 YAML 配置文件中指定，也可通过命令行参数覆盖。

## 使用方法

使用 `--config` 参数指定配置文件：

```bash
python3 train.py --config path/to/config.yaml
```

您可以通过命令行覆盖特定参数：

```bash
python3 train.py --config path/to/config.yaml actor.lr=1e-4 seed=42
```

详细示例请参阅 `examples/` 目录中的实验配置。

## 目录

""",
    }

    # Get header for selected language
    doc = headers.get(lang, headers["en"])

    # Generate table of contents automatically
    for category_name, class_list in categories.items():
        doc += f"### {category_name}\n"
        for class_name, cls in class_list:
            anchor = get_anchor_name(class_name)
            title = class_name.replace("Config", " Configuration").replace(
                "Spec", " Specification"
            )
            if title.endswith(" Configuration Configuration"):
                title = title.replace(" Configuration Configuration", " Configuration")
            doc += f"- [{title}](section-{anchor})\n"
        doc += "\n"

    doc += "---\n\n"

    # Generate documentation sections automatically
    for category_name, class_list in categories.items():
        for class_name, cls in class_list:
            doc += generate_config_section(cls, all_dataclasses)

    return doc


def main():
    """Generate the CLI documentation and save it to markdown files."""
    docs_dir = Path(__file__).parent
    output_paths = {
        "en": docs_dir / "en" / "cli_reference.md",
        "zh": docs_dir / "zh" / "cli_reference.md",
    }

    try:
        for lang in ["en", "zh"]:
            documentation = generate_cli_documentation(lang)
            documentation = mdformat.text(
                documentation,
                options={"wrap": 88},
                extensions=["gfm", "tables", "frontmatter"],
            )

            output_path = output_paths[lang]
            # Ensure parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                f.write(documentation)

            print(f"✅ CLI documentation ({lang}) generated at: {output_path}")

        return True

    except Exception as e:
        print(f"❌ Error generating documentation: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
