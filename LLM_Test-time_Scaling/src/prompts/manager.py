"""Prompt manager for loading and managing prompt templates."""

import json
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .templates import PromptTemplate, PromptType


class PromptManager:
    """Manager for loading and accessing prompt templates."""

    def __init__(self, templates_dir: Optional[Path] = None):
        """Initialize the prompt manager.

        Args:
            templates_dir: Directory containing prompt templates
        """
        if templates_dir is None:
            templates_dir = Path(__file__).parent / "templates"

        self.templates_dir = Path(templates_dir)
        self.templates: Dict[str, PromptTemplate] = {}
        self._load_templates()

    def _load_templates(self) -> None:
        """Load all prompt templates from the templates directory."""
        if not self.templates_dir.exists():
            self.templates_dir.mkdir(parents=True, exist_ok=True)
            return

        for template_file in self.templates_dir.glob("*.yaml"):
            self._load_template_file(template_file)

        for template_file in self.templates_dir.glob("*.json"):
            self._load_template_file(template_file)

    def _load_template_file(self, file_path: Path) -> None:
        """Load a single template file.

        Args:
            file_path: Path to the template file
        """

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                if file_path.suffix == ".yaml":
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)

            if isinstance(data, list):
                for template_data in data:
                    template = PromptTemplate(**template_data)
                    self.templates[template.name] = template
            else:
                template = PromptTemplate(**data)
                self.templates[template.name] = template

        except Exception as e:
            print(f"Error loading template from {file_path}: {e}")

    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get a template by name.

        Args:
            name: Name of the template

        Returns:
            PromptTemplate if found, None otherwise
        """
        return self.templates.get(name)

    def get_templates_by_type(self, prompt_type: PromptType) -> List[PromptTemplate]:
        """Get all templates of a specific type.

        Args:
            prompt_type: Type of prompts to retrieve

        Returns:
            List of matching templates
        """
        return [t for t in self.templates.values() if t.prompt_type == prompt_type]

    def get_templates_by_domain(self, domain: str) -> List[PromptTemplate]:
        """Get all templates for a specific task domain.

        Args:
            domain: Task domain (e.g., "math", "coding")

        Returns:
            List of matching templates
        """
        return [t for t in self.templates.values() if t.task_domain == domain]

    def register_template(self, template: PromptTemplate) -> None:
        """Register a new template.

        Args:
            template: Template to register
        """
        self.templates[template.name] = template

    def list_templates(self) -> List[str]:
        """List all available template names.

        Returns:
            List of template names
        """
        return list(self.templates.keys())
