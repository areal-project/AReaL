import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATTERN = re.compile(
    r"<!-- areal: offline-dataset-example start -->\s*"
    r"```python\r?\n(.*?)\r?\n```\s*"
    r"<!-- areal: offline-dataset-example end -->",
    re.DOTALL,
)


def _extract_example(path: Path) -> str:
    match = EXAMPLE_PATTERN.search(path.read_text(encoding="utf-8"))
    assert match is not None
    return match.group(1)


def test_offline_dataset_example_is_identical_and_executes():
    english_path = REPO_ROOT / "docs" / "en" / "customization" / "dataset.md"
    chinese_path = REPO_ROOT / "docs" / "zh" / "customization" / "dataset.md"
    english_code = _extract_example(english_path)
    chinese_code = _extract_example(chinese_path)

    assert chinese_code == english_code
    for path, code in ((english_path, english_code), (chinese_path, chinese_code)):
        namespace = {"__name__": "offline_dataset_docs_example"}
        exec(compile(code, str(path), "exec"), namespace)
