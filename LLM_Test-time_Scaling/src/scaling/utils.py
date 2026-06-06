"""Utility functions for solution processing."""


def extract_detailed_solution(solution: str, marker: str = "Detailed Solution", after: bool = True) -> str:
    """Extract the detailed solution section from a formatted solution.

    This function is used to extract the "Detailed Solution" part from solutions
    that follow a structured format with sections like "Summary" and "Detailed Solution".

    Based on the extract_detailed_solution function from the IMO25-reflection reference:
    https://github.com/openai/imo25-reflection

    Args:
        solution: The full solution text
        marker: The section marker to search for (default: "Detailed Solution")
        after: If True, return text after the marker; if False, return text before
               (default: True)

    Returns:
        Extracted text, or empty string if marker not found
    """
    # Search for the marker (case-insensitive, flexible formatting)
    # Common patterns: "### Detailed Solution ###", "**2. Detailed Solution**", etc.
    idx = solution.find(marker)

    if idx == -1:
        # Try with common formatting patterns if not found
        patterns = [
            f"### {marker} ###",
            f"## {marker} ##",
            f"**{marker}**",
            f"# {marker}",
            f"{marker}"
        ]
        for pattern in patterns:
            idx = solution.find(pattern)
            if idx != -1:
                marker = pattern
                break

    if idx == -1:
        # Marker not found, return full solution
        return solution.strip()

    if after:
        # Return everything after the marker
        extracted = solution[idx + len(marker):].strip()
        # Remove leading "#" symbols and whitespace from the extracted part
        extracted = extracted.lstrip('#').strip()
        return extracted
    else:
        # Return everything before the marker
        return solution[:idx].strip()


def extract_section(solution: str, section_marker: str) -> str:
    """Extract a specific section from a structured solution.

    Args:
        solution: The full solution text
        section_marker: The marker identifying the section (e.g., "Summary", "Method Sketch")

    Returns:
        The extracted section content, or empty string if not found
    """
    # Find the start of the section
    patterns = [
        f"### {section_marker} ###",
        f"## {section_marker} ##",
        f"**{section_marker}**",
        f"# {section_marker}",
        section_marker,
    ]

    start_idx = -1
    marker_used = None

    for pattern in patterns:
        start_idx = solution.find(pattern)
        if start_idx != -1:
            marker_used = pattern
            break

    if start_idx == -1:
        return ""

    # Find the end of the section (next section marker or end of text)
    content_start = start_idx + len(marker_used)

    # Look for the next section marker
    next_section_idx = len(solution)
    for pattern in ["###", "##", "**1.", "**2.", "**3."]:
        idx = solution.find(pattern, content_start)
        if idx != -1 and idx < next_section_idx:
            next_section_idx = idx

    extracted = solution[content_start:next_section_idx].strip()
    return extracted


def has_structured_format(solution: str) -> bool:
    """Check if a solution follows the structured format with Summary and Detailed Solution.

    Args:
        solution: The solution text to check

    Returns:
        True if the solution appears to have the structured format
    """
    markers = [
        "Summary",
        "Detailed Solution",
        "Method Sketch",
        "Verdict",
    ]

    found_markers = sum(1 for marker in markers if marker in solution)
    return found_markers >= 2
