"""
Converting markdown to other formats
"""

import json
import re
from pathlib import Path
from typing import Any, Callable, Optional, Union


def markdown_to_notebook(
    markdown: Union[str, Path, bytes], *, egress: Optional[Union[Callable, str]] = None
) -> Union[dict[str, Any], Any]:
    """Convert markdown content to Jupyter notebook format.

    Args:
        markdown: Markdown content as string, file path, or bytes
        egress: Optional output handler - callable or filepath string

    Returns:
        Notebook dict or result of egress function

    >>> # Basic usage with string content
    >>> content = "# Test\\n\\n```python\\nprint('hello')\\n```"
    >>> nb = markdown_to_notebook(content)
    >>> len(nb['cells'])
    2
    """

    def _create_save_egress(filepath: str) -> Callable[[dict], str]:
        """Create egress function that saves notebook to file."""

        def save_notebook(notebook_dict: dict[str, Any]) -> str:
            full_path = Path(filepath).expanduser().resolve()
            full_path.parent.mkdir(parents=True, exist_ok=True)

            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(notebook_dict, f, indent=1, ensure_ascii=False)

            return str(full_path)

        return save_notebook

    def _determine_output_path(input_path: Optional[str]) -> str:
        """Determine output path when egress is '.ipynb'."""
        if input_path:
            return str(Path(input_path).with_suffix('.ipynb'))
        return "./markdown_to_notebook.ipynb"

    def _parse_markdown_content(content: str) -> list[dict[str, Any]]:
        """Parse markdown content into notebook cells."""

        def _create_code_cell(code: str, language: str = "python") -> dict[str, Any]:
            """Create a code cell."""
            return {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in code.split("\n")],
            }

        def _create_markdown_cell(text: str) -> dict[str, Any]:
            """Create a markdown cell."""
            return {
                "cell_type": "markdown",
                "metadata": {},
                "source": [line + "\n" for line in text.split("\n")],
            }

        # Split content by code blocks
        pattern = r'```(\w+)?\n(.*?)\n```'
        parts = re.split(pattern, content, flags=re.DOTALL)

        cells = []
        i = 0

        while i < len(parts):
            if i % 3 == 0:  # Markdown content
                text = parts[i].strip()
                if text:
                    cells.append(_create_markdown_cell(text))
            elif i % 3 == 1:  # Language identifier
                language = parts[i] or "python"
                code = parts[i + 1].strip()
                if code:
                    cells.append(_create_code_cell(code, language))
                i += 1  # Skip the code content part
            i += 1

        return cells

    # Handle input type and extract content
    input_filepath = None

    if isinstance(markdown, bytes):
        content = markdown.decode('utf-8')
    elif isinstance(markdown, Path):
        input_filepath = str(markdown)
        content = markdown.read_text(encoding='utf-8')
    elif isinstance(markdown, str):
        # Check if it's a reasonable file path (no newlines, reasonable length)
        if '\n' not in markdown and len(markdown) < 260:
            markdown_path = Path(markdown)
            try:
                if markdown_path.exists():
                    input_filepath = str(markdown_path)
                    content = markdown_path.read_text(encoding='utf-8')
                else:
                    # Treat as string content
                    content = markdown
            except OSError:
                # File path too long or invalid, treat as content
                content = markdown
        else:
            # Definitely content, not a file path
            content = markdown
    else:
        raise TypeError("markdown must be str, Path, or bytes")

    # Parse markdown into cells
    cells = _parse_markdown_content(content)

    # Create notebook structure
    notebook_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.5",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    # Handle egress
    if egress is None:
        return notebook_dict
    elif callable(egress):
        return egress(notebook_dict)
    elif isinstance(egress, str):
        if egress == ".ipynb":
            output_path = _determine_output_path(input_filepath)
        else:
            output_path = egress

        save_egress = _create_save_egress(output_path)
        return save_egress(notebook_dict)
    else:
        raise TypeError("egress must be callable, string, or None")
