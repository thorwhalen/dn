"""
Converting markdown to other formats
"""

import json
import re
from pathlib import Path
from typing import Any, Optional, Union
from collections.abc import Callable


def markdown_to_notebook(
    markdown: str | Path | bytes, *, egress: Callable | str | None = None
) -> dict[str, Any] | Any:
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

            with open(full_path, "w", encoding="utf-8") as f:
                json.dump(notebook_dict, f, indent=1, ensure_ascii=False)

            return str(full_path)

        return save_notebook

    def _determine_output_path(input_path: str | None) -> str:
        """Determine output path when egress is '.ipynb'."""
        if input_path:
            return str(Path(input_path).with_suffix(".ipynb"))
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
        pattern = r"```(\w+)?\n(.*?)\n```"
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
        content = markdown.decode("utf-8")
    elif isinstance(markdown, Path):
        input_filepath = str(markdown)
        content = markdown.read_text(encoding="utf-8")
    elif isinstance(markdown, str):
        # Check if it's a reasonable file path (no newlines, reasonable length)
        if "\n" not in markdown and len(markdown) < 260:
            markdown_path = Path(markdown)
            try:
                if markdown_path.exists():
                    input_filepath = str(markdown_path)
                    content = markdown_path.read_text(encoding="utf-8")
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


# --- Markdown -> Slack mrkdwn ---------------------------------------------------------
# Slack's mrkdwn is not Markdown (``*bold*``, ``<url|text>``, no headings or tables, and
# ``&``, ``<``, ``>`` escaped). Stdlib only; every pattern is bounded, no output carries a
# live Slack mention, and input cannot forge the internal placeholders.

_FENCE_RE = re.compile(r"^(```|~~~)[^\n]*\n(.*?)^\1[ \t]*$", re.M | re.S)
_CODE_RE = re.compile(r"`([^`\n]{1,2000})`")
#: Link text has no brackets, so a run of ``[`` is scanned once; a URL may hold one level of parentheses.
_IMAGE_OR_LINK_RE = re.compile(
    r"!?\[([^\[\]\n]{0,1000})\]\(([^()\s<>]{1,2000}(?:\([^()\s<>]{0,200}\)[^()\s<>]{0,2000})*)(?:\s+\"[^\"\n]{0,200}\")?\)"
)
_AUTOLINK_RE = re.compile(r"<((?:https?://|mailto:)[^<>\s]{1,2000})>")
#: Link targets Slack is given as links; anything else (``!channel``, ``@U123``, ``#C1``) stays text.
_LINK_SCHEME_RE = re.compile(r"^(?:https?://|mailto:)", re.I)
_BACKSLASH_RE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!~|>])")
_HEADING_RE = re.compile(r"^#{1,6}[ \t]+([^\n]{1,1000}?)[ \t]*#*[ \t]*$", re.M)
_BOLD_ITALIC_RE = re.compile(r"\*\*\*(?=\S)([^*\n]{1,1000}?)(?<=\S)\*\*\*")
_BOLD_RE = re.compile(r"(\*\*|__)(?=\S)([^\n]{1,1000}?)(?<=\S)\1")
_ITALIC_RE = re.compile(r"(?<![\w*])\*(?=\S)([^*\n]{1,1000}?)(?<=\S)\*(?![\w*])")
_STRIKE_RE = re.compile(r"~~(?=\S)([^\n]{1,1000}?)(?<=\S)~~")
_RULE_RE = re.compile(
    r"^[ \t]*(?:(?:-[ \t]*){3,}|(?:\*[ \t]*){3,}|(?:_[ \t]*){3,})$", re.M
)
_BULLET_RE = re.compile(r"^([ \t]*)[-*+][ \t]+", re.M)
_TABLE_ROW_RE = re.compile(r"^[ \t]*\|.*\|[ \t]*$")
_TABLE_SEPARATOR_RE = re.compile(
    r"^[ \t]*\|?[ \t]*:?-{3,}:?[ \t]*(?:\|[ \t]*:?-{3,}:?[ \t]*)*\|?[ \t]*$"
)
_SLOT_RE = re.compile("\x00(\\d+)\x00")


def _escape(text: str) -> str:
    """Slack's three control characters escaped; an entity already written (``&amp;``) is read as its character first."""
    text = text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _url(url: str) -> str:
    """A link target safe inside ``<...|...>``: ``<``, ``>``, ``|`` and spaces percent-encoded."""
    return (
        url.replace("|", "%7C")
        .replace("<", "%3C")
        .replace(">", "%3E")
        .replace(" ", "%20")
    )


def markdown_to_mrkdwn(markdown: str) -> str:
    """Common Markdown as Slack ``mrkdwn``: bold, italic, strike, links, headings, lists, code, quotes, tables as code.

    Only ``http``, ``https`` and ``mailto`` targets become links: a Markdown link to
    ``!channel`` or ``@U123`` stays text, so no draft can notify a channel or a person.

    >>> markdown_to_mrkdwn("# Release\\n- **new**: _faster_ export\\n- ~~old~~ gone")
    '*Release*\\n• *new*: _faster_ export\\n• ~old~ gone'
    >>> markdown_to_mrkdwn("> quoted <b> & `a < b`")
    '> quoted &lt;b&gt; &amp; `a &lt; b`'
    >>> markdown_to_mrkdwn("[ping](!channel)")
    'ping (!channel)'
    """
    slots: list[str] = []

    def keep(text: str) -> str:
        slots.append(text)
        return f"\x00{len(slots) - 1}\x00"

    def resolve(text: str) -> str:
        # A slot refers only to slots made before it, so this ends.
        for _ in range(len(slots) + 1):
            if "\x00" not in text:
                return text
            text = _SLOT_RE.sub(lambda m: slots[int(m.group(1))], text)
        return text.replace("\x00", "")

    def link(m: re.Match) -> str:
        label, target = m.group(1), m.group(2)
        if not _LINK_SCHEME_RE.match(target):
            return keep(_escape(f"{label} ({target})" if label else target))
        return keep(
            f"<{_url(target)}|{_escape(label)}>" if label else f"<{_url(target)}>"
        )

    def table(lines: list[str]) -> str:
        return keep("```\n" + _escape("\n".join(lines)) + "\n```")

    text = markdown.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = _FENCE_RE.sub(lambda m: keep(f"```\n{_escape(m.group(2))}```"), text)
    lines, out, i = text.split("\n"), [], 0
    while i < len(lines):  # a table is a header row, a separator row, then rows
        if (
            i + 1 < len(lines)
            and _TABLE_ROW_RE.match(lines[i])
            and _TABLE_SEPARATOR_RE.match(lines[i + 1])
        ):
            j = i + 2
            while j < len(lines) and _TABLE_ROW_RE.match(lines[j]):
                j += 1
            out.append(table(lines[i:j]))
            i = j
        else:
            out.append(lines[i])
            i += 1
    text = "\n".join(out)
    text = _CODE_RE.sub(lambda m: keep(f"`{_escape(m.group(1))}`"), text)
    text = _BACKSLASH_RE.sub(lambda m: keep(_escape(m.group(1))), text)
    text = _AUTOLINK_RE.sub(lambda m: keep(f"<{_url(m.group(1))}>"), text)
    text = _IMAGE_OR_LINK_RE.sub(link, text)
    quoted = bool(
        re.search(r"^[ \t]*>", text, re.M)
    )  # Slack quotes a line starting with ">"
    text = _escape(text)
    if quoted:
        text = re.sub(r"^([ \t]*)&gt;", r"\1>", text, flags=re.M)
    text = _RULE_RE.sub(lambda m: keep("──────────"), text)

    def heading(m: re.Match) -> str:
        inner = _BOLD_RE.sub(lambda b: b.group(2), m.group(1))
        inner = _ITALIC_RE.sub(lambda it: f"_{it.group(1)}_", inner)
        return keep(f"*{inner}*")

    text = _HEADING_RE.sub(heading, text)
    text = _BOLD_ITALIC_RE.sub(lambda m: keep(f"*_{m.group(1)}_*"), text)
    text = _BOLD_RE.sub(lambda m: keep(f"*{m.group(2)}*"), text)
    text = _STRIKE_RE.sub(lambda m: keep(f"~{m.group(1)}~"), text)
    text = _ITALIC_RE.sub(lambda m: f"_{m.group(1)}_", text)
    text = _BULLET_RE.sub(r"\1• ", text)
    return resolve(text)


#: Short alias, the name this converter had in ``correspond``.
to_mrkdwn = markdown_to_mrkdwn
