"""Markdown repair and cleanup utilities.

Pure string→string transforms for fixing common markdown artifacts.
These arise from web scraping, AI-generated text, format conversion,
or copy-paste from rich-text sources.

Main entry point::

    >>> from dn.repair import repair_markdown
    >>> repair_markdown('[Title\\n\\nDescription.](http://x.com)')
    '[Title](http://x.com) — Description.'

Individual fixers can be used standalone or composed.
"""

import re


def repair_markdown(md: str) -> str:
    """Fix common markdown rendering issues.

    Applies all available fixers in sequence:

    1. **Multi-line links**: ``[text with\\nnewlines](url)`` → single-line link
       with description appended after an em-dash.
    2. **Empty links**: ``[](url)`` → removed.

    This is useful as a post-processing step on any markdown, regardless of
    how it was generated.

    Parameters
    ----------
    md : str
        Markdown text to repair.

    Returns
    -------
    str
        Repaired markdown text.

    Examples
    --------
    >>> repair_markdown('[Title\\n\\nDescription.](http://x.com)')
    '[Title](http://x.com) — Description.'
    >>> repair_markdown('[Good link](http://x.com)')
    '[Good link](http://x.com)'
    >>> repair_markdown('before [](http://empty) after')
    'before  after'
    """
    md = fix_multiline_links(md)
    md = fix_empty_links(md)
    return md


def fix_multiline_links(md: str) -> str:
    """Fix markdown links whose text spans multiple lines.

    Collapses ``[title\\ndescription](url)`` into
    ``[title](url) — description``.

    Parameters
    ----------
    md : str
        Markdown text.

    Returns
    -------
    str
        Text with multi-line links collapsed.

    Examples
    --------
    >>> fix_multiline_links('[Hello\\n\\nWorld](http://x.com)')
    '[Hello](http://x.com) — World'
    >>> fix_multiline_links('[Single line](http://x.com)')
    '[Single line](http://x.com)'
    """

    def _collapse_link(match):
        full_text = match.group(1)
        url = match.group(2)
        lines = [line.strip() for line in full_text.split('\n') if line.strip()]
        if len(lines) <= 1:
            return match.group(0)  # Not actually multi-line
        title = lines[0]
        desc = ' '.join(lines[1:])
        return f'[{title}]({url}) — {desc}'

    return re.sub(
        r'\[([^\]]*\n[^\]]*)\]\(([^)]+)\)',
        _collapse_link,
        md,
    )


def fix_empty_links(md: str) -> str:
    """Remove empty links like ``[](url)``.

    Examples
    --------
    >>> fix_empty_links('before [](http://x.com) after')
    'before  after'
    >>> fix_empty_links('[real](http://x.com)')
    '[real](http://x.com)'
    """
    return re.sub(r'\[\s*\]\([^)]+\)', '', md)


def remove_hyperlink_crap(string: str) -> str:
    r"""Remove unwanted hyperlink artifacts from text.

    Cleans up common artifacts from AI-generated or copy-pasted text:

    - ChatGPT UTM tracking: ``?utm_source=chatgpt.com``
    - ChatGPT citation markers: ``oai_citation:\d+‡``
    - Double hyperlinks: ``[[X](Y)](Y)`` → ``[X](Y)``
      (common when copying from Claude)

    Parameters
    ----------
    string : str
        Text to clean.

    Returns
    -------
    str
        Cleaned text.

    Examples
    --------
    >>> remove_hyperlink_crap('See [here](http://x.com?utm_source=chatgpt.com)')
    'See [here](http://x.com)'
    >>> remove_hyperlink_crap('[[Title](http://x.com)](http://x.com)')
    '[Title](http://x.com)'
    """
    string = string.replace('?utm_source=chatgpt.com', '')
    string = string.replace('&utm_source=chatgpt.com', '')
    string = re.sub(r'oai_citation:\d*‡', '', string)

    # Remove double hyperlinks: [[X](Y)](Y) -> [X](Y)
    # Tolerant of trailing-slash differences
    pattern = r'\[\[([^\]]+)\]\(([^)]+?)/?\)\]\(\2/?\)'
    replacement = r'[\1](\2)'
    string = re.sub(pattern, replacement, string)

    return string


def remove_improperly_double_newlines(string: str) -> str:
    r"""Remove spurious double newlines caused by whitespace-only lines.

    Replaces patterns like ``\n   \n`` (newline, spaces, newline) with a
    single newline. Also normalizes ``\r\n`` and ``\n\r`` to ``\n``.

    Parameters
    ----------
    string : str
        Text to clean.

    Returns
    -------
    str
        Text with spurious double newlines removed.

    Examples
    --------
    >>> remove_improperly_double_newlines("a\n\nb\n  \nc")
    'a\n\nb\nc'
    """
    string = string.replace('\n\r', '\n').replace('\r\n', '\n')
    return re.sub(r'\n +\n', '\n', string)
