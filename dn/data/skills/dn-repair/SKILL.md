---
name: dn-repair
description: Clean up and repair markdown text. Use when the user has messy markdown with broken links, ChatGPT citation artifacts, empty links, multi-line links, or double newlines. Triggers on "clean markdown", "fix markdown", "repair markdown", "remove chatgpt links", "fix broken links".
---

# Markdown Repair

Fix common markdown rendering issues using `dn.repair`.

## Quick start

```python
from dn.repair import repair_markdown

fixed = repair_markdown(messy_markdown)
```

## Available fixers

| Function | What it fixes |
|----------|--------------|
| `repair_markdown(md)` | Applies all fixers below in sequence |
| `fix_multiline_links(md)` | `[text\nmore](url)` → `[text](url) — more` |
| `fix_empty_links(md)` | `[](url)` → removed |
| `remove_hyperlink_crap(text)` | UTM tracking, `oai_citation` markers, double `[[X](Y)](Y)` links |
| `remove_improperly_double_newlines(text)` | `\n   \n` → `\n` |

## Usage patterns

### Fix a file in place
```python
from pathlib import Path
from dn.repair import repair_markdown

p = Path('messy.md')
p.write_text(repair_markdown(p.read_text()))
```

### Clean up AI-generated text
```python
from dn.repair import remove_hyperlink_crap
clean = remove_hyperlink_crap(chatgpt_output)
```

### Compose custom repair pipeline
```python
from dn.repair import fix_multiline_links, remove_hyperlink_crap

def my_repair(md):
    md = fix_multiline_links(md)
    md = remove_hyperlink_crap(md)
    # add your own fixers here
    return md
```
