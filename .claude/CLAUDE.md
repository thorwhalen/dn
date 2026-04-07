# dn - Document-to-Markdown Conversion

dn converts documents (PDF, Word, Excel, PowerPoint, HTML, Jupyter notebooks) to
markdown, and provides markdown repair/cleanup utilities.

## Package Structure

```
dn/
  src.py     # Format converters: bytes_to_markdown, pdf_to_markdown, etc.
  to.py      # Reverse: markdown_to_notebook
  repair.py  # Markdown repair: repair_markdown, remove_hyperlink_crap, etc.
  util.py    # Utilities: path handling, URL fetching
```

## Key Functions

- `bytes_to_markdown(data, input_format=None, key=None)` — smart converter with format detection
- `repair_markdown(md)` — fix broken links, empty links, and other artifacts
- `remove_hyperlink_crap(text)` — strip ChatGPT/Claude citation artifacts
- `remove_improperly_double_newlines(text)` — fix spurious whitespace-only blank lines
- `notebook_to_markdown(nb_source)` — Jupyter notebook to markdown

## Dependencies

Core: `dol`. Optional extras: `dn[pdf]`, `dn[word]`, `dn[html]`, `dn[all]`.
