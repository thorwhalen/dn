# dn.to

Converting markdown to other formats

### Module Attributes

| [`to_mrkdwn`](#dn.to.to_mrkdwn)(markdown)   | Short alias, the name this converter had in `correspond`.   |
|------------------------------------------------------------------------|-------------------------------------------------------------|

### Functions

| [`markdown_to_mrkdwn`](#dn.to.markdown_to_mrkdwn)(markdown)                 | Common Markdown as Slack `mrkdwn`: bold, italic, strike, links, headings, lists, code, quotes, tables as code.   |
|-----------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| [`markdown_to_notebook`](#dn.to.markdown_to_notebook)(markdown, \*[, egress]) | Convert markdown content to Jupyter notebook format.                                                             |
| [`to_mrkdwn`](#dn.to.to_mrkdwn)(markdown)                          | Short alias, the name this converter had in `correspond`.                                                        |

### dn.to.markdown_to_mrkdwn(markdown)

Common Markdown as Slack `mrkdwn`: bold, italic, strike, links, headings, lists, code, quotes, tables as code.

Only `http`, `https` and `mailto` targets become links: a Markdown link to
`!channel` or `@U123` stays text, so no draft can notify a channel or a person.

```pycon
>>> markdown_to_mrkdwn("# Release\n- **new**: _faster_ export\n- ~~old~~ gone")
'*Release*\n• *new*: _faster_ export\n• ~old~ gone'
>>> markdown_to_mrkdwn("> quoted <b> & `a < b`")
'> quoted &lt;b&gt; &amp; `a &lt; b`'
>>> markdown_to_mrkdwn("[ping](!channel)")
'ping (!channel)'
```

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### dn.to.markdown_to_notebook(markdown, , egress=None)

Convert markdown content to Jupyter notebook format.

* **Parameters:**
  * **markdown** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Path`](https://docs.python.org/3/library/pathlib.html#pathlib.Path) | [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)) – Markdown content as string, file path, or bytes
  * **egress** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable) | [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Optional output handler - callable or filepath string
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)] | [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)
* **Returns:**
  Notebook dict or result of egress function

```pycon
>>> # Basic usage with string content
>>> content = "# Test\n\n```python\nprint('hello')\n```"
>>> nb = markdown_to_notebook(content)
>>> len(nb['cells'])
2
```

### dn.to.to_mrkdwn(markdown)

Short alias, the name this converter had in `correspond`.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)
