# dn.to

Converting markdown to other formats

### Functions

| [`markdown_to_notebook`](#dn.to.markdown_to_notebook)(markdown, \*[, egress])   | Convert markdown content to Jupyter notebook format.   |
|-------------------------------------------------------------------------------------------------|--------------------------------------------------------|

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
