# dn.util

General utilities for dn

This module provides core utility functions used throughout the dn package, including:

- File path handling (fullpath)
- URL detection and content retrieval (is_url, url_to_contents)
- File saving utilities (save_to_file_and_return_file)
- Basic helper functions (identity)

### Functions

| [`fullpath`](#dn.util.fullpath)(path)                                    | Returns the full path of the given path.                                                              |
|----------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| [`identity`](#dn.util.identity)(x)                                       | Returns the input unchanged.                                                                          |
| [`is_url`](#dn.util.is_url)(path)                                      | Check if the given path is a URL.                                                                     |
| [`save_to_file_and_return_file`](#dn.util.save_to_file_and_return_file)([obj, encoder, key]) | Save `encoder(obj)` to a file using a random name in `rootdir` (or a temp directory if not provided). |
| [`url_to_contents`](#dn.util.url_to_contents)(url)                              | Fetch the content of a URL.                                                                           |

### dn.util.fullpath(path)

Returns the full path of the given path.

* **Parameters:**
  **path** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – The path to convert to a full path.
* **Returns:**
  The full path.
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### Example

```pycon
>>> fullpath('~/Downloads')
'/home/user/Downloads'
```

```pycon
>>> fullpath('.')
'/home/user/python_projects/aix/aix'
```

### dn.util.identity(x)

Returns the input unchanged.

* **Parameters:**
  **x** – Any input
* **Returns:**
  The input unchanged

### dn.util.is_url(path)

Check if the given path is a URL.

* **Parameters:**
  **path** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Path to check
* **Returns:**
  True if the path is a URL, False otherwise
* **Return type:**
  [`bool`](https://docs.python.org/3/builtins/functions.html#bool)

### dn.util.save_to_file_and_return_file(obj=None, \*, encoder=<function identity>, key=None)

Save `encoder(obj)` to a file using a random name in `rootdir` (or a temp directory if not provided).
Returns the full path to the saved file.
If `obj` is None, returns a partial function with preconfigured `encoder` and
`rootdir`.

* **Parameters:**
  * **obj** – The object to save. If None, return a partial function.
  * **encoder** – A function to encode the object into text or bytes. Defaults to identity.
  * **key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)) – The key (by default, filepath) to write to.
    If None, a temporary file is created.
    If a string starting with ‘\*’, the ‘\*’ is replaced with a unique temporary filename.
    If a string that has a ‘\*’ somewhere in the middle, what’s on the left of if is used as a directory
    and the ‘\*’ is replaced with a unique temporary filename. For example
    ‘/tmp/\*_file.ext’ would be replaced with ‘/tmp/oiu8fj9873_file.ext’.
    If a callable, it will be called with obj as input to get the key. One use case
    is to use a function that generates a key based on the object.
* **Returns:**
  Full path to the saved file, or a partial function if `obj` is None.
* **Return type:**
  [*str*](https://docs.python.org/3/builtins/stdtypes.html#str)

### Examples

```pycon
>>> from pathlib import Path
>>> filepath = save_to_file_and_return_file("hello world")
>>> import os
>>> Path(filepath).read_text()
'hello world'
```

The default encoder is identity, so you can save binary data as well:

```pycon
>>> filepath = save_to_file_and_return_file(b"binary data", encoder=lambda x: x)
>>> Path(filepath).read_bytes()
b'binary data'
```

### dn.util.url_to_contents(url)

Fetch the content of a URL.

* **Parameters:**
  **url** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – URL to fetch
* **Returns:**
  Content of the URL
* **Return type:**
  [*bytes*](https://docs.python.org/3/builtins/stdtypes.html#bytes)
* **Raises:**
  **HTTPError** – If the request fails
