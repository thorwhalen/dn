# dn.src

Converting things to markdown

This module provides tools to convert various file formats (PDF, Word, Excel, PowerPoint, HTML)
to markdown format. It includes:

- Default converters for common file formats
- Functions to convert bytes to markdown
- Store-based converters for processing multiple files

### Functions

| [`add_dflt_converter`](#dn.src.add_dflt_converter)(input_format, converter)     | Add (or change) a default converter for a given input format.                                                                 |
|--------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| [`bytes_store_to_markdown_store`](#dn.src.bytes_store_to_markdown_store)(src_files[, ...]) | Converts files to markdown using the enhanced bytes_to_markdown function.                                                     |
| [`bytes_to_markdown`](#dn.src.bytes_to_markdown)(b[, input_format, key, ...])  | Convert bytes to markdown text using a flexible detection strategy.                                                           |
| [`default_fallback`](#dn.src.default_fallback)(b, input_format)               | Fallback converter that attempts basic text extraction.                                                                       |
| [`extensions_not_supported_by_converters`](#dn.src.extensions_not_supported_by_converters)(paths)   | Returns a set of file extensions that are not supported by the given converters.                                              |
| [`get_extension`](#dn.src.get_extension)(path)                             | Get the extension of a file path.                                                                                             |
| [`notebook_to_markdown`](#dn.src.notebook_to_markdown)([src_notebook, ...])       | Converts a Jupyter notebook to Markdown, applying a output text processors.                                                   |
| `truncate_text`(text, \*[, max_chars])                                                           |                                                                                                                               |
| [`try_to_convert_to_markdown`](#dn.src.try_to_convert_to_markdown)(data[, key, ...])    | Attempts to identify the content type of the given bytes and convert it to markdown using appropriate converters from dn.src. |

### dn.src.add_dflt_converter(input_format, converter)

Add (or change) a default converter for a given input format.

### dn.src.bytes_store_to_markdown_store(src_files, target_store=None, \*, converters={'azw': functools.partial(<function ebook_to_markdown>, input_format='azw'), 'azw3': functools.partial(<function ebook_to_markdown>, input_format='azw3'), 'azw4': functools.partial(<function ebook_to_markdown>, input_format='azw4'), 'chm': functools.partial(<function ebook_to_markdown>, input_format='chm'), 'docx': functools.partial(<function ebook_to_markdown>, input_format='docx'), 'epub': functools.partial(<function ebook_to_markdown>, input_format='epub'), 'fb2': functools.partial(<function ebook_to_markdown>, input_format='fb2'), 'fbz': functools.partial(<function ebook_to_markdown>, input_format='fbz'), 'htmlz': functools.partial(<function ebook_to_markdown>, input_format='htmlz'), 'ipynb': <function notebook_to_markdown>, 'kepub': functools.partial(<function ebook_to_markdown>, input_format='kepub'), 'lit': functools.partial(<function ebook_to_markdown>, input_format='lit'), 'lrf': functools.partial(<function ebook_to_markdown>, input_format='lrf'), 'mobi': functools.partial(<function ebook_to_markdown>, input_format='mobi'), 'odt': functools.partial(<function ebook_to_markdown>, input_format='odt'), 'pmlz': functools.partial(<function ebook_to_markdown>, input_format='pmlz'), 'pobi': functools.partial(<function ebook_to_markdown>, input_format='pobi'), 'rtf': functools.partial(<function ebook_to_markdown>, input_format='rtf'), 'txtz': functools.partial(<function ebook_to_markdown>, input_format='txtz'), 'updb': functools.partial(<function ebook_to_markdown>, input_format='updb')}, fallback=<function default_fallback>, ext_to_input_format=<function <lambda>>, try_bytes_detection=True, verbose=False, old_to_new_key=<function <lambda>>, target_store_egress=<function identity>)

Converts files to markdown using the enhanced bytes_to_markdown function.

* **Parameters:**
  * **src_files** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)]) – Source files as a directory path or mapping
  * **target_store** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`MutableMapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)]) – Target store for markdown output
  * **converters** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – Dictionary of format-specific converters
  * **fallback** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)) – Fallback conversion method
  * **ext_to_input_format** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Function to extract format from key
  * **try_bytes_detection** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to try content detection from bytes
  * **verbose** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to print detection information
  * **old_to_new_key** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Function to transform source keys to target keys
  * **target_store_egress** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)], [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]) – Function to process the target store before returning
* **Return type:**
  [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)
* **Returns:**
  Result of applying target_store_egress to the target store

### Examples

# Convert all files in a directory using extension detection
result = bytes_store_to_markdown_store(‘path/to/files’, ‘path/to/output’)

### Convert using content-based detection

result = bytes_store_to_markdown_store(‘path/to/files’, ‘path/to/output’,
: ext_to_input_format=None)

### dn.src.bytes_to_markdown(b, input_format=None, \*, key=None, converters={'azw': functools.partial(<function ebook_to_markdown>, input_format='azw'), 'azw3': functools.partial(<function ebook_to_markdown>, input_format='azw3'), 'azw4': functools.partial(<function ebook_to_markdown>, input_format='azw4'), 'chm': functools.partial(<function ebook_to_markdown>, input_format='chm'), 'docx': functools.partial(<function ebook_to_markdown>, input_format='docx'), 'epub': functools.partial(<function ebook_to_markdown>, input_format='epub'), 'fb2': functools.partial(<function ebook_to_markdown>, input_format='fb2'), 'fbz': functools.partial(<function ebook_to_markdown>, input_format='fbz'), 'htmlz': functools.partial(<function ebook_to_markdown>, input_format='htmlz'), 'ipynb': <function notebook_to_markdown>, 'kepub': functools.partial(<function ebook_to_markdown>, input_format='kepub'), 'lit': functools.partial(<function ebook_to_markdown>, input_format='lit'), 'lrf': functools.partial(<function ebook_to_markdown>, input_format='lrf'), 'mobi': functools.partial(<function ebook_to_markdown>, input_format='mobi'), 'odt': functools.partial(<function ebook_to_markdown>, input_format='odt'), 'pmlz': functools.partial(<function ebook_to_markdown>, input_format='pmlz'), 'pobi': functools.partial(<function ebook_to_markdown>, input_format='pobi'), 'rtf': functools.partial(<function ebook_to_markdown>, input_format='rtf'), 'txtz': functools.partial(<function ebook_to_markdown>, input_format='txtz'), 'updb': functools.partial(<function ebook_to_markdown>, input_format='updb')}, fallback=<function default_fallback>, ext_to_input_format=<function <lambda>>, try_bytes_detection=True, verbose=False)

Convert bytes to markdown text using a flexible detection strategy.

The function follows this logic to find a converter:

1. If input_format is provided, use it directly to find a converter
2. If input_format is None but key is provided, extract format using ext_to_input_format
3. If try_bytes_detection is True, attempt content-type detection from bytes
4. If no converter found through above methods, use the fallback

* **Parameters:**
  * **b** ([`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)) – Input bytes to convert
  * **input_format** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Format of the input (e.g., ‘pdf’, ‘docx’)
  * **key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Filename or identifier to help with format detection
  * **converters** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – Dictionary of format-specific converters
  * **fallback** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)) – Fallback conversion method
  * **ext_to_input_format** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – Function to extract format from key
  * **try_bytes_detection** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to try content detection from bytes
  * **verbose** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to print detection information
* **Returns:**
  Markdown-formatted text
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### Examples

# Convert with explicit format
md = bytes_to_markdown(pdf_bytes, ‘pdf’)

### Convert using filename extension

md = bytes_to_markdown(file_bytes, key=’document.docx’)

### Convert using only content-based detection

md = bytes_to_markdown(file_bytes, input_format=None,
: ext_to_input_format=None)

### Convert with explicit format and disable content detection

md = bytes_to_markdown(file_bytes, ‘xlsx’, try_bytes_detection=False)

### dn.src.default_fallback(b, input_format)

Fallback converter that attempts basic text extraction.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### dn.src.extensions_not_supported_by_converters(paths, converters={'azw': functools.partial(<function ebook_to_markdown>, input_format='azw'), 'azw3': functools.partial(<function ebook_to_markdown>, input_format='azw3'), 'azw4': functools.partial(<function ebook_to_markdown>, input_format='azw4'), 'chm': functools.partial(<function ebook_to_markdown>, input_format='chm'), 'docx': functools.partial(<function ebook_to_markdown>, input_format='docx'), 'epub': functools.partial(<function ebook_to_markdown>, input_format='epub'), 'fb2': functools.partial(<function ebook_to_markdown>, input_format='fb2'), 'fbz': functools.partial(<function ebook_to_markdown>, input_format='fbz'), 'htmlz': functools.partial(<function ebook_to_markdown>, input_format='htmlz'), 'ipynb': <function notebook_to_markdown>, 'kepub': functools.partial(<function ebook_to_markdown>, input_format='kepub'), 'lit': functools.partial(<function ebook_to_markdown>, input_format='lit'), 'lrf': functools.partial(<function ebook_to_markdown>, input_format='lrf'), 'mobi': functools.partial(<function ebook_to_markdown>, input_format='mobi'), 'odt': functools.partial(<function ebook_to_markdown>, input_format='odt'), 'pmlz': functools.partial(<function ebook_to_markdown>, input_format='pmlz'), 'pobi': functools.partial(<function ebook_to_markdown>, input_format='pobi'), 'rtf': functools.partial(<function ebook_to_markdown>, input_format='rtf'), 'txtz': functools.partial(<function ebook_to_markdown>, input_format='txtz'), 'updb': functools.partial(<function ebook_to_markdown>, input_format='updb')})

Returns a set of file extensions that are not supported by the given converters.

* **Parameters:**
  * **paths** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]) – A list of file paths or the root directory to check for paths.
  * **converters** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)) – An iterable of extensions supported by the converters.
    By default, it’s the default converters (a dict, so keys are considered).
* **Returns:**
  A set of file extensions that are not supported by the given converters.
* **Return type:**
  [*set*](https://docs.python.org/3/builtins/stdtypes.html#set)

### dn.src.get_extension(path)

Get the extension of a file path.

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### dn.src.notebook_to_markdown(src_notebook=None, \*, target_file=None, output_processors=(<function truncate_text>, ), read_encoding='utf-8', write_encoding='utf-8')

Converts a Jupyter notebook to Markdown, applying a output text processors.

* **Parameters:**
  * **src_notebook** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Path to the source notebook or URL.
  * **target_file** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Path to save the converted Markdown file. If None, returns the Markdown content.
    If it starts with ‘\*’, the ‘\*’ is replaced with the source notebook name.
    Popular value for target_file is ‘

    ```
    *
    ```

    .md’.
  * **output_processors** ([`Sequence`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence)[[`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)], [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]]) – List of functions to process cell outputs.
  * **read_encoding** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Encoding for reading the source notebook.
  * **write_encoding** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Encoding for writing the Markdown file.

### dn.src.try_to_convert_to_markdown(data, key=None, , converters=None, verbose=False)

Attempts to identify the content type of the given bytes and convert it to markdown
using appropriate converters from dn.src.

* **Parameters:**
  * **data** ([`bytes`](https://docs.python.org/3/builtins/stdtypes.html#bytes)) – The byte data to convert
  * **key** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Optional identifier/filename to help with content detection and for verbose output
  * **converters** ([`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)] | [`None`](https://docs.python.org/3/builtins/constants.html#None)) – Optional dictionary of content type to converter function mappings
    If None, uses the default converters from contexts module
  * **verbose** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to print information about the conversion process
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`None`](https://docs.python.org/3/builtins/constants.html#None)
* **Returns:**
  Markdown string if conversion successful, None otherwise
