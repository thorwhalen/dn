# dn

Markdown parsing and generation

To install: `pip install dn`


## Optional Dependencies

This package supports converting various file formats to Markdown, with each format requiring specific dependencies:


    Format      Required Package(s)
    ----------- -----------------
    PDF         pypdf
    Word        mammoth
    Excel       pandas, openpyxl, tabulate
    PowerPoint  python-pptx
    HTML        html2text
    Notebooks   nbconvert, nbformat

**Installation Options**

You can install these dependencies after the fact, if and when package complains it needs some specific resource. 

You can also install these when installing `dn`, like so:

```bash
    # Install with minimal dependencies
    pip install dn

    # Install with support for specific formats
    pip install dn[pdf]               # PDF conversion support
    pip install dn[word]              # Word document support
    pip install dn[excel]             # Excel support
    pip install dn[powerpoint]        # PowerPoint support
    pip install dn[html]              # HTML conversion
    pip install dn[notebook]          # Jupyter Notebook support

    # Install multiple format support
    pip install dn[pdf,word,excel]    # Multiple formats

    # Install all optional dependencies
    pip install dn[all]
```

# Examples

## To and from jupyter notebooks


```python
from dn import markdown_to_notebook

sample_markdown = """# Sample Notebook

This is a markdown cell with some explanation.

```python
# This is a code cell
print("Hello, World!")
x = 42
print(f"The answer is {x}")
```

## Another Section

More markdown content here.

```python
# Another code cell
def greet(name):
    return f"Hello, {name}!"

print(greet("Jupyter"))
```

Final markdown cell."""
    

# Test basic functionality
notebook = markdown_to_notebook(sample_markdown)
print(f"Created notebook with {len(notebook['cells'])} cells")

# Test with file output
output_path = markdown_to_notebook(
    sample_markdown,
    egress="./sample_notebook.ipynb"
)
print(f"Saved notebook to: {output_path}")
```

    Created notebook with 5 cells
    Saved notebook to: /Users/thorwhalen/Dropbox/py/proj/t/dn/misc/sample_notebook.ipynb



```python
from dn import notebook_to_markdown

md_string = notebook_to_markdown(notebook)
print(md_string)
```

    # Sample Notebook
    
    This is a markdown cell with some explanation.
    
    
    
    ```python
    # This is a code cell
    print("Hello, World!")
    x = 42
    print(f"The answer 
    ...
    nt(greet("Jupyter"))
    
    ```
    
    Final markdown cell.
    
    


## ... and other formats


```python
from dn import pdf_to_markdown  # requires pypdf
```


```python
from dn import docx_to_markdown  # requires mammoth
```


```python
from dn import excel_to_markdown  # requires pandas
```


```python
from dn import pptx_to_markdown  # requires python-pptx
```


```python
from dn import html_to_markdown  # requires html2text
```

# Markdown stores

User story: I have a directory with multiple files in different formats.

I want to batch convert all supported files to markdown and store them in memory.


```python
from dn import Files, bytes_store_to_markdown_store

from dn.tests.utils_for_testing_dn import test_data_dir

# Setup source files from test directory
src_files = Files(test_data_dir)

# Setup target store as an in-memory dictionary
target_store = {}

# Convert all files in directory to markdown
result = bytes_store_to_markdown_store(src_files, target_store, verbose=False)

# Check that the result is the target_store
assert result is target_store

# Verify that the supported file types were converted correctly
supported_files = [
    "test.docx",
    "test.pptx",
    "test.pdf",
    "test.html",
    "test.xlsx",
    "test.txt",
    "test.md",
    "test.ipynb",
]

print(f"\nSupported files (given what packages are installed here): {supported_files}\n")

for filename in supported_files:
    assert f"{filename}.md" in target_store, f"{filename} not found in target_store"
    assert len(target_store[f"{filename}.md"]) > 0, f"{filename} conversion failed"

```

    invalid pdf header: b'PK\x03\x04\n'
    EOF marker not found
    EOF marker not found
    invalid pdf header: b'PK\x03\x04\x14'
    EOF marker not found
    invalid pdf h
    ...
    df header: b'PK\x03\x04\x14'
    EOF marker not found


    
    Supported files (given what packages are installed here): ['test.docx', 'test.pptx', 'test.pdf', 'test.html', 'test.xlsx', 'test.txt', 'test.md', 'test.ipynb']
    


# Convert this notebook into a markdown for the README.md


```python
from dn import notebook_to_markdown

notebook_to_markdown('~/Dropbox/py/proj/t/dn/misc/dn_readme.ipynb', target_file='../README.md')
```

    HTML output truncated. (Data removed)



```python

```
