# Configuration file for the Sphinx documentation builder.

# -- Project information

import sys
from pathlib import Path

sys.path.insert(0, "../../../")

with open(Path.cwd().parent.parent / "oml" / "__init__.py", "r") as f:
    version = f.read().split('"')[-2]

project = "Open Metric Learning"
author = "Shabanov Aleksei"

# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"

# -- Options for EPUB output
epub_show_urls = "footnote"
