# Configuration file for the Sphinx documentation builder.

# -- Project information

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent

sys.path.insert(0, str(project_root))

with open(project_root / "oml" / "__init__.py", "r") as f:
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
    "sphinx_mdinclude",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

docs_user = "OML-Team"
docs_repo = "open-metric-learning"
html_context = {
    "display_github": True,
    "source_url_prefix": f"https://github.com/{docs_user}/{docs_repo}/tree/master/docs",
    "github_host": "github.com",
    "github_user": docs_user,
    "github_repo": docs_repo,
    "github_version": "master",
    "conf_py_path": "/docs/source/",
    "source_suffix": ".rst",
}

html_show_sourcelink = True

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"
