# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'PySTL'
copyright = '2022, Vince Kurtz'
author = 'Vince Kurtz'

release = '0.2'
version = '0.2.0'

# -- General configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.graphviz',
    'sphinx.ext.autosectionlabel'
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

# The master toctree document
master_doc = 'index'
