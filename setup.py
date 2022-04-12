from setuptools import setup, find_packages

long_description = """
A python library for control from Signal Temporal Logic (STL) specifications.

This software is designed with the following goals in mind:

- Provide a simple python interface for dealing with STL formulas
- Provide high-quality implementations of several state-of-the-art synthesis 
  algorithms, including Mixed-Integer Convex Programming (MICP) and 
  gradient-based optimization.
- Make it easy to design and evaluate new synthesis algorithms.
- Provide a variety of benchmark scenarios that can be used to test new algorithms.
"""

setup(name='stlpy',
      version='0.3.0',
      description='A Python library for control from Signal Temporal Logic (STL) specifications',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://stlpy.readthedocs.io/en/latest/index.html',
      project_urls ={
          "Source Code": "https://github.com/vincekurtz/stlpy",
          "Documentation": "https://stlpy.readthedocs.io/en/latest/index.html"},
      author='Vince Kurtz',
      author_email='vjkurtz@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=['numpy', 'matplotlib', 'treelib'],
      python_requires='>=3.8',
      zip_safe=False)
