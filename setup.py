from setuptools import setup, find_packages

setup(name='stlpy',
      version='0.1.0',
      description='A Python library for control from Signal Temporal Logic (STL) specifications',
      url='http://github.com/vincekurtz/stlpy',
      author='Vince Kurtz',
      author_email='vjkurtz@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=['numpy', 'matplotlib', 'treelib'],
      zip_safe=False)
