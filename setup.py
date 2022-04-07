from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(name='stlpy',
      version='0.2.0',
      description='A Python library for control from Signal Temporal Logic (STL) specifications',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://stlpy.readthedocs.io/en/latest/index.html',
      author='Vince Kurtz',
      author_email='vjkurtz@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=['numpy', 'matplotlib', 'treelib'],
      python_requires='>=3.8',
      zip_safe=False)
