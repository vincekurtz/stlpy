1. Make sure version info is correct (setup.py, docs/source/conf.py)

2. Create source distribution: `python setup.py sdist`. That should put
   a `stlpy-[VERSION].tar.gz` file in the `dist/` directory. 

3. Upload: `twine upload dist/stlpy-[VERSION].tar.gz`
