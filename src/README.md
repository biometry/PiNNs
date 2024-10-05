# Creating a PRELES pytorch c++ extension
Here is a recipe for creating the pytorch c++ extension of PRELES. Make sure you have the project environment installed and loaded.

## Check compiler
This code runs with gcc compiler, e.g.
```console
@PiNNs~: gcc --version
gcc (GCC) 9.1.0
Copyright (C) 2019 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```

## Create pytorch c++ extension
Next we create the pytorch c++ extension. We use setuptools and the file setup.py to proceed as follows
```console
@PiNNs~: python setup.py install
running install
running bdist_egg
running egg_info
writing preles.egg-info/PKG-INFO
writing dependency_links to preles.egg-info/dependency_links.txt
writing top-level names to preles.egg-info/top_level.txt
reading manifest file 'preles.egg-info/SOURCES.txt'
writing manifest file 'preles.egg-info/SOURCES.txt'
installing library code to build/bdist.linux-x86_64/egg
running install_lib
running build_ext
building 'preles' extension
...
Processing dependencies for preles==0.0.0
Finished processing dependencies for preles==0.0.0
```

## Tests
We check if we can call PRELES now from python by
```console
@PiNNs~: python
Python 3.7.10 | packaged by conda-forge | (default, Feb 19 2021, 16:07:37)
[GCC 9.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> import preles
>>>
```
Note that it is required to import torch before importing preles. To make predictions with PRELES in python call *preles.preles()*. This is equivalent to the R function used to make PRELES predictions. Note that in python it takes torch tensors as inputs. 