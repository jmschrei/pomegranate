.. _install:

Installation
============

The easiest way to get pomegranate is through pip using the command

.. code-block:: bash
	pip install pomegranate

This should install all the dependencies in addition to the package.

You can also get pomegranate through conda using the command

.. code-block:: bash
	conda install pomegranate

This version may not be as up to date as the pip version though.

Lastly, you can get the bleeding edge from GitHub using the following commands:

.. code-block:: bash

	git clone https://github.com/jmschrei/pomegranate
	cd pomegranate
	python setup.py install

On Windows machines you may need to download a C++ compiler if you wish to build from source yourself. For Python 2 this `minimal version of Visual Studio 2008 works well <https://www.microsoft.com/en-us/download/details.aspx?id=44266>`_. For Python 3 `this version of the Visual Studio build tools <http://go.microsoft.com/fwlink/?LinkId=691126>`_ has been reported to work.

The requirements for pomegranate can be found in the requirements.txt file in the repository, and include numpy, scipy, networkx (below v2.0), joblib, cupy (if using a GPU), and cython (if building from source or on an Ubuntu machine). 

FAQ
---

Q. I'm on a Windows machine and I'm still encountering problems. What should I do?

A. If those do not work, it has been suggested that https://wiki.python.org/moin/WindowsCompilers may provide more information. Note that your compiler version must fit your python version. Run python --version to tell which python version you use. Don't forget to select the appropriate Windows version API you'd like to use. If you get an error message "ValueError: Unknown MS Compiler version 1900" remove your Python's Lib/distutils/distutil.cfg and retry. See http://stackoverflow.com/questions/34135280/valueerror-unknown-ms-compiler-version-1900 for details.


Q. I've been getting the following error: ```ModuleNotFoundError: No module named 'pomegranate.utils'.``` 

A. A reported solution is to uninstall and reinstall without cached files using the following:

.. code-block:: bash
	pip uninstall pomegranate
	pip install pomegranate --no-cache-dir

If that doesn't work for you, you may need to downgrade your version of numpy to 1.11.3 and try the above again.


Q. I've been getting the following error: ```MarkovChain.so: unknown file type, first eight bytes: 0x7F 0x45 0x4C 0x46 0x02 0x01 0x01 0x00.``` 

A .This can be fixed by removing the .so files from the pomegranate installation or by building pomegranate from source.


Q. I'm encountering some other error when I try to install pomegranate.

A. pomegranate has had some weird linker issues, particularly when users try to upgrade from an older version. In the following order, try:

1. Uninstalling pomegranate using pip and reiinstalling it with the option --no-cache-dir, like in the above question.
2. Removing all pomegranate files on your computer manually, including egg and cache files that cython may have left in your site-packages folder
3. Reinstalling the Anaconda distribution (usually only necessary in issues where libgfortran is not linking properly)
