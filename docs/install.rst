.. _install:

Installation
============

The easiest way to get pomegranate is through pip using the command

.. code-block:: bash

	pip install pomegranate

This should install all the dependencies in addition to the package.

You can also get the bleeding edge from GitHub using the following commands:

.. code-block:: bash

	git clone https://github.com/jmschrei/pomegranate
	cd pomegranate
	python setup.py install

Because pomegranate recently moved to a PyTorch backend, the most complicated installation step now is likely installing that and its CUDA dependencies. Please see the PyTorch documentation for help installing those.
