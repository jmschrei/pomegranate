PACKAGE_NAME=pomegranate
PY2_ENV=py2.7
PY3_ENV=py3.6

default:
	echo no default

.PHONY: install test bigclean nbtest nbclean
.PHONY: bigbuild py2build py3build
.PHONY: biginstall py2install py3install
.PHONY: biguninstall py2uninstall py3uninstall
.PHONY: bigtest py2test py3test
.PHONY: bignbtest py2nbtest py3nbtest

install:
	python setup.py install

test:
	python setup.py test

bigclean: nbclean
	rm -rf build
	rm -rf dist
	rm -rf .eggs
	rm -rf $(PACKAGE_NAME).egg-info
	find . -name '*.pyc' -print | xargs rm
	find . -name '*.so' -print | xargs rm
ifndef NO_REMOVE_CFILES
	find . -name '*.c' -print | xargs rm
endif

# Add python dependencies
ifdef PY2_ENV
bigbuild: py2build
biginstall: py2install
biguninstall: py2uninstall
bigtest: py2test
bignbtest: py2test
endif

ifdef PY3_ENV
bigbuild: py3build
biginstall: py3install
biguninstall: py3uninstall
bigtest: py3test
# Don't know how to override the kernelspec and the notebooks are python2 only anyway
# bignbtest: py3test
endif


py2build:
	(source activate $(PY2_ENV) ; python setup.py build ; python setup.py build_ext --inplace )
py3build:
	(source activate $(PY3_ENV) ; python setup.py build ; python setup.py build_ext --inplace )

py2install:
	(source activate $(PY2_ENV) ; python setup.py install )
py3install:
	(source activate $(PY3_ENV) ; python setup.py install )

py3uninstall:
	rm -rf ~/miniconda2/envs/$(PY3_ENV)/lib/python3.6/site-packages/$(PACKAGE_NAME)*
py2uninstall:
	rm -rf ~/miniconda2/envs/$(PY2_ENV)/lib/python2.7/site-packages/$(PACKAGE_NAME)*

py3test:
	(source activate $(PY3_ENV) ; python setup.py test )
py2test:
	(source activate $(PY2_ENV) ; python setup.py test )

## Notebook tests
PYTHON_NOTEBOOKS= examples/*.ipynb tutorials/*.ipynb benchmarks/*.ipynb
EXECUTE_TIMEOUT=600
ALLOW_ERRORS=
# Allow errors if you want to check as many cells as possible
#ALLOW_ERRORS=--allow-errors
# Required conda package installs:
# CONDA_INSTALL_BASE_PACKAGES=cython scipy scikit-learn pandas joblib nose networkx=1.11
# CONDA_INSTALL_NOTEBOOK_PACKAGES=jupyter jupyter_contrib_nbextensions jupyter_nbextensions_configurator seaborn xlrd pygraphviz pillow
nbtest:
	for nb in $(PYTHON_NOTEBOOKS) ; do time jupyter nbconvert $$nb --execute --ExecutePreprocessor.timeout=$(EXECUTE_TIMEOUT) --to html $(ALLOW_ERRORS) ; done
py2nbtest:
	(source activate $(PY2_ENV) ; for nb in $(PYTHON_NOTEBOOKS) ; do time jupyter nbconvert $$nb --execute --ExecutePreprocessor.timeout=$(EXECUTE_TIMEOUT) --to html $(ALLOW_ERRORS) ; done )
py3nbtest:
	(source activate $(PY3_ENV) ; for nb in $(PYTHON_NOTEBOOKS) ; do time jupyter nbconvert $$nb --execute --ExecutePreprocessor.timeout=$(EXECUTE_TIMEOUT) --to html $(ALLOW_ERRORS) ; done )
nbclean:
	for nb in $(PYTHON_NOTEBOOKS) ; do rm -f $${nb%%.ipynb}.html ; done
