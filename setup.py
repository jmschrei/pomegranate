from setuptools import setup

setup(
	name='pomegranate',
	version='1.1.1',
	author='Jacob Schreiber',
	author_email='jmschreiber91@gmail.com',
	packages=['pomegranate', 'pomegranate.distributions', 'pomegranate.hmm'],
	url='https://github.com/jmschrei/torchegranate',
	license='LICENSE.txt',
	description='A PyTorch implementation of probabilistic models.',
	install_requires=[
		'numpy >= 1.22.2', 
		'scipy >= 1.6.2',
		'scikit-learn >= 1.0.2',
		'torch >= 1.9.0',
		'apricot-select >= 0.6.1',
		'networkx >= 2.8.4'
	]
)