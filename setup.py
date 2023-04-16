from setuptools import setup

setup(
	name='torchegranate',
	version='0.5.0',
	author='Jacob Schreiber',
	author_email='jmschreiber91@gmail.com',
	packages=['torchegranate', 'torchegranate.distributions', 'torchegranate.hmm'],
	url='https://github.com/jmschrei/torchegranate',
	license='LICENSE.txt',
	description='A rewrite of pomegranate using PyTorch.',
	install_requires=[
		'numpy >= 1.22.2', 
		'scipy >= 1.6.2',
		'scikit-learn >= 1.0.2',
		'torch >= 1.9.0',
		'apricot-select >= 0.6.1',
		'networkx >= 2.8.4'
	]
)