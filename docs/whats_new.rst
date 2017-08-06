.. currentmodule:: pomegranate


===============
Release History
===============

Version 0.7.8
=============

Highlights
----------

This will serve as a log for the changes added for the release of version 0.7.8.


Changelog
---------

k-means
.......

	- k-means has been changed from using iterative computation to using the
	alternate formulation of euclidean distance, from ||a - b||^{2} to using
	||a||^{2} + ||b||^{2} - 2||a \cdot b||. This allows for the centroid norms
	to be cached, significantly speeding up computation, and for dgemm to be used
	to solve the matrix matrix multiplication. Initial attempts to add in GPU
	support appeared unsuccessful, but in theory it should be something that can
	be added in.


Hidden Markov Models
....................

	- Allowed labels for labeled training to take in string names of the states
	instead of the state objects themselves.

	- Added in `state_names` and `names` parameters to the `from_samples` method
	to allow for more control over the creation of the model.

	- Added in semi-supervised learning to the `fit` step that can be activated
	by passing in a list of labels where sequences that have no labels have a None
	value. This allows for training to occur where some sequences are fully labeled
	and others have no labels, not for training to occur on partially labeled
	sequences.

	- Supervised initialization followed by semi-supervised learning added in to the
	`from_samples` method similarly to other methods. One should do this by passing
	in string labels for state names, always starting with <model_name>-start, where
	model_name is the `name` parameter passed into the `from_samples` method. Sequences
	that do not have labels should have a None instead of a list of corresponding labels.
	While semi-supervised learning using the `fit` method can support arbitrary
	transitions amongst silent states, the `from_samples` method does not produce
	silent states, and so other than the start and end states, all states should be
	symbol emitting states. If using semi-supervised learning, one must also pass in a
	list of the state names using the `state_names` parameter that has been added in.


Tutorials
.........

	- Removed the PyData 2016 Chicago Tutorial due to it's similarity to
	tutorials_0_pomegranate_overview.