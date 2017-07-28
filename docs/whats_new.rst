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


Tutorials
.........

	- Removed the PyData 2016 Chicago Tutorial due to it's similarity to
	tutorials_0_pomegranate_overview.