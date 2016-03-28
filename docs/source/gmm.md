General Mixture Models
======================

[IPython Notebook Tutorial](https://github.com/jmschrei/pomegranate/blob/master/tutorials/Tutorial_2_General_Mixture_Models.ipynb)

General Mixture Models (GMMs) are an unsupervised model composed of multiple distributions (or components) and corresponding weights. This allows you to model more sophisticated phenomena probabilistically. A common task is to figure out which component a new data point comes from given only a large quantity of unlabelled data.

Initialization
--------------

General Mixture Models can be initialized in two ways depending on if you know the initial parameters of the distributions of not. If you do know the prior parameters of the distributions then you can pass them in as a list. You can also pass in the weights, or the prior probability of a sample belonging to that component of the model.

```
>>> gmm = GeneralMixtureModel([NormalDistribution(5, 2), NormalDistribution(1, 2)], weights=[0.33, 0.67])
```

If you do not know the initial parameters, then the components can be initialized using kmeans++. This algorithm involves picking a point randomly to be the center for the first class, and then randomly selecting a point with weights inversely proportional to the distance to this point for each of the remaining points. kmeans is then run until convergence, and the initial parameters are selected as MLE estimates on the points assigned to that weight. This is done in the following manner:

```
>>> gmm = GeneralMixtureModel( NormalDistribution, n_components=2 )
```

This allows all distributions in pomegranate to be natively used in GMMs.

API Reference
-------------

```eval_rst
.. automodule:: pomegranate.gmm
	:members:
```