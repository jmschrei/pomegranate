General Mixture Models
======================

[IPython Notebook Tutorial](https://github.com/jmschrei/pomegranate/blob/master/tutorials/Tutorial_2_General_Mixture_Models.ipynb)

General Mixture Models (GMMs) are an unsupervised model composed of multiple distributions (commonly also referred to as components) and corresponding weights. This allows you to model more sophisticated phenomena probabilistically. A common task is to figure out which component a new data point comes from given only a large quantity of unlabelled data.

Initialization
--------------

General Mixture Models can be initialized in two ways depending on if you know the initial parameters of the distributions of not. If you do know the prior parameters of the distributions then you can pass them in as a list. These do not have to be the same distribution--you can mix and match distributions as you want. You can also pass in the weights, or the prior probability of a sample belonging to that component of the model.

```
>>> gmm = GeneralMixtureModel([NormalDistribution(5, 2), NormalDistribution(1, 2)], weights=[0.33, 0.67])
```

If you do not know the initial parameters, then the components can be initialized using kmeans++. This algorithm involves picking a point randomly to be the center for the first class, and then randomly selecting a point with weights inversely proportional to the distance to this point for each of the remaining points. kmeans is then run until convergence, and the initial parameters are selected as MLE estimates on the points assigned to that weight. This is done in the following manner:

```
>>> gmm = GeneralMixtureModel( NormalDistribution, n_components=2 )
```

This allows any distribution in pomegranate to be natively used in GMMs.

Log Probability
---------------

The probability of a point is the sum of its probability under each of the components, multiplied by the weight of each component c, $P(D|M) = \sum\limits_{c \in M} P(D|c)$. This is easily calculated by summing the probability under each distribution in the mixture model and multiplying by the appropriate weights, and then taking the log.

Prediction
----------

The common prediction tasks involve predicting which component a new point falls under. This is done using Bayes rule to determine $P(M|D)$, not the more likely $P(D|M)$. This means that it is not simply which component gives the highest log probability when producing the point. Bayes Rule is as follows: $P(M|D) = \frac{P(D|M)P(M)}{P(D)}$. Since we're looking for a maximum and $P(D)$ is a constant we can cross that off, and we're left with $P(M|D) = P(D|M)P(M)$. $P(D|M)$ is just the probability of the point under the distribution and $P(M)$ are the prior model weights passed in upon initialization or learned from data. This adds a regularization term, meaning that components with fewer samples corresponding to them are less likely to have given this point its label.

We can get the component label assignments using `model.predict(data)`, which will return an array of indexes corresponding to the maximally likely component. If what we want is the full matrix of $P(M|D)$, then we can use `model.predict_proba(data)`, which will return a matrix with each row being a sample, each column being a component, and each cell being the probability that that model generated that data. If we want log probabilities instead we can use `model.predict_log_proba(data)` instead.

Fitting
-------

Training GMMs faces the classic chicken-and-egg problem that most unsupervised learning algorithms face. If we knew which component a sample belonged to, we could use MLE estimates to update the component. And if we knew the parameters of the components we could predict which sample belonged to which component. This problem is solved using expectation-maximization, which iterates between the two until convergence. In essence, an initialization point is chosen which usually is not a very good start, but through successive iteration steps, the parameters converge to a good ending.

These models are fit using `model.fit(data)`. A maximimum number of iterations can be specified as well as a stopping threshold for the improvement ratio. See the API reference for full documentation.


API Reference
-------------

```eval_rst
.. automodule:: pomegranate.gmm
	:members:
```