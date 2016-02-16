Naive Bayes Classifiers
=======================

The Naive Bayes classifier is a simple probabilistic classification model based on Bayes Theorem. It is a simple and fast classification model. Since Naive Bayes classifiers simply classifies sets of data by which class has the highest conditional probability, Naive Bayes classifiers can use any distribution or model which has a probabilistic interpretation of the data as one of its components. Basically if it can output a log probability, then it can be used in Naive Bayes.

An IPython notebook example demonstrating a Naive Bayes classifier using multivariate distributions can be [found here](https://github.com/jmschrei/pomegranate/blob/master/examples/naivebayes_multivariate_male_female.ipynb).

Naive Bayes implements the following methods:

```
predict_log_proba( X ) : Input a set of data and output the log probability of each sample under each component
predict_proba( X ) : Input a set of data and output the probability of each sample under each component
predict( X ) : Input a set of data and classify each sample under one of the components
fit( X, y ) : Fit the Naive Bayes classifier to a set of training data, with X being the sample, and y being the correct classification for each sample
```

## Initialization

Naive Bayes can be initialized a number of ways. The classifier can be initialized either by (1) passing in the initial distribution objects or by (2) passing in the constructor and the number of components.

For example, here is an initialization using an already initialized normal distribution for class 0, a uniform distribution for class 1, and an exponential distribution for class 2 .

```Python
clf = NaiveBayes([ NormalDistribution( 5, 2 ), UniformDistribution( 0, 10 ), ExponentialDistribution( 1.0 ) ])
```

Since Naive Bayes classifiers simply compares the likelihood of a sample occurring under different models, it can be initialized with any model in pomegranate. This is assuming that all the models take the same type of input.

**TODO: covariance matrix for gaussian distribution**

```Python
multi = MultivariateGaussianDistribution( means=[ 5, 5 ], covariance=[[ 2, 0 ], [ 0, 2 ]] )
indie = IndependentComponentsDistribution( distributions=[ UniformDistribution(0, 10), UniformDistribution(0, 10) ])
clf = NaiveBayes([ mutli, indie ])
```

Naive Bayes classifiers can also take Hidden Markov Models as input as well as Bayesian Networks.

If all the models in the Naive Bayes classifier use the same type of model, then Naive Bayes can be initialized by passing in the constructor for the model and the number of classses there are.

```Python
clf = NaiveBayes( NormalDistribution, n_components=5 )
```

**Warning!** keep in mind that if Naive Bayes is initialized this way then all the classes must be fitted to some data before any prediction methods are called.

Finally, Naive Bayes must be given at least a model or n_components must be specified otherwise a ValueError will be thrown.

## Fitting

Naive Bayes has a fit method, in which the models in the classifier are trained to "fit" to a set of data. The method takes two numpy arrays as input, an array of samples and an array of correct classifications for each sample. Here is an example for a Naive Bayes made up of two bivariate distributions.

```Python
multi = MultivariateGaussianDistribution( means=[ 5, 5 ], covariance=[[ 2, 0 ], [ 0, 2 ]] )
indie = IndependentComponentsDistribution( distributions=[ UniformDistribution(0, 10), UniformDistribution(0, 10) ])
clf = NaiveBayes([ mutli, indie ])

samples = np.array([[ 6, 5 ],
					[ 3.5, 4 ],
					[ 7.5, 1.5 ],
					[ 7, 7 ]])
classes = np.array([ 0, 0, 1, 1 ])
clf.fit( samples, classes )
```

As we can see, there are four samples, with the first two samples labeled as class 0 and the last two samples labeled as class 1. The training samples follow the same format as the input for the prediction methods; they must match the input for the models used in the Naive Bayes classifier. Additionally, there must be a corresponding number of correct classfications for each sample. In other words the length of both arrays must be the same. On a final note, it is a good idea to include samples for all the models in the Naive Bayes classifier since all models are retrained, even if no samples are supplied to retrain it with.

## Prediction

Naive Bayes has three different prediction methods which all take the same input, an numpy array of samples. These methods are predict_proba, predict_log_proba, and predict which output, respectively, the probability of each sample occurring under each model, the log probability of each sample occurring under each model, and the model each sample is classified as.

Since Naive Bayes uses Bayes Theorem, the probability of a sample occurring under each model is in respect to the total sample space of the sample occurring under all the models.

Calling predict_proba on five samples for a Naive Bayes with univariate components would look like the following.

```Python
clf = NaiveBayes([ NormalDistribution( 5, 2 ), UniformDistribution( 0, 10 ), ExponentialDistribution( 1.0 ) ])

probs = clf.predict_proba( np.array([ 0, 1, 2, 3, 4 ]) )
```
With the output being the following numpy array.

```Python
[[ 0.00790443  0.09019051  0.90190506]
 [ 0.05455011  0.20207126  0.74337863]
 [ 0.21579499  0.33322883  0.45097618]
 [ 0.44681566  0.36931382  0.18387052]
 [ 0.59804205  0.33973357  0.06222437]]
```

With models that take in multiple inputs, the input to predict_proba would take the inputs as a list of lists. This would look like the following.

```Python
multi = MultivariateGaussianDistribution( means=[ 5, 5 ], covariance=[[ 2, 0 ], [ 0, 2 ]] )
indie = IndependentComponentsDistribution( distributions=[ UniformDistribution(0, 10), UniformDistribution(0, 10) ])
clf = NaiveBayes([ mutli, indie ])

probs = clf.predict_proba( np.array([[ 0, 4 ],
									 [ 1, 3 ],
									 [ 2, 2 ],
									 [ 3, 1 ],
									 [ 4, 0 ]]) )
```

With the output having the same format as the example above. Keep in mind that the samples taken as input must be the same length as one another unless Naive Bayes is being used to compare Hidden Markov Models.

The method predict_log_proba is called in the exact same manner, and its output is formatted exactly the same as predict_proba. The only difference is the output values are the log probabilities rather than the probabitilies of each sample occurring.

```Python
clf = NaiveBayes([ NormalDistribution( 5, 2 ), UniformDistribution( 0, 10 ), ExponentialDistribution( 1.0 ) ])

log_probs = clf.predict_log_proba( np.array([ 0, 1, 2, 3, 4 ]) )
```

Finally, there is the method predict. Again, this takes the same formatted input as predict_proba and predict_log_proba. However instead of outputting a array of arrays, predict outputs an array of classifications for each sample. Here is a call using the Naive Bayes made up of three univariate distributions once again.

```Python
clf = NaiveBayes([ NormalDistribution( 5, 2 ), UniformDistribution( 0, 10 ), ExponentialDistribution( 1.0 ) ])

predictions = clf.predict( np.array([ 0, 1, 2, 3, 4 ]) )
```

The corresponding output would be.

```Python
[ 2, 2, 2, 0, 0 ]
```

Pretty straightforward.