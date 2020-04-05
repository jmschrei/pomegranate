# NeuralNetworkWrapper.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy

class NeuralNetworkWrapper():
    '''A wrapper for a neural network model for use in pomegranate.
    
    This wrapper will store a pointer to the model, as well as an indicator
    of what class it represents. It needs information about the number of
    dimensions of the input and the total number of classes in the input.
    It is currently built to work with keras models, but can be easily
    modified to work with the package of your choice.
    
    Training works in a somewhat hacky manner. Internally, pomegranate
    will scan over all components of the model, calling `summarize` for each
    one, and then `from_summaries` at the end. During the EM procedure, the
    samples and their associated weights are passed in to the `summarize`
    function. The associated weights are the responsibilities calculated
    during the EM algorithm. In theory, one could simply update the model
    using the samples and their corresponding weights. In practice, it's much
    better to reconstruct the whole responsibility matrix for the batch of
    data and then train on soft labels.
    
    The process for training is as follows. When pomegranate starts a round of
    optimization, this wrapper will store a pointer to the data set to the
    neural network model object. This data set is the same one passed to each
    NeuralNetworkWrapper, the only difference being the ~weights~. Thus, each
    successive call will store the weights that are passed in (the responsibilities
    from the EM algorithm) to an associated label matrix. The result is a single
    copy of the data batch and a corresponding matrix of soft labels. Keras
    allows us to train a classifier on soft labels, and this is the preferred
    strategy.
    
    Parameters
    ----------
    model : object
        The neural network model being utilized.
        
    i : int
        The class that this distribution represents.
    
    n_dimensions : int
        The number of dimensions in the input.
    
    n_classes : int
        The total number of classes that the model can output.
    '''
    
    def __init__(self, model, i, n_dimensions, n_classes):
        self.d = n_dimensions
        self.n_classes = n_classes
        self.model = model
        self.i = i
        self.model.X = []
        self.model.y = []
        self.model.w = []
    
    def log_probability(self, X):
        ''' Return pseudo-log probabilities from the neural network.
        
        This method returns the log probability of the class that this
        wrapper represents given the model. Thus, it's not strictly a
        likelihood function, but rather, a posterior. However, because
        the HMM takes log probabilities, multiplies them with a prior,
        and then normalizes them, mathematically they work out to be
        equivalent.

        This method uses the `predict` function from the neural network,
        which should take in a single batch of data, and returns the
        posterior probability of each class given the network. Typically,
        this is calculated using a softmax function on the outputs. The
        output of this function should be a matrix of size (n, k), where
        n is the number of samples and k is the number of classes, where
        the sum over each sample is equal to 1. 

        Parameters
        ----------
        X : numpy.ndarray, shape=(n, d)
            The batch of data to calculcate probabilities over.
        '''
        
        return numpy.log(self.model.predict(X)[:,self.i])
    
    def summarize(self, X, w):
        '''When shown a batch of data, store the data.

        This will store the batch of data, and associated weights, to the
        object. The actual update occurs when `from_summaries` is called.

        Parameters
        ----------
        X : numpy.ndarray, shape=(n, d)
            The batch of data to be passed in.

        w : numpy.ndarray, shape=(n,)
            The associated weights. These can be uniform if unweighted.

        Returns
        -------
        None
        '''

        if self.i == 0:
            self.model.X = X.copy()
            self.model.y = numpy.zeros((X.shape[0], self.n_classes))
            
        self.model.y[:, self.i] = w
        
    def from_summaries(self, inertia=0.0):
        '''Perform a single gradient update to the network.

        This will perform a single gradient update step, using the
        `train_on_batch` method from the network. This is already implemented
        for keras networks, but for other networks this method will have to
        be implemented. It should take in a single batch of data, along with
        associated sample weights, and update the model weights.

        Parameters
        ----------
        inertia : double, optional
            This parameter is ignored for neural networks, but required for
            compatibility reasons.

        Returns
        -------
        None
        '''


        if self.i == 0:
            self.model.train_on_batch(self.model.X, self.model.y)
            self.clear_summaries()
    
    def clear_summaries(self):
        self.model.X = None
        self.model.y = None


    @classmethod
    def from_samples(self, X, weights):
        '''The training of this wrapper on data should be performed in the main model.

        This method should not be used directly to train the network.
        '''

        return self
