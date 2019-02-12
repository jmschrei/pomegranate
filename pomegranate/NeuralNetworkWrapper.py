# NeuralNetworkWrapper.py


class NeuralNetworkWrapper():
    '''A wrapper for a neural network model for use in pomegranate.
    
    This wrapper will store a pointer to the model, as well as an indicator
    of what class it represents. It needs information about the number of
    dimensions of the input and the total number of classes in the input.
    It is currently built to work with keras models, but can be easily
    modified to work with the package of your choice.
    
    Note: Training works by having each call to `summarize` append the
    observed data and corresponding label to the neural network model 
    object. After multiple calls to `summarize`, the model object will
    now store a large chunk of the data set (if not the entire thing) 
    along with the corresponding labels. The `from_summaries` method then
    will have the model perform a single round of training on that stored
    data. To ensure that only a single round of training is performed
    (and not one round per wrapper object in the model) we only perform
    training if the class ID of the wrapper object is 0.
    
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

        self.model.X.append(X.copy())
        self.model.w.append(w.copy())
        
        y = numpy.zeros((X.shape[0], self.n_classes))
        y[:, self.i] = 1
        self.model.y.append(y)
        
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
            X = numpy.concatenate(self.model.X)
            w = numpy.concatenate(self.model.w)
            y = numpy.concatenate(self.model.y)
        
            self.model.train_on_batch(X, y, sample_weight=w)
        
        self.clear_summaries()
    
    def clear_summaries(self):
        self.model.X = []
        self.model.y = []
        self.model.w = []