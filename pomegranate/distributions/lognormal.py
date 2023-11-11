# normal.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch

from .._utils import _cast_as_tensor
from .._utils import _cast_as_parameter
from .._utils import _update_parameter
from .._utils import _check_parameter
from .._utils import _check_shapes

from .normal import Normal


class LogNormal(Normal):
    """A lognormal object.

    The parameters are the mu and sigma of the normal distribution, which
    is the the exponential of the log normal distribution. This
    distribution can assume that features are independent of the others if
    the covariance type is 'diag' or 'sphere', but if the type is 'full' then
    the features are not independent.

    There are two ways to initialize this object. The first is to pass in
    the tensor of probablity parameters, at which point they can immediately be
    used. The second is to not pass in the rate parameters and then call
    either `fit` or `summarize` + `from_summaries`, at which point the probability
    parameter will be learned from data.


    Parameters
    ----------
    means: list, numpy.ndarray, torch.Tensor or None, shape=(d,), optional
            The mean values of the normal distributions. Default is None.

    covs: list, numpy.ndarray, torch.Tensor, or None, optional
            The variances and covariances of the distribution. If covariance_type
            is 'full', the shape should be (self.d, self.d); if 'diag', the shape
            should be (self.d,); if 'sphere', it should be (1,). Note that this is
            the variances or covariances in all settings, and not the standard
            deviation, as may be more common for diagonal covariance matrices.
            Default is None.

    covariance_type: str, optional
            The type of covariance matrix. Must be one of 'full', 'diag', or
            'sphere'. Default is 'full'.

    min_cov: float or None, optional
            The minimum variance or covariance.

    inertia: float, [0, 1], optional
            Indicates the proportion of the update to apply to the parameters
            during training. When the inertia is 0.0, the update is applied in
            its entirety and the previous parameters are ignored. When the
            inertia is 1.0, the update is entirely ignored and the previous
            parameters are kept, equivalently to if the parameters were frozen.

    frozen: bool, optional
            Whether all the parameters associated with this distribution are frozen.
            If you want to freeze individual pameters, or individual values in those
            parameters, you must modify the `frozen` attribute of the tensor or
            parameter directly. Default is False.
    """

    def __init__(
        self,
        means=None,
        covs=None,
        covariance_type="full",
        min_cov=None,
        inertia=0.0,
        frozen=False,
        check_data=True,
    ):
        self.name = "LogNormal"
        super().__init__(
            means=means,
            covs=covs,
            covariance_type=covariance_type,
            min_cov=min_cov,
            inertia=inertia,
            frozen=frozen,
            check_data=check_data,
        )

    def sample(self, n):
        """Sample from the probability distribution.

        This method will return `n` samples generated from the underlying
        probability distribution.


        Parameters
        ----------
        n: int
                The number of samples to generate.


        Returns
        -------
        X: torch.tensor, shape=(n, self.d)
                Randomly generated samples.
        """

        if self.covariance_type == "diag":
            return torch.distributions.Normal(self.means, self.covs).sample([n]).exp()
        elif self.covariance_type == "full":
            return (
                torch.distributions.MultivariateNormal(self.means, self.covs)
                .sample([n])
                .exp()
            )

    def log_probability(self, X):
        """Calculate the log probability of each example.

        This method calculates the log probability of each example given the
        parameters of the distribution. The examples must be given in a 2D
        format.

        Note: This differs from some other log probability calculation
        functions, like those in torch.distributions, because it is not
        returning the log probability of each feature independently, but rather
        the total log probability of the entire example.


        Parameters
        ----------
        X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
                A set of examples to evaluate.


        Returns
        -------
        logp: torch.Tensor, shape=(-1,)
                The log probability of each example.
        """

        X = _check_parameter(
            _cast_as_tensor(X, dtype=self.means.dtype),
            "X",
            ndim=2,
            shape=(-1, self.d),
            check_parameter=self.check_data,
        )

        # take the log of X
        x_log = X.log()

        return super().log_probability(x_log)

    def summarize(self, X, sample_weight=None):
        """Extract the sufficient statistics from a batch of data.

        This method calculates the sufficient statistics from optionally
        weighted data and adds them to the stored cache. The examples must be
        given in a 2D format. Sample weights can either be provided as one
        value per example or as a 2D matrix of weights for each feature in
        each example.


        Parameters
        ----------
        X: list, tuple, numpy.ndarray, torch.Tensor, shape=(-1, self.d)
                A set of examples to summarize.

        sample_weight: list, tuple, numpy.ndarray, torch.Tensor, optional
                A set of weights for the examples. This can be either of shape
                (-1, self.d) or a vector of shape (-1,). Default is ones.
        """

        if self.frozen is True:
            return
        X = _cast_as_tensor(X, dtype=self.means.dtype)
        super().summarize(X.log(), sample_weight=sample_weight)
