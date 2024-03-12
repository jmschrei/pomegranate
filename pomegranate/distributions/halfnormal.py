# normal.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch

from .._utils import _cast_as_tensor
from .._utils import _cast_as_parameter
from .._utils import _update_parameter
from .._utils import _check_parameter
from .._utils import _check_shapes

from ._distribution import Distribution
from .normal import Normal


# Define some useful constants
LOG_2 = 0.6931471805599453


class HalfNormal(Normal):
    """A half-normal distribution object.

    A half-normal distribution is a distribution over positive real numbers that
    is zero for negative numbers. It is defined by a single parameter, sigma,
    which is the standard deviation of the distribution. The mean of the
    distribution is sqrt(2/pi) * sigma, and the variance is (1 - 2/pi) * sigma^2.

    This distribution can assume that features are independent of the others if
    the covariance type is 'diag' or 'sphere', but if the type is 'full' then
    the features are not independent.

    There are two ways to initialize this object. The first is to pass in
    the tensor of probablity parameters, at which point they can immediately be
    used. The second is to not pass in the rate parameters and then call
    either `fit` or `summarize` + `from_summaries`, at which point the probability
    parameter will be learned from data.


    Parameters
    ----------
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
        covs=None,
        covariance_type="full",
        min_cov=None,
        inertia=0.0,
        frozen=False,
        check_data=True,
    ):
        self.name = "HalfNormal"
        super().__init__(
            means=None,
            covs=covs,
            min_cov=min_cov,
            covariance_type=covariance_type,
            inertia=inertia,
            frozen=frozen,
            check_data=check_data,
        )

    def _initialize(self, d):
        """Initialize the probability distribution.

        This method is meant to only be called internally. It initializes the
        parameters of the distribution and stores its dimensionality. For more
        complex methods, this function will do more.


        Parameters
        ----------
        d: int
                The dimensionality the distribution is being initialized to.
        """
        super()._initialize(d)

    def _reset_cache(self):
        """Reset the internally stored statistics.

        This method is meant to only be called internally. It resets the
        stored statistics used to update the model parameters as well as
        recalculates the cached values meant to speed up log probability
        calculations.
        """
        super()._reset_cache()

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
        if self.covariance_type in ["diag", "full"]:
            return torch.distributions.HalfNormal(self.covs).sample([n])

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
            _cast_as_tensor(X, dtype=self.covs.dtype),
            "X",
            ndim=2,
            shape=(-1, self.d),
            check_parameter=self.check_data,
        )
        return super().log_probability(X) + LOG_2

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

        super().summarize(X, sample_weight=sample_weight)

    def from_summaries(self):
        """Update the model parameters given the extracted statistics.

        This method uses calculated statistics from calls to the `summarize`
        method to update the distribution parameters. Hyperparameters for the
        update are passed in at initialization time.

        Note: Internally, a call to `fit` is just a successive call to the
        `summarize` method followed by the `from_summaries` method.
        """

        if self.frozen == True:
            return

        #  the means are always zero for a half normal distribution
        means = torch.zeros(self.d, dtype=self.covs.dtype)

        if self.covariance_type == "full":
            v = self._xw_sum.unsqueeze(0) * self._xw_sum.unsqueeze(1)
            covs = self._xxw_sum / self._w_sum - v / self._w_sum**2.0

        elif self.covariance_type in ["diag", "sphere"]:
            covs = self._xxw_sum / self._w_sum - self._xw_sum**2.0 / self._w_sum**2.0
            if self.covariance_type == "sphere":
                covs = covs.mean(dim=-1)

        _update_parameter(self.covs, covs, self.inertia)
        _update_parameter(self.means, means, self.inertia)
        self._reset_cache()
