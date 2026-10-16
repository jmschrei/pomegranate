import numpy
import pytest
import torch

from pomegranate.distributions import LogNormal, Normal


@pytest.fixture
def X():
    return torch.tensor([[1., 2.], [2., 1.], [4., 8.], [8., 4.], [16., 16.]])


@pytest.mark.parametrize("covariance_type", ["full", "diag", "sphere"])
@pytest.mark.parametrize("input_type", [list, numpy.array, torch.tensor])
def test_fit_uninitialized(X, covariance_type, input_type):
    distribution = LogNormal(covariance_type=covariance_type)
    assert distribution.fit(input_type(X.tolist())) is distribution
    expected = Normal(covariance_type=covariance_type).fit(X.log())
    torch.testing.assert_close(distribution.means, expected.means)
    torch.testing.assert_close(distribution.covs, expected.covs)


@pytest.mark.parametrize("covariance_type", ["full", "diag", "sphere"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_summarize_uninitialized(X, covariance_type, dtype):
    X = X.to(dtype)
    weights = torch.tensor([1., 2., 0., 3., 1.], dtype=dtype)
    distribution = LogNormal(covariance_type=covariance_type).to(dtype)
    distribution.summarize(X[:3], sample_weight=weights[:3])
    distribution.summarize(X[3:], sample_weight=weights[3:])
    distribution.from_summaries()
    expected = Normal(covariance_type=covariance_type).to(dtype).fit(X.log(), weights)
    torch.testing.assert_close(distribution.means, expected.means)
    torch.testing.assert_close(distribution.covs, expected.covs)


@pytest.mark.parametrize("covariance_type", ["full", "diag"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fit_initialized(X, covariance_type, dtype):
    covs = torch.eye(2) if covariance_type == "full" else torch.ones(2)
    distribution = LogNormal(torch.zeros(2), covs, covariance_type=covariance_type).to(dtype)
    distribution.fit(X.to(dtype))
    expected = Normal(covariance_type=covariance_type).to(dtype).fit(X.to(dtype).log())
    torch.testing.assert_close(distribution.means, expected.means)
    torch.testing.assert_close(distribution.covs, expected.covs)


def test_initialized_means_dtype(X):
    distribution = LogNormal(numpy.zeros(2), numpy.ones(2), covariance_type="diag")
    expected = Normal(numpy.zeros(2), numpy.ones(2), covariance_type="diag")
    assert distribution.means.dtype == torch.float64
    assert distribution.dtype == torch.float32
    distribution.fit(X)
    expected.fit(X.to(torch.float64).log())
    torch.testing.assert_close(distribution.means, expected.means)
    torch.testing.assert_close(distribution.covs, expected.covs)


def test_fit_frozen(X):
    distribution = LogNormal([0., 0.], [1., 1.], covariance_type="diag", frozen=True)
    distribution.fit(X)
    torch.testing.assert_close(distribution.means, torch.zeros(2))
    torch.testing.assert_close(distribution.covs, torch.ones(2))
