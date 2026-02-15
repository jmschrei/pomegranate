# test_dtype_propagation.py
# Contact: Michael Enudi <okmich2002@yahoo.com>

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import numpy as np
import pomegranate
from pomegranate.distributions import Normal
from pomegranate.kmeans import KMeans
from pomegranate.hmm import DenseHMM, SparseHMM


# ---------------------------------------------------------------------------
# 1. Default dtype is float32 (backward compatibility)
# ---------------------------------------------------------------------------
def test_distribution_default_dtype_is_float32():
    d = Normal()
    assert d.dtype == torch.float32, f"Expected float32, got {d.dtype}"


def test_kmeans_default_dtype_is_float32():
    km = KMeans(k=2)
    assert km.dtype == torch.float32, f"Expected float32, got {km.dtype}"


def test_dense_hmm_default_dtype_is_float32():
    dists = [Normal(), Normal(), Normal()]
    model = DenseHMM(dists)
    assert model.dtype == torch.float32, f"Expected float32, got {model.dtype}"


def test_sparse_hmm_default_dtype_is_float32():
    dists = [Normal(), Normal(), Normal()]
    model = SparseHMM(dists)
    assert model.dtype == torch.float32, f"Expected float32, got {model.dtype}"


# ---------------------------------------------------------------------------
# 2. dtype=torch.float64 propagates correctly
# ---------------------------------------------------------------------------
def test_distribution_float64():
    d = Normal(dtype=torch.float64)
    assert d.dtype == torch.float64, f"Expected float64, got {d.dtype}"


def test_kmeans_float64():
    km = KMeans(k=2, dtype=torch.float64)
    assert km.dtype == torch.float64, f"Expected float64, got {km.dtype}"


def test_dense_hmm_float64_dtype():
    dists = [Normal(dtype=torch.float64), Normal(dtype=torch.float64),
             Normal(dtype=torch.float64)]
    model = DenseHMM(dists, dtype=torch.float64)
    assert model.dtype == torch.float64, f"Expected float64, got {model.dtype}"


def test_sparse_hmm_float64_dtype():
    dists = [Normal(dtype=torch.float64), Normal(dtype=torch.float64),
             Normal(dtype=torch.float64)]
    model = SparseHMM(dists, dtype=torch.float64)
    assert model.dtype == torch.float64, f"Expected float64, got {model.dtype}"


# ---------------------------------------------------------------------------
# 3. After fit, buffers and parameters are float64 when dtype=float64
# ---------------------------------------------------------------------------
def _make_synthetic_data(n=5000, n_states=3, seed=42):
    rng = np.random.RandomState(seed)
    means = [-5.0, 0.0, 5.0]
    data = []
    state = 0
    for _ in range(n):
        data.append(rng.normal(means[state], 0.5))
        if rng.random() < 0.02:
            state = rng.randint(0, n_states)
    return np.array(data, dtype=np.float64).reshape(-1, 1)


def test_dense_hmm_float64_buffers_after_fit():
    X = _make_synthetic_data()
    dists = [Normal(dtype=torch.float64), Normal(dtype=torch.float64),
             Normal(dtype=torch.float64)]
    model = DenseHMM(dists, dtype=torch.float64, max_iter=5, verbose=False)
    model.fit([torch.tensor(X, dtype=torch.float64)])

    assert model.edges.dtype == torch.float64, f"edges: {model.edges.dtype}"
    assert model.starts.dtype == torch.float64, f"starts: {model.starts.dtype}"
    assert model.ends.dtype == torch.float64, f"ends: {model.ends.dtype}"
    for i, d in enumerate(model.distributions):
        assert d.means.dtype == torch.float64, f"dist[{i}].means: {d.means.dtype}"


def _make_sparse_hmm(n_states=3, dtype=torch.float64):
    dists = [Normal(dtype=dtype) for _ in range(n_states)]
    # SparseHMM edges format: [dist_from, dist_to, probability]
    prob = 1.0 / n_states
    edges = [[dists[i], dists[j], prob]
             for i in range(n_states) for j in range(n_states)]
    starts = [1.0 / n_states] * n_states
    ends = [0.1] * n_states
    return SparseHMM(dists, edges=edges, starts=starts, ends=ends,
                     dtype=dtype, max_iter=5, verbose=False)


def test_sparse_hmm_float64_forward_backward_tensors():
    X = _make_synthetic_data(n=2000)
    X_t = torch.tensor(X, dtype=torch.float64)

    model = _make_sparse_hmm(n_states=3, dtype=torch.float64)
    model.fit([X_t])

    # forward() should return float64
    f = model.forward(X_t.unsqueeze(0))
    assert f.dtype == torch.float64, f"forward tensor dtype: {f.dtype}"

    # backward() should return float64
    b = model.backward(X_t.unsqueeze(0))
    assert b.dtype == torch.float64, f"backward tensor dtype: {b.dtype}"


# ---------------------------------------------------------------------------
# 4. State separation — 3 unique states on clean synthetic data
# ---------------------------------------------------------------------------

def test_dense_hmm_float64_state_separation():
    X = _make_synthetic_data(n=10000)
    X_t = torch.tensor(X, dtype=torch.float64)

    dists = [Normal(dtype=torch.float64), Normal(dtype=torch.float64),
             Normal(dtype=torch.float64)]
    model = DenseHMM(dists, dtype=torch.float64, max_iter=30, verbose=False)
    model.fit([X_t])

    # predict expects a 3D tensor (batch, length, features) or list of 2D
    predictions = model.predict(X_t.unsqueeze(0))
    unique_states = torch.unique(predictions).numel()
    assert unique_states == 3, (
        f"Expected 3 unique states, got {unique_states}. "
        f"State collapse likely — check dtype propagation."
    )


def test_sparse_hmm_float64_state_separation():
    X = _make_synthetic_data(n=10000)
    X_t = torch.tensor(X, dtype=torch.float64)

    model = _make_sparse_hmm(n_states=3, dtype=torch.float64)
    model.max_iter = 30
    model.fit([X_t])

    predictions = model.predict(X_t.unsqueeze(0))
    unique_states = torch.unique(predictions).numel()
    assert unique_states == 3, (
        f"Expected 3 unique states, got {unique_states}. "
        f"State collapse likely — check dtype propagation."
    )


# ---------------------------------------------------------------------------
# 5. KMeans float64 buffers
# ---------------------------------------------------------------------------

def test_kmeans_float64_buffers():
    X = torch.randn(500, 3, dtype=torch.float64)
    km = KMeans(k=3, dtype=torch.float64)
    km.fit(X)

    assert km.centroids.dtype == torch.float64, f"centroids: {km.centroids.dtype}"
    assert km._w_sum.dtype == torch.float64, f"_w_sum: {km._w_sum.dtype}"
    assert km._xw_sum.dtype == torch.float64, f"_xw_sum: {km._xw_sum.dtype}"