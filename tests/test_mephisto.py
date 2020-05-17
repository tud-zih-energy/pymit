import numpy as np
import pytest
from hypothesis import given, strategies as st

import mephisto as mp


def test_a_ndarray():
    a = np.sin(np.linspace(0, 10*np.pi, 1000))
    hist_np, bin_edges_np = np.histogram(a)
    hist_mp, bin_edges_mp = mp.histogram(a)
    assert bin_edges_np.dtype == bin_edges_mp.dtype
    assert bin_edges_np.ndim == bin_edges_mp.ndim
    assert bin_edges_np.shape == bin_edges_mp.shape
    assert np.array_equal(bin_edges_np, bin_edges_mp)
    assert hist_np.dtype == hist_mp.dtype
    assert hist_np.ndim == hist_mp.ndim
    assert hist_np.shape == hist_mp.shape
    assert hist_np.sum() == hist_mp.sum()
    assert np.array_equal(hist_np,  hist_mp)


def test_a_list():
    a = [1, 1, 2, 2, 3]
    hist_np, bin_edges_np = np.histogram(a)
    hist_mp, bin_edges_mp = mp.histogram(a)
    assert bin_edges_np.dtype == bin_edges_mp.dtype
    assert bin_edges_np.ndim == bin_edges_mp.ndim
    assert bin_edges_np.shape == bin_edges_mp.shape
    assert np.array_equal(bin_edges_np, bin_edges_mp)
    assert hist_np.dtype == hist_mp.dtype
    assert hist_np.ndim == hist_mp.ndim
    assert hist_np.shape == hist_mp.shape
    assert hist_np.sum() == hist_mp.sum()
    assert np.array_equal(hist_np, hist_mp)


@given(st.integers(min_value=1, max_value=2**16))
def test_bins_scalar(bins):
    rng = np.random.RandomState(42)
    a = rng.normal(size=1000)
    hist_np, bin_edges_np = np.histogram(a, bins)
    hist_mp, bin_edges_mp = mp.histogram(a, bins)
    assert bin_edges_np.dtype == bin_edges_mp.dtype
    assert bin_edges_np.ndim == bin_edges_mp.ndim
    assert bin_edges_np.shape == bin_edges_mp.shape
    assert np.array_equal(bin_edges_np, bin_edges_mp)
    assert hist_np.dtype == hist_mp.dtype
    assert hist_np.ndim == hist_mp.ndim
    assert hist_np.shape == hist_mp.shape
    assert hist_np.sum() == hist_mp.sum()
    assert np.array_equal(hist_np,  hist_mp)


@pytest.mark.xfail
def test_bins_array():
    rng = np.random.RandomState(42)
    a = rng.normal(size=1000)
    bins = np.linspace(-5, 5, 10)
    hist_np, bin_edges_np = np.histogram(a, bins)
    hist_mp, bin_edges_mp = mp.histogram(a, bins)
    assert bin_edges_np.dtype == bin_edges_mp.dtype
    assert bin_edges_np.ndim == bin_edges_mp.ndim
    assert bin_edges_np.shape == bin_edges_mp.shape
    assert np.array_equal(bin_edges_np, bin_edges_mp)
    assert hist_np.dtype == hist_mp.dtype
    assert hist_np.ndim == hist_mp.ndim
    assert hist_np.shape == hist_mp.shape
    assert hist_np.sum() == hist_mp.sum()
    assert np.array_equal(hist_np,  hist_mp)
