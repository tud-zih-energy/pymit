import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

import mephisto as mp


def test_histogram_a_ndarray():
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


def test_histogram_a_list():
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


@given(st.integers(min_value=1, max_value=16))
def test_histogram_bins_scalar(bins):
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
def test_histogram_bins_array():
    rng = np.random.RandomState(42)
    a = rng.normal(size=1000)
    bins = np.linspace(-5, 5, 10)
    hist_mp, bin_edges_mp = mp.histogram(a, bins)


def test_histogramdd_sample_ndarray():
    rng = np.random.RandomState(42)
    sample = rng.randn(100, 3)
    H_np, edges_np = np.histogramdd(sample)
    H_mp, edges_mp = mp.histogramdd(sample)
    assert type(edges_np) == type(edges_mp)
    assert len(edges_np) == len(edges_mp)
    assert all(n.dtype == m.dtype for n, m in zip(edges_np, edges_mp))
    assert all(n.ndim == m.ndim for n, m in zip(edges_np, edges_mp))
    assert all(n.shape == m.shape for n, m in zip(edges_np, edges_mp))
    assert all(np.array_equal(n, m) for n, m in zip(edges_np, edges_mp))
    assert H_np.dtype == H_mp.dtype
    assert H_np.ndim == H_mp.ndim
    assert H_np.shape == H_mp.shape
    assert H_np.sum() == H_mp.sum()
    assert np.array_equal(H_np,  H_mp)


@pytest.mark.xfail
def test_histogramdd_sample_list():
    sample = [[1, 1, 2, 2, 3], [1, 2, 2, 3, 3], [2, 2, 3, 3, 4]]
    H_mp, edges_mp = mp.histogramdd(sample)


@given(st.integers(min_value=1, max_value=16))
def test_histogramdd_bins_scalar(bins):
    rng = np.random.RandomState(42)
    sample = rng.randn(100, 3)
    H_np, edges_np = np.histogramdd(sample, bins)
    H_mp, edges_mp = mp.histogramdd(sample, bins)
    assert type(edges_np) == type(edges_mp)
    assert len(edges_np) == len(edges_mp)
    assert all(n.dtype == m.dtype for n, m in zip(edges_np, edges_mp))
    assert all(n.ndim == m.ndim for n, m in zip(edges_np, edges_mp))
    assert all(n.shape == m.shape for n, m in zip(edges_np, edges_mp))
    assert all(np.array_equal(n, m) for n, m in zip(edges_np, edges_mp))
    assert H_np.dtype == H_mp.dtype
    assert H_np.ndim == H_mp.ndim
    assert H_np.shape == H_mp.shape
    assert H_np.sum() == H_mp.sum()
    assert np.array_equal(H_np,  H_mp)


@settings(deadline=None)
@given(st.lists(st.integers(min_value=1, max_value=8), min_size=2, max_size=8))
def test_histogramdd_bins_array_of_scalars(bins):
    rng = np.random.RandomState(42)
    sample = rng.randn(100//len(bins), len(bins))
    H_np, edges_np = np.histogramdd(sample, bins)
    H_mp, edges_mp = mp.histogramdd(sample, bins)
    assert type(edges_np) == type(edges_mp)
    assert len(edges_np) == len(edges_mp)
    assert all(n.dtype == m.dtype for n, m in zip(edges_np, edges_mp))
    assert all(n.ndim == m.ndim for n, m in zip(edges_np, edges_mp))
    assert all(n.shape == m.shape for n, m in zip(edges_np, edges_mp))
    assert all(np.array_equal(n, m) for n, m in zip(edges_np, edges_mp))
    assert H_np.dtype == H_mp.dtype
    assert H_np.ndim == H_mp.ndim
    assert H_np.shape == H_mp.shape
    assert H_np.sum() == H_mp.sum()
    assert np.array_equal(H_np,  H_mp)


@pytest.mark.xfail
def test_histogramdd_bins_narray():
    rng = np.random.RandomState(42)
    sample = rng.randn(100, 3)
    bins = (np.arange(3), np.arange(5), np.arange(7))
    H_mp, edges_mp = mp.histogramdd(sample, bins)
