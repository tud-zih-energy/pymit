import numpy as np
import pytest
from hypothesis import assume, given, settings, strategies as st

import mephisto as mp


def test_histogram_a_ndarray():
    a = np.sin(np.linspace(0, 10*np.pi, 1000))
    hist_np, bin_edges_np = np.histogram(a)
    hist_mp, bin_edges_mp = mp.histogram(a)
    assert bin_edges_np.dtype == bin_edges_mp.dtype
    assert bin_edges_np.ndim == bin_edges_mp.ndim
    assert bin_edges_np.shape == bin_edges_mp.shape
    assert np.allclose(bin_edges_np, bin_edges_mp)
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
    assert np.allclose(bin_edges_np, bin_edges_mp)
    assert hist_np.dtype == hist_mp.dtype
    assert hist_np.ndim == hist_mp.ndim
    assert hist_np.shape == hist_mp.shape
    assert hist_np.sum() == hist_mp.sum()
    assert np.array_equal(hist_np, hist_mp)


@given(st.integers(min_value=1, max_value=2**21))
def test_histogram_bins_scalar(bins):
    rng = np.random.RandomState(42)
    a = rng.normal(size=1000)
    hist_np, bin_edges_np = np.histogram(a, bins)
    hist_mp, bin_edges_mp = mp.histogram(a, bins)
    assert bin_edges_np.dtype == bin_edges_mp.dtype
    assert bin_edges_np.ndim == bin_edges_mp.ndim
    assert bin_edges_np.shape == bin_edges_mp.shape
    assert np.allclose(bin_edges_np, bin_edges_mp)
    assert hist_np.dtype == hist_mp.dtype
    assert hist_np.ndim == hist_mp.ndim
    assert hist_np.shape == hist_mp.shape
    assert hist_np.sum() == hist_mp.sum()
    assert np.array_equal(hist_np,  hist_mp)


@pytest.mark.xfail(strict=True, raises=NotImplementedError)
def test_histogram_bins_array():
    rng = np.random.RandomState(42)
    a = rng.normal(size=1000)
    bins = np.linspace(-5, 5, 10)
    hist_mp, bin_edges_mp = mp.histogram(a, bins)


@pytest.mark.xfail(strict=True, raises=NotImplementedError)
def test_histogram_bins_string():
    rng = np.random.RandomState(42)
    a = rng.normal(size=1000)
    bins = 'auto'
    hist_mp, bin_edges_mp = mp.histogram(a, bins)


def test_histogram2d_x_ndarray_y_ndarray():
    x = np.sin(np.linspace(0, 10*np.pi, 1000))
    y = np.sin(np.linspace(0, 10*np.pi, 1000))
    H_np, xedges_np, yedges_np = np.histogram2d(x, y)
    H_mp, xedges_mp, yedges_mp = mp.histogram2d(x, y)
    assert xedges_np.dtype == xedges_mp.dtype
    assert xedges_np.ndim == xedges_mp.ndim
    assert xedges_np.shape == xedges_mp.shape
    assert np.allclose(xedges_np, xedges_mp)
    assert yedges_np.dtype == yedges_mp.dtype
    assert yedges_np.ndim == yedges_mp.ndim
    assert yedges_np.shape == yedges_mp.shape
    assert np.allclose(yedges_np, yedges_mp)
    assert H_np.dtype == H_mp.dtype
    assert H_np.ndim == H_mp.ndim
    assert H_np.shape == H_mp.shape
    assert H_np.sum() == H_mp.sum()
    assert np.array_equal(H_np,  H_mp)


def test_histogram2d_x_list_y_list():
    x = [1, 1, 2, 2, 3]
    y = [1, 1, 2, 2, 3]
    H_np, xedges_np, yedges_np = np.histogram2d(x, y)
    H_mp, xedges_mp, yedges_mp = mp.histogram2d(x, y)
    assert xedges_np.dtype == xedges_mp.dtype
    assert xedges_np.ndim == xedges_mp.ndim
    assert xedges_np.shape == xedges_mp.shape
    assert np.allclose(xedges_np, xedges_mp)
    assert yedges_np.dtype == yedges_mp.dtype
    assert yedges_np.ndim == yedges_mp.ndim
    assert yedges_np.shape == yedges_mp.shape
    assert np.allclose(yedges_np, yedges_mp)
    assert H_np.dtype == H_mp.dtype
    assert H_np.ndim == H_mp.ndim
    assert H_np.shape == H_mp.shape
    assert H_np.sum() == H_mp.sum()
    assert np.array_equal(H_np,  H_mp)


@given(st.integers(min_value=1, max_value=1024))
def test_histogram2d_bins_scalar(bins):
    rng = np.random.RandomState(42)
    x = rng.normal(size=1000)
    y = rng.normal(size=1000)
    H_np, xedges_np, yedges_np = np.histogram2d(x, y, bins)
    H_mp, xedges_mp, yedges_mp = mp.histogram2d(x, y, bins)
    assert xedges_np.dtype == xedges_mp.dtype
    assert xedges_np.ndim == xedges_mp.ndim
    assert xedges_np.shape == xedges_mp.shape
    assert np.allclose(xedges_np, xedges_mp)
    assert yedges_np.dtype == yedges_mp.dtype
    assert yedges_np.ndim == yedges_mp.ndim
    assert yedges_np.shape == yedges_mp.shape
    assert np.allclose(yedges_np, yedges_mp)
    assert H_np.dtype == H_mp.dtype
    assert H_np.ndim == H_mp.ndim
    assert H_np.shape == H_mp.shape
    assert H_np.sum() == H_mp.sum()
    assert np.array_equal(H_np,  H_mp)


@pytest.mark.xfail(strict=True, raises=NotImplementedError)
def test_histogram2d_bins_1Darray():
    rng = np.random.RandomState(42)
    x = rng.normal(size=1000)
    y = rng.normal(size=1000)
    bins = np.linspace(-5, 5, 10)
    H_mp, xedges_mp, yedges_mp = mp.histogram2d(x, y, bins)


@given(st.lists(st.integers(min_value=1, max_value=1024), min_size=2, max_size=2))
def test_histogram2d_bins_list_of_2_ints(bins):
    rng = np.random.RandomState(42)
    x = rng.normal(size=1000)
    y = rng.normal(size=1000)
    H_np, xedges_np, yedges_np = np.histogram2d(x, y, bins)
    H_mp, xedges_mp, yedges_mp = mp.histogram2d(x, y, bins)
    assert xedges_np.dtype == xedges_mp.dtype
    assert xedges_np.ndim == xedges_mp.ndim
    assert xedges_np.shape == xedges_mp.shape
    assert np.allclose(xedges_np, xedges_mp)
    assert yedges_np.dtype == yedges_mp.dtype
    assert yedges_np.ndim == yedges_mp.ndim
    assert yedges_np.shape == yedges_mp.shape
    assert np.allclose(yedges_np, yedges_mp)
    assert H_np.dtype == H_mp.dtype
    assert H_np.ndim == H_mp.ndim
    assert H_np.shape == H_mp.shape
    assert H_np.sum() == H_mp.sum()
    assert np.array_equal(H_np,  H_mp)


@pytest.mark.xfail(strict=True)#, raises=NotImplementedError)
def test_histogram2d_bins_list_of_1Darrays():
    rng = np.random.RandomState(42)
    x = rng.normal(size=1000)
    y = rng.normal(size=1000)
    bins = [np.linspace(-5, 5, 10), np.linspace(-1, 1, 100)]
    H_mp, xedges_mp, yedges_mp = mp.histogram2d(x, y, bins)


@pytest.mark.xfail(strict=True)#, raises=RuntimeError)
def test_histogram2d_bins_list_mixed():
    rng = np.random.RandomState(42)
    x = rng.normal(size=1000)
    y = rng.normal(size=1000)
    bins = [np.linspace(-5, 5, 10), 100]
    H_mp, xedges_mp, yedges_mp = mp.histogram2d(x, y, bins)


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
    assert all(np.allclose(n, m) for n, m in zip(edges_np, edges_mp))
    assert H_np.dtype == H_mp.dtype
    assert H_np.ndim == H_mp.ndim
    assert H_np.shape == H_mp.shape
    assert H_np.sum() == H_mp.sum()
    assert np.allclose(H_np,  H_mp)


@pytest.mark.xfail(strict=True, raises=TypeError)
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
    assert all(np.allclose(n, m) for n, m in zip(edges_np, edges_mp))
    assert H_np.dtype == H_mp.dtype
    assert H_np.ndim == H_mp.ndim
    assert H_np.shape == H_mp.shape
    assert H_np.sum() == H_mp.sum()
    assert np.allclose(H_np,  H_mp)


@settings(deadline=None, database=None)
@given(st.lists(st.integers(min_value=1, max_value=8), min_size=2, max_size=8))
def test_histogramdd_bins_array_of_scalars(bins):
    assume(np.prod(bins) * 8 < 1024 * 1024 * 128)  # don't stress RAM too much
    rng = np.random.RandomState(42)
    sample = rng.randn(1024*1024*16//len(bins), len(bins))
    H_np, edges_np = np.histogramdd(sample, bins)
    H_mp, edges_mp = mp.histogramdd(sample, bins)
    assert type(edges_np) == type(edges_mp)
    assert len(edges_np) == len(edges_mp)
    assert all(n.dtype == m.dtype for n, m in zip(edges_np, edges_mp))
    assert all(n.ndim == m.ndim for n, m in zip(edges_np, edges_mp))
    assert all(n.shape == m.shape for n, m in zip(edges_np, edges_mp))
    assert all(np.allclose(n, m) for n, m in zip(edges_np, edges_mp))
    assert H_np.dtype == H_mp.dtype
    assert H_np.ndim == H_mp.ndim
    assert H_np.shape == H_mp.shape
    assert H_np.sum() == H_mp.sum()
    assert np.allclose(H_np,  H_mp)


@pytest.mark.xfail(strict=True)#, raises=NotImplementedError)
def test_histogramdd_bins_ndarray():
    rng = np.random.RandomState(42)
    sample = rng.randn(100, 3)
    bins = (np.arange(3), np.arange(5), np.arange(7))
    H_mp, edges_mp = mp.histogramdd(sample, bins)


def test_histogram_bin_edges_a_ndarray():
    a = np.sin(np.linspace(0, 10*np.pi, 1000))
    bin_edges_np = np.histogram_bin_edges(a)
    bin_edges_mp = mp.histogram_bin_edges(a)
    assert bin_edges_np.dtype == bin_edges_mp.dtype
    assert bin_edges_np.ndim == bin_edges_mp.ndim
    assert bin_edges_np.shape == bin_edges_mp.shape
    assert np.allclose(bin_edges_np, bin_edges_mp)


def test_histogram_bin_edges_a_list():
    a = [1, 1, 2, 2, 3]
    bin_edges_np = np.histogram_bin_edges(a)
    bin_edges_mp = mp.histogram_bin_edges(a)
    assert bin_edges_np.dtype == bin_edges_mp.dtype
    assert bin_edges_np.ndim == bin_edges_mp.ndim
    assert bin_edges_np.shape == bin_edges_mp.shape
    assert np.allclose(bin_edges_np, bin_edges_mp)


@given(st.integers(min_value=1, max_value=2**21))
def test_histogram_bin_edges_bins_scalar(bins):
    rng = np.random.RandomState(42)
    a = rng.normal(size=1000)
    bin_edges_np = np.histogram_bin_edges(a, bins)
    bin_edges_mp = mp.histogram_bin_edges(a, bins)
    assert bin_edges_np.dtype == bin_edges_mp.dtype
    assert bin_edges_np.ndim == bin_edges_mp.ndim
    assert bin_edges_np.shape == bin_edges_mp.shape
    assert np.allclose(bin_edges_np, bin_edges_mp)


@pytest.mark.xfail(strict=True, raises=NotImplementedError)
def test_histogram_bin_edges_bins_array():
    rng = np.random.RandomState(42)
    a = rng.normal(size=1000)
    bins = np.linspace(-5, 5, 10)
    hist_mp, bin_edges_mp = mp.histogram(a, bins)


@pytest.mark.xfail(strict=True, raises=NotImplementedError)
def test_histogram_bin_edges_bins_string():
    rng = np.random.RandomState(42)
    a = rng.normal(size=1000)
    bins = 'auto'
    hist_mp, bin_edges_mp = mp.histogram(a, bins)


def test_digitize_x_ndarray():
    x = np.sin(np.linspace(0, 10*np.pi, 1000))
    bins = np.linspace(-1, 1, 11)
    indices_np = np.digitize(x, bins)
    indices_mp = mp.digitize(x, bins)
    assert indices_np.dtype == indices_mp.dtype
    assert indices_np.ndim == indices_mp.ndim
    assert indices_np.shape == indices_mp.shape
    assert np.array_equal(indices_np, indices_mp)


def test_digitize_x_list():
    x = [1, 1, 2, 2, 3, 4]
    bins = np.linspace(1, 4, 4)
    indices_np = np.digitize(x, bins)
    indices_mp = mp.digitize(x, bins)
    assert indices_np.dtype == indices_mp.dtype
    assert indices_np.ndim == indices_mp.ndim
    assert indices_np.shape == indices_mp.shape
    assert np.array_equal(indices_np, indices_mp)


def test_digitize_bins_list():
    x = np.sin(np.linspace(0, 10*np.pi, 1000))
    bins = [-1, 0, 1]
    indices_np = np.digitize(x, bins)
    indices_mp = mp.digitize(x, bins)
    assert indices_np.dtype == indices_mp.dtype
    assert indices_np.ndim == indices_mp.ndim
    assert indices_np.shape == indices_mp.shape
    assert np.array_equal(indices_np, indices_mp)


def test_digitize_bins_decreasing():
    x = np.sin(np.linspace(0, 10*np.pi, 1000))
    bins = np.linspace(1, -1, 10)
    indices_np = np.digitize(x, bins)
    indices_mp = mp.digitize(x, bins)
    assert indices_np.dtype == indices_mp.dtype
    assert indices_np.ndim == indices_mp.ndim
    assert indices_np.shape == indices_mp.shape
    assert np.array_equal(indices_np, indices_mp)


@pytest.mark.parametrize('right', [False, True])
def test_digitize_bins_increasing_right(right):
    x = np.array([1, 1, 2, 2, 3, 4])
    bins = np.linspace(1, 4, 4)
    indices_np = np.digitize(x, bins, right)
    indices_mp = mp.digitize(x, bins, right)
    assert indices_np.dtype == indices_mp.dtype
    assert indices_np.ndim == indices_mp.ndim
    assert indices_np.shape == indices_mp.shape
    assert np.array_equal(indices_np, indices_mp)


@pytest.mark.parametrize('right', [False, True])
def test_digitize_bins_decreasing_right(right):
    x = np.array([1, 1, 2, 2, 3, 4])
    bins = np.linspace(4, 1, 4)
    indices_np = np.digitize(x, bins, right)
    indices_mp = mp.digitize(x, bins, right)
    assert indices_np.dtype == indices_mp.dtype
    assert indices_np.ndim == indices_mp.ndim
    assert indices_np.shape == indices_mp.shape
    assert np.array_equal(indices_np, indices_mp)
