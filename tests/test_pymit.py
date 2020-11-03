import numpy as np
import pytest

import pymit

import mephisto as mp


def test_set_library():
    assert pymit._lib is mp
    pymit._set_library('numpy')
    assert pymit._lib is np
    pymit._set_library('mephisto')
    assert pymit._lib is mp


@pytest.mark.parametrize('library', ['numpy', 'mephisto'])
def test_hjmi(library, madelon):
    pymit._set_library(library)
    data, labels = madelon
    bins = 10
    expected_features = [241, 338, 378, 105, 472]#, 475, 433, 64, 128, 442, 453, 336, 48, 493, 281, 318, 153, 28, 451, 455]

    [num_examples, num_features] = data.shape
    data_discrete = np.zeros([num_examples, num_features])
    for i in range(num_features):
        _, bin_edges = pymit._lib.histogram(data[:, i], bins=bins)
        data_discrete[:, i] = pymit._lib.digitize(data[:, i], bin_edges, right=False)

    max_features = len(expected_features)
    selected_features = []
    j_h = 0
    hjmi = None

    for i in range(0, max_features):
        jmi = np.zeros([num_features], dtype=np.float)
        for X_k in range(num_features):
            if X_k in selected_features:
                continue
            jmi_1 = pymit.I(data_discrete[:, X_k], labels, bins=[bins, 2])
            jmi_2 = 0
            for X_j in selected_features:
                tmp1 = pymit.I(data_discrete[:, X_k], data_discrete[:, X_j], bins=[bins, bins])
                tmp2 = pymit.I_cond(data_discrete[:, X_k], data_discrete[:, X_j], labels, bins=[bins, bins, 2])
                jmi_2 += tmp1 - tmp2
            if len(selected_features) == 0:
                jmi[X_k] = j_h + jmi_1
            else:
                jmi[X_k] = j_h + jmi_1 - jmi_2/len(selected_features)
        f = jmi.argmax()
        j_h = jmi[f]
        if hjmi is None or (j_h - hjmi)/hjmi > 0.03:
            hjmi = j_h
            selected_features.append(f)
        else:
            break

    assert np.array_equal(expected_features, selected_features)


@pytest.mark.parametrize('library', ['numpy', 'mephisto'])
def test_jmi(library, madelon):
    pymit._set_library(library)

    data, labels = madelon
    bins = 10
    expected_features = [241, 338, 378, 105, 472]#, 475, 433, 64, 128, 442, 453, 336, 48, 493, 281, 318, 153, 28, 451, 455]

    [num_examples, num_features] = data.shape
    data_discrete = np.zeros([num_examples, num_features])
    for i in range(num_features):
        _, bin_edges = pymit._lib.histogram(data[:, i], bins=bins)
        data_discrete[:, i] = pymit._lib.digitize(data[:, i], bin_edges, right=False)

    max_features = len(expected_features)
    selected_features = []

    mi = np.zeros([num_features], dtype=np.float)
    for i in range(num_features):
        mi[i] = pymit.I(data_discrete[:, i], labels, bins=[bins, 2])
    f = mi.argmax()
    selected_features.append(f)

    for i in range(1, max_features):
        jmi = np.zeros([num_features], dtype=np.float)
        for X_k in range(num_features):
            if X_k in selected_features:
                continue
            for X_j in selected_features:
                sum1 = pymit.I(data_discrete[:, X_j], labels, bins=[bins, 2])
                sum2 = pymit.I_cond(data_discrete[:, X_k], labels, data_discrete[:, X_j], bins=[bins, 2, bins])
                jmi[X_k] += sum1 + sum2
        f = jmi.argmax()
        selected_features.append(f)

    assert np.array_equal(expected_features, selected_features)

def test_transform3D():
    X = np.arange(start=0, stop=10)
    Y = np.arange(start=10, stop=20)
    Z = np.arange(start=20, stop=30)
    pymit._set_library('np')
    XYZ_np = pymit._lib._transform3D(X, Y, Z).astype('float64')
    pymit._set_library('mp')
    XYZ_mp = pymit._lib._transform3D(X, Y, Z)
    assert XYZ_np.dtype == XYZ_mp.dtype
    assert XYZ_np.ndim == XYZ_mp.ndim
    assert XYZ_np.shape == XYZ_mp.shape
    assert np.array_equal(XYZ_np, XYZ_mp)
