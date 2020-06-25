import inspect
import os
import shutil
import zipfile

import numpy as np
import requests


def madelon(num_examples=None, num_features=None):
    if num_examples is None and num_features is None:
        ARCHIVE = 'MADELON.zip'
        URL = 'http://clopinet.com/isabelle/Projects/NIPS2003/MADELON.zip'
        FOLDER = 'MADELON'

        if not os.path.exists(ARCHIVE):
            with requests.get(URL, stream=True) as r:
                with open(ARCHIVE, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)

        shutil.rmtree(FOLDER, ignore_errors=True)

        with zipfile.ZipFile(ARCHIVE, 'r') as z:
            z.extractall(FOLDER)

        data = np.loadtxt(os.path.join(FOLDER, 'MADELON', 'madelon_train.data'), dtype=np.int)
        labels = np.loadtxt(os.path.join(FOLDER, 'MADELON', 'madelon_train.labels'), dtype=np.int)
    else:
        num_examples = num_examples if num_examples else 2000
        num_features = num_features if num_features else 500
        useful = max(int(0.01 * num_features), 1)
        redundant = int(0.01 * num_features)
        repeated = int(0.02 * num_features)
        useless = num_features - repeated - redundant - useful
        s = 'lambda '
        for i in range(useful-1):
            s += 'x{}, '.format(i)
        s += 'x{}: '.format(useful-1)
        for i in range(useful-1):
            s += 'x{} + '.format(i)
        s += 'x{}'.format(useful-1)
        fun = eval(s)
        labels, data, *_ = _gen_features(fun, useless, redundant, repeated, num_examples)
    return data, labels


def _gen_useful(fun, n, correlation, zeros, value_range=(0, 1)):
    if correlation is None and zeros is None:
        parameters = len(inspect.signature(fun).parameters)
        rng = np.random.default_rng()
        value_range_min, value_range_max = value_range
        param = (rng.uniform(value_range_min, value_range_max, n),)
        for elem in range(1, parameters):
            param = (*param, rng.uniform(value_range_min, value_range_max, n))
        return fun(*param), param
    elif correlation is not None:
        parameters = len(inspect.signature(fun).parameters)
        if len(correlation) != parameters:
            raise ValueError("If \"correlation\" is given, there must be a correlation for each feature")

        rng = np.random.default_rng()
        value_range_min, value_range_max = value_range
        param = (rng.uniform(value_range_min, value_range_max, n),)
        for elem in range(1, parameters):
            param = (*param, rng.uniform(value_range_min, value_range_max, n))
        Y = fun(*param)
        for i, cor in enumerate(correlation):
            for j in range(n):
                if not rng.random() < cor:  # nur austauschen, wenn notwendig
                    param[i][j] = rng.uniform(value_range_min, value_range_max)
        return Y, param
    elif zeros is not None:
        parameters = len(inspect.signature(fun).parameters)
        if len(zeros) != parameters:
            raise ValueError("If \"zeros\" is given, there must be a zeros for each feature")
        rng = np.random.default_rng()
        value_range_min, value_range_max = value_range
        param = (rng.uniform(value_range_min, value_range_max, n),)
        for elem in range(1, parameters):
            param = (*param, rng.uniform(value_range_min, value_range_max, n))
        Y = fun(*param)
        for i, zero in enumerate(zeros):
            for j in range(n):
                if not rng.random() < zero:  # nur austauschen, wenn notwendig
                    param[i][j] = 0
        return Y, param


def _gen_redundant(X, num_redundant_feat):
    num_useful_feat = X.shape[0]
    rng = np.random.default_rng()
    B = 2*rng.random((num_redundant_feat, num_useful_feat))-1  # Greetings from MADELON
    C = np.matmul(B, X)
    return C, B


def _gen_repeated(X, num_repeat_feat):
    num_feat = X.shape[0]
    rng = np.random.default_rng()
    ind = rng.integers(num_feat, size=num_repeat_feat)
    return X[ind, :], ind


def _gen_useless_uniform(features, n, value_range=(0, 1)):
    value_range_min, value_range_max = value_range
    rng = np.random.default_rng()
    param = (rng.uniform(value_range_min, value_range_max, n),)
    for elem in range(1, features):
        param = (*param, rng.uniform(value_range_min, value_range_max, n))
    return param


def _gen_useless_norm(features, n, value_range=(0, 1)):
    mean, std = value_range
    rng = np.random.default_rng()
    param = (rng.normal(mean, std, n),)
    for elem in range(1, features):
        param = (*param, rng.normal(mean, std, n))
    return param


def _gen_features(fun, useless, redundant, repeated, n, correlation=None, zeros=None, value_range=(0, 1)):
    """
    Generated Features
    @param fun function to generate use full features, amount of features will be determined from input function.
    @param useless amount of random features. Half normal distributed, half uniform
    @param redundant linear combination of useful features (unsign random matrix to multiply)
    @param repeated repeated features drawn from useful and redundant
    @param correlation list with percentage of correlated for each relevant features, None if 100% for all.
    """
    labels, data = _gen_useful(fun, n, correlation, zeros, value_range)
    data = np.asarray(data)

    C, B = _gen_redundant(data, redundant)
    data = np.concatenate((data, C))

    C, ind = _gen_repeated(data, repeated)
    data = np.concatenate((data, C))

    lower, upper = data.min(), data.max()
    mean, std = np.mean(data), np.std(data)

    if useless != 0:
        C = _gen_useless_uniform(useless//2, n, (lower, upper))
        data = np.concatenate((data, C))

    if useless > 1:
        C = _gen_useless_norm(useless//2, n, (mean, std))
        data = np.concatenate((data, C))

        if useless % 2 != 0:
            C = _gen_useless_uniform(useless % 2, n, (lower, upper))
            data = np.concatenate((data, C))

    return labels, data.transpose(), B.transpose(), ind
