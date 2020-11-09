import math

import numpy as np

import pymit

LOG_BASE = 2


def H(X, bins):
    """
    Calculates the entropy of X.
    If X is already discretised, set \ref bins to the amount of bins, aka states of X.
    If X is not discretised, \ref bins will be used, to diskretise X into  \ref bins states

    @param X np vector
    @param bins If X is already diskretised, amount of bins of X. If X is not discretised, amount of bins to diskretise X into
    @return Returns the entropy of X
    """
    p_x, _ = pymit._lib.histogram(X, bins=bins)
    p_x = p_x / len(X)

    H_ = 0
    for p in p_x:
        if p > 0:
            H_ -= p * math.log(p)
    H_ = H_ / math.log(bins)
    return H_


def H_cond(X, Y, bins):
    """
    Calculates the conditional entropy of X depending on Y.
    If X and Y are already discretised, set \ref bins to the amount of bins, aka states of X and Y.
    If X and Y are not discretised, \ref bins will be used, to diskretise X and Y into \ref bins states

    @param X np vector depending on \ref Y
    @param Y np vector
    @param bins If X and Y are already diskretised specify the amount of bins of X and Y. If X and Y are not discretised specifies the amount of bins to diskretise X into
                bins can be spcified as tuple, e.g. bins = (bins_x, bins_y), to diskretise X and Y independently, or as a single value to diskretise X and Y similar, e.g. bins = amount_of_bins
    @return Returns the conditional entropy
    """
    if type(bins) == list:
        ybins = bins[1]
        xbins = bins[0]
        base = LOG_BASE
    else:
        ybins = bins
        xbins = bins
        base = bins

    p_xy, _, _ = pymit._lib.histogram2d(X, Y, bins=[xbins, ybins])
    p_xy = p_xy / len(X)
    p_y, _ = np.histogram(Y, bins=ybins)
    p_y = p_y / len(Y)

    H = 0
    for i in range(ybins):
        for j in range(xbins):
            if p_xy[j, i] > 0:
                p_cond = p_xy[j, i] / p_y[i]
                H -= p_y[i] * p_cond * math.log(p_cond)
    H = H / math.log(base)
    return H
