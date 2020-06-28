import math

import numpy as np

import pymit

LOG_BASE = 2


def I(X, Y, bins):
    """
    Calculates the mutual information of X and Y.
    If X and Y are already discretised, set \ref bins to the amount of bins, aka states of X and Y.
    If X and Y are not discretised, \ref bins will be used, to diskretise X and Y into \ref bins states

    @param X np vector
    @param Y np vector
    @param bins If X and Y are already diskretised specify the amount of bins of X and Y. If X and Y are not discretised specifies the amount of bins to diskretise X and Y into
                bins can be spcified as tuple, e.g. bins = (bins_x, bins_y), to diskretise X and Y independently, or as a single value to diskretise X and Y similar, e.g. bins = amount_of_bins
    @return Returns the mutual information
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
    p_y, _ = pymit._lib.histogram(Y, bins=ybins)
    p_y = p_y / len(Y)
    p_x, _ = pymit._lib.histogram(X, bins=xbins)
    p_x = p_x / len(X)

    I_ = 0
    for i in range(xbins):
        for j in range(ybins):
            if p_x[i] > 0 and p_y[j] > 0 and p_xy[i, j] > 0:
                I_ += p_xy[i, j] * math.log(p_xy[i, j] / (p_x[i] * p_y[j]))
    I_ = I_ / math.log(base)
    return I_


def I_cond(X, Y, Z, bins):
    """
    Calculates the conditional mutual information of X and Y depending on Z.
    If X, Y and Z are already discretised, set \ref bins to the amount of bins, aka states of X, Y and Z.
    If X, Y and Z are not discretised, \ref bins will be used, to diskretise X, Y and Z into \ref bins states

    @param X np vector
    @param Y np vector
    @param Z np vector
    @param bins If X, Y and Z are already diskretised specify the amount of bins of X, Y and Z. If X, Y and Z are not discretised specifies the amount of bins to diskretise X, Y and Z into
                bins can be spcified as tuple, e.g. bins = (bins_x, bins_y, bins_z), to diskretise X, Y and Z independently, or as a single value to diskretise X, Y and Z similar, e.g. bins = amount_of_bins
    @return Returns the mutual information
    """
    if type(bins) == list:
        xbins = bins[0]
        ybins = bins[1]
        zbins = bins[2]
        base = LOG_BASE
    else:
        ybins = bins
        xbins = bins
        zbins = bins
        base = bins

    XYZ = np.transpose(np.array([X, Y, Z]))
    p_xyz, _ = pymit._lib.histogramdd(XYZ, bins=[xbins, ybins, zbins])
    p_xyz = p_xyz / len(Z)
    p_xz, _, _ = pymit._lib.histogram2d(X, Z, bins=[xbins, zbins])
    p_xz = p_xz / len(X)
    p_yz, _, _ = pymit._lib.histogram2d(Y, Z, bins=[ybins, zbins])
    p_yz = p_yz / len(Y)
    p_z, _ = pymit._lib.histogram(Z, bins=zbins)
    p_z = p_z / len(Z)

    I = 0
    for i in range(xbins):
        for j in range(ybins):
            for k in range(zbins):
                if p_xyz[i, j, k] > 0 and p_xz[i, k] > 0 and p_yz[j, k] > 0 and p_z[k] > 0:
                    I += p_xyz[i, j, k] * math.log((p_z[k] * p_xyz[i, j, k]) / (p_xz[i, k] * p_yz[j, k]))
    I = I / math.log(base)
    return I
