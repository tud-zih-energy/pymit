#include <cmath>
#include <cstdlib>
#include <limits>

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#define THREAD_CHUNK_SIZE   4096

static Py_ssize_t cache_size_kb;

bool
histogram_bin_edges_impl(
    PyArrayObject* a, PyObject* bins, double range_lower, double range_upper, PyArrayObject* weights,
    PyArrayObject** bin_edges, bool* bin_edges_arg
);

/**
 * @brief Compute the histogram of a set of data.
 *
 * @see https://numpy.org/doc/stable/reference/generated/numpy.histogram.html
 * @see https://github.com/numpy/numpy/blob/v1.18.1/numpy/lib/histograms.py#L680-L931
 *
 * @param self
 * @param args
 * @return PyObject*
 */
static PyObject*
histogram(PyObject *self, PyObject *args, PyObject *kwds) {
    /* input */
    PyObject *a = nullptr;
    PyObject *bins = nullptr;
    double range_lower = -std::numeric_limits< double >::infinity();
    double range_upper = std::numeric_limits< double >::infinity();
    int normed = false;
    PyObject *weights = nullptr;
    int density = false;
    static char *kwlist[] = {"a", "bins", "range", "normed", "weights", "density", NULL};

    /* output */
    PyArrayObject *hist = nullptr;
    PyArrayObject *bin_edges = nullptr;

    /* helpers */
    PyArrayObject *a_np = nullptr;
    PyArrayObject *weights_np = nullptr;
    bool bin_edges_arg = false;
    bool status = true;
    double *bin_edges_data = nullptr;
    npy_intp hist_length;
    double a_min;
    double hist_bin_width;
    double *a_data = nullptr;
    npy_intp a_length;
    int64_t *hist_data = nullptr;
    int64_t *bin_priv = nullptr;

    /* argument parsing */
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O(ff)pOp", kwlist, &a, &bins, &range_lower, &range_upper, &normed, &weights, &density))
        goto fail;

    if (normed || range_lower != -std::numeric_limits< double >::infinity() || range_upper != std::numeric_limits< double >::infinity() || weights || density)
    {
        PyErr_SetString(PyExc_NotImplementedError, "Supported arguments: (a, bins)");
        goto fail;
    }

    /* obtain ndarray behind `a` */
    a_np = reinterpret_cast< PyArrayObject* >(PyArray_FROM_OTF(a, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED));
    if (!a_np)
        goto fail;
    if (PyArray_NDIM(a_np) != 1)
    {
        PyErr_SetString(PyExc_ValueError, "`a` array must be 1D");
        goto fail;
    }

    status = histogram_bin_edges_impl(a_np, bins, range_lower, range_upper, weights_np,
                                      &bin_edges, &bin_edges_arg);
    if (!status)
        goto fail;
    bin_edges_data = (double *)PyArray_DATA(bin_edges);
    if (!bin_edges_data)
        goto fail;
    hist_length = PyArray_SIZE(bin_edges) - 1;
    a_min = bin_edges_data[0];
    hist_bin_width = (bin_edges_data[hist_length] - bin_edges_data[0]) / hist_length;

    /* allocate output array */
    hist = reinterpret_cast< PyArrayObject* >(PyArray_ZEROS(1, &hist_length, weights ? PyArray_TYPE(weights_np) : NPY_INT64, 0));
    if (!hist)
        goto fail;

    /* workwörk */
    a_data = (double *)PyArray_DATA(a_np);
    if (!a_data)
        goto fail;
    a_length = PyArray_SIZE(a_np);

    hist_data = (int64_t *)PyArray_DATA(hist);
    if (bin_edges_arg)
    {
        PyErr_SetString(PyExc_NotImplementedError, "Arbitrarily spaced bins not supported yet");
        goto fail;
    } else
    {
        if (PyArray_NBYTES(hist) < cache_size_kb * 1024) // histogram easily fits in CPU caches - use private histograms
        {
            bin_priv = new int64_t[hist_length];
            for (npy_intp i = 0; i < hist_length; ++i)
                bin_priv[i] = 0;
            #pragma omp parallel for simd schedule(static, THREAD_CHUNK_SIZE) aligned(a_data) reduction(+: bin_priv[0:hist_length])
            for (npy_intp i = 0; i < a_length; ++i) {
                auto bin = static_cast< npy_intp >((a_data[i] - a_min) / hist_bin_width);
                auto idx = (bin < hist_length-1) ? bin : hist_length-1;
                ++bin_priv[idx];
            }
            for (npy_intp i = 0; i < hist_length; ++i)
                hist_data[i] = bin_priv[i];
        } else // histogram likely spills to higher level caches or RAM - use shared histogram
        {
            #pragma omp parallel for schedule(static, THREAD_CHUNK_SIZE)
            for (npy_intp i = 0; i < a_length; ++i) {
                auto bin = static_cast< npy_intp >((a_data[i] - a_min) / hist_bin_width);
                auto idx = (bin < hist_length-1) ? bin : hist_length-1;
                #pragma omp atomic update
                ++hist_data[idx];
            }
        }
    }

    goto success;

success:
    if (bin_priv != nullptr) { delete[] bin_priv; bin_priv = nullptr; }
    Py_DECREF(a_np);
    return Py_BuildValue("NN", hist, bin_edges);

fail:
    if (bin_priv != nullptr) { delete[] bin_priv; bin_priv = nullptr; }
    Py_XDECREF(a_np);
    Py_XDECREF(bin_edges);
    Py_XDECREF(hist);
    return NULL;
}

/**
 * @brief Compute the bi-dimensional histogram of two data samples.
 *
 * @see https://numpy.org/doc/stable/reference/generated/numpy.histogram2d.html
 * @see https://github.com/numpy/numpy/blob/v1.18.1/numpy/lib/twodim_base.py#L584-L716
 *
 * @param self
 * @param args
 * @param kwds
 * @return PyObject*
 */
static PyObject*
histogram2d(PyObject *self, PyObject *args, PyObject *kwds) {
    /* input */
    PyObject *x = nullptr;
    PyObject *y = nullptr;
    PyObject *bins = nullptr;
    PyObject *range = nullptr;
    npy_intp density = false;
    npy_intp normed = false;
    PyObject *weights = nullptr;
    static char *kwlist[] = {"x", "y", "bins", "range", "density", "normed", "weights", NULL};

    /* output */
    PyArrayObject *H = nullptr;
    PyArrayObject *xedges = nullptr;
    PyArrayObject *yedges = nullptr;

    /* helpers */
    PyArrayObject *x_np = nullptr;
    npy_intp x_length;
    PyArrayObject *y_np = nullptr;
    npy_intp y_length;
    npy_intp H_shape[2];
    PyArrayObject *bins_np = nullptr;
    bool bin_edges_arg = false;
    npy_intp edges_shape[2];
    double *x_data = nullptr;
    double *y_data = nullptr;
    double xmin = std::numeric_limits< double >::infinity();
    double xmax = -std::numeric_limits< double >::infinity();
    double ymin = std::numeric_limits< double >::infinity();
    double ymax = -std::numeric_limits< double >::infinity();
    double xbin_width;
    double ybin_width;
    double *xedges_data = nullptr;
    double *yedges_data = nullptr;
    double *H_data = nullptr;
    npy_intp H_length;
    int64_t *bin_priv = nullptr;

    /* argument parsing */
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|OOppO", kwlist, &x, &y, &bins, &range, &density, &normed, &weights))
        goto fail;

    if (range || density || normed || weights)
    {
        PyErr_SetString(PyExc_NotImplementedError, "Supported arguments: (x, y, bins)");
        goto fail;
    }

    /* obtain ndarray behind `x` */
    x_np = reinterpret_cast< PyArrayObject* >(PyArray_FROM_OTF(x, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED));
    if (!x_np)
        goto fail;
    if (PyArray_NDIM(x_np) != 1)
    {
        PyErr_SetString(PyExc_ValueError, "`x` array must be 1D");
        goto fail;
    }
    x_length = PyArray_SIZE(x_np);
    /* obtain ndarray behind `y` */
    y_np = reinterpret_cast< PyArrayObject* >(PyArray_FROM_OTF(y, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED));
    if (!y_np)
        goto fail;
    if (PyArray_NDIM(y_np) != 1)
    {
        PyErr_SetString(PyExc_ValueError, "`x` array must be 1D");
        goto fail;
    }
    y_length = PyArray_SIZE(y_np);

    if (x_length != y_length)
    {
        PyErr_SetString(PyExc_ValueError, "`x` and `y` arrays must have identical shape");
        goto fail;
    }

    /* process `bins` */
    if (bins)
    {
        if (PyArray_IsIntegerScalar(bins)) // the number of bins for the two dimensions (nx=ny=bins)
        {
            auto length = PyLong_AsLong(bins);
            H_shape[0] = length;
            H_shape[1] = length;
        } else if(PySequence_Check(bins) && PySequence_Size(bins) == 2 && !PyArray_Check(bins))
        {
            bins_np = reinterpret_cast< PyArrayObject* >(PyArray_FROM_OT(bins, NPY_NOTYPE));
            if (!bins_np)
                goto fail;

            if (PyArray_NDIM(bins_np) == 1 && PyArray_SIZE(bins_np) == 2) // the number of bins in each dimension (nx, ny = bins)
            {
                Py_DECREF(bins_np);

                bins_np = reinterpret_cast< PyArrayObject* >(PyArray_FROM_OT(bins, NPY_INT64));
                if (!bins_np)
                    goto fail;
                int64_t *bins_data = (int64_t *)PyArray_DATA(bins_np);

                H_shape[0] = bins_data[0];
                H_shape[1] = bins_data[1];
            }
            else if (PyArray_NDIM(bins_np) == 2) // he bin edges in each dimension (x_edges, y_edges = bins)
            {
                bin_edges_arg = true;
                H_shape[0] = PyArray_DIM(bins_np, 0);
                H_shape[1] = PyArray_DIM(bins_np, 1);
            } else
            {
                PyErr_SetString(PyExc_RuntimeError, "`bins` sequence not understood (must be either [int, int] or [array, array])");
                goto fail;
            }
        } else
        {
            bins_np = reinterpret_cast< PyArrayObject* >(PyArray_FROM_OT(bins, NPY_NOTYPE));
            if (!bins_np)
                goto fail;

            if (PyArray_NDIM(bins_np) == 1) // the bin edges for the two dimensions (x_edges=y_edges=bins)
            {
                bin_edges_arg = true;
                auto length = PyArray_DIM(bins_np, 0);
                H_shape[0] = length;
                H_shape[1] = length;
            }
            else
            {
                PyErr_SetString(PyExc_ValueError, "`bins` array must be 1d");
                goto fail;
            }
        }
    } else
    {
        H_shape[0] = 10;
        H_shape[1] = 10;
    }
    edges_shape[0] = H_shape[0] + 1;
    edges_shape[1] = H_shape[1] + 1;

    /* allocate output arrays */
    H = reinterpret_cast< PyArrayObject* >(PyArray_ZEROS(2, H_shape, NPY_DOUBLE, 0));
    if (!H)
        goto fail;
    xedges = reinterpret_cast< PyArrayObject* >(PyArray_SimpleNew(1, &edges_shape[0], NPY_DOUBLE));
    if (!xedges)
        goto fail;
    yedges = reinterpret_cast< PyArrayObject* >(PyArray_SimpleNew(1, &edges_shape[1], NPY_DOUBLE));
    if (!yedges)
        goto fail;

    /* workwörk */
    x_data = (double *)PyArray_DATA(x_np);
    if (!x_data)
        goto fail;
    y_data = (double *)PyArray_DATA(y_np);
    if (!y_data)
        goto fail;

    if (bin_edges_arg)
    {
        PyErr_SetString(PyExc_NotImplementedError, "Arbitrarily spaced bins not supported yet");
        goto fail;
    } else
    {
        // TODO might be more efficient to split this loop into x and y
        #pragma omp parallel for simd schedule(static, THREAD_CHUNK_SIZE) aligned(x_data, y_data) reduction(min: xmin) reduction(max: xmax) reduction(min: ymin) reduction(max: ymax)
        for (npy_intp i = 0; i < x_length; ++i)
        {
            double const x_tmp = x_data[i];
            xmin = (x_tmp < xmin) ? x_tmp : xmin;
            xmax = (x_tmp > xmax) ? x_tmp : xmax;
            double const y_tmp = y_data[i];
            ymin = (y_tmp < ymin) ? y_tmp : ymin;
            ymax = (y_tmp > ymax) ? y_tmp : ymax;
        }
        xbin_width = (xmax-xmin)/H_shape[0];
        ybin_width = (ymax-ymin)/H_shape[1];

        xedges_data = (double *)PyArray_DATA(reinterpret_cast< PyArrayObject* >(xedges));
        if (!xedges_data)
            goto fail;
        xedges_data[0] = xmin;
        for (npy_intp i = 1; i < edges_shape[0] - 1; ++i)
            xedges_data[i] = xmin + xbin_width * i;
        xedges_data[edges_shape[0] - 1] = xmax;

        yedges_data = (double *)PyArray_DATA(reinterpret_cast< PyArrayObject* >(yedges));
        if (!yedges_data)
            goto fail;
        yedges_data[0] = ymin;
        for (npy_intp i = 1; i < edges_shape[1] - 1; ++i)
            yedges_data[i] = ymin + ybin_width * i;
        yedges_data[edges_shape[1] - 1] = ymax;
    }

    H_data = (double *)PyArray_DATA(H);
    H_length = PyArray_SIZE(H);
    if (bin_edges_arg)
    {
        PyErr_SetString(PyExc_NotImplementedError, "Arbitrarily spaced bins not supported yet");
        goto fail;
    } else
    {
        if (PyArray_NBYTES(H) < cache_size_kb * 1024) // histogram easily fits in CPU caches - use private histograms
        {
            bin_priv = new int64_t[H_length];
            for (npy_intp i = 0; i < H_length; ++i)
                bin_priv[i] = 0;
            #pragma omp parallel for simd schedule(static, THREAD_CHUNK_SIZE) aligned(x_data, y_data) reduction(+: bin_priv[0:H_length])
            for (npy_intp i = 0; i < x_length; ++i) {
                auto xbin = static_cast< npy_intp >((x_data[i] - xmin) / xbin_width);
                auto xidx = (xbin < H_shape[0]-1) ? xbin : H_shape[0]-1;
                auto ybin = static_cast< npy_intp >((y_data[i] - ymin) / ybin_width);
                auto yidx = (ybin < H_shape[1]-1) ? ybin : H_shape[1]-1;
                auto idx = xidx * H_shape[1] + yidx;
                ++bin_priv[idx];
            }
            for (npy_intp i = 0; i < H_length; ++i)
                H_data[i] = bin_priv[i];
        } else // histogram likely spills to higher level caches or RAM - use shared histogram
        {
            #pragma omp parallel for schedule(static, THREAD_CHUNK_SIZE)
            for (npy_intp i = 0; i < x_length; ++i) {
                auto xbin = static_cast< npy_intp >((x_data[i] - xmin) / xbin_width);
                auto xidx = (xbin < H_shape[0]-1) ? xbin : H_shape[0]-1;
                auto ybin = static_cast< npy_intp >((y_data[i] - ymin) / ybin_width);
                auto yidx = (ybin < H_shape[1]-1) ? ybin : H_shape[1]-1;
                auto idx = xidx * H_shape[1] + yidx;
                #pragma omp atomic update
                ++H_data[idx]; // floating point addition is not associative - might be too restrictive for compiler optimization
            }
        }
    }


    goto success;

success:
    if (bin_priv != nullptr) { delete[] bin_priv; bin_priv = nullptr; }
    Py_DECREF(y_np);
    Py_DECREF(x_np);
    Py_XDECREF(bins_np);
    return Py_BuildValue("NNN", H, xedges, yedges);

fail:
    if (bin_priv != nullptr) { delete[] bin_priv; bin_priv = nullptr; }
    Py_XDECREF(y_np);
    Py_XDECREF(x_np);
    Py_XDECREF(bins_np);
    Py_XDECREF(yedges);
    Py_XDECREF(xedges);
    Py_XDECREF(H);
    return NULL;
}

/**
 * @brief Compute the multidimensional histogram of some data.
 *
 * @see https://numpy.org/doc/stable/reference/generated/numpy.histogramdd.html
 * @see https://github.com/numpy/numpy/blob/v1.18.1/numpy/lib/histograms.py#L945-L1123
 *
 * @param self
 * @param args
 * @param kwds
 * @return PyObject*
 */
static PyObject*
histogramdd(PyObject *self, PyObject *args, PyObject *kwds) {
    /* input */
    PyObject *sample = nullptr;
    PyObject *bins = nullptr;
    PyObject *range = nullptr;
    npy_intp normed = false;
    PyObject *weights = nullptr;
    npy_intp density = false;
    static char *kwlist[] = {"sample", "bins", "range", "normed", "weights", "density", NULL};

    /* output */
    PyArrayObject *H = nullptr;
    PyObject *edges = nullptr;

    /* helpers */
    PyArrayObject *sample_np = nullptr;
    npy_intp sample_N;
    npy_intp sample_D;
    npy_intp sample_length;
    npy_intp *H_dims = nullptr;
    npy_intp bins_;
    PyArrayObject *bins_np = nullptr;
    double *bins_data = nullptr;
    bool bin_edges_arg = false;
    double *sample_min = nullptr;
    double *sample_max = nullptr;
    PyObject **bin_edges_np = nullptr;
    double *sample_data = nullptr;
    double *hist_bin_width = nullptr;
    double *bin_edges_data = nullptr;
    npy_intp bin_edges_dims;
    double *H_data = nullptr;
    int64_t *bin_priv = nullptr;

    /* argument parsing */
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OOpOp", kwlist, &sample, &bins, &range, &normed, &weights, &density))
        goto fail;

    if (range || normed || weights || density)
    {
        PyErr_SetString(PyExc_NotImplementedError, "Supported arguments: (sample, bins)");
        goto fail;
    }

    /* obtain ndarray behind `sample` */
    if (!PyArray_CheckExact(sample))
    {
        PyErr_SetString(PyExc_TypeError, "Only array-form of `sample` is allowed");
        goto fail;
    }
    sample_np = reinterpret_cast< PyArrayObject* >(PyArray_FROM_OTF(sample, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED));
    if (!sample_np)
        goto fail;
    if (PyArray_NDIM(sample_np) != 2)
    {
        PyErr_SetString(PyExc_ValueError, "`sample` array must be 2D");
        goto fail;
    }
    sample_N = PyArray_DIM(sample_np, 0);
    sample_D = PyArray_DIM(sample_np, 1);
    H_dims = new npy_intp[sample_D];

    /* process `bins` */
    if (bins)
    {
        if (PyArray_IsIntegerScalar(bins)) // The number of bins for all dimensions (nx=ny=…=bins).
        {
            bins_ = PyLong_AsLong(bins);
            for (npy_intp i = 0; i < sample_D; ++i)
                H_dims[i] = bins_;
        }
        else
        {
            bins_np = reinterpret_cast< PyArrayObject* >(PyArray_FROM_OT(bins, NPY_DOUBLE));
            if (!bins_np)
                goto fail;

            if (PyArray_NDIM(bins_np) == 1) // The number of bins for each dimension (nx, ny, … =bins)
            {
                if (PyArray_DIM(bins_np, 0) != sample_D)
                {
                    PyErr_SetString(PyExc_ValueError, "The dimension of `bins` must be equal to the dimension of `sample`");
                    goto fail;
                }
                bins_data = (double *)PyArray_DATA(bins_np);
                for (npy_intp i = 0; i < sample_D; ++i)
                    H_dims[i] = bins_data[i];
            }
            else // A sequence of arrays describing the monotonically increasing bin edges along each dimension.
            {
                bin_edges_arg = true;
                /*
                if (PyArray_NDIM(bins_np) != 2)
                {
                    PyErr_SetString(PyExc_ValueError, "`bins` array must be 2D");
                    goto fail;
                }
                if (PyArray_DIM(bins_np, 0) != sample_D)
                {
                    PyErr_SetString(PyExc_ValueError, "The dimension of `bins` must be equal to the dimension of `sample`");
                    goto fail;
                }

                for (npy_intp i = 0; i < sample_D; ++i)
                    H_dims = ...;
                */
            }
        }
    } else
    {
        for (npy_intp i = 0; i < sample_D; ++i)
            H_dims[i] = 10;
    }

    /* allocate output arrays */
    H = reinterpret_cast< PyArrayObject* >(PyArray_ZEROS(sample_D, H_dims, NPY_DOUBLE, 0));
    if (!H)
        goto fail;
    edges = PyList_New(sample_D);
    if (!edges)
        goto fail;
    bin_edges_np = new PyObject*[sample_D];
    for (npy_intp i = 0; i < sample_D; ++i)
    {
        H_dims[i] += 1;
        bin_edges_np[i] = PyArray_SimpleNew(1, &H_dims[i], NPY_DOUBLE);
        if (!bin_edges_np[i])
            goto fail;
        PyList_SetItem(edges, i, bin_edges_np[i]);
        H_dims[i] -= 1;
    }

    /* workwörk */
    sample_data = (double *)PyArray_DATA(sample_np);
    if (!sample_data)
        goto fail;
    sample_length = PyArray_SIZE(sample_np);
    if (bin_edges_arg)
    {
        PyErr_SetString(PyExc_NotImplementedError, "Arbitrarily spaced bins not supported yet");
        goto fail;
    } else
    {
        sample_min = new double[sample_D];
        sample_max = new double[sample_D];
        for (npy_intp i = 0; i < sample_D; ++i)
        {
            sample_min[i] = std::numeric_limits< double >::infinity();
            sample_max[i] = -std::numeric_limits< double >::infinity();
        }
        #pragma omp parallel for simd schedule(static, THREAD_CHUNK_SIZE)  aligned(sample_data) reduction(min: sample_min[0:sample_D]) reduction(max: sample_max[0:sample_D])
        for (npy_intp i = 0; i < sample_length; ++i)
        {
            auto const dim = i % sample_D;
            double const tmp = sample_data[i];
            sample_min[dim] = (tmp < sample_min[dim]) ? tmp : sample_min[dim];
            sample_max[dim] = (tmp > sample_max[dim]) ? tmp : sample_max[dim];
        }
        hist_bin_width = new double[sample_D];
        for (npy_intp i = 0; i < sample_D; ++i)
        {
            hist_bin_width[i] = (sample_max[i]-sample_min[i])/H_dims[i];

            bin_edges_data = (double *)PyArray_DATA(reinterpret_cast< PyArrayObject* >(bin_edges_np[i]));
            bin_edges_dims = H_dims[i] + 1;
            if (!bin_edges_data)
                goto fail;
            bin_edges_data[0] = sample_min[i];
            for (npy_intp j = 1; j < bin_edges_dims-1; ++j)
                bin_edges_data[j] = sample_min[i] + hist_bin_width[i] * j;
            bin_edges_data[bin_edges_dims-1] = sample_max[i];
        }
    }

    H_data = (double *)PyArray_DATA(H);
    if (bin_edges_arg)
    {
        PyErr_SetString(PyExc_NotImplementedError, "Arbitrarily spaced bins not supported yet");
        goto fail;
    } else
    {
        if (PyArray_NBYTES(H) < cache_size_kb * 1024) // histogram easily fits in CPU caches - use private histograms
        {
            auto H_length = PyArray_SIZE(H);
            bin_priv = new int64_t[H_length];
            for (npy_intp i = 0; i < H_length; ++i)
                bin_priv[i] = 0;
            #pragma omp parallel for simd schedule(static, THREAD_CHUNK_SIZE) aligned(sample_data) reduction(+: bin_priv[0:H_length])
            for (npy_intp i = 0; i < sample_length; i += sample_D)
            {
                npy_intp idx = 0;
                for (npy_int dim = 0; dim < sample_D; ++dim)
                {
                    auto bin = static_cast< npy_intp >((sample_data[i+dim] - sample_min[dim]) / hist_bin_width[dim]);
                    bin = (bin < H_dims[dim]-1) ? bin : H_dims[dim]-1;
                    idx += bin;
                    if (dim != sample_D - 1)
                        idx *= H_dims[dim+1];
                }
                ++bin_priv[idx];
            }
            for (npy_intp i = 0; i < H_length; ++i)
                H_data[i] = bin_priv[i];
        } else // histogram likely spills to higher level caches or RAM - use shared histogram
        {
            #pragma omp parallel for schedule(static, THREAD_CHUNK_SIZE)
            for (npy_intp i = 0; i < sample_length; i += sample_D)
            {
                npy_intp idx = 0;
                for (npy_int dim = 0; dim < sample_D; ++dim)
                {
                    auto bin = static_cast< npy_intp >((sample_data[i+dim] - sample_min[dim]) / hist_bin_width[dim]);
                    bin = (bin < H_dims[dim]-1) ? bin : H_dims[dim]-1;
                    idx += bin;
                    if (dim != sample_D - 1)
                        idx *= H_dims[dim+1];
                }
                #pragma omp atomic update
                ++H_data[idx]; // floating point addition is not associative - might be too restrictive for compiler optimization
            }
        }
    }

    goto success;

success:
    if (bin_priv != nullptr) { delete[] bin_priv; bin_priv = nullptr; }
    if (hist_bin_width != nullptr) { delete[] hist_bin_width; hist_bin_width = nullptr; }
    if (bin_edges_np != nullptr) { delete[] bin_edges_np; bin_edges_np = nullptr; }
    if (sample_max != nullptr) { delete[] sample_max; sample_max = nullptr; }
    if (sample_min != nullptr) { delete[] sample_min; sample_min = nullptr; }
    Py_XDECREF(bins_np);
    if (H_dims != nullptr) { delete[] H_dims; H_dims = nullptr; }
    Py_DECREF(sample_np);
    return Py_BuildValue("NN", H, edges);

fail:
    if (bin_priv != nullptr) { delete[] bin_priv; bin_priv = nullptr; }
    if (hist_bin_width != nullptr) { delete[] hist_bin_width; hist_bin_width = nullptr; }
    if (bin_edges_np != nullptr) { delete[] bin_edges_np; bin_edges_np = nullptr; } // possible memory leak (list elements not GC'ed)
    if (sample_max != nullptr) { delete[] sample_max; sample_max = nullptr; }
    if (sample_min != nullptr) { delete[] sample_min; sample_min = nullptr; }
    Py_XDECREF(bins_np);
    if (H_dims != nullptr) { delete[] H_dims; H_dims = nullptr; }
    Py_XDECREF(sample_np);
    Py_XDECREF(edges);
    Py_XDECREF(H);
    return NULL;
}

/**
 * @brief Function to calculate only the edges of the bins used by the histogram function.
 *
 * @see https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html
 * @see https://github.com/numpy/numpy/blob/v1.18.1/numpy/lib/histograms.py#L473-L672
 *
 * @param self
 * @param args
 * @return PyObject*
 */
static PyObject*
histogram_bin_edges(PyObject *self, PyObject *args, PyObject *kwds) {
    /* input */
    PyObject *a = nullptr;
    PyObject *bins = nullptr;
    double range_lower = -std::numeric_limits< double >::infinity();
    double range_upper = std::numeric_limits< double >::infinity();
    PyObject *weights = nullptr;
    static char *kwlist[] = {"a", "bins", "range", "weights", NULL};

    /* output */
    PyArrayObject *bin_edges = nullptr;

    /* helpers */
    PyArrayObject *a_np = nullptr;
    PyArrayObject *weights_np = nullptr;
    bool bin_edges_arg = false;
    bool status = true;

    /* argument parsing */
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O(ff)O", kwlist, &a, &bins, &range_lower, &range_upper, &weights))
        goto fail;

    if (range_lower != -std::numeric_limits< double >::infinity() || range_upper != std::numeric_limits< double >::infinity() || weights)
    {
        PyErr_SetString(PyExc_NotImplementedError, "Supported arguments: (a, bins)");
        goto fail;
    }

    /* obtain ndarray behind `a` */
    // TODO not necessary when bins is present as array
    a_np = reinterpret_cast< PyArrayObject* >(PyArray_FROM_OTF(a, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED));
    if (!a_np)
        goto fail;

    status = histogram_bin_edges_impl(a_np, bins, range_lower, range_upper, weights_np,
                                      &bin_edges, &bin_edges_arg);
    if (!status)
        goto fail;

    goto success;

success:
    Py_DECREF(a_np);
    return Py_BuildValue("N", bin_edges);

fail:
    Py_XDECREF(a_np);
    Py_XDECREF(bin_edges);
    return NULL;
}

/**
 * @brief Return the indices of the bins to which each value in input array belongs.
 *
 * @see https://numpy.org/doc/stable/reference/generated/numpy.digitize.html
 * @see https://github.com/numpy/numpy/blob/v1.18.1/numpy/lib/function_base.py#L4700-L4808
 *
 * @param self
 * @param args
 * @param kwds
 * @return PyObject*
 */
static PyObject*
digitize(PyObject *self, PyObject *args, PyObject *kwds) {
    /* input */
    PyObject *x = nullptr;
    PyObject *bins = nullptr;
    npy_intp right = false;
    static char *kwlist[] = {"x", "bins", "right", NULL};

    /* output */
    PyArrayObject *indices = nullptr;

    /* helpers */
    PyArrayObject *x_np = nullptr;
    PyArrayObject *bins_np = nullptr;
    double *bins_data = nullptr;
    npy_intp bins_length;
    bool increasing;
    npy_intp *x_shape = nullptr;
    double *x_data = nullptr;
    npy_intp x_length;
    int64_t *indices_data = nullptr;

    /* argument parsing */
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|p", kwlist, &x, &bins, &right))
        goto fail;

    /* obtain ndarray behind `x` */
    x_np = reinterpret_cast< PyArrayObject* >(PyArray_FROM_OTF(x, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED));
    if (!x_np)
        goto fail;
    /* obtain ndarray behind `bins` */
    bins_np = reinterpret_cast< PyArrayObject* >(PyArray_FROM_OTF(bins, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED));
    if (!bins_np)
        goto fail;
    if (PyArray_NDIM(bins_np) != 1)
    {
        PyErr_SetString(PyExc_ValueError, "`bins` must be 1d array");
        goto fail;
    }
    bins_data = (double*)PyArray_DATA(bins_np);
    bins_length = PyArray_SIZE(bins_np);
    if (bins_length < 2)
    {
        PyErr_SetString(PyExc_ValueError, "`bins` must contain at least 2 values");
        goto fail;
    }
    increasing = bins_data[0] < bins_data[1];
    for (npy_intp i = 1; i < bins_length - 1; ++i)
        if ((bins_data[i] >= bins_data[i+1]) == increasing)
        {
            PyErr_SetString(PyExc_ValueError, "`bins` must be monotonic");
            goto fail;
        }

    /* allocate output arrays */
    x_shape = PyArray_DIMS(x_np);
    if (!x_shape)
        goto fail;
    indices = reinterpret_cast< PyArrayObject* >(PyArray_SimpleNew(PyArray_NDIM(x_np), x_shape, NPY_INT64));
    if (!indices)
        goto fail;

    /* wörkwörk */
    x_data = (double*)PyArray_DATA(x_np);
    if (!x_data)
        goto fail;
    x_length = PyArray_SIZE(x_np);
    indices_data = (int64_t*)PyArray_DATA(indices);
    if (!indices_data)
        goto fail;
    if (increasing && !right)
    {
        #pragma omp parallel for schedule(static, THREAD_CHUNK_SIZE)
        for (npy_intp i = 0; i < x_length; ++i)
        {
            auto const tmp = x_data[i];
            int64_t lower = 0;
            int64_t upper = bins_length;
            int64_t bin = 0;
            while (true)
            {
                bin = lower + (upper - lower) / 2;
                if (bin == 0 || bin == bins_length || (bins_data[bin-1] <= tmp && bins_data[bin] > tmp))
                    break;

                if (bins_data[bin] > tmp)
                    upper = bin;
                else
                    lower = bin + 1;
            }
            indices_data[i] = bin;
        }
    } else if (increasing && right)
    {
        #pragma omp parallel for schedule(static, THREAD_CHUNK_SIZE)
        for (npy_intp i = 0; i < x_length; ++i)
        {
            auto const tmp = x_data[i];
            int64_t lower = 0;
            int64_t upper = bins_length;
            int64_t bin = 0;
            while (true)
            {
                bin = lower + (upper - lower) / 2;
                if (bin == 0 || bin == bins_length || (bins_data[bin-1] < tmp && bins_data[bin] >= tmp))
                    break;

                if (bins_data[bin] >= tmp)
                    upper = bin;
                else
                    lower = bin;
            }
            indices_data[i] = bin;
        }
    } else if (!increasing && !right)
    {
        #pragma omp parallel for schedule(static, THREAD_CHUNK_SIZE)
        for (npy_intp i = 0; i < x_length; ++i)
        {
            auto const tmp = x_data[i];
            int64_t lower = 0;
            int64_t upper = bins_length;
            int64_t bin = 0;
            while (true)
            {
                bin = lower + (upper - lower) / 2;
                if (bin == 0 || bin == bins_length || (bins_data[bin-1] > tmp && bins_data[bin] <= tmp))
                    break;

                if (bins_data[bin] <= tmp)
                    upper = bin;
                else
                    lower = bin;
            }
            indices_data[i] = bin;
        }
    } else
    {
        #pragma omp parallel for schedule(static, THREAD_CHUNK_SIZE)
        for (npy_intp i = 0; i < x_length; ++i)
        {
            auto const tmp = x_data[i];
            int64_t lower = 0;
            int64_t upper = bins_length;
            int64_t bin = 0;
            while (true)
            {
                bin = lower + (upper - lower) / 2;
                if (bin == 0 || bin == bins_length || (bins_data[bin-1] >= tmp && bins_data[bin] < tmp))
                    break;

                if (bins_data[bin] < tmp)
                    upper = bin;
                else
                    lower = bin + 1;
            }
            indices_data[i] = bin;
        }
    }

    goto success;

success:
    Py_DECREF(x_shape);
    Py_DECREF(bins_np);
    Py_DECREF(x_np);
    return Py_BuildValue("N", indices);

fail:
    Py_XDECREF(x_shape);
    Py_XDECREF(bins_np);
    Py_XDECREF(x_np);
    Py_XDECREF(indices);
    return NULL;
}

bool
histogram_bin_edges_impl(
    PyArrayObject* a, PyObject* bins, double range_lower, double range_upper, PyArrayObject* weights,
    PyArrayObject** bin_edges, bool* bin_edges_arg
) {
    PyArrayObject *bins_np = nullptr;
    npy_intp hist_length = 0;
    npy_intp bin_edges_length = 0;
    double *a_data = nullptr;
    double *bin_edges_data = nullptr;
    npy_intp a_length = 0;
    double a_min = std::numeric_limits< double >::infinity();
    double a_max = -std::numeric_limits< double >::infinity();
    double hist_bin_width;

    /* process `bins` */
    if (bins)
    {
        if (PyArray_IsIntegerScalar(bins))
            hist_length = PyLong_AsLong(bins);
        else if(PyUnicode_Check(bins))
        {
            PyErr_SetString(PyExc_NotImplementedError, "Bin estimation algorithms not implemented.");
            goto fail;
        } else
        {
            bins_np = reinterpret_cast< PyArrayObject* >(PyArray_FROM_OT(bins, NPY_DOUBLE));
            if (!bins_np)
                goto fail;

            if (PyArray_NDIM(bins_np) == 1)
            {
                *bin_edges_arg = true;
                hist_length = PyArray_DIM(bins_np, 0) - 1;
            }
            else
            {
                PyErr_SetString(PyExc_ValueError, "`bins` must be int or 1d array");
                goto fail;
            }
        }
    } else
        hist_length = 10;
    bin_edges_length = hist_length + 1;

    /* allocate output array */
    *bin_edges = reinterpret_cast< PyArrayObject* >(PyArray_SimpleNew(1, &bin_edges_length, NPY_DOUBLE));
    if (!*bin_edges)
        goto fail;

    /* workwörk */
    a_data = (double *)PyArray_DATA(a);
    if (!a_data)
        goto fail;
    a_length = PyArray_SIZE(a);

    bin_edges_data = (double *)PyArray_DATA(*bin_edges);
    if (!bin_edges_data)
        goto fail;

    if (*bin_edges_arg)
    {
        if (PyArray_CopyInto(*bin_edges, bins_np) == -1)
        {
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy bin edges");
            goto fail;
        }
    } else
    {
        #pragma omp parallel for simd schedule(static, THREAD_CHUNK_SIZE) aligned(a_data) reduction(min: a_min) reduction(max: a_max)
        for (npy_intp i = 0; i < a_length; ++i)
        {
            double const tmp = a_data[i];
            a_min = (tmp < a_min) ? tmp : a_min;
            a_max = (tmp > a_max) ? tmp : a_max;
        }
        hist_bin_width = (a_max-a_min)/hist_length;

        bin_edges_data[0] = a_min;
        for (npy_intp i = 1; i < bin_edges_length-1; ++i)
            bin_edges_data[i] = a_min + hist_bin_width * i;
        bin_edges_data[bin_edges_length-1] = a_max;
    }

    goto success;

success:
    Py_XDECREF(bins_np);
    return true;

fail:
    Py_XDECREF(*bin_edges);
    Py_XDECREF(bins_np);
    return false;
}

static PyObject*
get_cache_size_kb(PyObject *self, PyObject*) {
    return Py_BuildValue("n", cache_size_kb);
}

static PyObject*
set_cache_size_kb(PyObject *self, PyObject *args, PyObject *kwds) {
    /* input */
    Py_ssize_t size;
    static char *kwlist[] = {"size", NULL};

    /* argument parsing */
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "n", kwlist, &size))
        goto fail;

    if (size < 0)
    {
        PyErr_SetString(PyExc_ValueError, "`size` must be non-negative");
        goto fail;
    }

    cache_size_kb = size;
    goto success;

success:
    return Py_None;

fail:
    return NULL;
}

static PyObject*
_I_impl(PyObject *self, PyObject *args, PyObject *kwds) {
    /* input */
    PyObject *p_xy = nullptr;
    PyObject *p_x = nullptr;
    PyObject *p_y = nullptr;
    PyObject *xbins = nullptr;
    PyObject *ybins = nullptr;
    PyObject *base = nullptr;
    static char *kwlist[] = {"p_xy", "p_x", "p_y", "xbins", "ybins", "base", NULL};

    /* output */
    double I_ = 0.;

    /* helpers */
    long xbins_;
    long ybins_;
    long base_;
    PyArrayObject *p_xy_np = nullptr;
    PyArrayObject *p_x_np = nullptr;
    PyArrayObject *p_y_np = nullptr;
    double *p_xy_data = nullptr;
    double *p_x_data = nullptr;
    double *p_y_data = nullptr;

    /* argument parsing */
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOOOO", kwlist, &p_xy, &p_x, &p_y, &xbins, &ybins, &base))
        goto fail;

    xbins_ = PyLong_AsLong(xbins);
    if(xbins_ == -1)
        goto fail;
    ybins_ = PyLong_AsLong(ybins);
    if(ybins_ == -1)
        goto fail;
    base_ = PyLong_AsLong(base);
    if(base_ == -1)
        goto fail;

    /* obtain ndarray behind `p_xy` */
    p_xy_np = reinterpret_cast< PyArrayObject* >(PyArray_FROM_OTF(p_xy, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED));
    if (!p_xy_np)
        goto fail;
    /* obtain ndarray behind `p_x` */
    p_x_np = reinterpret_cast< PyArrayObject* >(PyArray_FROM_OTF(p_x, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED));
    if (!p_x_np)
        goto fail;
    /* obtain ndarray behind `p_y` */
    p_y_np = reinterpret_cast< PyArrayObject* >(PyArray_FROM_OTF(p_y, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED));
    if (!p_y_np)
        goto fail;

    p_xy_data = (double *)PyArray_DATA(p_xy_np);
    p_x_data = (double *)PyArray_DATA(p_x_np);
    p_y_data = (double *)PyArray_DATA(p_y_np);
    #pragma omp parallel for schedule(static, THREAD_CHUNK_SIZE) collapse(2) reduction(+: I_)
    for (long x = 0; x < xbins_; ++x)
    {
        for (long y = 0; y < ybins_; ++y)
        {
            double const xy_ = *(p_xy_data + x * ybins_ + y);
            double const x_ = *(p_x_data + x);
            double const y_ = *(p_y_data + y);
            if(xy_ > 0 && x_ > 0 && y_ > 0)
                I_ += xy_ * std::log(xy_ / (x_ * y_));
        }
    }
    I_ /= std::log(base_);

success:
    Py_DECREF(p_y_np);
    Py_DECREF(p_x_np);
    Py_DECREF(p_xy_np);
    return Py_BuildValue("d", I_);

fail:
    Py_XDECREF(p_y_np);
    Py_XDECREF(p_x_np);
    Py_XDECREF(p_xy_np);
    return NULL;
}

static PyObject*
_I_cond_impl(PyObject *self, PyObject *args, PyObject *kwds) {
    /* input */
    PyObject *p_xyz = nullptr;
    PyObject *p_xz = nullptr;
    PyObject *p_yz = nullptr;
    PyObject *p_z = nullptr;
    PyObject *xbins = nullptr;
    PyObject *ybins = nullptr;
    PyObject *zbins = nullptr;
    PyObject *base = nullptr;
    static char *kwlist[] = {"p_xyz", "p_xz", "p_yz", "p_z", "xbins", "ybins", "zbins", "base", NULL};

    /* output */
    double I_ = 0.;

    /* helpers */
    long xbins_;
    long ybins_;
    long zbins_;
    long base_;
    PyArrayObject *p_xyz_np = nullptr;
    PyArrayObject *p_xz_np = nullptr;
    PyArrayObject *p_yz_np = nullptr;
    PyArrayObject *p_z_np = nullptr;
    double *p_xyz_data = nullptr;
    double *p_xz_data = nullptr;
    double *p_yz_data = nullptr;
    double *p_z_data = nullptr;

    /* argument parsing */
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOOOOOO", kwlist, &p_xyz, &p_xz, &p_yz, &p_z, &xbins, &ybins, &zbins, &base))
        goto fail;

    xbins_ = PyLong_AsLong(xbins);
    if(xbins_ == -1)
        goto fail;
    ybins_ = PyLong_AsLong(ybins);
    if(ybins_ == -1)
        goto fail;
    zbins_ = PyLong_AsLong(zbins);
    if(zbins_ == -1)
        goto fail;
    base_ = PyLong_AsLong(base);
    if(base_ == -1)
        goto fail;

    /* obtain ndarray behind `p_xyz` */
    p_xyz_np = reinterpret_cast< PyArrayObject* >(PyArray_FROM_OTF(p_xyz, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED));
    if (!p_xyz_np)
        goto fail;
    /* obtain ndarray behind `p_xz` */
    p_xz_np = reinterpret_cast< PyArrayObject* >(PyArray_FROM_OTF(p_xz, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED));
    if (!p_xz_np)
        goto fail;
    /* obtain ndarray behind `p_yz` */
    p_yz_np = reinterpret_cast< PyArrayObject* >(PyArray_FROM_OTF(p_yz, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED));
    if (!p_yz_np)
        goto fail;
    /* obtain ndarray behind `p_z` */
    p_z_np = reinterpret_cast< PyArrayObject* >(PyArray_FROM_OTF(p_z, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED));
    if (!p_z_np)
        goto fail;

    p_xyz_data = (double *)PyArray_DATA(p_xyz_np);
    p_xz_data = (double *)PyArray_DATA(p_xz_np);
    p_yz_data = (double *)PyArray_DATA(p_yz_np);
    p_z_data = (double *)PyArray_DATA(p_z_np);
    #pragma omp parallel for schedule(static, THREAD_CHUNK_SIZE) collapse(3) reduction(+: I_)
    for (long x = 0; x < xbins_; ++x)
        for (long y = 0; y < ybins_; ++y)
            for (long z = 0; z < zbins_; ++z)
            {
                double const xyz_ = *(p_xyz_data + x * ybins_ * zbins_ + y * zbins_ + z);
                double const xz_ = *(p_xz_data + x * zbins_ + z);
                double const yz_ = *(p_yz_data + y * zbins_ + z);
                double const z_ = *(p_z_data + z);
                if(xyz_ > 0 && xz_ > 0 && yz_ > 0 && z_ > 0)
                    I_ += xyz_ * std::log((z_ * xyz_) / (xz_ * yz_));
            }
    I_ /= std::log(base_);

success:
    Py_DECREF(p_z_np);
    Py_DECREF(p_yz_np);
    Py_DECREF(p_xz_np);
    Py_DECREF(p_xyz_np);
    return Py_BuildValue("d", I_);

fail:
    Py_XDECREF(p_z_np);
    Py_XDECREF(p_yz_np);
    Py_XDECREF(p_xz_np);
    Py_XDECREF(p_xyz_np);
    return NULL;
}

static PyObject *
_transform3D(PyObject *self, PyObject *args, PyObject *kwds)
{
    /* input */
    PyObject *x = nullptr;
    PyObject *y = nullptr;
    PyObject *z = nullptr;
    static char *kwlist[] = {"X", "Y", "Z", NULL};

    /* output */
    PyArrayObject *xyz_np = nullptr;

    /* helpers */
    PyArrayObject *x_np = nullptr;
    PyArrayObject *y_np = nullptr;
    PyArrayObject *z_np = nullptr;
    npy_intp length;
    npy_intp XYZ_dims[2];
    double *x_data = nullptr;
    double *y_data = nullptr;
    double *z_data = nullptr;
    double *xyz_data = nullptr;

    /* argument parsing */
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOO", kwlist, &x, &y, &z))
        goto fail;

    /* obtain ndarray behind `X` */
    x_np = reinterpret_cast<PyArrayObject *>(PyArray_FROM_OTF(x, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED));
    if (!x_np)
        goto fail;
    if (PyArray_NDIM(x_np) != 1)
    {
        PyErr_SetString(PyExc_ValueError, "`X` array must be 1D");
        goto fail;
    }
    /* obtain ndarray behind `Y` */
    y_np = reinterpret_cast<PyArrayObject *>(PyArray_FROM_OTF(y, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED));
    if (!y_np)
        goto fail;
    if (PyArray_NDIM(y_np) != 1)
    {
        PyErr_SetString(PyExc_ValueError, "`Y` array must be 1D");
        goto fail;
    }
    /* obtain ndarray behind `Z` */
    z_np = reinterpret_cast<PyArrayObject *>(PyArray_FROM_OTF(z, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED));
    if (!z_np)
        goto fail;
    if (PyArray_NDIM(z_np) != 1)
    {
        PyErr_SetString(PyExc_ValueError, "`Y` array must be 1D");
        goto fail;
    }

    length = PyArray_SIZE(x_np);
    if (length != PyArray_SIZE(y_np) || length != PyArray_SIZE(z_np))
    {
        PyErr_SetString(PyExc_ValueError, "X, Y and Z must be of identical length");
        goto fail;
    }

    XYZ_dims[0] = length;
    XYZ_dims[1] = 3;
    xyz_np = reinterpret_cast<PyArrayObject *>(PyArray_EMPTY(2, XYZ_dims, NPY_DOUBLE, 0));
    if (!xyz_np)
        goto fail;

    /* workwörk */
    x_data = (double *)PyArray_DATA(x_np);
    if (!x_data)
        goto fail;
    y_data = (double *)PyArray_DATA(y_np);
    if (!y_data)
        goto fail;
    z_data = (double *)PyArray_DATA(z_np);
    if (!z_data)
        goto fail;
    xyz_data = (double *)PyArray_DATA(xyz_np);
    if (!xyz_data)
        goto fail;

    #pragma omp parallel for simd schedule(static, THREAD_CHUNK_SIZE) aligned(x_data) aligned(y_data) aligned(z_data) aligned(xyz_data)
    for (npy_intp i = 0; i < length; ++i)
    {
        xyz_data[i * 3] = x_data[i];
        xyz_data[i * 3 + 1] = y_data[i];
        xyz_data[i * 3 + 2] = z_data[i];
    }

success:
    Py_DECREF(z_np);
    Py_DECREF(y_np);
    Py_DECREF(x_np);
    return Py_BuildValue("N", xyz_np);

fail:
    Py_XDECREF(xyz_np);
    Py_XDECREF(z_np);
    Py_XDECREF(y_np);
    Py_XDECREF(x_np);
    return NULL;
}

PyMethodDef methods[] = {
    {
        "histogram",
        (PyCFunction) histogram,
        METH_VARARGS | METH_KEYWORDS,
        "Method docstring"
    },
    {
        "histogram2d",
        (PyCFunction) histogram2d,
        METH_VARARGS | METH_KEYWORDS,
        "Method docstring"
    },
    {
        "histogramdd",
        (PyCFunction) histogramdd,
        METH_VARARGS | METH_KEYWORDS,
        "Method docstring"
    },
    {
        "histogram_bin_edges",
        (PyCFunction) histogram_bin_edges,
        METH_VARARGS | METH_KEYWORDS,
        "Method docstring"
    },
    {
        "digitize",
        (PyCFunction) digitize,
        METH_VARARGS | METH_KEYWORDS,
        "Method docstring"
    },
    {
        "get_cache_size_kb",
        (PyCFunction) get_cache_size_kb,
        METH_NOARGS,
        "Method docstring"
    },
    {
        "set_cache_size_kb",
        (PyCFunction) set_cache_size_kb,
        METH_VARARGS | METH_KEYWORDS,
        "Method docstring"
    },
    {
        "_I_impl",
        (PyCFunction) _I_impl,
        METH_VARARGS | METH_KEYWORDS,
        "Method docstring"
    },
    {
        "_I_cond_impl",
        (PyCFunction) _I_cond_impl,
        METH_VARARGS | METH_KEYWORDS,
        "Method docstring"
    },
    {
        "_transform3D",
        (PyCFunction) _transform3D,
        METH_VARARGS | METH_KEYWORDS,
        "Method docstring"
    },
    {NULL, NULL, 0, NULL} /* sentinel */
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "mephisto", /* name of module */
    "Module docstring",
    -1, /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    methods
};

PyMODINIT_FUNC
PyInit_mephisto(void) {
    char* env_cache_size_kb;
    PyObject *ret = PyModule_Create(&module);
    import_array();

    cache_size_kb = 128;
    env_cache_size_kb = std::getenv("PYMIT_CACHE_SIZE_KB");
    if (env_cache_size_kb)
    {
        auto tmp = std::atoi(env_cache_size_kb);
        if (0 < tmp)
            cache_size_kb = tmp;
    }

    return ret;
}
