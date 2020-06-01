#include <cmath>
#include <limits>

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

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
    bool normed = false;
    PyObject *weights = nullptr;
    bool density = false;
    static char *kwlist[] = {"a", "bins", "range", "normed", "weights", "density", NULL};

    /* output */
    PyArrayObject *hist = nullptr;
    PyArrayObject *bin_edges = nullptr;

    /* helpers */
    PyArrayObject *a_np = nullptr;
    PyArrayObject *weights_np = nullptr;
    bool bin_edges_arg;
    bool status;
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
        if (PyArray_NBYTES(hist) < (512 * 1024)) // histogram easily fits in CPU caches - use private histograms
        {
            bin_priv = new int64_t[hist_length];
            for (npy_intp i = 0; i < hist_length; ++i)
                bin_priv[i] = 0;
            #pragma omp parallel for simd aligned(a_data) reduction(+: bin_priv[0:hist_length])
            for (npy_intp i = 0; i < a_length; ++i) {
                auto bin = static_cast< npy_intp >((a_data[i] - a_min) / hist_bin_width);
                auto idx = (bin < hist_length-1) ? bin : hist_length-1;
                ++bin_priv[idx];
            }
            for (npy_intp i = 0; i < hist_length; ++i)
                hist_data[i] = bin_priv[i];
        } else // histogram likely spills to higher level caches or RAM - use shared histogram
        {
            #pragma omp parallel for
            for (npy_intp i = 0; i < a_length; ++i) {
                auto bin = static_cast< npy_intp >((a_data[i] - a_min) / hist_bin_width);
                auto idx = (bin < hist_length-1) ? bin : hist_length-1;
                #pragma omp atomic
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
 * @brief Compute the multidimensional histogram of some data.
 *
 * @see https://numpy.org/doc/stable/reference/generated/numpy.histogramdd.html
 * @see https://github.com/numpy/numpy/blob/v1.18.1/numpy/lib/histograms.py#L945-L1123
 *
 * @param self
 * @param args
 * @return PyObject*
 */
static PyObject*
histogramdd(PyObject *self, PyObject *args, PyObject *kwds) {
    /* input */
    PyObject *sample = nullptr;
    PyObject *bins = nullptr;
    PyObject *range = nullptr;
    bool normed = false;
    PyObject *weights = nullptr;
    bool density = false;
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
        #pragma omp parallel for simd aligned(sample_data) reduction(min: sample_min[0:sample_D]) reduction(max: sample_max[0:sample_D])
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
        if (PyArray_NBYTES(H) < (512 * 1024)) // histogram easily fits in CPU caches - use private histograms
        {
            auto H_length = PyArray_SIZE(H);
            bin_priv = new int64_t[H_length];
            for (npy_intp i = 0; i < H_length; ++i)
                bin_priv[i] = 0;
            #pragma omp parallel for simd aligned(sample_data) reduction(+: bin_priv[0:H_length])
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
            #pragma omp parallel for
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
                #pragma omp atomic
                ++H_data[idx];
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
    if (bin_edges_np != nullptr) { delete[] bin_edges_np; bin_edges_np = nullptr; }
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
    bool bin_edges_arg;
    bool status;

    /* argument parsing */
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O(ff)O", kwlist, &a, &bins, &range_lower, &range_upper, &weights))
        goto fail;

    if (range_lower != -std::numeric_limits< double >::infinity() || range_upper != std::numeric_limits< double >::infinity() || weights)
    {
        PyErr_SetString(PyExc_NotImplementedError, "Supported arguments: (a, bins)");
        goto fail;
    }

    /* obtain ndarray behind `a` */
    a_np = reinterpret_cast< PyArrayObject* >(PyArray_FROM_OTF(a, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED));
    if (!a_np)
        goto fail;

    status = histogram_bin_edges_impl(a_np, bins, range_lower, range_upper, weights_np,
                                      &bin_edges, &bin_edges_arg);
    if (!status)
        goto fail;

    goto success;

success:
    return Py_BuildValue("N", bin_edges);

fail:
    Py_XDECREF(bin_edges);
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
        #pragma omp parallel for simd aligned(a_data) reduction(min: a_min) reduction(max: a_max)
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

PyMethodDef methods[] = {
    {
        "histogram",
        (PyCFunction) histogram,
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
    auto ret = PyModule_Create(&module);
    import_array();
    return ret;
}
