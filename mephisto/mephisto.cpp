#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>

#include <iostream>

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

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
    bool normed;
    PyObject *weights = nullptr;
    bool density;
    static char *kwlist[] = {"a", "bins", "range", "normed", "weights", "density", NULL};

    /* output */
    PyArrayObject *hist = nullptr;
    PyArrayObject *bin_edges = nullptr;

    /* helpers */
    PyArrayObject *a_np = nullptr;
    PyArrayObject *bins_np = nullptr;
    bool bin_edges_arg = false;
    npy_intp hist_dims = 0;
    npy_intp bin_edges_dims = 0;
    PyArrayObject *weights_np = nullptr;
    double *a_data = nullptr;
    npy_intp a_length;
    double a_min = std::numeric_limits< double >::infinity();
    double a_max = -std::numeric_limits< double >::infinity();
    double *bin_edges_data = nullptr;
    int64_t *hist_data = nullptr;
    double hist_bin_width;

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

    /* process `bins` */
    if (bins)
    {
        if (PyArray_IsIntegerScalar(bins))
            hist_dims = PyLong_AsLong(bins);
        else
        {
            bins_np = reinterpret_cast< PyArrayObject* >(PyArray_FROM_OT(bins, NPY_DOUBLE));
            if (!bins_np)
                goto fail;

            if (PyArray_NDIM(bins_np) == 1)
            {
                bin_edges_arg = true;
                hist_dims = PyArray_DIM(bins_np, 0) - 1;
            }
            else
            {
                PyErr_SetString(PyExc_ValueError, "`bins` must be int or 1d array");
                goto fail;
            }
        }
    } else
        hist_dims = 10;
    bin_edges_dims = hist_dims + 1;

    /* allocate output arrays */
    hist = reinterpret_cast< PyArrayObject* >(PyArray_ZEROS(1, &hist_dims, weights ? PyArray_TYPE(weights_np) : NPY_INT64, 0));
    if (!hist)
        goto fail;
    bin_edges = reinterpret_cast< PyArrayObject* >(PyArray_SimpleNew(1, &bin_edges_dims, NPY_DOUBLE));
    if (!bin_edges)
        goto fail;

    /* workwörk */
    a_data = (double *)PyArray_DATA(a_np);
    if (!a_data)
        goto fail;
    a_length = PyArray_SIZE(a_np);

    bin_edges_data = (double *)PyArray_DATA(bin_edges);
    if (!bin_edges_data)
        goto fail;

    if (bin_edges_arg)
    {
        if (PyArray_CopyInto(bin_edges, bins_np) == -1)
        {
            PyErr_SetString(PyExc_RuntimeError, "Failed to copy bin edges");
            goto fail;
        }
    } else
    {
        for (npy_intp i = 0; i < a_length; ++i)
        {
            if (a_data[i] < a_min)
                a_min = a_data[i];
            if (a_data[i] > a_max)
                a_max = a_data[i];
        }
        hist_bin_width = (a_max-a_min)/hist_dims;

        bin_edges_data[0] = a_min;
        for (npy_intp i = 1; i < bin_edges_dims-1; ++i)
            bin_edges_data[i] = a_min + hist_bin_width * i;
        bin_edges_data[bin_edges_dims-1] = a_max;
    }

    hist_data = (int64_t *)PyArray_DATA(hist);
    if (bin_edges_arg)
    {
        PyErr_SetString(PyExc_NotImplementedError, "Arbitrarily spaced bins not supported yet");
        goto fail;
    } else
    {
        for (npy_intp i = 0; i < a_length; ++i)
        {
            auto bin = static_cast< npy_intp >((a_data[i] - a_min) / hist_bin_width);
            bin = std::min(bin, hist_dims-1);
            ++hist_data[bin];
        }
    }

    goto success;

success:
    Py_XDECREF(bins_np);
    Py_DECREF(a_np);
    return Py_BuildValue("NN", hist, bin_edges);

fail:
    Py_XDECREF(bins_np);
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
    bool normed;
    PyObject *weights = nullptr;
    bool density;
    static char *kwlist[] = {"sample", "bins", "range", "normed", "weights", "density", NULL};

    /* output */
    PyArrayObject *H = nullptr;
    PyObject *edges = nullptr;

    /* helpers */
    PyArrayObject *sample_np = nullptr;
    npy_intp sample_N;
    npy_intp sample_D;
    npy_intp sample_length;
    std::unique_ptr< npy_intp[] > H_dims;
    npy_intp bins_;
    PyArrayObject *bins_np = nullptr;
    double *bins_data = nullptr;
    bool bin_edges_arg = false;
    std::unique_ptr< double[] > sample_min = nullptr;
    std::unique_ptr< double[] > sample_max = nullptr;
    std::unique_ptr< PyObject*[] > bin_edges_np = nullptr;
    double *sample_data = nullptr;
    std::unique_ptr< double[] > hist_bin_width = nullptr;
    double *bin_edges_data = nullptr;
    npy_intp bin_edges_dims;
    double *H_data;

    /* argument parsing */
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OOpOp", kwlist, &sample, &bins, &range, &normed, &weights, &density))
        goto fail;

    if (range || normed || weights || density)
    {
        PyErr_SetString(PyExc_NotImplementedError, "Supported arguments: (sample)");
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
    H_dims = std::make_unique< npy_intp[] >(sample_D);

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
    H = reinterpret_cast< PyArrayObject* >(PyArray_ZEROS(sample_D, H_dims.get(), NPY_DOUBLE, 0));
    if (!H)
        goto fail;
    edges = PyList_New(sample_D);
    if (!edges)
        goto fail;
    bin_edges_np = std::make_unique< PyObject*[] >(sample_D);
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
        sample_min = std::make_unique< double[] >(sample_D);
        sample_max = std::make_unique< double[] >(sample_D);
        for (npy_intp i = 0; i < sample_length; ++i)
        {
            auto dim = i % sample_D;
            if (sample_data[i] < sample_min[dim])
                sample_min[dim] = sample_data[i];
            if (sample_data[i] > sample_max[dim])
                sample_max[dim] = sample_data[i];
        }
        hist_bin_width = std::make_unique< double[] >(sample_D);
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
        for (npy_intp i = 0; i < sample_length; i += sample_D)
        {
            npy_intp bin = 0;
            for (npy_int j = 0; j < sample_D; ++j)
            {
                auto bin_ = static_cast< npy_intp >((sample_data[i+j] - sample_min[j]) / hist_bin_width[j]);
                bin_ = std::min(bin_, H_dims[j]-1);
                bin += bin_;
                if (j != sample_D - 1)
                    bin *= H_dims[j+1];
            }
            H_data[bin] += 1;
        }
    }

    goto success;

success:
    Py_XDECREF(bins_np);
    Py_DECREF(sample_np);
    return Py_BuildValue("NN", H, edges);

fail:
    Py_XDECREF(bins_np);
    Py_XDECREF(sample_np);
    Py_XDECREF(edges);
    Py_XDECREF(H);
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
        "histogramdd",
        (PyCFunction) histogramdd,
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