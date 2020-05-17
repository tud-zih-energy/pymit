#include <algorithm>
#include <cmath>
#include <limits>

#define PY_SSIZE_T_CLEAN
#include <Python.h>
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
    bool a_ref = false;
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

    if (normed || weights || density)
    {
        PyErr_SetString(PyExc_NotImplementedError, "Supported arguments: (a)");
        goto fail;
    }

    /* obtain ndarray behind `a` */
    if (PyArray_CHKFLAGS(reinterpret_cast< PyArrayObject* >(a), NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED))
    {
        a_ref = true;
        Py_INCREF(a);
        a_np = reinterpret_cast< PyArrayObject* >(a);
    } else
    {
        a_np = reinterpret_cast< PyArrayObject* >(PyArray_FROM_OTF(a, NPY_DOUBLE, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED));
        if (!a_np)
            goto fail;
    }

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
    hist = reinterpret_cast< PyArrayObject* >(PyArray_SimpleNew(1, &hist_dims, weights ? PyArray_TYPE(weights_np) : NPY_INT64));
    if (!hist)
        goto fail;
    PyArray_FILLWBYTE(hist, 0);
    bin_edges = reinterpret_cast< PyArrayObject* >(PyArray_SimpleNew(1, &bin_edges_dims, NPY_DOUBLE));
    if (!bin_edges)
        goto fail;

    /* workwörk */
    a_data = (double *)PyArray_DATA(a_np);
    if (!a_data)
        goto fail;

    a_length = 1;
    for (int i = 0; i < PyArray_NDIM(a_np); ++i)
        a_length *= PyArray_DIM(a_np, i);

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
            a_min = std::fmin(a_min, a_data[i]);
            a_max = std::fmax(a_max, a_data[i]);
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
    if (a_ref)
        Py_DECREF(a);
    else
        Py_DECREF(a_np);
     Py_XDECREF(bins_np);
    return Py_BuildValue("NN", hist, bin_edges);

fail:
    Py_XDECREF(hist);
    Py_XDECREF(bin_edges);
    Py_XDECREF(a_np);
    Py_XDECREF(bins_np);
    return NULL;
}

PyMethodDef methods[] = {
    {"histogram", (PyCFunction) histogram, METH_VARARGS | METH_KEYWORDS, "Method docstring"},
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