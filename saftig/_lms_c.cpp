// Relevant documentation: https://docs.python.org/3/extending/newtypes_tutorial.html


#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#include <vector>
#include <iostream>

// /// LeastMeanSquares implementation
// class LMSFilter{
// private:
//     uint32_t n_filter, idx_target, n_channel;
//     float step_scale;
//     bool normalized;
// 
//     std::vector<std::vector<double>> filter_coefficients;
// public:
//     LMSFilter(uint32_t n_filter,
//             uint32_t idx_target,
//             uint32_t n_channel,
//             float step_scale,
//             bool normalized=true)
//         : n_filter(n_filter),
//           idx_target(idx_target),
//           n_channel(n_channel),
//           step_scale(step_scale),
//           normalized(normalized),
//           filter_coefficients(n_channel, std::vector<double>(n_filter, 0)) {
//     }
// 
//     /**
//      * @brief reset filter parameters
//      */
//     void reset() {
//         for(auto channel: filter_coefficients) {
//             std::fill(channel.begin(), channel.end(), 0.);
//         }
//     }
// 
//     uint32_t get_n_filter() {return n_filter;}
//     uint32_t get_idx_target() {return idx_target;}
//     uint32_t get_n_channel() {return n_channel;}
// 
//     double step() {
//         return 0.;
//     }
// };



typedef struct {
    PyObject_HEAD;
    int n_filter, idx_target, n_channel;
    float step_scale;
    bool normalized;

    std::vector<std::vector<double>> filter_coefficients;
    /* Type-specific fields go here. */
} LMS_C_OBJECT;

/**
 * check that the array is 2D with the given shape and dtype double
 * raise an exception if not
 * returns false if an exception was raised
 */
bool check_array_properties(PyArrayObject *array, int shape0, int shape1) {
    // check dtype
    if(PyArray_TYPE(array) != NPY_FLOAT64) {
        PyErr_SetString(PyExc_ValueError, "np.float64 is the only supported dtype");
        return false;
    }

    // Check that it's a 2D array
    if (PyArray_NDIM(array) != 2) {
        std::cout << "2+" << std::endl;
        PyErr_SetString(PyExc_ValueError, "Input must be a 2D array.");
        return false;
    }

    // check that dimensions match
    npy_intp* dims = PyArray_DIMS(array);
    int channels = dims[0], n_filter = dims[1];

    if(channels != shape0) {
        PyErr_SetString(PyExc_ValueError, "Input channel count missmatch");
        return false;
    }
    if(n_filter != shape1) {
        PyErr_SetString(PyExc_ValueError, "Input sample count missmatch");
        return false;
    }
    return true;
}

static PyObject *
LMS_C_step(LMS_C_OBJECT *self, PyObject *args)
{
    PyArrayObject* array;
    NpyIter *iter;
    NpyIter_IterNextFunc *iternext;
    NpyIter_GetMultiIndexFunc *get_multi_index;
    npy_intp multi_index[2];
    double** dataptr;

    // get parameter
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &array)) {
        return NULL;
    }

    if(not check_array_properties(array, self->n_channel, self->n_filter)) {
        return NULL;
    }

    // get iterator
    iter = NpyIter_New( array,
            NPY_ITER_READONLY | NPY_ITER_MULTI_INDEX | NPY_ITER_REFS_OK,
            NPY_KEEPORDER,
            NPY_NO_CASTING,
            NULL);
    if (iter == NULL) {
        return NULL;
    }

    if (NpyIter_GetIterSize(iter) == 0) {
        NpyIter_Deallocate(iter);
        return NULL;
    }

    iternext = NpyIter_GetIterNext(iter, NULL);
    dataptr = (double**)NpyIter_GetDataPtrArray(iter);
    get_multi_index = NpyIter_GetGetMultiIndex(iter, NULL);

    if ((iternext == NULL) || (dataptr == NULL) || (get_multi_index == NULL)) {
        NpyIter_Deallocate(iter);
        return NULL;
    }

    // calculate prediction
    do {
        get_multi_index(iter, multi_index);
        printf("multi_index is [%" NPY_INTP_FMT ", %" NPY_INTP_FMT "], %lf\n",
               multi_index[0], multi_index[1], **dataptr);
    } while (iternext(iter));

    // update filter
    char *reset_fail_msg = "Iterator reset failed";
    if(NpyIter_Reset(iter, &reset_fail_msg) != NPY_SUCCEED) {
        return NULL;
    }
    puts("");
    do {
        get_multi_index(iter, multi_index);
        printf("multi_index is [%" NPY_INTP_FMT ", %" NPY_INTP_FMT "], %lf\n",
               multi_index[0], multi_index[1], **dataptr);
    } while (iternext(iter));

    // dealloc the iter instance
    if (!NpyIter_Deallocate(iter)) {
        return NULL;
    }

    return PyFloat_FromDouble(0.);
}

static PyMethodDef LMS_C_methods[] = {
    {"step",
     (PyCFunction) LMS_C_step,
     METH_VARARGS,
     "Return the name, combining the first and last name", },
    {NULL}  /* Sentinel */
};

static int
LMS_C_init(LMS_C_OBJECT *self, PyObject *args, PyObject *kwds)
{
    static char * kwlist[] = {"n_filter", "idx_target", "n_channel", "step_scale", "normalized"};
    unsigned int n_filter, idx_target, n_channel;
    float step_scale;
    bool normalized = true;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "IIIf|p", kwlist,
                                     &n_filter,
                                     &idx_target,
                                     &n_channel,
                                     &step_scale,
                                     &normalized)) {
        return -1;
    }

    self->n_filter = n_filter;
    self->idx_target = idx_target;
    self->n_channel = n_channel;
    self->step_scale = step_scale;
    self->normalized = normalized;

    self->filter_coefficients.insert(self->filter_coefficients.begin(),
            n_channel,
            std::vector<double>(n_filter, 0));

    return 0;
}

static void
LMS_C_dealloc(LMS_C_OBJECT *self)
{
    // this must call Py_XDECREF(object) for all held python objects
    Py_TYPE(self)->tp_free((PyObject *) self);
}


static PyTypeObject LMS_C_TYPE = {
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "LMS_C.LMS_C",
    .tp_basicsize = sizeof(LMS_C_OBJECT),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor) LMS_C_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = PyDoc_STR("LMS Filter implemented in C"),
    .tp_methods = LMS_C_methods,
    .tp_init = (initproc) LMS_C_init,
    .tp_new = PyType_GenericNew,
};

static int
custom_module_exec(PyObject *m)
{
    if (PyType_Ready(&LMS_C_TYPE) < 0) {
        return -1;
    }

    if (PyModule_AddObjectRef(m, "LMS_C", (PyObject *) &LMS_C_TYPE) < 0) {
        return -1;
    }

    return 0;
}

static PyModuleDef_Slot module_slots[] = {
    {Py_mod_exec, (void *) custom_module_exec},
    // Just use this while using static types
    {Py_mod_multiple_interpreters, Py_MOD_MULTIPLE_INTERPRETERS_NOT_SUPPORTED},
    {0, NULL},
};

static struct PyModuleDef _lms_c = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "lms_c",
    .m_doc = "An LMS filter module",
    .m_size = 0,
    .m_slots = module_slots,
};

PyMODINIT_FUNC PyInit__lms_c(void)
{
    import_array(); // enable numpy support, otherwise we get a lot of segfaults
    return PyModuleDef_Init(&_lms_c);
}
