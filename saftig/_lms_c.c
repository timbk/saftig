// #define PY_SSIZE_T_CLEAN
#include <Python.h>

static PyObject *lms_step_c(PyObject *self, PyObject *args)
{
    return Py_BuildValue("i", 0);
}

static PyMethodDef module_methods[] = {
    {"lms_step_c", lms_step_c, METH_VARARGS, "Multiply two numbers."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef _lms_c =
    {
        PyModuleDef_HEAD_INIT,
        "_lms_c", // the name of the module in Python
        "",            // The docstring in Python
        -1,            /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
        module_methods};

PyMODINIT_FUNC PyInit__lms_c(void)
{
    return PyModule_Create(&_lms_c);
}
