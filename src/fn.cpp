#include <pybind11/pybind11.h>

namespace py = pybind11;

float fn(float arg1, float arg2)
{
    return arg1 + arg2;
}

PYBIND11_MODULE(sum_module, handle)
{
    handle.doc() = "This is the module docs. Teehee";
    handle.def("fn_python_name", &fn);
}