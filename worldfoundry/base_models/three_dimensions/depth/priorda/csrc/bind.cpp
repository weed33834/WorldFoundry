/*
 * Minimal in-tree binding for PriorDA's KNN utility.
 *
 * The upstream VIPE extension exposes many unrelated modules. PriorDA only
 * needs utils_ext.nearest_neighbours during inference, so WorldFoundry builds
 * this small subset to keep the runtime self-contained.
 */

#include <torch/extension.h>

void pybind_utils_ext(py::module &m);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::module m_utils = m.def_submodule("utils_ext");
    pybind_utils_ext(m_utils);
}
