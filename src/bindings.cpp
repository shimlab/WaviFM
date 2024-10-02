#include "bindings.hpp"

namespace py = pybind11;

PYBIND11_MODULE(WaviFM, m)
{
    py::class_<CaviDimensions>(m, "CaviDimensions")
        .def(py::init<int, int, int, int>())
        .def_readonly("n_factors", &CaviDimensions::n_factors)
        .def_readonly("n_resolutions", &CaviDimensions::n_resolutions)
        .def_readonly("n_features", &CaviDimensions::n_features)
        .def_readonly("n_spots", &CaviDimensions::n_spots)
        .def_readonly("p_pi_shape", &CaviDimensions::p_pi_shape)
        .def_readonly("p_eta_shape", &CaviDimensions::p_eta_shape)
        .def_readonly("ab_t_shape", &CaviDimensions::ab_t_shape)
        .def_readonly("ab_tau_shape", &CaviDimensions::ab_tau_shape)
        .def_readonly("F_shape", &CaviDimensions::F_shape)
        .def_readonly("L_skeleton", &CaviDimensions::L_skeleton)
        .def_readonly("Y_skeleton", &CaviDimensions::Y_skeleton);

    py::class_<Parameters>(m, "Parameters")
        .def(py::init<int, int, int, const Tensor4D &, const Tensor1D &, const Tensor1D &, const Tensor2D &,
                      const Tensor2D &, const Tensor2D &, const Tensor2D &, const Tensor4D &,
                      const Tensor4D &, const Tensor4D &, const Tensor2D &, const Tensor2D &,
                      const Tensor2D &, const Tensor2D &, const Tensor2D &, const Tensor2D &,
                      const Tensor2D &>())
        .def_readwrite("n_resolutions", &Parameters::n_resolutions)
        .def_readwrite("n_factors", &Parameters::n_factors)
        .def_readwrite("n_features", &Parameters::n_features)
        .def_readwrite("Y", &Parameters::Y)
        .def_readwrite("log_p_pi", &Parameters::log_p_pi)
        .def_readwrite("log_p_eta", &Parameters::log_p_eta)
        .def_readwrite("alpha_t", &Parameters::alpha_t)
        .def_readwrite("beta_t", &Parameters::beta_t)
        .def_readwrite("alpha_tau", &Parameters::alpha_tau)
        .def_readwrite("beta_tau", &Parameters::beta_tau)
        .def_readwrite("mu_L", &Parameters::mu_L)
        .def_readwrite("sigma_squared_L", &Parameters::sigma_squared_L)
        .def_readwrite("log_r_pi", &Parameters::log_r_pi)
        .def_readwrite("mu_F", &Parameters::mu_F)
        .def_readwrite("sigma_squared_F", &Parameters::sigma_squared_F)
        .def_readwrite("log_r_eta", &Parameters::log_r_eta)
        .def_readwrite("alpha_hat_t", &Parameters::alpha_hat_t)
        .def_readwrite("beta_hat_t", &Parameters::beta_hat_t)
        .def_readwrite("alpha_hat_tau", &Parameters::alpha_hat_tau)
        .def_readwrite("beta_hat_tau", &Parameters::beta_hat_tau);

    py::class_<CaviResult>(m, "CaviResult")
        .def(py::init<Parameters, std::vector<double>, double>())
        .def_readonly("parameters", &CaviResult::parameters)
        .def_readonly("elbo_record", &CaviResult::elbo_record)
        .def_readonly("elbo", &CaviResult::elbo);

    m.def("cavi", &cavi, "Run the CAVI (coordinate ascent variational inference) algorithm for model inference",
          py::arg("parameters"), py::arg("max_iterations"), py::arg("relative_elbo_threshold"));
}