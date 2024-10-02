#ifndef TENSOR_HPP_INCLUDED
#define TENSOR_HPP_INCLUDED

#include <vector>

using Tensor1D = std::vector<double>;
using Tensor2D = std::vector<Tensor1D>;
using Tensor3D = std::vector<Tensor2D>;
using Tensor4D = std::vector<Tensor3D>;

#endif /*TENSOR_HPP_INCLUDED*/