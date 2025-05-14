#ifndef TESTING_UTILITIES_INCLUDED
#define TESTING_UTILITIES_INCLUDED

#include "gtest/gtest.h"
#include "tensor.hpp"
#include "parameters.hpp"

// Functions to compare equivalency (up to certain precision) of multidimensional tensors/vectors containing double type values
void compareVectorDouble(const std::vector<double> &v1, const std::vector<double> &v2, double precision);
void compareTensor1D(const Tensor1D &t1, const Tensor1D &t2, double precision);
void compareTensor2D(const Tensor2D &t1, const Tensor2D &t2, double precision);
void compareTensor3D(const Tensor3D &t1, const Tensor3D &t2, double precision);
void compareTensor4D(const Tensor4D &t1, const Tensor4D &t2, double precision);

// Class and function to throw an exception that prints out contents of a std::vector<double>
class VectorException : public std::runtime_error
{
public:
    // Constructor that takes a vector of doubles
    explicit VectorException(const std::vector<double> &vec);

    // Getter to access the stored vector
    const std::vector<double> &getVector() const;

private:
    // Member to hold the vector
    std::vector<double> vector_;

    // Helper function to format the error message
    static std::string formatMessage(const std::vector<double> &vec);
};

void throwVectorException(const std::vector<double> &vec);

#endif /*TESTING_UTILITIES_INCLUDED*/