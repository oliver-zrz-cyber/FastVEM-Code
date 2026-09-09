#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>

#define DOFSCALE 1

#ifdef VEMFLUID_USE_FLOAT
using Real = float;
#else
using Real = double;
#endif

// Element generation, dense VEM algebra, interpolation, and host-side vectors
// stay in Real. Global sparse matrices and GPU solver buffers may use float.
using GlobalReal = float;

namespace Eigen {
using MatrixXr = Matrix<Real, Dynamic, Dynamic>;
using VectorXr = Matrix<Real, Dynamic, 1>;
using LocalMatrixXr = Matrix<Real, Dynamic, Dynamic>;
using LocalVectorXr = Matrix<Real, Dynamic, 1>;
using GlobalMatrixXr = Matrix<GlobalReal, Dynamic, Dynamic>;
using GlobalVectorXr = Matrix<GlobalReal, Dynamic, 1>;
using SparseMatrixR = SparseMatrix<GlobalReal, RowMajor, int>;
using SparseMatrixCr = SparseMatrix<GlobalReal, ColMajor, int>;
}
