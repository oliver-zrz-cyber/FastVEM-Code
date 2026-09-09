#pragma once
#include "dfn.h"
#include <Eigen/Sparse>
void save_pressure_vti(const int nx,
                        const int ny,
                        const int nz,
                        const Real dx,
                       const Eigen::VectorXr& pressure,
                       const std::string& filename);

void visualize_A_sparsity(const Eigen::SparseMatrixR &A,const std::string& filename);


