#pragma once
#include <Eigen/Sparse>
// #include <amgcl/make_solver.hpp>
#include <amgcl/preconditioner/runtime.hpp>
#include <amgcl/backend/vexcl.hpp>
#include "dfn.h"

void eigen_to_csr64(CSR64& csr,Eigen::SparseMatrixR &A);

Eigen::VectorXr amgclSolve(const Eigen::SparseMatrixCr& matrix_col,
    const Eigen::VectorXr& erhs, const Real relativeTol, const Real absTol, const int iter,
    const Real eps_strong = -1, const int block_size = -1, const Real add_diagonal_epsilon = 0,
    const std::vector<GlobalReal> null_spaces = std::vector<GlobalReal>());

// Eigen::VectorXr amgclSolveCPU(const Eigen::SparseMatrixCr& matrix_col,
//     const Eigen::VectorXr& erhs, const Real relativeTol, const Real absTol, const int iter,
//     const Real eps_strong = -1, const int block_size = -1, const Real add_diagonal_epsilon = 0,
//     const std::vector<Real> null_spaces = std::vector<Real>());

// ============================================================
// 初始化部分：只做一次
// ============================================================
struct AMGCLContext {
    using ValueType = GlobalReal;
    using ColumnType = int;
    using PointerType = ptrdiff_t;
    using Backend = amgcl::backend::vexcl<ValueType,ColumnType,PointerType>;
    //
    using Solver  = amgcl::make_solver<
        amgcl::runtime::preconditioner<Backend>,
        amgcl::solver::gmres<Backend>>;
    // using Solver  = amgcl::make_solver<
    // amgcl::runtime::preconditioner<Backend>,
    // amgcl::solver::cg<Backend>>;
    std::shared_ptr<Solver> solver;
    CSR64 my_csr;
    std::shared_ptr<amgcl::backend::crs<ValueType, ColumnType,PointerType>> A_holder;
    typename Backend::params bprm;
    std::shared_ptr<typename Backend::vector> d_rhs;
    std::shared_ptr<typename Backend::vector> d_x;
    std::shared_ptr<vex::Context> q;
    int nrows = 0;
};

// // ============================================================
// // CUDA Context
// // ============================================================
// struct AMGCLContext_CUDA {
//     using Value   = float;
//     using Index   = int;
//     using Backend = amgcl::backend::cuda<Value>;
//
//     using Solver  = amgcl::make_solver<
//         amgcl::runtime::preconditioner<Backend>,
//         amgcl::solver::gmres<Backend>
//     >;
//
//
//     // AMGCL 矩阵 holder（zero_copy_direct 返回的类型）
//     std::shared_ptr<amgcl::backend::crs<Value, Index>> A_holder;
//
//     std::shared_ptr<Solver> solver;
//
//     typename Backend::params bprm; // CUDA backend params
//     std::shared_ptr<typename Backend::vector> d_rhs;
//     std::shared_ptr<typename Backend::vector> d_x;
//
//     int nrows = 0;
//
//     // （可选但更安全）把 nullspace 也绑到 ctx，避免悬垂指针
//     std::vector<Real> nullspace_holder;
// };

// void amgclInit(
//     AMGCLContext_CUDA &ctx,
//     const CSR64 &csr,
//     Real relativeTol, int iter,
//     Real add_diagonal_epsilon,
//     const std::vector<Real> &null_spaces
// );
//
// struct AMGCLContextCPU {
//     using Backend    = amgcl::backend::builtin<Real>;  // CPU 后端
//     using Precond    = amgcl::runtime::preconditioner<Backend>;
//     using SolverIter = amgcl::solver::cg<Backend>;
//
//     std::shared_ptr<Precond>    precond;    // AMG 预条件器（近似逆）
//     std::shared_ptr<SolverIter> solver;     // Krylov 迭代器
//     typename Backend::params bprm;
//
//     std::vector<Real> h_rhs, h_x;         // Host 向量
//     int nrows = 0;
// };
//
// void amgclInitCPU(AMGCLContextCPU &ctx,
//                   const Eigen::SparseMatrixCr &matrix_col,
//                   Real relativeTol, int iter,
//                   Real add_diagonal_epsilon = 0,
//                const std::vector<Real> &null_spaces= std::vector<Real>());

void amgclInit(AMGCLContext &ctx,
               CSR64 &csr,
               Real relativeTol, int iter,
               Real add_diagonal_epsilon,
               const std::vector<GlobalReal> &null_spaces= std::vector<GlobalReal>());

//
// Eigen::VectorXr amgclSolveCPU(const AMGCLContextCPU &ctx,
//                               const Eigen::VectorXr &erhs);

void amgclInit(AMGCLContext &ctx,
                Eigen::SparseMatrixR &matrix,
               Real relativeTol, int iter,
               Real add_diagonal_epsilon = 0,
               const std::vector<GlobalReal> &null_spaces= std::vector<GlobalReal>());


Eigen::VectorXr amgclSolve(const AMGCLContext &ctx,
                           const Eigen::VectorXr &erhs,
                           const Eigen::VectorXr *x0 = nullptr);
