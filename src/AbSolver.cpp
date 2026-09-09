#include "AbSolver.h"
// #include <amgcl/adapter/eigen.hpp>
#include <amgcl/adapter/zero_copy.hpp>

void eigen_to_csr64(CSR64& csr,Eigen::SparseMatrixR &A)
{
    FUNC_TIMER();
    A.makeCompressed();
    csr.nrows = static_cast<CSR64::PointerType>(A.rows());
    csr.ncols = static_cast<CSR64::PointerType>(A.cols());
    const int nnz = static_cast<int>(A.nonZeros());
    csr.row_ptr.resize(csr.nrows + 1);
    csr.col_idx.resize(nnz);
    csr.values.resize(nnz);

    // Eigen row pointers
    const int *outer = A.outerIndexPtr();
    const int *inner = A.innerIndexPtr();
    const GlobalReal *vals = A.valuePtr();

    for (ptrdiff_t i = 0; i <= csr.nrows; ++i)
        csr.row_ptr[i] = static_cast<CSR64::PointerType>(outer[i]);

    for (int k = 0; k < nnz; ++k) {
        csr.col_idx[k] = static_cast<CSR64::ColumnType>(inner[k]);
        csr.values[k]  = vals[k];
    }

    // === 关键一步：释放 Eigen matrix 内存 ===
    Eigen::SparseMatrixR().swap(A);

}



void amgclInit(AMGCLContext &ctx,
               Eigen::SparseMatrixR &matrix,
               Real relativeTol, int iter,
               Real add_diagonal_epsilon,
               const std::vector<GlobalReal> &null_spaces)
{
    using Backend = AMGCLContext::Backend;
    using Solver  = AMGCLContext::Solver;
    eigen_to_csr64(ctx.my_csr,matrix);
    // 再释放矩阵 holder（zero-copy adapter，本身不一定占显存，但安全）
    ctx.A_holder.reset();

    // ---------- 2) 重建矩阵 adapter ----------
    ctx.A_holder = amgcl::adapter::zero_copy_direct(
        ctx.my_csr.nrows,
        ctx.my_csr.ncols,
        ctx.my_csr.row_ptr.data(),
        ctx.my_csr.col_idx.data(),
        ctx.my_csr.values.data()
    );
    amgcl::backend::sort_rows(*ctx.A_holder);
    // if (add_diagonal_epsilon != 0.0) {
    //     for (int i=0;i<matrix.rows();++i)
    //         matrix.coeffRef(i,i)+=add_diagonal_epsilon;
    // }

    vex::Context ctx_vex(vex::Filter::Env);
    ctx.bprm.q = ctx_vex;

    boost::property_tree::ptree prm;
    prm.put("solver.tol", relativeTol);
    prm.put("solver.maxiter", iter);
    // prm.put("precond.class", "relaxation");
    // prm.put("precond.type", "chebyshev");
    // prm.put("precond.degree", "5");
    // prm.put("precond.power_iters", 200);
    // prm.put("precond.higher", 2.0f);
    // prm.put("precond.lower", 1.0f / 80.0f);
    // prm.put("precond.scale", true);
    //
    //
    prm.put("precond.class", "amg");
     prm.put("precond.relax.type", "spai0");
     prm.put("precond.relax.degree", 5);
     prm.put("precond.relax.power_iters", 200);
     prm.put("precond.relax.higher", 2.0f);
     prm.put("precond.relax.lower", 1.0f / 120.0f);
     prm.put("precond.relax.scale", true);
     prm.put("precond.max_levels", 5);
     prm.put("precond.direct_coarse", false);
     prm.put("precond.coarsening.type", "aggregation");
    // 这里用不用smoothed aggregation区别很大 那么在非RAP矩阵上呢 也会更好吗
    // prm.put("precond.coarsening.type", "smoothed_aggregation");
    // prm.put("precond.coarsening.estimate_spectral_radius", true);
    // prm.put("precond.coarsening.relax", 0.5f);
    // prm.put("precond.coarsening.power_iters", 100);
    prm.put("precond.pre_cycles", 1);
    prm.put("precond.npre", 3);
    prm.put("precond.npost", 3);
    prm.put("precond.coarsening.verbose", true);
    prm.put("precond.verbose", true);
    prm.put("solver.verbose", true);

    if (!null_spaces.empty()) {
        RZ_INFO("null cols:",null_spaces.size() / matrix.rows());
        prm.put("precond.coarsening.nullspace.cols", null_spaces.size() / matrix.rows());
        prm.put("precond.coarsening.nullspace.rows", matrix.rows());
        prm.put("precond.coarsening.nullspace.B", &null_spaces[0]);
    }

    prm.put("precond.ncycle", 1);

    ctx.solver = std::make_shared<Solver>(ctx.A_holder,prm);
    ctx.nrows  = ctx.A_holder->nrows;
    ctx.d_rhs = std::make_shared<typename Backend::vector>(ctx.bprm.q, ctx.nrows);
    ctx.d_x   = std::make_shared<typename Backend::vector>(ctx.bprm.q, ctx.nrows);

}


static bool first_init = true;


void amgclInit(AMGCLContext &ctx,
               CSR64 &csr,
               Real relativeTol, int iter,
               Real add_diagonal_epsilon,
               const std::vector<GlobalReal> &null_spaces)
{
    FUNC_TIMER();
    using Backend = AMGCLContext::Backend;
    using Solver  = AMGCLContext::Solver;

    // ---------- 0) 确保 context 只建一次 ----------
    if (first_init) {
        first_init = false;
        ctx.q = std::make_shared<vex::Context>(vex::Filter::Env);
    }
    ctx.bprm.q = *ctx.q;

    // ---------- 1) 释放上一轮 GPU 显存（关键） ----------
    // 1.1 先确保队列完成，避免“延迟释放/看起来不掉显存”
    // vex::Context 可能有多个设备/queue；建议全 finish
    for (uint i = 0; i < ctx.q->size(); ++i) {
        ctx.q->queue(i).finish();
    }

    // 1.2 按“依赖关系”逆序 reset（vector -> solver -> matrix）
    // 先释放向量（它们直接持 cl buffer）
    ctx.d_rhs.reset();
    ctx.d_x.reset();

    // 再释放 solver（AMG hierarchy / 临时 buffer 最大头）
    ctx.solver.reset();

    // 再释放矩阵 holder（zero-copy adapter，本身不一定占显存，但安全）
    ctx.A_holder.reset();

    // ---------- 2) 重建矩阵 adapter ----------
    ctx.A_holder = amgcl::adapter::zero_copy_direct(
        csr.nrows,
        csr.ncols,
        csr.row_ptr.data(),
        csr.col_idx.data(),
        csr.values.data()
    );
    amgcl::backend::sort_rows(*ctx.A_holder);

    // ---------- 3) 参数 ----------
    boost::property_tree::ptree prm;
    prm.put("solver.tol", relativeTol);
    prm.put("solver.maxiter", iter);

    // Match the GMG Chebyshev smoother. All other AMG settings use AMGCL
    // defaults: smoothed aggregation, unlimited levels, direct coarse solve,
    // one pre-/post-smoothing step, and one V-cycle.
    prm.put("precond.relax.type", "chebyshev");
    prm.put("precond.relax.degree", 3);
    prm.put("precond.relax.power_iters", 200);
    prm.put("precond.relax.higher", 2.0f);
    prm.put("precond.relax.lower", 1.0f / 120.0f);
    prm.put("precond.relax.scale", true);

    if (!null_spaces.empty()) {
        prm.put("precond.coarsening.nullspace.cols", null_spaces.size() / csr.nrows);
        prm.put("precond.coarsening.nullspace.rows", csr.nrows);
        prm.put("precond.coarsening.nullspace.B", null_spaces.data());
    }

    // ---------- 4) 重建 solver + vectors ----------
    ctx.solver = std::make_shared<Solver>(ctx.A_holder, prm);
    std::cout << (*ctx.solver) << std::endl;

    ctx.nrows = csr.nrows;
    ctx.d_rhs = std::make_shared<typename Backend::vector>(ctx.bprm.q, ctx.nrows);
    ctx.d_x   = std::make_shared<typename Backend::vector>(ctx.bprm.q, ctx.nrows);
}



Eigen::VectorXr amgclSolve(const AMGCLContext &ctx,
                           const Eigen::VectorXr &erhs,
                           const Eigen::VectorXr *x0)
{
    FUNC_TIMER();
    const int n = ctx.nrows;

    // 1️⃣ 上传 RHS
    vex::copy(std::vector<GlobalReal>(erhs.data(), erhs.data() + n), *ctx.d_rhs);

    // 2️⃣ 若给定初值，则上传到 GPU
    if (x0 && x0->size() == n) {
        vex::copy(std::vector<GlobalReal>(x0->data(), x0->data() + n), *ctx.d_x);
        std::cout << "[AMGCL] Using user-provided warm start." << std::endl;
    } else if (x0) {
        std::cerr << "[AMGCL] Warning: warm start vector size mismatch, ignored."
                  << std::endl;
    } else {
        std::cout << "[AMGCL] Using existing device x as warm start." << std::endl;
    }

    // 3️⃣ 调用求解器
    size_t iters = 0;
    GlobalReal resid = 0.0;
    std::tie(iters, resid) = (*ctx.solver)(*ctx.d_rhs, *ctx.d_x);

    std::cout << "[AMGCL] Solver converged in " << iters
              << " iterations with residual " << resid << std::endl;

    // 4️⃣ 拷回结果
    std::vector<GlobalReal> x_host(n);
    vex::copy(*ctx.d_x, x_host);
    Eigen::VectorXr xvec(n);
    for (int i = 0; i < n; ++i) {
        xvec[i] = static_cast<Real>(x_host[i]);
    }
    return xvec;
}




Eigen::VectorXr amgclSolve(const Eigen::SparseMatrixCr& matrix_col,
    const Eigen::VectorXr& erhs, const Real relativeTol, const Real absTol, const int iter,
    const Real eps_strong, const int block_size, const Real add_diagonal_epsilon ,
    const std::vector<GlobalReal> null_spaces)
{
    // typedef amgcl::backend::vexcl<Real> Backend;
    //     typedef amgcl::make_solver<
    //         amgcl::runtime::preconditioner<Backend>,
    //         amgcl::solver::cg<Backend>>
    //         Solver;
    //     Eigen::SparseMatrixR matrix(matrix_col);
    //     // {
    //     //     if (add_diagonal_epsilon) {
    //     //         for (int i = 0; i < matrix.rows(); ++i)
    //     //             matrix.coeffRef(i, i) += add_diagonal_epsilon;
    //     //     }
    //     // }
    //     CSR64 csr;
    //     eigen_to_csr64(csr,matrix);
    //     std::shared_ptr<amgcl::backend::crs<Real, int,ptrdiff_t>> A_holder;
    //
    //     // ---------- 2) 重建矩阵 adapter ----------
    //     A_holder = amgcl::adapter::zero_copy_direct(
    //         csr.nrows,
    //         csr.ncols,
    //         csr.row_ptr.data(),
    //         csr.col_idx.data(),
    //         csr.values.data()
    //     );
    //
    //     typename Backend::params bprm;
    //
    //     vex::Context ctx(vex::Filter::Env);
    //     bprm.q = ctx;
    //
    //     std::vector<Real> rhs(erhs.data(), erhs.data() + erhs.rows() * erhs.cols());
    //     std::vector<Real> x(matrix.rows());
    //
    //
    //     {
    //         boost::property_tree::ptree prm;
    //         prm.put("solver.tol", relativeTol);
    //         prm.put("solver.maxiter", iter);
    //         prm.put("precond.class", "amg");
    //         prm.put("precond.relax.type", "chebyshev");
    //         prm.put("precond.relax.degree", 5);
    //         prm.put("precond.relax.power_iters", 100);
    //         prm.put("precond.relax.higher", 2.0f);
    //         prm.put("precond.relax.lower", 1.0f / 120.0f);
    //         prm.put("precond.relax.scale", true);
    //         prm.put("precond.max_levels", 6);
    //         prm.put("precond.direct_coarse", false);
    //         prm.put("precond.coarsening.type", "smoothed_aggregation");
    //         prm.put("precond.coarsening.estimate_spectral_radius", true);
    //         prm.put("precond.coarsening.relax", 1.0f);
    //         prm.put("precond.coarsening.power_iters", 100);
    //         if (!null_spaces.empty()) {
    //             prm.put("precond.coarsening.nullspace.cols", null_spaces.size() / matrix.rows());
    //             prm.put("precond.coarsening.nullspace.rows", matrix.rows());
    //             prm.put("precond.coarsening.nullspace.B", &null_spaces[0]);
    //         }
    //
    //         prm.put("precond.ncycle", 2);
    //         {
    //             Solver solve(A_holder, prm);
    //
    //             auto f_b = Backend::copy_vector(rhs, bprm);
    //             auto x_b = Backend::copy_vector(x, bprm);
    //
    //             size_t iters;
    //             Real resid;
    //
    //             {
    //                 std::tie(iters, resid) = solve(*f_b, *x_b);
    //             }
    //             ;
    //
    //             Eigen::VectorXr xvec = Eigen::VectorXr::Zero(csr.nrows);
    //             vex::copy(*x_b, x);
    //             for (uint i = 0; i < x.size(); i++)
    //                 xvec[i] = x[i];
    //             return xvec;
    //         }
    //     }


}


//
// Eigen::VectorXr amgclSolveCPU(const Eigen::SparseMatrixCr& matrix_col,
//     const Eigen::VectorXr& erhs, const Real relativeTol, const Real absTol, const int iter,
//     const Real eps_strong, const int block_size, const Real add_diagonal_epsilon ,
//     const std::vector<Real> null_spaces) {
//     FUNC_TIMER();
//     using Backend = amgcl::backend::builtin<Real>;
//     using AmgclSolver = amgcl::make_solver<
//         amgcl::runtime::preconditioner<Backend>,
//         amgcl::solver::cg<Backend>>;
//
//
//     std::vector<Real>rhs(erhs.rows(),0.0);
//     for (int i = 0; i < erhs.rows(); ++i) rhs[i] = erhs[i];
//
//     Eigen::SparseMatrixR matrix(matrix_col);
//     if (add_diagonal_epsilon) {
//         for (int i = 0; i < matrix.rows(); ++i)
//             matrix.coeffRef(i, i) += add_diagonal_epsilon;
//         // TStack diag_vec = TStack::Ones(matrix.rows());
//         // diag_vec *= add_diagonal_epsilon;
//         // matrix += diag_vec.asDiagonal();
//     }
//     matrix.makeCompressed();
//
//     // Eigen::VectorXr x = Eigen::VectorXr::Zero(matrix.rows());
//     std::vector<Real> x(matrix.rows(), 0.0);
//
//     boost::property_tree::ptree prm;
//     prm.put("solver.tol", relativeTol);
//     prm.put("solver.maxiter", iter);
//     prm.put("precond.class", "amg");
//     prm.put("precond.relax.type", "chebyshev");
//     prm.put("precond.relax.degree", 16);
//     prm.put("precond.relax.power_iters", 100);
//     prm.put("precond.relax.higher", 2.0f);
//     prm.put("precond.relax.lower", 1.0f / 120.0f);
//     prm.put("precond.relax.scale", true);
//     prm.put("precond.max_levels", 6);
//     prm.put("precond.direct_coarse", false);
//     prm.put("precond.coarsening.type", "smoothed_aggregation");
//     prm.put("precond.coarsening.estimate_spectral_radius", true);
//     prm.put("precond.coarsening.relax", 1.0f);
//     prm.put("precond.coarsening.power_iters", 100);
//     if (!null_spaces.empty()) {
//         prm.put("precond.coarsening.nullspace.cols", null_spaces.size() / matrix.rows());
//         prm.put("precond.coarsening.nullspace.rows", matrix.rows());
//         prm.put("precond.coarsening.nullspace.B", &null_spaces[0]);
//     }
//     prm.put("precond.ncycle", 2);
//
//     AmgclSolver amgcl_solver(matrix, prm);
//
//     std::cout << amgcl_solver.precond() << std::endl;
//     size_t iters;
//     Real resid;
//
//     std::tie(iters, resid) = amgcl_solver(rhs, x);
//     // // ---- 收敛性检测 ----
//     // if (resid <= relativeTol) {
//     //     std::cout << "[AMGCL] Solver converged in " << iters
//     //               << " iterations with residual " << resid << std::endl;
//     // } else {
//     //     std::cerr << "[AMGCL] Solver did NOT converge! "
//     //               << "Final residual: " << resid
//     //               << ", after " << iters << " iterations" << std::endl;
//     // }
//     Eigen::VectorXr xvec(matrix.rows());
//     for (int i = 0; i < matrix.rows(); ++i) xvec[i] = x[i];
//     return xvec;
// }
//
// void amgclInitCPU(AMGCLContextCPU &ctx,
//                   const Eigen::SparseMatrixCr &matrix_col,
//                   Real relativeTol, int iter,
//                   Real add_diagonal_epsilon,
//                   const std::vector<Real> &null_spaces)
// {
//     using Backend = AMGCLContextCPU::Backend;
//     using Precond = AMGCLContextCPU::Precond;
//     using SolverIter = AMGCLContextCPU::SolverIter;
//
//     Eigen::SparseMatrixR A(matrix_col);
//     if (add_diagonal_epsilon != 0.0) {
//         for (int i = 0; i < A.rows(); ++i)
//             A.coeffRef(i, i) += add_diagonal_epsilon;
//     }
//
//     boost::property_tree::ptree prm;
//     prm.put("solver.tol", relativeTol);
//     prm.put("solver.maxiter", iter);
//     prm.put("precond.class", "amg");
//     prm.put("precond.relax.type", "chebyshev");
//     prm.put("precond.relax.degree", 16);
//     prm.put("precond.relax.power_iters", 100);
//     prm.put("precond.relax.higher", 2.0f);
//     prm.put("precond.relax.lower", 1.0f / 120.0f);
//     prm.put("precond.relax.scale", true);
//     prm.put("precond.max_levels", 6);
//     prm.put("precond.direct_coarse", false);
//     prm.put("precond.coarsening.type", "smoothed_aggregation");
//     prm.put("precond.coarsening.estimate_spectral_radius", true);
//     prm.put("precond.coarsening.relax", 1.0f);
//     prm.put("precond.coarsening.power_iters", 100);
//     if (!null_spaces.empty()) {
//         prm.put("precond.coarsening.nullspace.cols", null_spaces.size() / A.rows());
//         prm.put("precond.coarsening.nullspace.rows", A.rows());
//         prm.put("precond.coarsening.nullspace.B", &null_spaces[0]);
//     }
//     prm.put("precond.ncycle", 2);
//
//     ctx.nrows = A.rows();
//
//     // 构建预条件器（近似逆）
//     ctx.precond = std::make_shared<Precond>(A, prm.get_child("precond"));
//
//     // 构建迭代器
//     ctx.solver = std::make_shared<SolverIter>(ctx.nrows, prm.get_child("solver"));
//
//     // 分配 host 向量
//     ctx.h_rhs.resize(ctx.nrows, 0.0);
//     ctx.h_x.resize(ctx.nrows, 0.0);
// }
// Eigen::VectorXr amgclSolveCPU(const AMGCLContextCPU &ctx,
//                               const Eigen::VectorXr &erhs)
// {
//     FUNC_TIMER();
//     const int n = ctx.nrows;
//
//     // 拷贝右端项到 std::vector
//     std::vector<Real> rhs(erhs.data(), erhs.data() + n);
//     std::vector<Real> x(n, 0.0);
//
//     size_t iters = 0;
//     Real resid = 0.0;
//     std::tie(iters, resid) = (*ctx.solver)(*ctx.precond, rhs, x);
//
//     std::cout << "[CPU AMGCL] Iterations = " << iters
//               << ", Residual = " << resid << std::endl;
//
//     Eigen::VectorXr xvec(n);
//     for (int i = 0; i < n; ++i) xvec[i] = x[i];
//     return xvec;
// }


// Eigen::VectorXr amgclSolve(const AMGCLContext &ctx, const Eigen::VectorXr &erhs)
// {
//     FUNC_TIMER();
//     using Backend = AMGCLContext::Backend;
//     const int n = ctx.nrows;
//
//     // === 仅更新 rhs ===
//     vex::copy(
//     std::vector<Real>(erhs.data(), erhs.data() + n),
//     *ctx.d_rhs);
//
//     size_t iters = 0;
//     Real resid = 0.0;
//     std::tie(iters, resid) = (*ctx.solver)(*ctx.d_rhs, *ctx.d_x);
//
//     // ---- 收敛性检测 ----
//     std::cout << "[AMGCL] Solver converged in " << iters
//               << " iterations with residual " << resid << std::endl;
//
//     Eigen::VectorXr xvec(n);
//     vex::copy(*ctx.d_x, xvec.data());  // 若后续CPU用，否则可省略
//     return xvec;
// }
