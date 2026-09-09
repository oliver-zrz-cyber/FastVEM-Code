#pragma once
#include <stdexcept>
#include <amgcl/backend/vexcl.hpp>
#include <vexcl/vexcl.hpp>
#include <amgcl/backend/interface.hpp>

#include "dfn.h"
#include "Eigen/Sparse"
typedef amgcl::backend::vexcl<GlobalReal, int, int> VEX_BACKEND;
typedef VEX_BACKEND::vector VEX_VECTOR;
typedef VEX_BACKEND::matrix VEX_MATRIX;
std::shared_ptr<VEX_MATRIX> eigentovex(const Eigen::SparseMatrixR& A,
    const typename VEX_BACKEND::params &bprm_) ;




std::shared_ptr<typename amgcl::backend::builtin<VEX_BACKEND::value_type, VEX_BACKEND::col_type,
VEX_BACKEND::ptr_type>::matrix> eigentobuildin(const Eigen::SparseMatrixR& A,
    const typename VEX_BACKEND::params &bprm_) ;


class VEX_PCG {
    public:
    using SpMat = Eigen::SparseMatrixR;
    using Vec   = Eigen::VectorXr;
    using Backend   = VEX_BACKEND;
    using value_type = Backend::value_type;
    using col_type  = Backend::col_type;
    using ptr_type = Backend::ptr_type;
    using CRS       = amgcl::backend::crs<value_type,col_type,ptr_type>;
    using Matrix       = Backend::matrix;

    struct params {
        /// Maximum number of iterations.
        size_t maxiter;

        /// Target relative residual error.
        value_type tol;


        /// Verbose output (show iterations and error)
        bool verbose;
        bool diagnostic_checks = false;

        params()
            : maxiter(1000), tol(1e-3),verbose(true)
        {}
    };

    enum class InitialGuess { Host, Device };
    // Call after changing pressure on the CPU or switching simulation state.
    void reset_device_initial_guess() const { device_guess_valid_ = false; }

    template <class Precond>
    std::tuple<size_t, value_type> solve(
                const Precond &P, const Eigen::VectorXr& rhs, Eigen::VectorXr& x,
                InitialGuess initial_guess = InitialGuess::Host) const;

    void build(const Eigen::SparseMatrixR& A,size_t max_iter
        ,Real rel_tol,bool verbose) ;
    void build(const Eigen::SparseMatrixR& A,const std::shared_ptr<Matrix>& shared_gpu_A,
        size_t max_iter,Real rel_tol,bool verbose) ;
    void clear() ;

    int nrows_ = 0;
    int ncols_ = 0;
    int nnz_   = 0;
    int n_ = 0;
    std::shared_ptr<typename Backend::vector> d_r;
    std::shared_ptr<typename Backend::vector> d_s;
    std::shared_ptr<typename Backend::vector> d_p;
    std::shared_ptr<typename Backend::vector> d_q;
    std::shared_ptr<typename Backend::vector> d_rhs;
    std::shared_ptr<typename Backend::vector> d_x;
    std::shared_ptr<Matrix> gpu_A;
    typename Backend::params bprm_;
    params prm;
private:
    // Reused by sequential solves on this instance; solve is not reentrant.
    mutable bool device_guess_valid_ = false;
    mutable std::vector<value_type> host_rhs_, host_x_;
};

template<class Precond>
std::tuple<size_t, VEX_PCG::value_type> VEX_PCG::solve(
    const Precond &P, const Eigen::VectorXr &rhs, Eigen::VectorXr &x,
    InitialGuess initial_guess) const {
    FUNC_TIMER();
    if (!gpu_A || !d_x || rhs.size() != n_ || x.size() != n_)
        throw std::invalid_argument("PCG not built or vector size mismatch");
    if (!(prm.tol > 0) || !std::isfinite(prm.tol))
        throw std::invalid_argument("PCG tolerance must be finite and positive");

    std::copy(rhs.data(), rhs.data() + n_, host_rhs_.begin());
    vex::copy(host_rhs_, *d_rhs);
    if (initial_guess == InitialGuess::Host || !device_guess_valid_) {
        std::copy(x.data(), x.data() + n_, host_x_.begin());
        vex::copy(host_x_, *d_x);
    }
    // A failed solve must not become the next frame's cached initial guess.
    device_guess_valid_ = false;
    auto norm = [](const auto &v) -> value_type {
        const value_type squared = amgcl::backend::inner_product(v, v);
        if (!std::isfinite(squared) || squared < 0)
            throw std::runtime_error("PCG nonfinite vector norm");
        return std::sqrt(squared);
    };
    const value_type norm_rhs = norm(*d_rhs);
    const value_type threshold = prm.tol * norm_rhs;
    if (norm_rhs == 0) amgcl::backend::clear(*d_x);
    amgcl::backend::residual(*d_rhs, *gpu_A, *d_x, *d_r);
    value_type res_norm = norm(*d_r);
    value_type previous_rho = 0;
    bool restart = true;
    size_t iter = 0;
    while (iter < prm.maxiter && res_norm > threshold) {
        if (prm.diagnostic_checks) (void)norm(*d_r);
        P.apply(*d_r, *d_s, n_);
        if (prm.diagnostic_checks) (void)norm(*d_s);
        const value_type rho = amgcl::backend::inner_product(*d_r, *d_s);
        if (!std::isfinite(rho) || rho <= 0)
            throw std::runtime_error("PCG breakdown: nonpositive r dot z");
        if (restart) amgcl::backend::copy(*d_s, *d_p);
        else amgcl::backend::axpby(1.0, *d_s, rho / previous_rho, *d_p);
        restart = false;
        previous_rho = rho;
        amgcl::backend::spmv(1.0, *gpu_A, *d_p, 0.0, *d_q);
        const value_type denominator = amgcl::backend::inner_product(*d_p, *d_q);
        if (!std::isfinite(denominator) || denominator <= 0)
            throw std::runtime_error("PCG breakdown: nonpositive p dot A p");
        const value_type alpha = rho / denominator;
        if (!std::isfinite(alpha)) throw std::runtime_error("PCG nonfinite step");
        amgcl::backend::axpby(alpha, *d_p, 1.0, *d_x);
        amgcl::backend::axpby(-alpha, *d_q, 1.0, *d_r);
        ++iter;
        res_norm = norm(*d_r);
        // Verify apparent convergence against the actual system. If recurrence
        // drift is significant, restart PCG from the true residual.
        if (res_norm <= threshold || iter == prm.maxiter) {
            amgcl::backend::residual(*d_rhs, *gpu_A, *d_x, *d_r);
            res_norm = norm(*d_r);
            restart = true;
        }
        if (prm.verbose && iter % 5 == 0)
            std::cout << iter << "\t" << res_norm / norm_rhs << std::endl;
    }
    const value_type relative_residual = norm_rhs == 0 ? res_norm : res_norm / norm_rhs;
    if (prm.verbose)
        std::cout << iter << "\t" << relative_residual << std::endl;
    if (res_norm > threshold)
        std::cerr << "[PCG] Iteration limit reached; true relative residual = "
                  << relative_residual << std::endl;

    // CPU pressure application still needs the result. Device mode avoids
    // uploading this same pressure again on the next frame.
    vex::copy(*d_x, host_x_);
    for (int i = 0; i < n_; ++i) x[i] = static_cast<Real>(host_x_[i]);
    device_guess_valid_ = res_norm <= threshold;
    return std::make_tuple(iter, relative_residual);
}
