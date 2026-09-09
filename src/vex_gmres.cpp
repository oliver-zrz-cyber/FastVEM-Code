#include "vex_gmres.h"

void VEX_GMRES::build(const Eigen::SparseMatrixR& A,
                      const std::shared_ptr<Matrix>& shared_gpu_A,
                      size_t max_iter, Real rel_tol, unsigned restart, bool verbose) {
    if (A.rows() != A.cols() || A.rows() == 0)
        throw std::invalid_argument("GMRES requires a nonempty square matrix");
    if (!(rel_tol > 0) || restart == 0)
        throw std::invalid_argument("GMRES tolerance and restart must be positive");
    if (bprm_.q.empty()) {
        vex::Context context(vex::Filter::Env);
        if (context.empty()) throw std::runtime_error("No OpenCL device for GMRES");
        bprm_.q = context;
    }

    clear();
    n_ = A.rows();
    prm.maxiter = max_iter;
    prm.tol = rel_tol;
    prm.restart = restart;
    prm.verbose = verbose;
    bprm_.fast_matrix_setup = false;
    gpu_A_ = shared_gpu_A ? shared_gpu_A : eigentovex(A, bprm_);

    Solver::params solver_params;
    solver_params.maxiter = prm.maxiter;
    solver_params.tol = prm.tol;
    solver_params.M = prm.restart;
    solver_params.pside = amgcl::preconditioner::side::right;
    solver_params.verbose = false; // Wrapper controls the summary output.
    solver_ = std::make_unique<Solver>(n_, solver_params, bprm_);
    d_rhs_ = std::make_shared<Backend::vector>(bprm_.q, n_);
    d_x_ = std::make_shared<Backend::vector>(bprm_.q, n_);
    host_rhs_.resize(n_);
    host_x_.resize(n_);
}

void VEX_GMRES::clear() {
    reset_device_initial_guess();
    d_rhs_.reset();
    d_x_.reset();
    solver_.reset();
    gpu_A_.reset();
    host_rhs_.clear();
    host_x_.clear();
    n_ = 0;
}
