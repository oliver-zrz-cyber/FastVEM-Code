#pragma once

#include <amgcl/solver/gmres.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "vex_pcg.h"

// GPU GMRES wrapper for a matrix with an externally supplied preconditioner.
// The preconditioner must provide: apply(const VEX_VECTOR&, VEX_VECTOR&) const.
class VEX_GMRES {
public:
    using Backend = VEX_BACKEND;
    using Matrix = VEX_MATRIX;
    using value_type = Backend::value_type;
    using Solver = amgcl::solver::gmres<Backend>;

    struct params {
        size_t maxiter = 1000;
        value_type tol = 1e-3;
        unsigned restart = 30;
        bool verbose = false;
    };

    enum class InitialGuess { Host, Device };

    void build(const Eigen::SparseMatrixR& A, const std::shared_ptr<Matrix>& shared_gpu_A,
               size_t max_iter, Real rel_tol, unsigned restart, bool verbose);
    void clear();
    void reset_device_initial_guess() const { device_guess_valid_ = false; }

    template <class Precond>
    std::tuple<size_t, value_type> solve(const Precond& precond, const Eigen::VectorXr& rhs,
                                         Eigen::VectorXr& x,
                                         InitialGuess initial_guess = InitialGuess::Host) const {
        if (!solver_ || !gpu_A_ || rhs.size() != n_ || x.size() != n_)
            throw std::invalid_argument("GMRES not built or vector size mismatch");
        if (!(prm.tol > 0) || prm.restart == 0)
            throw std::invalid_argument("GMRES tolerance and restart must be positive");

        std::copy(rhs.data(), rhs.data() + n_, host_rhs_.begin());
        vex::copy(host_rhs_, *d_rhs_);
        if (initial_guess == InitialGuess::Host || !device_guess_valid_) {
            std::copy(x.data(), x.data() + n_, host_x_.begin());
            vex::copy(host_x_, *d_x_);
        }
        device_guess_valid_ = false;

        const auto result = (*solver_)(*gpu_A_, precond, *d_rhs_, *d_x_);
        vex::copy(*d_x_, host_x_);
        for (int i = 0; i < n_; ++i) x[i] = static_cast<Real>(host_x_[i]);

        const auto [iterations, relative_residual] = result;
        device_guess_valid_ = std::isfinite(relative_residual) && relative_residual <= prm.tol;
        if (prm.verbose)
            std::cout << "[GMRES] " << iterations << " iterations, relative residual "
                      << relative_residual << '\n';
        return result;
    }

    typename Backend::params bprm_;
    params prm;

private:
    int n_ = 0;
    std::shared_ptr<Matrix> gpu_A_;
    std::unique_ptr<Solver> solver_;
    std::shared_ptr<Backend::vector> d_rhs_, d_x_;
    mutable bool device_guess_valid_ = false;
    mutable std::vector<value_type> host_rhs_, host_x_;
};
