#include <stdexcept>
#include"amgcl_smoother.h"
// #include <vexcl/vexcl.hpp>
#include "dfn.h"
//
// void AMGCLIL0Smoother_CPU::build(const SpMat &A) {
//     {
//         const int n = A.rows();
//         n_ = n;
//         std::vector<int>    ptr(n + 1, 0);
//         std::vector<int>    col;
//         std::vector<Real> val;
//         col.reserve(A.nonZeros());
//         val.reserve(A.nonZeros());
//         int nnz = 0;
//         for (int i = 0; i < n; ++i) {
//             ptr[i] = nnz;
//             for (SpMat::InnerIterator it(A, i); it; ++it) {
//                 col.push_back(it.col());
//                 val.push_back(it.value());
//                 ++nnz;
//             }
//         }
//         ptr[n] = nnz;
//         A_host_ = std::make_shared<CRS>(n, n, ptr, col, val);
//         typename Relax::params rprm;
//         rprm.damping = 0.1;
//         relax_ = std::make_shared<Relax>(*A_host_, rprm,bprm_);
//     }
// }
//
// void AMGCLIL0Smoother_CPU::smooth(const Eigen::VectorXr &b, Eigen::VectorXr &x, int iterations) {
//     if (!relax_) {
//         std::cerr << "smoother not built.\n";
//         return;
//     }
//     const int n = n_;
//     if (x.size() != n) {
//         exit(-1);
//         x = Eigen::VectorXr::Zero(n);
//     }
//     std::vector<Real> val(n), rhs(n),tmp(n,0.0);
//     for (int i = 0; i < n; ++i) {
//         rhs[i] = b[i];
//     }
//     for (int i = 0; i < n; ++i) {
//         val[i] = x[i];
//     }
//     for (int it = 0; it < iterations; ++it) {
//         relax_->RZ_apply(*A_host_,rhs, val,tmp);
//     }
//     for (int i = 0; i < n; ++i) {
//         x[i] = val[i];
//     }
// }
//
// void AMGCLGSSmoother_CPU::build(const SpMat &A) {
//     const int n = A.rows();
//     n_ = n;
//     std::vector<int>    ptr(n + 1, 0);
//     std::vector<int>    col;
//     std::vector<Real> val;
//     col.reserve(A.nonZeros());
//     val.reserve(A.nonZeros());
//     int nnz = 0;
//     for (int i = 0; i < n; ++i) {
//         ptr[i] = nnz;
//         for (SpMat::InnerIterator it(A, i); it; ++it) {
//             col.push_back(it.col());
//             val.push_back(it.value());
//             ++nnz;
//         }
//     }
//     ptr[n] = nnz;
//     A_host_ = std::make_shared<CRS>(n, n, ptr, col, val);
//     typename Relax::params rprm;
//     rprm.serial = false;
//     relax_ = std::make_shared<Relax>(*A_host_, rprm,bprm_);
// }
//
// void AMGCLGSSmoother_CPU::smooth(const Eigen::VectorXr &b, Eigen::VectorXr &x, int iterations) {
//     {
//         if (!relax_) {
//             std::cerr << "smoother not built.\n";
//             return;
//         }
//         const int n = n_;
//         if (x.size() != n) {
//             exit(-1);
//             x = Eigen::VectorXr::Zero(n);
//         }
//
//         std::vector<Real> val(n), rhs(n);
//         for (int i = 0; i < n; ++i) {
//             rhs[i] = b[i];
//         }
//         for (int i = 0; i < n; ++i) {
//             val[i] = x[i];
//         }
//         for (int it = 0; it < iterations; ++it) {
//             relax_->RZ_apply(*A_host_,rhs, val);
//         }
//         for (int i = 0; i < n; ++i) {
//             x[i] = val[i];
//         }
//     }
// }

void AMGCLChebyshevSmoother_CPU::build(const SpMat &A, int degree, int power_iter, Real higher, Real lower,
bool scale) {
    const int n = A.rows();
    n_ = n;
    std::vector<int>    ptr(n + 1, 0);
    std::vector<int>    col;
    std::vector<Backend::value_type> val;
    col.reserve(A.nonZeros());
    val.reserve(A.nonZeros());
    int nnz = 0;
    for (int i = 0; i < n; ++i) {
        ptr[i] = nnz;
        for (SpMat::InnerIterator it(A, i); it; ++it) {
            col.push_back(it.col());
            val.push_back(it.value());
            ++nnz;
        }
    }
    ptr[n] = nnz;
    A_host_ = std::make_shared<CRS>(n, n, ptr, col, val);
    typename Relax::params rprm;
    rprm.degree = degree;
    rprm.power_iters = power_iter;
    rprm.higher = higher;
    rprm.lower = lower;
    rprm.scale = scale;
    relax_ = std::make_shared<Relax>(*A_host_, rprm,bprm_);
}

void AMGCLChebyshevSmoother_CPU::smooth(const Eigen::VectorXr &b, Eigen::VectorXr &x, int iterations) {
    if (!relax_) {
        std::cerr << "smoother not built.\n";
        return;
    }
    const int n = n_;
    if (x.size() != n) {
        exit(-1);
        x = Eigen::VectorXr::Zero(n);
    }

    std::vector<Backend::value_type> val(n), rhs(n);
    for (int i = 0; i < n; ++i) {
        rhs[i] = b[i];
    }
    for (int i = 0; i < n; ++i) {
        val[i] = x[i];
    }
    for (int it = 0; it < iterations; ++it) {
        relax_->RZ_apply(*A_host_,rhs, val);
    }
    for (int i = 0; i < n; ++i) {
        x[i] = val[i];
    }
}

void AMGCLChebyshevSmoother_GPU::build(const SpMat &A, int degree, int power_iter, Real higher, Real lower,
bool scale) {
    build(A, nullptr, degree, power_iter, higher, lower, scale);
}

void AMGCLChebyshevSmoother_GPU::build(const SpMat &A, const std::shared_ptr<Matrix> &shared_gpu_A,
int degree, int power_iter, Real higher, Real lower, bool scale) {
    clear();
    const int n = A.rows();
    n_ = n;
    if (bprm_.q.empty()) {
        RZ_INFO("new contex for chev smoother");
        vex::Context ctx(vex::Filter::Env);
        if (ctx.empty())exit(-1);
        bprm_.q = ctx;
    }
    if (!A.isCompressed()) {
        RZ_FATAL("AMGCLChebyshevSmoother_GPU::build requires a compressed RowMajor matrix.");
    }

    A_host_ = amgcl::adapter::zero_copy_direct(
        static_cast<size_t>(A.rows()),
        static_cast<size_t>(A.cols()),
        A.outerIndexPtr(),
        A.innerIndexPtr(),
        A.valuePtr()
    );

    bprm_.fast_matrix_setup = false;
    gpu_A = shared_gpu_A ? shared_gpu_A : Backend::copy_matrix(A_host_,bprm_);
    typename Relax::params rprm;
    rprm.degree = degree;
    rprm.power_iters = power_iter;
    rprm.higher = higher;
    rprm.lower = lower;
    rprm.scale = scale;

    relax_ = std::make_shared<Relax>(*A_host_, rprm, bprm_);
}

void AMGCLChebyshevSmoother_GPU::smooth(const Vec &b, Vec &x, int iterations) const {
    if (!relax_) {
        std::cerr << "[AMGCLJacobi] smoother not built.\n";
        return;
    }

    const int n = n_;
    if (x.size() != n) {
        x = Vec::Zero(n);
    }

    if (b.size() != n) throw std::invalid_argument("Chebyshev RHS size mismatch");
    if (!host_rhs_device_) {
        host_rhs_device_ = std::make_shared<Backend::vector>(bprm_.q, n);
        host_x_device_ = std::make_shared<Backend::vector>(bprm_.q, n);
        host_rhs_.resize(n);
        host_x_.resize(n);
    }
    auto &d_rhs = *host_rhs_device_;
    auto &d_x = *host_x_device_;
    std::copy(b.data(), b.data() + n, host_rhs_.begin());
    std::copy(x.data(), x.data() + n, host_x_.begin());
    vex::copy(host_rhs_, d_rhs);
    vex::copy(host_x_, d_x);

    for (int it = 0; it < iterations; ++it) {
        relax_->RZ_apply(*gpu_A, d_rhs, d_x);
    }
    // 拷回结果
    vex::copy(d_x, host_x_);
    for (int i = 0; i < n; ++i) x[i] = static_cast<Real>(host_x_[i]);
}

void AMGCLChebyshevSmoother_GPU::smooth(const Backend::vector &d_rhs, Backend::vector &d_x, int iterations) const  {
    if (!relax_) {
        std::cerr << "[AMGCLJacobi] smoother not built.\n";
        return;
    }

    for (int it = 0; it < iterations; ++it) {
        relax_->RZ_apply(*gpu_A, d_rhs, d_x);
    }

}

void AMGCLChebyshevSmoother_GPU::clear() {
    host_rhs_device_.reset();
    host_x_device_.reset();
    host_rhs_.clear();
    host_x_.clear();
    relax_.reset();

    gpu_A.reset();
    A_host_.reset();

    n_ = 0;
}
