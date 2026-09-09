#include"vex_pcg.h"
#include "dfn.h"
#include <amgcl/adapter/zero_copy.hpp>

std::shared_ptr<VEX_MATRIX> eigentovex(const Eigen::SparseMatrixR &A,
const VEX_BACKEND::params &bprm_){
    using SpMat = Eigen::SparseMatrixR;
    using Vec   = Eigen::VectorXr;
    using Backend   = VEX_BACKEND;
    using value_type = Backend::value_type;
    using col_type  = Backend::col_type;
    using ptr_type = Backend::ptr_type;
    using CRS       = amgcl::backend::crs<value_type,col_type,ptr_type>;
    using Matrix       = Backend::matrix;
    const ptr_type n = static_cast<ptr_type>(A.rows());
    auto copy_prm = bprm_;
    copy_prm.fast_matrix_setup = false;

    if (A.isCompressed()) {
        auto A_host_ = amgcl::adapter::zero_copy_direct(
            static_cast<size_t>(A.rows()),
            static_cast<size_t>(A.cols()),
            A.outerIndexPtr(),
            A.innerIndexPtr(),
            A.valuePtr()
        );
        RZ_ASSERT(A_host_->nrows == A.rows()&&A_host_->ncols == A.cols(),"");
        return Backend::copy_matrix(A_host_, copy_prm);
    }

    // Fallback for uncompressed Eigen matrices.
    std::vector<ptr_type> ptr(n + 1, 0);
    std::vector<col_type> col;
    std::vector<value_type> val;
    col.reserve(A.nonZeros());
    val.reserve(A.nonZeros());

    ptr_type nnz = 0;
    for (ptr_type i = 0; i < n; ++i) {
        ptr[i] = nnz;
        for (SpMat::InnerIterator it(A, i); it; ++it) {
            col.push_back(it.col());
            val.push_back(it.value());
            ++nnz;
        }
    }
    ptr[n] = nnz;
    auto A_host_ =
                std::make_shared<typename amgcl::backend::builtin<value_type, col_type, ptr_type>::matrix>(A.rows(),A.cols(),ptr,col,val);
    amgcl::backend::sort_rows(*A_host_);
    RZ_ASSERT(A_host_->nrows == A.rows()&&A_host_->ncols == A.cols(),"");
    return Backend::copy_matrix(A_host_,copy_prm);
}


std::shared_ptr<amgcl::backend::builtin<VEX_BACKEND::value_type, VEX_BACKEND
::col_type, VEX_BACKEND::ptr_type>::matrix> eigentobuildin(
    const Eigen::SparseMatrixR &A, const VEX_BACKEND::params &bprm_) {
    {
        using SpMat = Eigen::SparseMatrixR;
        using Vec   = Eigen::VectorXr;
        using Backend   = VEX_BACKEND;
        using value_type = Backend::value_type;
        using col_type  = Backend::col_type;
        using ptr_type = Backend::ptr_type;
        using CRS       = amgcl::backend::crs<value_type,col_type,ptr_type>;
        using Matrix       = Backend::matrix;
        const ptr_type n = static_cast<ptr_type>(A.rows());

        // Eigen -> CRS
        std::vector<ptr_type> ptr(n + 1, 0);
        std::vector<col_type> col;
        std::vector<value_type> val;
        col.reserve(A.nonZeros());
        val.reserve(A.nonZeros());

        ptr_type nnz = 0;
        for (ptr_type i = 0; i < n; ++i) {
            ptr[i] = nnz;
            for (SpMat::InnerIterator it(A, i); it; ++it) {
                col.push_back(it.col());
                val.push_back(it.value());
                ++nnz;
            }
        }
        ptr[n] = nnz;
        auto A_host_ =
                    std::make_shared<typename amgcl::backend::builtin<value_type, col_type, ptr_type>::matrix>(A.rows(),A.cols(),ptr,col,val);
        amgcl::backend::sort_rows(*A_host_);
        return std::move(A_host_);
    }
}

void VEX_PCG::build(const Eigen::SparseMatrixR &A, size_t max_iter, Real rel_tol,
bool verbose) {
    build(A, nullptr, max_iter, rel_tol, verbose);
}

void VEX_PCG::build(const Eigen::SparseMatrixR &A, const std::shared_ptr<Matrix> &shared_gpu_A,
size_t max_iter, Real rel_tol, bool verbose) {
    FUNC_TIMER();
    FUNC_MEMORY_TIMER();
    prm.maxiter = max_iter;
    prm.verbose = verbose;
    prm.tol = rel_tol;
    reset_device_initial_guess();
    host_rhs_.resize(A.rows());
    host_x_.resize(A.rows());
    // 基本信息
    n_ = A.rows();
    nrows_ = A.rows();
    ncols_ = A.cols();
    nnz_   = A.nonZeros();

    //分配 GPU 内存
    // vex::Context ctx(vex::Filter::Env);
    // if (ctx.empty())exit(-1);
    // bprm_.q = ctx;
    if (bprm_.q.empty()) {
        vex::Context ctx(vex::Filter::Env);
        if (ctx.empty()) exit(-1);
        bprm_.q = ctx;
    }
    bprm_.fast_matrix_setup = false;

    // GPU 向量分配
    d_rhs = std::make_shared<typename Backend::vector>(bprm_.q, n_);
    d_x = std::make_shared<typename Backend::vector>(bprm_.q, n_);
    d_s = std::make_shared<typename Backend::vector>(bprm_.q, n_);
    d_r = std::make_shared<typename Backend::vector>(bprm_.q, n_);
    d_q = std::make_shared<typename Backend::vector>(bprm_.q, n_);
    d_p = std::make_shared<typename Backend::vector>(bprm_.q, n_);

    gpu_A = shared_gpu_A ? shared_gpu_A : eigentovex(A,bprm_);
}

void VEX_PCG::clear() {
    reset_device_initial_guess();
    host_rhs_.clear();
    host_x_.clear();
    // 只释放 GPU 数据，不动 context
    d_r.reset();
    d_s.reset();
    d_p.reset();
    d_q.reset();
    d_rhs.reset();
    d_x.reset();
    gpu_A.reset();

    // 保留 bprm_ 和 bprm_.q（GPU context）
    // bprm_ = typename Backend::params();   // ❌ 不要这样

    n_ = 0;
    nrows_ = 0;
    ncols_ = 0;
    nnz_ = 0;

    prm = params();
}
