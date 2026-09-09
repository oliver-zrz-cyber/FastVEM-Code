#pragma once
#include <memory>
#include <Eigen/Sparse>
#include <amgcl/backend/vexcl.hpp>
#include <amgcl/relaxation/chebyshev.hpp>

#include "config.h"

// #include <vector>
// #include <iostream>

// #include <amgcl/backend/cuda.hpp>
// #include <amgcl/backend/interface.hpp>

// #include <amgcl/relaxation/damped_jacobi.hpp>
// #include <amgcl/relaxation/gauss_seidel.hpp>
// #include <amgcl/adapter/eigen.hpp>
// #include <amgcl/relaxation/ilu0.hpp>
// #include "vem3d.h"
// #include "vex_pcg.h"

#include <amgcl/adapter/zero_copy.hpp>
// class AMGCLIL0Smoother_CPU {
// public:
//     using Backend   = amgcl::backend::builtin<Real>;
//     using Relax     = amgcl::relaxation::ilu0<Backend>;
//     using SpMat = Eigen::SparseMatrixR;
//     using CRS       = amgcl::backend::crs<Real>;
//     void build(const SpMat& A) ;
//     void smooth(const Eigen::VectorXr& b, Eigen::VectorXr& x, int iterations = 3);
//
// private:
//     int n_ = -1;
//     std::shared_ptr<CRS> A_host_;
//     std::shared_ptr<Relax> relax_;
//     typename Backend::params bprm_;
// };
//
// class AMGCLGSSmoother_CPU {
// public:
//     using Backend   = amgcl::backend::builtin<Real>;
//     using Relax     = amgcl::relaxation::gauss_seidel<Backend>;
//     using SpMat = Eigen::SparseMatrixR;
//     using CRS       = amgcl::backend::crs<Real>;
//     void build(const SpMat& A) ;
//     void smooth(const Eigen::VectorXr& b, Eigen::VectorXr& x, int iterations = 3) ;
//
// private:
//     int n_ = -1;
//     std::shared_ptr<CRS> A_host_;
//     std::shared_ptr<Relax> relax_;
//     typename Backend::params bprm_;
// };

class AMGCLChebyshevSmoother_CPU {
public:
    using Backend   = amgcl::backend::builtin<GlobalReal>;
    using Relax     = amgcl::relaxation::chebyshev<Backend>;
    using SpMat = Eigen::SparseMatrixR;
    using CRS       = amgcl::backend::crs<GlobalReal>;
    void build(const SpMat& A,int degree = 5,int power_iter = 100,
        Real higher=2.0f,Real lower=1.0/120.0,bool scale = true) ;
    void smooth(const Eigen::VectorXr& b, Eigen::VectorXr& x, int iterations = 3) ;

private:
    int n_ = -1;
    std::shared_ptr<CRS> A_host_;
    std::shared_ptr<Relax> relax_;
    typename Backend::params bprm_;
};

class AMGCLChebyshevSmoother_GPU {
public:
    using ValueType = GlobalReal;
    using ColumnType = int;
    using PointerType = int;
    using SpMat = Eigen::SparseMatrixR;
    using Vec   = Eigen::VectorXr;
    using Backend = amgcl::backend::vexcl<ValueType, ColumnType, PointerType>;
    // using Backend   = amgcl::backend::vexcl<Real>;
    using Relax     = amgcl::relaxation::chebyshev<Backend>;
    using Matrix       = Backend::matrix;
    // ---------- 构建：Eigen 矩阵转 CRS 并初始化 AMGCL 平滑器 ----------
    void build(const SpMat& A,int degree = 3,int power_iter = 200,
    Real higher=2.0f,Real lower=1.0/120.0,bool scale = true) ;
    void build(const SpMat& A,const std::shared_ptr<Matrix>& shared_gpu_A,
        int degree = 3,int power_iter = 200,
        Real higher=2.0f,Real lower=1.0/120.0,bool scale = true) ;

    // ---------- 平滑：接口与 DampedJacobiSmoother 相同 ----------
    void smooth(const Vec& b, Vec& x, int iterations = 3) const ;

    void smooth(const Backend::vector& d_rhs, Backend::vector& d_x, int iterations = 3) const;
    typename Backend::params bprm_;
private:
    void clear() ;
    // Only allocated when the Eigen-vector interface is used; not reentrant.
    mutable std::shared_ptr<Backend::vector> host_rhs_device_, host_x_device_;
    mutable std::vector<ValueType> host_rhs_, host_x_;
    int n_ = 0;
    std::shared_ptr<Matrix> gpu_A;
    // CSR64 csr;
    // std::shared_ptr<amgcl::backend::crs<ValueType, ColumnType,PointerType>> A_holder;
    std::shared_ptr< typename amgcl::backend::builtin<ValueType, ColumnType, PointerType>::matrix > A_host_;
    std::shared_ptr<Relax> relax_;
};



//
// class AMGCLSpai0Smoother_GPU {
// public:
//
//     using ValueType = Real;
//     using ColumnType = int;
//     using PointerType = int;
//     using SpMat = Eigen::SparseMatrixR;
//     using Vec   = Eigen::VectorXr;
//     using Backend = amgcl::backend::vexcl<ValueType,ColumnType,PointerType>;
//     // using Backend   = amgcl::backend::vexcl<Real>;
//     using Relax     = amgcl::relaxation::chebyshev<Backend>;
//     using Matrix       = Backend::matrix;
//     // ---------- 构建：Eigen 矩阵转 CRS 并初始化 AMGCL 平滑器 ----------
//     void build(const SpMat& A) {
//         const int n = A.rows();
//         n_ = n;
//         // Eigen -> CRS
//         std::vector<int>    ptr(n + 1, 0);
//         std::vector<int>    col;
//         std::vector<Real> val;
//         col.reserve(A.nonZeros());
//         val.reserve(A.nonZeros());
//
//         int nnz = 0;
//         for (int i = 0; i < n; ++i) {
//         ptr[i] = nnz;
//         for (SpMat::InnerIterator it(A, i); it; ++it) {
//         col.push_back(it.col());
//         val.push_back(it.value());
//         ++nnz;
//         }
//         }
//         ptr[n] = nnz;
//
//         if (bprm_.q.empty()) {
//             RZ_INFO("new contex for chev smoother");
//             vex::Context ctx(vex::Filter::Env);
//             if (ctx.empty())exit(-1);
//             bprm_.q = ctx;
//         }
//         A_host_ = std::make_shared<typename amgcl::backend::builtin<ValueType, ColumnType, PointerType>::matrix>(n,n,ptr,col,val);
//
//
//         gpu_A = Backend::copy_matrix(A_host_,bprm_);
//         bprm_.fast_matrix_setup = false;
//         typename Relax::params rprm;
//
//
//         relax_ = std::make_shared<Relax>(*A_host_, rprm, bprm_);
//         // GPU 向量分配
//         d_rhs_ = std::make_shared<typename Backend::vector>(bprm_.q, n);
//         d_x_   = std::make_shared<typename Backend::vector>(bprm_.q, n);
//         residual_temp = std::make_shared<typename Backend::vector>(bprm_.q, n);
//     }
//
//     // ---------- 平滑：接口与 DampedJacobiSmoother 相同 ----------
//     void smooth(const Vec& b, Vec& x, int iterations = 3) const {
//         if (!relax_) {
//             std::cerr << "[AMGCLJacobi] smoother not built.\n";
//             return;
//         }
//
//         const int n = n_;
//         if (x.size() != n) {
//             x = Vec::Zero(n);
//         }
//
//         // 上传到 GPU
//         vex::copy(std::vector<Real>(b.data(), b.data() + n), *d_rhs_);
//         vex::copy(std::vector<Real>(x.data(), x.data() + n), *d_x_);
//
//         for (int it = 0; it < iterations; ++it) {
//             relax_->apply_pre(*gpu_A, *d_rhs_, *d_x_,*residual_temp);
//         }
//         // 拷回结果
//         vex::copy(*d_x_, x.data());
//     }
//
//     void smooth(const Backend::vector& d_rhs, Backend::vector& d_x, int iterations = 3) const {
//         if (!relax_) {
//             std::cerr << "[AMGCLJacobi] smoother not built.\n";
//             return;
//         }
//
//         for (int it = 0; it < iterations; ++it) {
//             relax_->apply_pre(*gpu_A, *d_rhs_, *d_x_,*residual_temp);
//         }
//
//     }
//     typename Backend::params bprm_;
// private:
//     int n_ = 0;
//     std::shared_ptr<Matrix> gpu_A;
//     // CSR64 csr;
//     // std::shared_ptr<amgcl::backend::crs<ValueType, ColumnType,PointerType>> A_holder;
//     std::shared_ptr< typename amgcl::backend::builtin<ValueType, ColumnType, PointerType>::matrix > A_host_;
//     std::shared_ptr<Relax> relax_;
//     std::shared_ptr<typename Backend::vector> d_rhs_;
//     std::shared_ptr<typename Backend::vector> d_x_;
//     std::shared_ptr<typename Backend::vector> residual_temp;
// };
//
//
// class AMGCLJacobiSmoother {
// public:
//     using SpMat = Eigen::SparseMatrixR;
//     using Vec   = Eigen::VectorXr;
//     using Backend   = amgcl::backend::vexcl<Real>;
//     using Relax     = amgcl::relaxation::damped_jacobi<Backend>;
//     using value_type = Backend::value_type;
//     using col_type  = Backend::col_type;
//     using ptr_type = Backend::ptr_type;
//     using CRS       = amgcl::backend::crs<value_type,col_type,ptr_type>;
//     using Matrix       = Backend::matrix;
//     // ---------- 构建：Eigen 矩阵转 CRS 并初始化 AMGCL 平滑器 ----------
//     void build(const SpMat& A) {
//         const int n = A.rows();
//         n_ = n;
//
//         // Eigen -> CRS
//         std::vector<int>    ptr(n + 1, 0);
//         std::vector<int>    col;
//         std::vector<Real> val;
//         col.reserve(A.nonZeros());
//         val.reserve(A.nonZeros());
//
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
//
//
//         // vexcl 上下文
//         if (bprm_.q.empty()) {
//             vex::Context ctx(vex::Filter::Env);
//             if (ctx.empty())exit(-1);
//             bprm_.q = ctx;
//         }
//         A_host_ = std::make_shared<typename amgcl::backend::builtin<value_type, col_type, ptr_type>::matrix >(n, n, ptr, col, val);
//         gpu_A = Backend::copy_matrix(A_host_,bprm_);
//         // Jacobi 参数（damping 默认 0.7，可在 smooth 时传入）
//         typename Relax::params rprm;
//         rprm.damping = 1.0;  // 默认值，稍后 smooth() 可以改
//
//         relax_ = std::make_shared<Relax>(*A_host_, rprm, bprm_);
//
//         // GPU 向量分配
//         d_rhs_ = std::make_shared<typename Backend::vector>(bprm_.q, n);
//         d_x_   = std::make_shared<typename Backend::vector>(bprm_.q, n);
//         d_x_temp = std::make_shared<typename Backend::vector>(bprm_.q, n);
//         residual   = std::make_shared<typename Backend::vector>(bprm_.q, n);
//         residual_temp = std::make_shared<typename Backend::vector>(bprm_.q, n);
//     }
//
//     // ---------- 平滑：接口与 DampedJacobiSmoother 相同 ----------
//     void smooth(const Vec& b, Vec& x, int iterations = 3, Real omega = 0.666) const {
//         if (!relax_) {
//             std::cerr << "[AMGCLJacobi] smoother not built.\n";
//             return;
//         }
//
//         const int n = n_;
//         if (x.size() != n) {
//             x = Vec::Zero(n);
//         }
//
//         relax_->prm.damping = omega;
//
//         vex::copy(std::vector<Real>(b.data(), b.data() + n), *d_rhs_);
//         vex::copy(std::vector<Real>(x.data(), x.data() + n), *d_x_);
//
//         for (int it = 0; it < iterations; ++it) {
//             amgcl::backend::residual(*d_rhs_,*gpu_A, *d_x_, *residual);
//             relax_->RZ_apply(*d_x_,*residual);
//         }
//
//         // for (int it = 0; it < iterations; ++it) {
//         //     amgcl::backend::residual(*d_rhs_,*gpu_A, *d_x_, *residual);
//         //     Real min_res = std::numeric_limits<Real>::infinity();
//         //     Real best_om = 1.0f;
//         //     for (Real om = 0.1f;om<0.8f;om+=0.1f) {
//         //         relax_->prm.damping = om;
//         //         vex::copy(*d_x_,*d_x_temp);
//         //         relax_->RZ_apply(*gpu_A, *d_rhs_, *d_x_temp,*residual);
//         //         amgcl::backend::residual(*d_rhs_,*gpu_A, *d_x_temp, *residual_temp);
//         //         Eigen::VectorXr temp_rs;
//         //         temp_rs.resize(n);
//         //         vex::copy(*residual_temp,temp_rs.data());
//         //         Real sum = temp_rs.norm();
//         //         if (sum < min_res) {
//         //             min_res = sum;
//         //             best_om = om;
//         //         }
//         //     }
//         //     relax_->prm.damping =best_om;
//         //     relax_->RZ_apply(*gpu_A, *d_rhs_, *d_x_,*residual);
//         // }
//
//         vex::copy(*d_x_, x.data());
//     }
//     typename Backend::params bprm_;
// private:
//     int n_ = 0;
//     std::shared_ptr<Matrix> gpu_A;
//     std::shared_ptr< typename amgcl::backend::builtin<value_type, col_type, ptr_type>::matrix > A_host_;
//
//     std::shared_ptr<Relax> relax_;
//     std::shared_ptr<typename Backend::vector> d_rhs_;
//     std::shared_ptr<typename Backend::vector> d_x_temp;
//     std::shared_ptr<typename Backend::vector> d_x_;
//     std::shared_ptr<typename Backend::vector> residual;
//     std::shared_ptr<typename Backend::vector> residual_temp;
// };
//
//
//
//
// class AMGCLJacobiBlockSmoother {
// public:
//     using SpMat = Eigen::SparseMatrixR;
//     using Vec   = Eigen::VectorXr;
//     using Backend   = amgcl::backend::vexcl<Real>;
//     using Relax     = amgcl::relaxation::damped_jacobi<Backend>;
//     using value_type = Backend::value_type;
//     using col_type  = Backend::col_type;
//     using ptr_type = Backend::ptr_type;
//     using CRS       = amgcl::backend::crs<value_type,col_type,ptr_type>;
//     using Matrix       = Backend::matrix;
//     // ---------- 构建：Eigen 矩阵转 CRS 并初始化 AMGCL 平滑器 ----------
//     void build(const SpMat& A,int v_size,int e_size,int f_size,int c_size) {
//         const int n = A.rows();
//         A_ = A;
//         n_ = n;
//         const int e_begin = v_size;
//         const int f_begin = e_begin+e_size;
//         const int c_begin = f_begin+f_size;
//         v_num = v_size;
//         e_num = e_size;
//         f_num = f_size;
//         c_num = c_size;
//         ratio_c_f = Real(c_size)/Real(f_size);
//         ratio_f_e = Real(f_size)/Real(e_size);
//         ratio_e_v = Real(e_size)/Real(v_size);
//         // vexcl 上下文
//         vex::Context ctx(vex::Filter::Env);
//         bprm_.q = ctx;
//         typename Relax::params rprm;
//         rprm.damping = 0.666;
//         //----------------------------v---------------------------------
//         {
//             std::vector<int>    ptrv(n + 1, 0);
//             std::vector<int>    colv;
//             std::vector<Real> valv;
//             colv.reserve(A.nonZeros());
//             valv.reserve(A.nonZeros());
//             int nnz_v = 0;
//             for (int i = 0; i < n; ++i) {
//                 ptrv[i] = nnz_v;
//                 // if (i>=v_size) {
//                 //     continue;
//                 // }
//                 std::vector<int> v_link_cs;
//                 for (SpMat::InnerIterator it(A, i); it; ++it) {
//                     if (i<v_num&&it.col()>=e_begin&&it.col()<f_begin)v_link_cs.push_back(it.col());
//                     colv.push_back(it.col());
//                     valv.push_back(it.value());
//                     // if (i==7812) {
//                     //     if (it.col()>=e_begin&&it.col()<f_begin)RZ_INFO("--------edge--------");
//                     //     if (it.col()<e_begin)RZ_INFO("--------ver--------");
//                     //     if (it.col()>=f_begin&&it.col()<c_begin)RZ_INFO("--------face--------");
//                     //     if (it.col()>=c_begin)RZ_INFO("--------cell--------");
//                     //     RZ_INFO("row: ",i,"col: ",it.col(),"value: ",it.value());
//                     // }
//                     ++nnz_v;
//                 }
//                 if (i<v_num)vs_link_cs.push_back(v_link_cs);
//             }
//             ptrv[n] = nnz_v;
//             A_host_v =
//                 std::make_shared<typename amgcl::backend::builtin<value_type, col_type, ptr_type>::matrix >
//             (n, n, ptrv, colv, valv);
//             gpu_A_v = Backend::copy_matrix(A_host_v,bprm_);
//             relax_v = std::make_shared<Relax>(*A_host_v, rprm, bprm_);
//         }
//         //----------------------------e---------------------------------
//         {// v&e
//             std::vector<int>    ptre(n + 1, 0);
//             std::vector<int>    cole;
//             std::vector<Real> vale;
//             cole.reserve(A.nonZeros());
//             vale.reserve(A.nonZeros());
//             int nnz_e = 0;
//             for (int i = 0; i < n; ++i) {
//                 ptre[i] = nnz_e;
//                 if (i>=f_begin) {
//                     continue;
//                 }
//                 if (i<e_begin) {
//                     continue;
//                 }
//                 for (SpMat::InnerIterator it(A, i); it; ++it) {
//                     cole.push_back(it.col());
//                     vale.push_back(it.value());
//                     ++nnz_e;
//                 }
//             }
//             ptre[n] = nnz_e;
//             A_host_e =
//                 std::make_shared<typename amgcl::backend::builtin<value_type, col_type, ptr_type>::matrix >
//             (n, n, ptre, cole, vale);
//             gpu_A_e = Backend::copy_matrix(A_host_e,bprm_);
//             relax_e = std::make_shared<Relax>(*A_host_e, rprm, bprm_);
//         }
//         //----------------------------f---------------------------------
//         {
//             std::vector<int>    ptrf(n + 1, 0);
//             std::vector<int>    colf;
//             std::vector<Real> valf;
//             colf.reserve(A.nonZeros());
//             valf.reserve(A.nonZeros());
//             int nnz_f = 0;
//             for (int i = 0; i < n; ++i) {
//                 ptrf[i] = nnz_f;
//                 if (i<f_begin||i>=c_begin) {
//                     continue;
//                 }
//                 if (i<f_begin) {
//                     continue;
//                 }
//                 for (SpMat::InnerIterator it(A, i); it; ++it) {
//                     colf.push_back(it.col());
//                     valf.push_back(it.value());
//                     ++nnz_f;
//                 }
//             }
//             ptrf[n] = nnz_f;
//             A_host_f =
//                 std::make_shared<typename amgcl::backend::builtin<value_type, col_type, ptr_type>::matrix >
//             (n, n, ptrf, colf, valf);
//             gpu_A_f = Backend::copy_matrix(A_host_f,bprm_);
//             relax_f = std::make_shared<Relax>(*A_host_f, rprm, bprm_);
//         }
//         //----------------------------c---------------------------------
//         {
//             std::vector<int>    ptrc(n + 1, 0);
//             std::vector<int>    colc;
//             std::vector<Real> valc;
//             colc.reserve(A.nonZeros());
//             valc.reserve(A.nonZeros());
//             int nnz_c = 0;
//             for (int i = 0; i < n; ++i) {
//                 ptrc[i] = nnz_c;
//                 if (i<c_begin) {// 出了管控范围 只是对角线设置为1
//                     continue;
//                 }
//                 for (SpMat::InnerIterator it(A, i); it; ++it) {
//                     colc.push_back(it.col());
//                     valc.push_back(it.value());
//                     ++nnz_c;
//                 }
//             }
//             ptrc[n] = nnz_c;
//             A_host_c =
//                 std::make_shared<typename amgcl::backend::builtin<value_type, col_type, ptr_type>::matrix >
//             (n, n, ptrc, colc, valc);
//             gpu_A_c = Backend::copy_matrix(A_host_c,bprm_);
//             relax_c = std::make_shared<Relax>(*A_host_c, rprm, bprm_);
//         }
//
//         // GPU 向量分配
//         d_rhs_v = std::make_shared<typename Backend::vector>(bprm_.q, n);
//         d_rhs_e = std::make_shared<typename Backend::vector>(bprm_.q, n);
//         d_rhs_f = std::make_shared<typename Backend::vector>(bprm_.q, n);
//         d_rhs_c = std::make_shared<typename Backend::vector>(bprm_.q, n);
//         d_x_   = std::make_shared<typename Backend::vector>(bprm_.q, n);
//         residual   = std::make_shared<typename Backend::vector>(bprm_.q, n);
//         residual_temp   = std::make_shared<typename Backend::vector>(bprm_.q, n);
//         d_x_temp = std::make_shared<typename Backend::vector>(bprm_.q, n);
//
//
//     }
//
//     // ---------- 平滑：接口与 DampedJacobiSmoother 相同 ----------
//     void smooth(const Vec& b, Vec& x, int iterations = 3, Real omega = 0.666) const {
//         if (!relax_v||!relax_e||!relax_f||!relax_c) {
//             std::cerr << "[AMGCLJacobi] smoother not built.\n";
//             return;
//         }
//
//         const int n = n_;
//         if (x.size() != n) {
//             x = Vec::Zero(n);
//             exit(-1);
//         }
//
//         Vec b_v = b;
//         // b_v.tail(e_num+f_num+c_num).setZero();
//
//         Vec b_e = b;
//         // b_e.head(v_num).setZero();
//         b_e.tail(f_num+c_num).setZero();
//
//         Vec b_f = b;
//         b_f.head(v_num+e_num).setZero();
//         b_f.tail(c_num).setZero();
//
//         Vec b_c = b;
//         b_c.head(v_num+e_num+f_num).setZero();
//
//
//         // 更新 damping 参数（每次可动态调整）
//         relax_c->prm.damping = 0.666;
//         relax_f->prm.damping = 0.666;
//         relax_e->prm.damping = 0.666;
//
//
//         // 上传到 GPU
//         vex::copy(std::vector<Real>(b_v.data(), b_v.data() + n), *d_rhs_v);
//         vex::copy(std::vector<Real>(b_e.data(), b_e.data() + n), *d_rhs_e);
//         vex::copy(std::vector<Real>(b_f.data(), b_f.data() + n), *d_rhs_f);
//         vex::copy(std::vector<Real>(b_c.data(), b_c.data() + n), *d_rhs_c);
//         vex::copy(std::vector<Real>(x.data(), x.data() + n), *d_x_);
//
//         for (int it = 0; it < iterations; ++it) {
//             relax_v->prm.damping = 0.666;
//             amgcl::backend::residual(*d_rhs_v,*gpu_A_v,*d_x_,*residual);
//             Vec res,res_temp;
//             res.resize(n);
//             vex::copy(*residual,res.data());
//
//             // if ((it%2) ==0) {
//             //     vex::copy(*d_x_, x.data());
//             //     for (int i=0;i<v_num;++i) {
//             //         Real weight = 1/(Real)vs_link_cs[i].size();
//             //         // std::cout<<"weight "<<weight<<std::endl;
//             //         x[i] = 0.0;
//             //         for (auto&c:vs_link_cs[i]) {
//             //             x[i] += weight * x[c];
//             //         }
//             //     }
//             //     vex::copy(std::vector<Real>(x.data(), x.data() + n), *d_x_);
//             // }
//
//             // --------------------- volume
//
//             res_temp = res;
//             res_temp.head(v_num+e_num+f_num).setZero();
//             vex::copy(std::vector<Real>(res_temp.data(), res_temp.data()+ n ), *residual);
//             relax_v->RZ_apply(*d_x_,*residual);
//
// //----------------------------------------f
//
//             res_temp = res;
//             res_temp.head(v_num+e_num).setZero();
//             res_temp.tail(c_num).setZero();
//             vex::copy(std::vector<Real>(res_temp.data(), res_temp.data()+ n ), *residual);
//             relax_v->RZ_apply(*d_x_,*residual);
//
//
//             // res_temp = res;
//             // res_temp.head(v_num).setZero();
//             // res_temp.tail(f_num+c_num).setZero();
//             // vex::copy(std::vector<Real>(res_temp.data(), res_temp.data()+ n ), *residual);
//             // relax_v->RZ_apply(*d_x_,*residual);
//             //----------------------------------------e
//             // if ((it%2) ==0) {
//             //     res_temp = res;
//             //     res_temp.head(v_num).setZero();
//             //     res_temp.tail(f_num+c_num).setZero();
//             //     vex::copy(std::vector<Real>(res_temp.data(), res_temp.data()+ n ), *residual);
//             //     relax_v->RZ_apply(*d_x_,*residual);
//             // }
//             //----------------------------------------v
//             // for (int k=0;k<5;++k) {
//             //     amgcl::backend::residual(*d_rhs_v,*gpu_A_v,*d_x_,*residual);
//             //     vex::copy(*residual,res.data());
//             //     relax_v->prm.damping = 0.2;
//             //     res_temp = res;
//             //     res_temp.tail(f_num+c_num).setZero();
//             //     vex::copy(std::vector<Real>(res_temp.data(), res_temp.data()+ n ), *residual);
//             //     relax_v->RZ_apply(*d_x_,*residual);
//             // }
//
//
//             //就是说 一更新体和面 只更新一次 节点和边会出问题  但是更新两次就不会爆炸！
//             // 直接 利用 每个节点 所在的体来更新节点上的值
//
//         }
//
//         // 拷回结果
//         vex::copy(*d_x_, x.data());
//         // Vec res_temp;
//         // res_temp.resize(n);
//         // amgcl::backend::residual(*d_rhs_v,*gpu_A_v,*d_x_,*residual);
//         // vex::copy(*residual,res_temp.data());
//         // int max_id =0;
//         // Real max_value = 0.0;
//         // for (int k=0;k<res_temp.rows();++k) {
//         //     if (std::fabs(res_temp[k])>max_value) {
//         //         max_value=std::fabs(res_temp[k]);
//         //         max_id=k;
//         //     }
//         // }
//         // RZ_INFO("x max id: ",max_id);
//         // for (SpMat::InnerIterator it(A_, max_id); it; ++it) {
//         //     if (it.col()>=v_num&&it.col()<v_num+e_num)RZ_INFO("--------edge--------");
//         //     if (it.col()<v_num)RZ_INFO("--------ver--------");
//         //     if (it.col()>=v_num+e_num&&it.col()<v_num+e_num+f_num)RZ_INFO("--------face--------");
//         //     if (it.col()>=v_num+e_num+f_num)RZ_INFO("--------cell--------");
//         //     RZ_INFO("row: ",max_id,"col: ",it.col(),"A value: ",it.value());
//         //     RZ_INFO("p col value: ",x[it.col()]);
//         // }
//         // RZ_INFO("x max id dia: ",A_.coeff(max_id,max_id));
//         // RZ_INFO("x max: ",x[max_id]);
//         // RZ_INFO("res max: ",max_value);
//     }
//
// private:
//     int n_ = 0;
//     SpMat A_;
//     std::vector<std::vector<int>> vs_link_cs;
//     std::shared_ptr<Matrix> gpu_A_v,gpu_A_e,gpu_A_f,gpu_A_c;
//     std::shared_ptr< typename amgcl::backend::builtin<value_type, col_type, ptr_type>::matrix >
//         A_host_v,A_host_e,A_host_f,A_host_c;
//     typename Backend::params bprm_;
//     int v_num=-1,e_num=-1,f_num=-1,c_num=-1;
//     Real ratio_c_f = 0.0,ratio_f_e= 0.0,ratio_e_v= 0.0;
//     std::shared_ptr<Relax> relax_v,relax_e,relax_f,relax_c;
//     std::shared_ptr<typename Backend::vector> d_rhs_v;
//     std::shared_ptr<typename Backend::vector> d_rhs_e;
//     std::shared_ptr<typename Backend::vector> d_rhs_f;
//     std::shared_ptr<typename Backend::vector> d_rhs_c;
//     std::shared_ptr<typename Backend::vector> residual;
//     std::shared_ptr<typename Backend::vector> residual_temp;
//     std::shared_ptr<typename Backend::vector> d_x_;
//     std::shared_ptr<typename Backend::vector> d_x_temp;
// };
//
// class AMGCLChebyshevBlockSmoother {
// public:
//     using SpMat = Eigen::SparseMatrixR;
//     using Vec   = Eigen::VectorXr;
//     using Backend   = amgcl::backend::vexcl<Real>;
//     using Relax     = amgcl::relaxation::chebyshev<Backend>;
//     using value_type = Backend::value_type;
//     using col_type  = Backend::col_type;
//     using ptr_type = Backend::ptr_type;
//     using CRS       = amgcl::backend::crs<value_type,col_type,ptr_type>;
//     using Matrix       = Backend::matrix;
//     // ---------- 构建：Eigen 矩阵转 CRS 并初始化 AMGCL 平滑器 ----------
//     void build(const SpMat& A,int v_size,int e_size,int f_size,int c_size) {
//         const int n = A.rows();
//         n_ = n;
//         const int e_begin = v_size;
//         const int f_begin = e_begin+e_size;
//         const int c_begin = f_begin+f_size;
//         v_num = v_size;
//         e_num = e_size;
//         f_num = f_size;
//         c_num = c_size;
//         ratio_c_f = Real(c_size)/Real(f_size);
//         ratio_f_e = Real(f_size)/Real(e_size);
//         ratio_e_v = Real(e_size)/Real(v_size);
//         // vexcl 上下文
//         vex::Context ctx(vex::Filter::Env);
//         bprm_.q = ctx;
//         typename Relax::params rprm;
//         rprm.degree = 2;
//         rprm.power_iters = 500;
//         rprm.higher = 1.0;
//         rprm.lower = 1.0/30.0;
//         rprm.scale = true;
//         //----------------------------v---------------------------------
//         {
//             std::vector<int>    ptrv(n + 1, 0);
//             std::vector<int>    colv;
//             std::vector<Real> valv;
//             colv.reserve(A.nonZeros());
//             valv.reserve(A.nonZeros());
//             int nnz_v = 0;
//             for (int i = 0; i < n; ++i) {
//                 ptrv[i] = nnz_v;
//                 if (i>=v_size) {
//                     continue;
//                 }
//                 for (SpMat::InnerIterator it(A, i); it; ++it) {
//                     colv.push_back(it.col());
//                     valv.push_back(it.value());
//                     ++nnz_v;
//                 }
//             }
//             ptrv[n] = nnz_v;
//             A_host_v =
//                 std::make_shared<typename amgcl::backend::builtin<value_type, col_type, ptr_type>::matrix >
//             (n, n, ptrv, colv, valv);
//             gpu_A_v = Backend::copy_matrix(A_host_v,bprm_);
//             relax_v = std::make_shared<Relax>(*A_host_v, rprm, bprm_);
//             // relax_v->prm.degree
//         }
//         //----------------------------e---------------------------------
//         {
//             std::vector<int>    ptre(n + 1, 0);
//             std::vector<int>    cole;
//             std::vector<Real> vale;
//             cole.reserve(A.nonZeros());
//             vale.reserve(A.nonZeros());
//             int nnz_e = 0;
//             for (int i = 0; i < n; ++i) {
//                 ptre[i] = nnz_e;
//                 if (i<e_begin||i>=f_begin) {
//                     continue;
//                 }
//                 if (i<e_begin) {
//                     continue;
//                 }
//                 for (SpMat::InnerIterator it(A, i); it; ++it) {
//                     cole.push_back(it.col());
//                     vale.push_back(it.value());
//                     ++nnz_e;
//                 }
//             }
//             ptre[n] = nnz_e;
//             A_host_e =
//                 std::make_shared<typename amgcl::backend::builtin<value_type, col_type, ptr_type>::matrix >
//             (n, n, ptre, cole, vale);
//             gpu_A_e = Backend::copy_matrix(A_host_e,bprm_);
//             relax_e = std::make_shared<Relax>(*A_host_e, rprm, bprm_);
//         }
//         //----------------------------f---------------------------------
//         {
//             std::vector<int>    ptrf(n + 1, 0);
//             std::vector<int>    colf;
//             std::vector<Real> valf;
//             colf.reserve(A.nonZeros());
//             valf.reserve(A.nonZeros());
//             int nnz_f = 0;
//             for (int i = 0; i < n; ++i) {
//                 ptrf[i] = nnz_f;
//                 if (i<f_begin||i>=c_begin) {
//                     continue;
//                 }
//                 if (i<f_begin) {
//                     continue;
//                 }
//                 for (SpMat::InnerIterator it(A, i); it; ++it) {
//                     colf.push_back(it.col());
//                     valf.push_back(it.value());
//                     ++nnz_f;
//                 }
//             }
//             ptrf[n] = nnz_f;
//             A_host_f =
//                 std::make_shared<typename amgcl::backend::builtin<value_type, col_type, ptr_type>::matrix >
//             (n, n, ptrf, colf, valf);
//             gpu_A_f = Backend::copy_matrix(A_host_f,bprm_);
//             relax_f = std::make_shared<Relax>(*A_host_f, rprm, bprm_);
//         }
//         //----------------------------c---------------------------------
//         {
//             std::vector<int>    ptrc(n + 1, 0);
//             std::vector<int>    colc;
//             std::vector<Real> valc;
//             colc.reserve(A.nonZeros());
//             valc.reserve(A.nonZeros());
//             int nnz_c = 0;
//             for (int i = 0; i < n; ++i) {
//                 ptrc[i] = nnz_c;
//                 if (i<c_begin) {// 出了管控范围 只是对角线设置为1
//                     continue;
//                 }
//                 for (SpMat::InnerIterator it(A, i); it; ++it) {
//                     colc.push_back(it.col());
//                     valc.push_back(it.value());
//                     ++nnz_c;
//                 }
//             }
//             ptrc[n] = nnz_c;
//             A_host_c =
//                 std::make_shared<typename amgcl::backend::builtin<value_type, col_type, ptr_type>::matrix >
//             (n, n, ptrc, colc, valc);
//             gpu_A_c = Backend::copy_matrix(A_host_c,bprm_);
//             relax_c = std::make_shared<Relax>(*A_host_c, rprm, bprm_);
//         }
//
//         // GPU 向量分配
//         d_rhs_v = std::make_shared<typename Backend::vector>(bprm_.q, n);
//         d_rhs_e = std::make_shared<typename Backend::vector>(bprm_.q, n);
//         d_rhs_f = std::make_shared<typename Backend::vector>(bprm_.q, n);
//         d_rhs_c = std::make_shared<typename Backend::vector>(bprm_.q, n);
//         d_x_   = std::make_shared<typename Backend::vector>(bprm_.q, n);
//         residual   = std::make_shared<typename Backend::vector>(bprm_.q, n);
//         d_x_temp = std::make_shared<typename Backend::vector>(bprm_.q, n);
//     }
//
//     // ---------- 平滑：接口与 DampedJacobiSmoother 相同 ----------
//     void smooth(const Vec& b, Vec& x, int iterations = 3, Real omega = 0.666) const {
//         if (!relax_v||!relax_e||!relax_f||!relax_c) {
//             std::cerr << "[AMGCLJacobi] smoother not built.\n";
//             return;
//         }
//
//         const int n = n_;
//         if (x.size() != n) {
//             x = Vec::Zero(n);
//             exit(-1);
//         }
//
//         Vec b_v = b;
//         b_v.tail(e_num+f_num+c_num).setZero();
//         Vec b_e = b;b_e.head(v_num).setZero();
//         b_e.tail(f_num+c_num).setZero();
//         Vec b_f = b;b_f.head(v_num+e_num).setZero();
//         b_f.tail(c_num).setZero();
//         Vec b_c = b;b_c.head(v_num+e_num+f_num).setZero();
//
//
//
//         // 上传到 GPU
//         vex::copy(std::vector<Real>(b_v.data(), b_v.data() + n), *d_rhs_v);
//         vex::copy(std::vector<Real>(b_e.data(), b_e.data() + n), *d_rhs_e);
//         vex::copy(std::vector<Real>(b_f.data(), b_f.data() + n), *d_rhs_f);
//         vex::copy(std::vector<Real>(b_c.data(), b_c.data() + n), *d_rhs_c);
//         vex::copy(std::vector<Real>(x.data(), x.data() + n), *d_x_);
//
//         for (int it = 0; it < iterations; ++it) {
//             // 先用 k=1 的 更新一下节点上的value 我们需要对应的 rhs和A矩阵
//             relax_c->apply_post(*gpu_A_c, *d_rhs_c, *d_x_,*residual);
//             relax_f->apply_post(*gpu_A_f, *d_rhs_f, *d_x_,*residual);
//             relax_e->apply_post(*gpu_A_e, *d_rhs_e, *d_x_,*residual);
//             relax_v->apply_post(*gpu_A_v, *d_rhs_v, *d_x_,*residual);
//         }
//         vex::copy(*d_x_, x.data());
//     }
//
// private:
//     int n_ = 0;
//     std::vector<std::vector<int>> v_link_cs;
//     std::shared_ptr<Matrix> gpu_A_v,gpu_A_e,gpu_A_f,gpu_A_c;
//     std::shared_ptr< typename amgcl::backend::builtin<value_type, col_type, ptr_type>::matrix >
//         A_host_v,A_host_e,A_host_f,A_host_c;
//     typename Backend::params bprm_;
//     int v_num=-1,e_num=-1,f_num=-1,c_num=-1;
//     Real ratio_c_f = 0.0,ratio_f_e= 0.0,ratio_e_v= 0.0;
//     std::shared_ptr<Relax> relax_v,relax_e,relax_f,relax_c;
//     std::shared_ptr<typename Backend::vector> d_rhs_v;
//     std::shared_ptr<typename Backend::vector> d_rhs_e;
//     std::shared_ptr<typename Backend::vector> d_rhs_f;
//     std::shared_ptr<typename Backend::vector> d_rhs_c;
//     std::shared_ptr<typename Backend::vector> residual;
//     std::shared_ptr<typename Backend::vector> d_x_;
//     std::shared_ptr<typename Backend::vector> d_x_temp;
// };
//
// class BlockJacobiSmoother {
//     public:
//     using SpMat = Eigen::SparseMatrixR;
//     using Vec   = Eigen::VectorXr;
//     using Backend   = amgcl::backend::vexcl<Real>;
//     using Relax     = amgcl::relaxation::damped_jacobi<Backend>;
//     using value_type = Backend::value_type;
//     using col_type  = Backend::col_type;
//     using ptr_type = Backend::ptr_type;
//     using CRS       = amgcl::backend::crs<value_type,col_type,ptr_type>;
//     using Matrix       = Backend::matrix;
//
//
//
//     // ---------- 构建：Eigen 矩阵转 CRS 并初始化 AMGCL 平滑器 ----------
//     void build(const SpMat& A,const VEM_Domain_3d& domain) {
//         FUNC_TIMER();
//         const int n = A.rows();
//         n_ = n;
//         // Eigen -> CRS
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
//         // vexcl 上下文
//         if (bprm_.q.empty()) {
//             vex::Context ctx(vex::Filter::Env);
//             if (ctx.empty())exit(-1);
//             bprm_.q = ctx;
//         }
//         A_host_ = std::make_shared<typename amgcl::backend::builtin<value_type, col_type, ptr_type>::matrix >(n, n, ptr, col, val);
//         gpu_A = Backend::copy_matrix(A_host_,bprm_);
//         typename Relax::params rprm;
//         rprm.damping = 0.666;
//
//         relax_ = std::make_shared<Relax>(*A_host_, rprm, bprm_);
//         d_rhs_ = std::make_shared<typename Backend::vector>(bprm_.q, n);
//
//         residual   = std::make_shared<typename Backend::vector>(bprm_.q, n);
//         residual_temp = std::make_shared<typename Backend::vector>(bprm_.q, n);
//
//         // auto compute_inv = [](const Eigen::MatrixXr& M) -> Eigen::MatrixXr {
//         //     // Try SPD solve first
//         //     Eigen::LLT<Eigen::MatrixXr> llt;
//         //     llt.compute(M);
//         //     if (llt.info() == Eigen::Success) {
//         //         Eigen::MatrixXr I = Eigen::MatrixXr::Identity(M.rows(), M.cols());
//         //         return llt.solve(I);
//         //     }
//         //
//         //     // fallback: robust LU
//         //     Eigen::FullPivLU<Eigen::MatrixXr> lu(M);
//         //     if (!lu.isInvertible()) {
//         //         // add small diagonal regularization and retry LU
//         //         Eigen::MatrixXr Mr = M;
//         //         Mr.diagonal().array() += 1e-12;
//         //         Eigen::FullPivLU<Eigen::MatrixXr> lu2(Mr);
//         //         if (!lu2.isInvertible()) {
//         //             throw std::runtime_error("Block matrix is not invertible even after regularization.");
//         //         }
//         //         return lu2.inverse();
//         //     }
//         //     return lu.inverse();
//         // };
//         auto compute_inv = [](const Eigen::MatrixXr& M) -> Eigen::MatrixXr {
//             const int n = (int)M.rows();
//             if (M.rows() != M.cols())
//                 throw std::runtime_error("Block matrix must be square.");
//
//             // // Try SPD solve first
//             // Eigen::LLT<Eigen::MatrixXr> llt;
//             // llt.compute(M);
//             // if (llt.info() == Eigen::Success) {
//             //     Eigen::MatrixXr I = Eigen::MatrixXr::Identity(n, n);
//             //     return llt.solve(I);
//             // }
//             // RZ_INFO("not spd inv");
//             // fallback: robust LU
//             Eigen::FullPivLU<Eigen::MatrixXr> lu(M);
//             if (lu.isInvertible()) {
//                 return lu.inverse();
//             }
//
//             // add small diagonal regularization and retry LU
//             Eigen::MatrixXr Mr = M;
//
//             // --- 更鲁棒：自适应尺度的正则化，而不是固定 1e-12 ---
//             const Real reg = 1e-12;
//             Mr.diagonal().array() += reg;
//
//             Eigen::FullPivLU<Eigen::MatrixXr> lu2(Mr);
//             if (lu2.isInvertible()) {
//                 return lu2.inverse();
//             }
//             // =========================================================
//             // NEW: approximate inverse (SVD pseudo-inverse with truncation)
//             // =========================================================
//             Eigen::JacobiSVD<Eigen::MatrixXr> svd(
//                 M, Eigen::ComputeThinU | Eigen::ComputeThinV
//             );
//             const auto& S = svd.singularValues();
//
//             // 阈值：相对最大奇异值截断（可按你的 block 尺度调）
//             const Real sigma_max = (S.size() > 0) ? S(0) : 0.0;
//             const Real tol = 1e-12;
//
//             Eigen::VectorXr Sinv = S;
//             for (int i = 0; i < S.size(); ++i) {
//                 Sinv(i) = (S(i) > tol) ? (1.0 / S(i)) : 0.0;
//             }
//
//             // 伪逆：V * S^+ * U^T
//             return svd.matrixV() * Sinv.asDiagonal() * svd.matrixU().transpose();
//         };
//
//         // V W A
//         std::map<int,Real>dof_cnt;
//         const int vdofs = domain.points.size();
//         const int edofs = domain.edges.size();
//         const int fdofs = domain.polygons.size();
//         RZ_ASSERT(vdofs+edofs+fdofs+domain.vem_elements.size() == A.rows(),"");
//         std::vector<_3d_vec> dof_pos;
//         dof_pos.resize(vdofs+edofs+fdofs+domain.vem_elements.size());
//         for (int vid=0;vid<domain.points.size();++vid)dof_pos[vid] = domain.points[vid];
//         for (auto&edge:domain.edges) dof_pos[vdofs+edge_id_map[edge]] = (domain.points[edge.v1]+domain.points[edge.v2])/2.0;
//         for (int pid = 0;pid < domain.polygons.size();++pid) dof_pos[vdofs+edofs+pid] = domain.polygons_plane[pid]._3d_centroid;
//         for (int cid=0;cid<domain.vem_elements.size();++cid) dof_pos[vdofs+edofs+fdofs+cid] = domain.vem_elements[cid].centroid;
//
//         std::vector<_3d_vec> cell_centroids;
//         for (int cell_id = 0; cell_id < domain.vem_elements.size(); ++cell_id) {
//             const auto& cell = domain.vem_elements[cell_id];
//             auto cell_centroid = cell.centroid;
//             std::vector<int> cell_V;
//             for (const auto& gvid:cell.vertex_local_to_global_dof) {
//                 cell_V.push_back(gvid);
//                 // Real dis = (dof_pos[gvid]-cell_centroid).norm();
//                 dof_cnt[gvid] += 1.0/cell.diameter;;
//             }
//             for (int local_edge_id=0;local_edge_id<cell.edges.size();++local_edge_id) {
//                 // Real dis = (dof_pos[geid+vdofs]-cell_centroid).norm();
//                 cell_V.push_back(geid+vdofs);
//                 dof_cnt[geid+vdofs] += 1.0/cell.diameter;;
//             }
//             // Vs.push_back(cell_V);
//             // cell_centroids.push_back(cell_centroid);
//             // cell_V.clear();
//             for (const auto& gfid:cell.polygon_local_to_global_dof) {
//                 cell_V.push_back(gfid+vdofs+edofs);
//                 // Real dis = (dof_pos[gfid+vdofs+edofs]-cell_centroid).norm();
//                 dof_cnt[gfid+vdofs+edofs] += 1.0/cell.diameter;
//             }
//             dof_cnt[fdofs+vdofs+edofs+cell_id] += 0.0;
//             cell_V.push_back(fdofs+vdofs+edofs+cell_id);
//             Vs.push_back(cell_V);
//             cell_centroids.push_back(cell_centroid);
//         }
//         RZ_ASSERT(dof_cnt.size() == A.rows(),dof_cnt.size()," ",A.rows());
//         RZ_ASSERT(Vs.size() == cell_centroids.size(),"");
//         for (int cell_id =0; cell_id<Vs.size();++cell_id) {
//             const auto& cell_V = Vs[cell_id];
//             std::vector<Real> cell_W;
//             for (auto&dof:cell_V) {
//                 Real weight = 1.0;
//                 if (dof<fdofs+vdofs+edofs) {
//                     Real dis = (cell_centroids[cell_id] - dof_pos[dof]).norm();
//                     weight = 1.0 / dis / dof_cnt[dof];
//                 }
//                 cell_W.push_back(weight);
//             }
//             Ws.push_back(cell_W);
//         }
//         // precompute ∑Vi * Aii_inverse * Wi_T
//         SpMat eigen_M_matrix;
//         eigen_M_matrix.resize(A.rows(),A.cols());
//         eigen_M_matrix.setZero();
//         std::vector<Eigen::Triplet<Real>> M_trips;
//         for (int cell_id=0;cell_id<Vs.size();++cell_id) {
//             const auto& cell_V = Vs[cell_id];
//             const auto& cell_W = Ws[cell_id];
//             Eigen::MatrixXr cell_Aii;
//             cell_Aii.resize(cell_V.size(),cell_V.size());
//             cell_Aii.setZero();
//             for (int i=0;i<cell_V.size();++i) {
//                 for (int j=0;j<cell_V.size();++j)
//                     cell_Aii(i,j) = A.coeff(cell_V[i],cell_V[j]);
//             }
//             Aiis.push_back(cell_Aii);
//             //gpt part:将 cell_Aii在cpu上做好分解 或者求逆 然后放到 gpu上
//             Eigen::MatrixXr Aii_inverse = compute_inv(cell_Aii);
//             // 其实是自由度的映射 只要把映射过去就行 Vi * Aii_inverse * WiT
//             std::map<int,int> lg_dof_map;
//             for (int i=0;i<cell_V.size();++i) {
//                 lg_dof_map[i] = cell_V[i];
//             }
//             for (int i=0;i<Aii_inverse.rows();++i) {
//                 for (int j=0;j<Aii_inverse.rows();++j)
//                     M_trips.emplace_back(lg_dof_map[i],lg_dof_map[j],
//                         Aii_inverse.coeff(i,j) * cell_W[j]);
//             }
//         }
//         eigen_M_matrix.setFromTriplets(M_trips.begin(),M_trips.end());
//         eigen_M_matrix.makeCompressed();
//         gpu_M = eigentovex(eigen_M_matrix,bprm_);
//     }
//
//     void smooth(const Backend::vector& d_rhs, Backend::vector& d_x, int iterations = 3) const {
//         for (int it = 0; it < iterations; ++it) {
//             amgcl::backend::residual(d_rhs, *gpu_A, d_x, *residual);
//             amgcl::backend::spmv(0.3,*gpu_M,*residual,1.0,d_x);
//         }
//     }
//
//     typename Backend::params bprm_;
// private:
//         Eigen::VectorXr solve_block(
//         const Eigen::MatrixXr &Aii,
//         const Eigen::VectorXr &rhs
//         ) const{
//             Eigen::FullPivLU<Eigen::MatrixXr> lu(Aii);
//             if (!lu.isInvertible()) {
//                 // 可选：轻微正则化
//                 Eigen::MatrixXr Ar = Aii;
//                 Ar.diagonal().array() += 1e-12;
//                 Eigen::FullPivLU<Eigen::MatrixXr> lu2(Ar);
//                 return lu2.solve(rhs);
//             }
//             return lu.solve(rhs);
//         }
//     int n_ = 0;
//     std::shared_ptr<Matrix> gpu_A;
//     std::vector<std::vector<int>> Vs;
//     std::vector<std::vector<Real>> Ws;
//     std::shared_ptr<Matrix> gpu_M;
//     std::vector<Eigen::MatrixXr> Aiis;
//     std::shared_ptr< typename amgcl::backend::builtin<value_type, col_type, ptr_type>::matrix > A_host_;
//
//     std::shared_ptr<Relax> relax_;
//     std::shared_ptr<typename Backend::vector> d_rhs_;
//     std::shared_ptr<typename Backend::vector> residual;
//     std::shared_ptr<typename Backend::vector> residual_temp;
// };
