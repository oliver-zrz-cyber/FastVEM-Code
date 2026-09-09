#pragma once
#include <Eigen/Sparse>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <chrono>

#include "config.h"


// ============================================================
// CUDA Jacobi 核函数（声明为 __global__）
// ============================================================
__global__ void jacobiKernelIter(
    int n,
    const int* row_ptr,
    const int* col_idx,
    const GlobalReal* val,
    const GlobalReal* b,
    GlobalReal* x,
    const GlobalReal* inv_diag,
    Real omega,
    int iterations);


// ============================================================
// DampedJacobiSmootherGPU 类定义
// ============================================================
class DampedJacobiSmootherGPU {
public:
    using SpMat = Eigen::SparseMatrixR;
    using Vec   = Eigen::VectorXr;

    DampedJacobiSmootherGPU();
    ~DampedJacobiSmootherGPU();

    void build(const SpMat& A);
    void smooth(const Vec& b, Vec& x, int iterations = 10, Real omega = 0.2);

private:
    void release();

private:
    int n_ = 0, nnz_ = 0;
    std::vector<int> row_ptr_, col_idx_;
    std::vector<GlobalReal> val_, diag_, inv_diag_;

    // GPU device pointers
    int *d_row_ptr_ = nullptr, *d_col_idx_ = nullptr;
    GlobalReal *d_val_ = nullptr, *d_inv_diag_ = nullptr;
    GlobalReal *d_x_old_ = nullptr, *d_x_new_ = nullptr, *d_b_ = nullptr;
};

#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err__) \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        throw std::runtime_error("CUDA error"); \
    } \
} while (0)

__global__ void k_set_zero(GlobalReal* x, int n);
__global__ void k_copy(const GlobalReal* src, GlobalReal* dst, int n);
__global__ void k_axpy(GlobalReal* y, const GlobalReal* x, Real alpha, int n);
__global__ void k_combine(GlobalReal* y, const GlobalReal* a, const GlobalReal* b, Real alpha, Real beta, int n);
__global__ void k_residual(const GlobalReal* b, const GlobalReal* y, GlobalReal* r, int n);
__global__ void k_spmv_csr(int n, const int* row_ptr, const int* col_idx, const GlobalReal* val,
                           const GlobalReal* x, GlobalReal* y);
__global__ void jacobiKernelIterDevice(int n, const int* row_ptr, const int* col_idx, const GlobalReal* val,
                                       const GlobalReal* b, GlobalReal* x, const GlobalReal* inv_diag,
                                       Real omega, int iters);

struct CsrDevice {
    int n=0, m=0, nnz=0;
    int *d_row=nullptr, *d_col=nullptr;
    GlobalReal *d_val=nullptr, *d_invdiag=nullptr;
    void freeAll();
    void uploadFromEigenA(const Eigen::SparseMatrixR& A);
    void uploadFromEigenPR(const Eigen::SparseMatrixR& M);
};

class MAC_MGSolverCUDA {
public:
    using SpMat = Eigen::SparseMatrixR;
    using Vec   = Eigen::VectorXr;
    MAC_MGSolverCUDA();
    ~MAC_MGSolverCUDA();
    bool isSymmetric(const SpMat& A);
    void build_hierarchy(const SpMat& A_fine,int nx,int ny,int nz,int levels);
    void set_rhs(const Vec& b_fine);
    void VC(Vec& xH,int nu1=2,int nu2=2,int cycles=6);
private:
    static constexpr int TPB=256;
    struct TimingStats{ Real smooth_time=0.0,other_time=0.0; void accumulate(const TimingStats&s){smooth_time+=s.smooth_time;other_time+=s.other_time;} };
    int L_=0;
    std::vector<int> nx_,ny_,nz_;
    std::vector<CsrDevice> As_,P_,R_;
    std::vector<GlobalReal*> d_x_,d_b_,d_r_,d_y_,d_t_;


    TimingStats V_cycle(int lev,int nu1,int nu2);

    void setZeroDevice(GlobalReal* p,int n);

    void allocVectors();

    void release();

    static inline int idx3(int i,int j,int k,int nx,int ny,int nz){return (k*ny+j)*nx+i;}
    Eigen::SparseMatrixR prolongation(int coarse,int fine);

};
