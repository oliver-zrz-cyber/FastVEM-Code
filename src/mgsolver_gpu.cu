#include "mgsolver_gpu.h"
#include "dfn.h"
#include <cmath>


void printGpuMemoryUsage(const char* msg = nullptr) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);

    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err)
                  << " in printGpuMemoryUsage()" << std::endl;
        return;
    }

    Real free_mb = static_cast<Real>(free_bytes) / (1024.0 * 1024.0);
    Real total_mb = static_cast<Real>(total_bytes) / (1024.0 * 1024.0);
    Real used_mb = total_mb - free_mb;

    if (msg) std::cout << "[" << msg << "] ";
    std::cout << "GPU memory: used = " << used_mb
              << " MB, free = " << free_mb
              << " MB, total = " << total_mb << " MB" << std::endl;
}
// ============================================================
// GPU Jacobi Kernel
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
    int iterations)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    for (int it = 0; it < iterations; ++it) {
        Real sum_off = 0.0;
        const int rs = row_ptr[i];
        const int re = row_ptr[i + 1];
        for (int k = rs; k < re; ++k) {
            int j = col_idx[k];
            if (j != i)
                sum_off += val[k] * x[j];
        }
        Real xi_new = (1.0 - omega) * x[i] + omega * (b[i] - sum_off) * inv_diag[i];
        x[i] = xi_new;
        __syncthreads(); // 同步所有线程，确保 x 已更新（Jacobi 不严格需要）
    }
}
// 单次 Jacobi：读 x_old，写 x_new。无 __syncthreads()，无内部循环。
__global__ void jacobiKernelOnce(
    int n,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const GlobalReal* __restrict__ val,
    const GlobalReal* __restrict__ b,
    const GlobalReal* __restrict__ x_old,
    GlobalReal* __restrict__ x_new,
    const GlobalReal* __restrict__ inv_diag,
    Real omega)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    Real sum_off = 0.0;
    const int rs = row_ptr[i];
    const int re = row_ptr[i + 1];

    // 稀疏行累加（排除对角）。保持分支，但极少触发，分支代价可忽略
    // todo : 这里是主要瓶颈
    for (int k = rs; k < re; ++k) {
        const int& j = col_idx[k];
        const GlobalReal aij = val[k];
        if (j != i) {
            // 读 x_old，尽量让编译器用只读缓存
            sum_off += aij * x_old[j];
        }
    }

    const Real xi = x_old[i];
    const Real ri = b[i] - sum_off;  // 注意：对角项不在 sum_off 内
    const Real xi_new = (1.0 - omega) * xi + omega * (ri * inv_diag[i]);

    x_new[i] = xi_new;
}

// ============================================================
// 构造 & 析构
// ============================================================
DampedJacobiSmootherGPU::DampedJacobiSmootherGPU() = default;

DampedJacobiSmootherGPU::~DampedJacobiSmootherGPU() {
    release();
}
// ============================================================
// build(A): 上传固定矩阵 A
// ============================================================
void DampedJacobiSmootherGPU::build(const SpMat& A) {
    release();

    n_ = A.rows();
    row_ptr_.assign(n_ + 1, 0);
    int nnz = 0;
    for (int i = 0; i < n_; ++i) {
        for (SpMat::InnerIterator it(A, i); it; ++it)
            ++nnz;
        row_ptr_[i + 1] = nnz;
    }

    col_idx_.resize(nnz);
    val_.resize(nnz);
    diag_.assign(n_, 0.0);
    inv_diag_.assign(n_, 0.0);

    int k = 0;
    for (int i = 0; i < n_; ++i) {
        for (SpMat::InnerIterator it(A, i); it; ++it) {
            col_idx_[k] = it.col();
            val_[k]     = it.value();
            if (it.col() == i) diag_[i] = it.value();
            ++k;
        }
        inv_diag_[i] = (std::abs(diag_[i]) > 1e-30) ? (1.0 / diag_[i]) : 0.0;
    }

    nnz_ = nnz;

    // GPU 分配并上传
    cudaMalloc(&d_row_ptr_, (n_ + 1) * sizeof(int));
    cudaMalloc(&d_col_idx_, nnz_ * sizeof(int));
    cudaMalloc(&d_val_, nnz_ * sizeof(GlobalReal));
    cudaMalloc(&d_inv_diag_, n_ * sizeof(GlobalReal));

    cudaMemcpy(d_row_ptr_, row_ptr_.data(), (n_ + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx_, col_idx_.data(), nnz_ * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val_, val_.data(), nnz_ * sizeof(GlobalReal), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inv_diag_, inv_diag_.data(), n_ * sizeof(GlobalReal), cudaMemcpyHostToDevice);

    cudaMalloc(&d_x_old_, n_ * sizeof(GlobalReal));
    cudaMalloc(&d_x_new_, n_ * sizeof(GlobalReal));
    cudaMalloc(&d_b_, n_ * sizeof(GlobalReal));
}
void DampedJacobiSmootherGPU::smooth(const Vec& b, Vec& x, int iterations, Real omega) {
    if (n_ == 0) return;
    if (x.size() != n_) x = Vec::Zero(n_);

    // 一次性上传
    cudaMemcpy(d_b_, b.data(), n_ * sizeof(GlobalReal), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x_old_, x.data(), n_ * sizeof(GlobalReal), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n_ + blockSize - 1) / blockSize;

    // 单次 launch 完成多次迭代
    jacobiKernelIter<<<gridSize, blockSize>>>(
        n_, d_row_ptr_, d_col_idx_, d_val_,
        d_b_, d_x_old_, d_inv_diag_, omega, iterations);

    // 取回结果
    cudaMemcpy(x.data(), d_x_old_, n_ * sizeof(GlobalReal), cudaMemcpyDeviceToHost);
}
// ===========================================================
// 内存释放
// ============================================================
void DampedJacobiSmootherGPU::release() {
    if (d_row_ptr_) cudaFree(d_row_ptr_);
    if (d_col_idx_) cudaFree(d_col_idx_);
    if (d_val_) cudaFree(d_val_);
    if (d_inv_diag_) cudaFree(d_inv_diag_);
    if (d_x_old_) cudaFree(d_x_old_);
    if (d_x_new_) cudaFree(d_x_new_);
    if (d_b_) cudaFree(d_b_);
    d_row_ptr_ = d_col_idx_ = nullptr;
    d_val_ = d_inv_diag_ = nullptr;
    d_x_old_ = d_x_new_ = d_b_ = nullptr;
    n_ = nnz_ = 0;
}


// ================= CUDA 核函数实现 =================
__global__ void k_reduce_sum_squares(const GlobalReal* __restrict__ x, GlobalReal* __restrict__ partial, int n) {
    extern __shared__ GlobalReal sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    GlobalReal val = 0.0;
    if (i < n) {
        GlobalReal v = x[i];
        val = v * v;
    }
    sdata[tid] = val;
    __syncthreads();

    // 归约
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset)
            sdata[tid] += sdata[tid + offset];
        __syncthreads();
    }

    if (tid == 0) partial[blockIdx.x] = sdata[0];
}

Real gpuL2Norm(const GlobalReal* d_x, int n) {
    const int TPB = 256;
    int nBlocks = (n + TPB - 1) / TPB;
    GlobalReal* d_partial = nullptr;
    CUDA_CHECK(cudaMalloc(&d_partial, nBlocks * sizeof(GlobalReal)));

    k_reduce_sum_squares<<<nBlocks, TPB, TPB * sizeof(GlobalReal)>>>(d_x, d_partial, n);
    CUDA_CHECK(cudaGetLastError());

    std::vector<GlobalReal> h_partial(nBlocks);
    CUDA_CHECK(cudaMemcpy(h_partial.data(), d_partial, nBlocks * sizeof(GlobalReal), cudaMemcpyDeviceToHost));
    cudaFree(d_partial);

    Real sum = 0.0;
    for (GlobalReal v : h_partial) sum += v;
    return std::sqrt(sum);
}

__global__ void k_set_zero(GlobalReal* x,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)x[i]=0.0;}
__global__ void k_copy(const GlobalReal* src,GlobalReal* dst,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)dst[i]=src[i];}
__global__ void k_axpy(GlobalReal* y,const GlobalReal* x,Real alpha,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]+=alpha*x[i];}
__global__ void k_combine(GlobalReal* y,const GlobalReal* a,const GlobalReal* b,Real alpha,Real beta,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=alpha*a[i]+beta*b[i];}
__global__ void k_residual(const GlobalReal* b,const GlobalReal* y,GlobalReal* r,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)r[i]=b[i]-y[i];}

__global__ void k_spmv_csr(int n,const int* row_ptr,const int* col_idx,const GlobalReal* val,
                           const GlobalReal* x,GlobalReal* y){
    int row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row>=n)return;
    Real sum=0.0;
    for(int k=row_ptr[row];k<row_ptr[row+1];++k) sum+=val[k]*x[col_idx[k]];
    y[row]=sum;
}


// ================= CsrDevice =================
void CsrDevice::freeAll(){if(d_row)cudaFree(d_row);if(d_col)cudaFree(d_col);if(d_val)cudaFree(d_val);if(d_invdiag)cudaFree(d_invdiag);
    d_row=d_col=nullptr;d_val=d_invdiag=nullptr;n=m=nnz=0;}

void CsrDevice::uploadFromEigenA(const Eigen::SparseMatrixR&A){
    freeAll();n=A.rows();m=A.cols();nnz=A.nonZeros();
    std::vector<int> hr(n+1);hr[0]=0;std::vector<int> hc(nnz);std::vector<GlobalReal> hv(nnz),hinv(n,0.0);
    int pos=0;for(int i=0;i<n;++i){
        for(Eigen::SparseMatrixR::InnerIterator it(A,i);it;++it){
            hc[pos]=it.col();
            hv[pos]=it.value();
            if(it.col()==i)
                hinv[i]=1.0/it.value();
            ++pos;}
        hr[i+1]=pos;
    }
    CUDA_CHECK(cudaMalloc(&d_row,(n+1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col,nnz*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_val,nnz*sizeof(GlobalReal)));
    CUDA_CHECK(cudaMalloc(&d_invdiag,n*sizeof(GlobalReal)));
    CUDA_CHECK(cudaMemcpy(d_row,hr.data(),(n+1)*sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col,hc.data(),nnz*sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_val,hv.data(),nnz*sizeof(GlobalReal),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_invdiag,hinv.data(),n*sizeof(GlobalReal),cudaMemcpyHostToDevice));
}

void CsrDevice::uploadFromEigenPR(const Eigen::SparseMatrixR&M){
    freeAll();n=M.rows();m=M.cols();nnz=M.nonZeros();
    std::vector<int> hr(n+1);hr[0]=0;std::vector<int> hc(nnz);std::vector<GlobalReal> hv(nnz);
    int pos=0;for(int i=0;i<n;++i) {
        for(Eigen::SparseMatrixR::InnerIterator it(M,i);it;++it) {
            hc[pos]=it.col();hv[pos]=it.value();++pos;
        }
        hr[i+1]=pos;
    }
    CUDA_CHECK(cudaMalloc(&d_row,(n+1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col,nnz*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_val,nnz*sizeof(GlobalReal)));
    CUDA_CHECK(cudaMemcpy(d_row,hr.data(),(n+1)*sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col,hc.data(),nnz*sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_val,hv.data(),nnz*sizeof(GlobalReal),cudaMemcpyHostToDevice));
    d_invdiag=nullptr;
}

// ================= MAC_MGSolverCUDA =================
MAC_MGSolverCUDA::MAC_MGSolverCUDA()=default;
MAC_MGSolverCUDA::~MAC_MGSolverCUDA(){release();}

bool MAC_MGSolverCUDA::isSymmetric(const SpMat&A){
    for(int k=0;k<A.outerSize();++k)
        for(SpMat::InnerIterator it(A,k);it;++it){
            int i=it.row(),j=it.col();
            if(i!=j && std::abs(A.coeff(i,j)-A.coeff(j,i))>1e-12) return false;
        }
    return true;
}

void MAC_MGSolverCUDA::build_hierarchy(const SpMat&A_fine,int nx,int ny,int nz,int levels){
    release();
    L_=levels;
    nx_.resize(L_);
    ny_.resize(L_);
    nz_.resize(L_);
    As_.resize(L_);
    P_.resize(L_-1);
    R_.resize(L_-1);
    nx_[L_-1]=nx;
    ny_[L_-1]=ny;
    nz_[L_-1]=nz;
    for(int lev=L_-2;lev>=0;--lev) {
        nx_[lev]=std::max(1,(nx_[lev+1]+1)/2);
        ny_[lev]=std::max(1,(ny_[lev+1]+1)/2);
        nz_[lev]=std::max(1,(nz_[lev+1]+1)/2);
    }
    std::vector<SpMat>A_cpu(L_),P_cpu(L_-1),R_cpu(L_-1);
    A_cpu[L_-1] = A_fine;
    for(int lev=L_-2;lev>=0;--lev) {
        P_cpu[lev]=prolongation(lev,lev+1);
        R_cpu[lev]=P_cpu[lev].transpose();
        A_cpu[lev]=R_cpu[lev]*A_cpu[lev+1]*P_cpu[lev];
        A_cpu[lev].makeCompressed();
    }
    printGpuMemoryUsage("A");
    for(int lev=0;lev<L_;++lev) {
        As_[lev].uploadFromEigenA(A_cpu[lev]);
        RZ_INFO("lev: ",lev," nozeros: ",A_cpu[lev].nonZeros()," shape: ",A_cpu[lev].rows());
    }
    printGpuMemoryUsage("A");
    for(int lev=0;lev<L_-1;++lev)
    {
        P_[lev].uploadFromEigenPR(P_cpu[lev]);
        printGpuMemoryUsage("P");
        R_[lev].uploadFromEigenPR(R_cpu[lev]);
        printGpuMemoryUsage("R");
    }
    allocVectors();
}

void MAC_MGSolverCUDA::set_rhs(const Vec&b_fine){
    if(L_==0)throw std::runtime_error("Call build_hierarchy() first.");
    const int n_fin=As_[L_-1].n;
    CUDA_CHECK(cudaMemcpy(d_b_[L_-1],b_fine.data(),n_fin*sizeof(GlobalReal),cudaMemcpyHostToDevice));
    for(int lev=L_-2;lev>=0;--lev){
        int ncoarse=As_[lev].n;int nblocks=(ncoarse+TPB-1)/TPB;
        k_spmv_csr<<<nblocks,TPB>>>(ncoarse,R_[lev].d_row,R_[lev].d_col,R_[lev].d_val,d_b_[lev+1],d_b_[lev]);
        CUDA_CHECK(cudaGetLastError());
    }
}

// void MAC_MGSolverCUDA::VC(Vec&xH,int nu1,int nu2,int cycles){
//     const int n_fin=As_[L_-1].n;
//     if(xH.size()==n_fin)CUDA_CHECK(cudaMemcpy(d_x_[L_-1],xH.data(),n_fin*sizeof(GlobalReal),cudaMemcpyHostToDevice));
//     else setZeroDevice(d_x_[L_-1],n_fin);
//     for(int lev=L_-2;lev>=0;--lev)setZeroDevice(d_x_[lev],As_[lev].n);
//     Real ts=0,to=0;
//     for(int t=0;t<cycles;++t) {
//         auto tic=std::chrono::high_resolution_clock::now();
//         auto st=V_cycle(L_-1,nu1,nu2);
//         auto toc=std::chrono::high_resolution_clock::now();
//         ts+=st.smooth_time;
//         to+=st.other_time;
//     }
//     if(xH.size()!=n_fin)xH.resize(n_fin);
//     CUDA_CHECK(cudaMemcpy(xH.data(),d_x_[L_-1],n_fin*sizeof(GlobalReal),cudaMemcpyDeviceToHost));
// }
void MAC_MGSolverCUDA::VC(Vec& xH, int nu1, int nu2, int cycles) {
    FUNC_TIMER();
    const int n_fin = As_[L_-1].n;
    if (xH.size() == n_fin)
        CUDA_CHECK(cudaMemcpy(d_x_[L_-1], xH.data(), n_fin * sizeof(GlobalReal), cudaMemcpyHostToDevice));
    else
        setZeroDevice(d_x_[L_-1], n_fin);

    for (int lev = L_-2; lev >= 0; --lev)
        setZeroDevice(d_x_[lev], As_[lev].n);

    Real ts = 0.0, to = 0.0;
    Real prev_res = 0.0;

    for (int t = 0; t < cycles; ++t) {
        auto tic = std::chrono::high_resolution_clock::now();
        auto st = V_cycle(L_-1, nu1, nu2);
        auto toc = std::chrono::high_resolution_clock::now();
        ts += st.smooth_time;
        to += st.other_time;

        // ===============================
        //  GPU 上计算残差范数
        // ===============================
        const CsrDevice& A = As_[L_-1];
        const int n = n_fin;
        const dim3 grid((n + TPB - 1) / TPB);

        // y = A * x
        k_spmv_csr<<<grid, TPB>>>(n, A.d_row, A.d_col, A.d_val, d_x_[L_-1], d_y_[L_-1]);
        CUDA_CHECK(cudaGetLastError());

        // r = b - y
        k_residual<<<grid, TPB>>>(d_b_[L_-1], d_y_[L_-1], d_r_[L_-1], n);
        CUDA_CHECK(cudaGetLastError());

        // 计算 ||r||² 和 ||b||² （在 GPU 上）
        Real res_norm = gpuL2Norm(d_r_[L_-1], n);
        Real b_norm   = gpuL2Norm(d_b_[L_-1], n);
        Real rel_res  = res_norm / std::max<Real>(1e-30, b_norm);

        Real smooth_rate = (t == 0) ? 1.0 : (rel_res / prev_res);
        std::cout << "[Cycle " << t << "] Relative residual = " << rel_res
                  << "   Smooth rate = " << smooth_rate << std::endl;
        prev_res = rel_res;
    }

    if (xH.size() != n_fin) xH.resize(n_fin);
    CUDA_CHECK(cudaMemcpy(xH.data(), d_x_[L_-1], n_fin * sizeof(GlobalReal), cudaMemcpyDeviceToHost));

    Real total = ts + to;
    std::cout << "----------------------------------\n";
    std::cout << "Smooth time: " << ts << " s\n";
    std::cout << "Other time : " << to << " s\n";
    std::cout << "Total time : " << total << " s\n";
    std::cout << "----------------------------------\n";
}



// MAC_MGSolverCUDA::TimingStats MAC_MGSolverCUDA::V_cycle(int lev, int nu1, int nu2) {
//          TimingStats stats;
//          auto s_tic = std::chrono::high_resolution_clock::now();
//
//          const CsrDevice& A = As_[lev];
//          const int n = A.n;
//          const dim3 gridA((n + TPB - 1) / TPB);
//
//          // 预平滑：纯设备 Jacobi
//          {
//              CUDA_CHECK(cudaDeviceSynchronize());
//              auto t1 = std::chrono::high_resolution_clock::now();
//              jacobiKernelIter<<<gridA, TPB>>>(n, A.d_row, A.d_col, A.d_val,
//                                                     d_b_[lev], d_x_[lev], A.d_invdiag,
//                                                     0.666, nu1);
//              CUDA_CHECK(cudaGetLastError());
//              CUDA_CHECK(cudaDeviceSynchronize());
//              auto t2 = std::chrono::high_resolution_clock::now();
//              stats.smooth_time += std::chrono::duration<Real>(t2 - t1).count();
//          }
//
//          if (lev == 0) {
//              // 最粗层：多做些迭代（也可换成小型 direct/LDLT）
//              CUDA_CHECK(cudaDeviceSynchronize());
//              auto t1 = std::chrono::high_resolution_clock::now();
//              jacobiKernelIter<<<gridA, TPB>>>(n, A.d_row, A.d_col, A.d_val,
//                                                     d_b_[lev], d_x_[lev], A.d_invdiag,
//                                                     0.7, 150);
//              CUDA_CHECK(cudaGetLastError());
//              CUDA_CHECK(cudaDeviceSynchronize());
//              auto t2 = std::chrono::high_resolution_clock::now();
//              stats.smooth_time += std::chrono::duration<Real>(t2 - t1).count();
//
//              auto s_toc = std::chrono::high_resolution_clock::now();
//              stats.other_time += std::chrono::duration<Real>(s_toc - s_tic).count() - stats.smooth_time;
//              return stats;
//          }
//
//          // 计算 y = A*x
//          k_spmv_csr<<<gridA, TPB>>>(n, A.d_row, A.d_col, A.d_val, d_x_[lev], d_y_[lev]);
//          CUDA_CHECK(cudaGetLastError());
//
//          // r = b - y
//          k_residual<<<gridA, TPB>>>(d_b_[lev], d_y_[lev], d_r_[lev], n);
//          CUDA_CHECK(cudaGetLastError());
//
//          // rH = R * r
//          {
//              const CsrDevice& Rm = R_[lev - 1];
//              const int nc = Rm.n;
//              const dim3 gridR((nc + TPB - 1) / TPB);
//              k_spmv_csr<<<gridR, TPB>>>(nc, Rm.d_row, Rm.d_col, Rm.d_val, d_r_[lev], d_b_[lev - 1]);
//              CUDA_CHECK(cudaGetLastError());
//          }
//
//          // eH 初值 = 0
//          setZeroDevice(d_x_[lev - 1], As_[lev - 1].n);
//
//          // 递归调用
//          TimingStats sub = V_cycle(lev - 1, nu1, nu2);
//          stats.accumulate(sub);
//
//          // 校正：t = P * eH
//          {
//              const CsrDevice& Pm = P_[lev - 1]; // 细 <- 粗
//              const int nf = Pm.n;
//              const dim3 gridP((nf + TPB - 1) / TPB);
//              k_spmv_csr<<<gridP, TPB>>>(nf, Pm.d_row, Pm.d_col, Pm.d_val, d_x_[lev - 1], d_t_[lev]);
//              CUDA_CHECK(cudaGetLastError());
//          }
//          // x += t
//          k_axpy<<<gridA, TPB>>>(d_x_[lev], d_t_[lev], 1.0, n);
//          CUDA_CHECK(cudaGetLastError());
//
//          // 后平滑
//          {
//              CUDA_CHECK(cudaDeviceSynchronize());
//              auto t1 = std::chrono::high_resolution_clock::now();
//              jacobiKernelIter<<<gridA, TPB>>>(n, A.d_row, A.d_col, A.d_val,
//                                                     d_b_[lev], d_x_[lev], A.d_invdiag,
//                                                     0.666, nu2);
//              CUDA_CHECK(cudaGetLastError());
//              CUDA_CHECK(cudaDeviceSynchronize());
//              auto t2 = std::chrono::high_resolution_clock::now();
//              stats.smooth_time += std::chrono::duration<Real>(t2 - t1).count();
//          }
//
//          auto s_toc = std::chrono::high_resolution_clock::now();
//          stats.other_time += std::chrono::duration<Real>(s_toc - s_tic).count() - stats.smooth_time;
//          return stats;
// }
MAC_MGSolverCUDA::TimingStats MAC_MGSolverCUDA::V_cycle(int lev, int nu1, int nu2) {
    TimingStats stats;
    auto s_tic = std::chrono::high_resolution_clock::now();

    const CsrDevice& A = As_[lev];
    const int n = A.n;
    const dim3 gridA((n + TPB - 1) / TPB);

    // -----------------------
    // 预平滑：多次单迭代 Jacobi（ping-pong）
    // 复用 d_y_[lev] 作为 x_new 临时缓冲，之后会被重新计算为 A*x
    // -----------------------
    {
        // x_old 指向当前解，x_new 指向临时缓冲
        GlobalReal* x_old = d_x_[lev];
        GlobalReal* x_new = d_y_[lev];

        auto t1 = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < nu1; ++it) {
            jacobiKernelOnce<<<gridA, TPB>>>(
                n, A.d_row, A.d_col, A.d_val,
                d_b_[lev],
                x_old, x_new,
                A.d_invdiag,
                0.9
            );
            // CUDA_CHECK(cudaGetLastError());
            // 交换指针（ping-pong）
            // GlobalReal* tmp = x_old; x_old = x_new; x_new = tmp;
        }
        // 最后一次的结果在 x_old 指针上；若它不是 d_x_[lev]，拷回去
        if (x_old != d_x_[lev]) {
            CUDA_CHECK(cudaMemcpyAsync(d_x_[lev], x_old, n * sizeof(GlobalReal), cudaMemcpyDeviceToDevice));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t2 = std::chrono::high_resolution_clock::now();
        RZ_INFO("lev: ",lev," time: ",std::chrono::duration<Real>(t2 - t1).count());
        stats.smooth_time += std::chrono::duration<Real>(t2 - t1).count();
    }

    if (lev == 0) {
        // 最粗层：多做一些迭代
        GlobalReal* x_old = d_x_[lev];
        GlobalReal* x_new = d_y_[lev];

        auto t1 = std::chrono::high_resolution_clock::now();
        const int iters = 150;
        for (int it = 0; it < iters; ++it) {
            jacobiKernelOnce<<<gridA, TPB>>>(
                n, A.d_row, A.d_col, A.d_val,
                d_b_[lev],
                x_old, x_new,
                A.d_invdiag,
                0.666
            );
            CUDA_CHECK(cudaGetLastError());
            GlobalReal* tmp = x_old; x_old = x_new; x_new = tmp;
        }
        if (x_old != d_x_[lev]) {
            CUDA_CHECK(cudaMemcpyAsync(d_x_[lev], x_old, n * sizeof(GlobalReal), cudaMemcpyDeviceToDevice));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t2 = std::chrono::high_resolution_clock::now();
        stats.smooth_time += std::chrono::duration<Real>(t2 - t1).count();

        auto s_toc = std::chrono::high_resolution_clock::now();
        stats.other_time += std::chrono::duration<Real>(s_toc - s_tic).count() - stats.smooth_time;
        return stats;
    }

    // ======================
    // 计算 y = A*x
    // ======================
    k_spmv_csr<<<gridA, TPB>>>(n, A.d_row, A.d_col, A.d_val, d_x_[lev], d_y_[lev]);
    CUDA_CHECK(cudaGetLastError());

    // r = b - y
    k_residual<<<gridA, TPB>>>(d_b_[lev], d_y_[lev], d_r_[lev], n);
    CUDA_CHECK(cudaGetLastError());

    // rH = R * r
    {
        const CsrDevice& Rm = R_[lev - 1];
        const int nc = Rm.n;
        const dim3 gridR((nc + TPB - 1) / TPB);
        k_spmv_csr<<<gridR, TPB>>>(nc, Rm.d_row, Rm.d_col, Rm.d_val, d_r_[lev], d_b_[lev - 1]);
        CUDA_CHECK(cudaGetLastError());
    }

    // eH 初值 = 0
    setZeroDevice(d_x_[lev - 1], As_[lev - 1].n);

    // 递归
    {
        TimingStats sub = V_cycle(lev - 1, nu1, nu2);
        stats.accumulate(sub);
    }

    // 校正：t = P * eH
    {
        const CsrDevice& Pm = P_[lev - 1]; // 细 <- 粗
        const int nf = Pm.n;
        const dim3 gridP((nf + TPB - 1) / TPB);
        k_spmv_csr<<<gridP, TPB>>>(nf, Pm.d_row, Pm.d_col, Pm.d_val, d_x_[lev - 1], d_t_[lev]);
        CUDA_CHECK(cudaGetLastError());
    }
    // x += t
    k_axpy<<<gridA, TPB>>>(d_x_[lev], d_t_[lev], 1.0, n);
    CUDA_CHECK(cudaGetLastError());

    // ======================
    // 后平滑（同样用 ping-pong）
    // ======================
    {
        GlobalReal* x_old = d_x_[lev];
        GlobalReal* x_new = d_y_[lev];

        auto t1 = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < nu2; ++it) {
            jacobiKernelOnce<<<gridA, TPB>>>(
                n, A.d_row, A.d_col, A.d_val,
                d_b_[lev],
                x_old, x_new,
                A.d_invdiag,
                0.9
            );
            CUDA_CHECK(cudaGetLastError());
            GlobalReal* tmp = x_old; x_old = x_new; x_new = tmp;
        }
        if (x_old != d_x_[lev]) {
            CUDA_CHECK(cudaMemcpyAsync(d_x_[lev], x_old, n * sizeof(GlobalReal), cudaMemcpyDeviceToDevice));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t2 = std::chrono::high_resolution_clock::now();
        stats.smooth_time += std::chrono::duration<Real>(t2 - t1).count();
    }

    auto s_toc = std::chrono::high_resolution_clock::now();
    stats.other_time += std::chrono::duration<Real>(s_toc - s_tic).count() - stats.smooth_time;
    return stats;
}




Eigen::SparseMatrixR MAC_MGSolverCUDA::prolongation(int coarse_level, int fine_level) {
        const int nxc = nx_[coarse_level], nyc = ny_[coarse_level], nzc = nz_[coarse_level];
        const int nxf = nx_[fine_level],   nyf = ny_[fine_level],   nzf = nz_[fine_level];
        const int Nc = nxc * nyc * nzc;
        const int Nf = nxf * nyf * nzf;

        // 比例（cell-centered）
        const Real rx = static_cast<Real>(nxc) / std::max(1, nxf);
        const Real ry = static_cast<Real>(nyc) / std::max(1, nyf);
        const Real rz = static_cast<Real>(nzc) / std::max(1, nzf);

        std::vector<Eigen::Triplet<GlobalReal>> trip;
        trip.reserve(Nf * 8);

        for (int k = 0; k < nzf; ++k)
            for (int j = 0; j < nyf; ++j)
                for (int i = 0; i < nxf; ++i) {
                    const int row = idx3(i, j, k, nxf, nyf, nzf);

                    // 细到粗的“索引空间”坐标（cell-centered 简化版）
                    Real x = (i ) * rx;
                    Real y = (j ) * ry;
                    Real z = (k ) * rz;

                    int I = static_cast<int>(std::floor(x ));
                    int J = static_cast<int>(std::floor(y ));
                    int K = static_cast<int>(std::floor(z ));

                    Real a = x - I , b = y - J , c = z - K ;


                    if (I < 0) { I = 0; a = 0.0; }
                    if (J < 0) { J = 0; b = 0.0; }
                    if (K < 0) { K = 0; c = 0.0; }
                    if (I >= nxc - 1) { I = nxc - 1; a = 0.0; } // 只有 I
                    if (J >= nyc - 1) { J = nyc - 1; b = 0.0; }
                    if (K >= nzc - 1) { K = nzc - 1; c = 0.0; }

                    Real total_w = 0.0;
                    struct C { int Ic, Jc, Kc; Real w; };
                    C buf[8]; int cnt = 0;

                    for (int dK = 0; dK <= (K < nzc - 1 ? 1 : 0); ++dK)
                        for (int dJ = 0; dJ <= (J < nyc - 1 ? 1 : 0); ++dJ)
                            for (int dI = 0; dI <= (I < nxc - 1 ? 1 : 0); ++dI) {
                                int Ic = I + dI, Jc = J + dJ, Kc = K + dK;
                                Real wx = (dI == 0) ? (1 - a) : a;
                                Real wy = (dJ == 0) ? (1 - b) : b;
                                Real wz = (dK == 0) ? (1 - c) : c;

                                Real w = wx * wy * wz;
                                buf[cnt++] = {Ic, Jc, Kc, w};
                                total_w += w;
                            }

                    if (total_w <= 0) { // 极端兜底
                        buf[0] = {I, J, K, 1.0};
                        cnt = 1; total_w = 1.0;
                    }

                    for (int t = 0; t < cnt; ++t) {
                        Real wnorm = buf[t].w / total_w;
                        // Real wnorm = buf[t].w;
                        trip.emplace_back(row, idx3(buf[t].Ic, buf[t].Jc, buf[t].Kc, nxc, nyc, nzc), wnorm);
                    }
                }
        Eigen::SparseMatrixR Pm(Nf, Nc);
        Pm.setFromTriplets(trip.begin(), trip.end());
        Pm.makeCompressed();
        return Pm;
}

// ============= 内部辅助 =============
void MAC_MGSolverCUDA::setZeroDevice(GlobalReal*p,int n)
{dim3 g((n+TPB-1)/TPB);k_set_zero<<<g,TPB>>>(p,n);}

void MAC_MGSolverCUDA::allocVectors(){
    d_x_.resize(L_);d_b_.resize(L_);d_r_.resize(L_);d_y_.resize(L_);d_t_.resize(L_);
    for(int lev=0;lev<L_;++lev){
        int n=As_[lev].n;
        cudaMalloc(&d_x_[lev],n*sizeof(GlobalReal));
        cudaMalloc(&d_b_[lev],n*sizeof(GlobalReal));
        cudaMalloc(&d_r_[lev],n*sizeof(GlobalReal));
        cudaMalloc(&d_y_[lev],n*sizeof(GlobalReal));
        cudaMalloc(&d_t_[lev],n*sizeof(GlobalReal));
        setZeroDevice(d_x_[lev],n);
        setZeroDevice(d_b_[lev],n);
        setZeroDevice(d_r_[lev],n);
        setZeroDevice(d_y_[lev],n);
        setZeroDevice(d_t_[lev],n);}
}
void MAC_MGSolverCUDA::release(){
    for(auto&A:As_)A.freeAll();for(auto&M:P_)M.freeAll();for(auto&M:R_)M.freeAll();
    for(int lev=0;lev<d_x_.size();++lev){if(d_x_[lev])cudaFree(d_x_[lev]);if(d_b_[lev])cudaFree(d_b_[lev]);
        if(d_r_[lev])cudaFree(d_r_[lev]);if(d_y_[lev])cudaFree(d_y_[lev]);if(d_t_[lev])cudaFree(d_t_[lev]);}
    d_x_.clear();d_b_.clear();d_r_.clear();d_y_.clear();d_t_.clear();As_.clear();P_.clear();R_.clear();L_=0;
}
