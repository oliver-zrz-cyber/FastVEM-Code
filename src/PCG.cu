#include <cuda_runtime.h>
#include <tuple>
#include "PCG.h"
/**
     * \brief Computes the linear combination of two vectors: y = ax + by.
     *
     * @param y Output vector.
     * @param x Input vector.
     * @param b Input vector.
     * @param a Scalar multiplier for x.
     * @param b Scalar multiplier for y.
     * @param size Size of the vectors.
     */
__global__ void axpby_kernel(GlobalReal alpha,const GlobalReal* x, GlobalReal beta, GlobalReal* y,  size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        y[idx] = alpha * x[idx] + beta * y[idx];
    }
}


/**
 * \brief Computes the residual error: r = rhs - A x.
 *
 * @param r Output residual vector.
 * @param rhs Right-hand side vector.
 * @param A Matrix.
 * @param x Input vector.
 * @param size Size of the vectors/matrix.
 */
__global__ void residual_kernel(GlobalReal* r, const GlobalReal* rhs, const GlobalReal* A, const GlobalReal* x, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        GlobalReal Ax = 0.0;
        // Perform matrix-vector multiplication for A * x
        // Here we assume A is a dense matrix and the rows are stored in a contiguous array.
        for (int i = 0; i < size; ++i) {
            Ax += A[idx * size + i] * x[i];
        }
        r[idx] = rhs[idx] - Ax;  // Compute residual
    }
}


/**
 * \brief Computes the matrix-vector product: y = alpha * A * x + beta * y.
 *
 * @param y Output vector (updated).
 * @param A Matrix.
 * @param x Input vector.
 * @param alpha Scalar multiplier for matrix-vector product.
 * @param beta Scalar multiplier for the vector y.
 * @param size Size of the vectors and matrix.
 */
__global__ void aAxby_kernel(GlobalReal* y, const GlobalReal* A, const GlobalReal* x, GlobalReal alpha, GlobalReal beta, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        GlobalReal temp = 0.0;
        // Perform the matrix-vector multiplication (A * x)
        for (int i = 0; i < size; ++i) {
            temp += A[idx * size + i] * x[i];
        }
        // Update the output vector y
        y[idx] = alpha * temp + beta * y[idx];
    }
}

__device__ __forceinline__
GlobalReal atomicAdd_double(GlobalReal* addr, GlobalReal val) {
    return atomicAdd(addr, val);
}
/**
 * \brief CUDA kernel to compute dot product of two vectors using parallel reduction.
 * This version performs block-level reduction and accumulates results directly on GPU using atomicAdd.
 *
 * \f[ \text{dot}(x, y) = \sum_{i=0}^{n-1} x_i \cdot y_i \f]
 *
 * @param x Input vector.
 * @param y Input vector.
 * @param result Output dot product.
 * @param size Size of the vectors.
 */
__global__ void dot_product_kernel(const GlobalReal* x, const GlobalReal* y, GlobalReal* result, size_t size) {
    extern __shared__ GlobalReal shared_mem[];  // Shared memory for block-wise reduction
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    GlobalReal temp = 0.0;

    // Ensure the index is within bounds
    if (idx < size) {
        temp = x[idx] * y[idx];  // Compute the element-wise product
    }

    // Store intermediate result in shared memory
    shared_mem[tid] = temp;
    __syncthreads();  // Ensure all threads have written to shared memory

    // Perform reduction within the block
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();  // Synchronize threads
    }

    // The first thread writes the result to global memory using atomicAdd
    if (tid == 0) {
        atomicAdd_double(result, shared_mem[0]);
    }
}


/**
 * \brief Computes residual r = rhs - A x using CSR format.
 *
 * @param r        Output residual vector (size n)
 * @param rhs      Right-hand side vector (size n)
 * @param row_ptr  CSR row pointer (size n+1)
 * @param col_idx  CSR column indices (size nnz)
 * @param val      CSR values (size nnz)
 * @param x        Input vector x (size n)
 * @param n        Number of rows
 */
__global__
void csr_residual_kernel(
    GlobalReal* r,
    const GlobalReal* rhs,
    const int* row_ptr,
    const int* col_idx,
    const GlobalReal* val,
    const GlobalReal* x,
    int n
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        GlobalReal Ax = 0.0;

        // CSR SpMV
        for (int jj = row_ptr[row]; jj < row_ptr[row + 1]; ++jj) {
            Ax += val[jj] * x[col_idx[jj]];
        }

        r[row] = rhs[row] - Ax;
    }
}


/**
 * \brief Computes y = alpha * A x + beta * y using CSR format.
 *
 * @param y        Output vector y (updated, size n)
 * @param row_ptr  CSR row pointer (size n+1)
 * @param col_idx  CSR column indices (size nnz)
 * @param val      CSR values (size nnz)
 * @param x        Input vector x (size n)
 * @param alpha    Scalar multiplier for A x
 * @param beta     Scalar multiplier for y
 * @param n        Number of rows
 */
__global__
void csr_aAxby_kernel(
    GlobalReal alpha,
    const int* row_ptr,
    const int* col_idx,
    const GlobalReal* val,
    const GlobalReal* x,
    GlobalReal beta,
    GlobalReal* y,
    int n
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        GlobalReal Ax = 0.0;

        // CSR SpMV
        for (int jj = row_ptr[row]; jj < row_ptr[row + 1]; ++jj) {
            Ax += val[jj] * x[col_idx[jj]];
        }

        y[row] = alpha * Ax + beta * y[row];
    }
}

void copyHostToDevice(const GlobalReal* a,GlobalReal* d_a,  size_t n) {
    cudaError_t err = cudaMemcpy(
        d_a,                   // dst: device
        a,                     // src: host
        n * sizeof(GlobalReal),
        cudaMemcpyHostToDevice
    );

#ifndef NDEBUG
    if (err != cudaSuccess) {
        fprintf(stderr,
            "copyHostToDevice failed: %s\n",
            cudaGetErrorString(err));
    }
#endif
}
void copyDeviceToHost(const GlobalReal* d_a,GlobalReal* a,  size_t n) {
    cudaError_t err = cudaMemcpy(
        a,                     // dst: host
        d_a,                   // src: device
        n * sizeof(GlobalReal),
        cudaMemcpyDeviceToHost
    );

#ifndef NDEBUG
    if (err != cudaSuccess) {
        fprintf(stderr,
            "copyDeviceToHost failed: %s\n",
            cudaGetErrorString(err));
    }
#endif
}

void copyDeviceToDevice(
    const GlobalReal* d_src,
    GlobalReal* d_dst,
    size_t n
) {
    cudaError_t err = cudaMemcpy(
        d_dst,                 // dst: device
        d_src,                 // src: device
        n * sizeof(GlobalReal),
        cudaMemcpyDeviceToDevice
    );

#ifndef NDEBUG
    if (err != cudaSuccess) {
        fprintf(stderr,
            "copyDeviceToDevice failed: %s\n",
            cudaGetErrorString(err));
    }
#endif
}
//
// template<class Precond>
// std::tuple<size_t, PCG::scalar_type> PCG::solve(const Precond &P, const GlobalReal *rhs, GlobalReal *x) const {
//     copyHostToDevice(rhs,d_rhs,n_);
//     copyHostToDevice(x,d_x,n_);
//     scalar_type norm_rhs = cuda_norm(d_rhs,n_);
//     scalar_type eps = prm.tol;
//     scalar_type rho1 = 2 * eps * 1.0;
//     scalar_type rho2 = 0.0;
//     csr_residual_kernel<<<cuda_grid, cuda_block>>>(d_r,d_rhs,
//         d_row_ptr_, d_col_idx_, d_val_,
//         d_x, n_);
//     scalar_type res_norm = cuda_norm(d_r,n_);
//     size_t iter = 0;
//     for(; iter < prm.maxiter && res_norm / norm_rhs > eps; ++iter) {
//         P.apply(d_r, d_s,n_);// s = z_iter
//
//         rho2 = rho1;
//         rho1 = gpu_dot(d_r,d_s);
//
//         if (iter) // get direction
//             axpby_kernel<<<cuda_grid, cuda_block>>>(1.0,d_s,rho1/rho2,d_p,n_);
//         else
//             copyDeviceToDevice(d_s,d_p,n_);
//
//         csr_aAxby_kernel<<<cuda_grid, cuda_block>>>(1.0,d_row_ptr_,d_col_idx_,d_val_,d_p,0.0,d_q,n_);
//         scalar_type alpha = rho1 /gpu_dot(d_p,d_q);
//
//         axpby_kernel<<<cuda_grid, cuda_block>>>(alpha,d_p,1.0,d_x,n_);
//         axpby_kernel<<<cuda_grid, cuda_block>>>(-alpha,d_q,1.0,d_r,n_);
//
//         res_norm = cuda_norm(d_r,n_);
//         if (prm.verbose && iter % 5 == 0)
//             std::cout << iter << "\t" << res_norm / norm_rhs << std::endl;
//     }
//     std::cout << iter << "\t" <<  res_norm / norm_rhs << std::endl;
//     copyDeviceToHost(d_x,x,n_);
//     return std::make_tuple(iter, res_norm / norm_rhs);
//
// }

void PCG::build(const Eigen::SparseMatrixR& A,
    size_t max_iter,GlobalReal rel_tol,bool verbose)
{
    prm.maxiter = max_iter;
    prm.verbose = verbose;
    prm.tol = rel_tol;
    // 1. 基本信息
    n_ = A.rows();
    nrows_ = A.rows();
    ncols_ = A.cols();
    nnz_   = A.nonZeros();
    cuda_block = 256;
    cuda_grid  = (n_ + cuda_block - 1) / cuda_block;
    cuda_shmem = cuda_block * sizeof(GlobalReal);
    // 2. 确保是压缩格式（非常重要）
    //    否则 outerIndexPtr / innerIndexPtr 可能无效
    if (!A.isCompressed()) {
        throw std::runtime_error("Eigen sparse matrix must be compressed before build()");
    }

    // 3. Eigen CSR 数据指针（host）
    const int*    h_row_ptr = A.outerIndexPtr(); // size = nrows + 1
    const int*    h_col_idx = A.innerIndexPtr(); // size = nnz
    const GlobalReal* h_val     = A.valuePtr();      // size = nnz

    // 4. 分配 GPU 内存
    cudaMalloc(&d_row_ptr_, (nrows_ + 1) * sizeof(int));
    cudaMalloc(&d_col_idx_, nnz_ * sizeof(int));
    cudaMalloc(&d_val_,     nnz_ * sizeof(GlobalReal));
    cudaMalloc(&d_x,     n_ * sizeof(GlobalReal));
    cudaMalloc(&d_s,     n_ * sizeof(GlobalReal));
    cudaMalloc(&d_r,     n_ * sizeof(GlobalReal));
    cudaMalloc(&d_q,     n_ * sizeof(GlobalReal));
    cudaMalloc(&d_p,     n_ * sizeof(GlobalReal));
    cudaMalloc(&d_rhs,     n_ * sizeof(GlobalReal));

    // 5. 拷贝到 GPU
    cudaMemcpy(d_row_ptr_, h_row_ptr,
               (nrows_ + 1) * sizeof(int),
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_col_idx_, h_col_idx,
               nnz_ * sizeof(int),
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_val_, h_val,
               nnz_ * sizeof(GlobalReal),
               cudaMemcpyHostToDevice);

}

GlobalReal PCG::gpu_dot(const GlobalReal *d_x, const GlobalReal *d_y)const {
    GlobalReal* d_result;
    cudaMalloc(&d_result, sizeof(GlobalReal));
    cudaMemset(d_result, 0, sizeof(GlobalReal));

    dot_product_kernel<<<cuda_grid, cuda_block, cuda_shmem>>>(
        d_x, d_y, d_result, n_
    );

    GlobalReal result;
    cudaMemcpy(&result, d_result, sizeof(GlobalReal), cudaMemcpyDeviceToHost);

    cudaFree(d_result);
    return result;
}

PCG::scalar_type PCG::cuda_norm(const GlobalReal *d_x,const int n)const {
    GlobalReal norm = gpu_dot(d_x,d_x);
    return std::sqrt(norm);
}

void PCG::crs_residual() const{
    csr_residual_kernel<<<cuda_grid, cuda_block>>>(d_r,d_rhs,
    d_row_ptr_, d_col_idx_, d_val_,
    d_x, n_);
}

void PCG::csr_aAxby(GlobalReal alpha, GlobalReal *d_x, GlobalReal beta, GlobalReal *d_y) const{
    csr_aAxby_kernel<<<cuda_grid, cuda_block>>>(alpha,d_row_ptr_,d_col_idx_,d_val_,d_x,beta,d_y,n_);
}

void PCG::axpby(GlobalReal alpha, GlobalReal *d_x, GlobalReal beta, GlobalReal *d_y) const {
    axpby_kernel<<<cuda_grid, cuda_block>>>(alpha,d_x,beta,d_y,n_);
}
