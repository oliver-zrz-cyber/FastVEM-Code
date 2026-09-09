#include "dfn.h"
#include "amgcl/amg.hpp"
#include "amgcl/adapter/block_matrix.hpp"

void copyHostToDevice(const GlobalReal* a,GlobalReal* d_a,  size_t n);
void copyDeviceToHost(const GlobalReal* d_a,GlobalReal* a,  size_t n);


void copyDeviceToDevice(
    const GlobalReal* d_src,
    GlobalReal* d_dst,
    size_t n
);



#include "dfn.h"
#include "Eigen/Sparse"

template < class Vec1, class Vec2>
struct IdentityPrecond {
    void apply(const Vec1& d_r, Vec2& d_s, int n) const {
        d_s = d_r;
    }
};

class PCG {
    public:
    typedef GlobalReal scalar_type;

    struct params {
        /// Maximum number of iterations.
        size_t maxiter;

        /// Target relative residual error.
        scalar_type tol;


        /// Verbose output (show iterations and error)
        bool verbose;

        params()
            : maxiter(1000), tol(1e-3),verbose(true)
        {}
    };

    template <class Precond>
    std::tuple<size_t, scalar_type> solve(
                const Precond &P, const GlobalReal* rhs, GlobalReal* x) const {
        FUNC_TIMER();
        copyHostToDevice(rhs,d_rhs,n_);
        copyHostToDevice(x,d_x,n_);
        scalar_type norm_rhs = cuda_norm(d_rhs,n_);
        scalar_type eps = prm.tol;
        scalar_type rho1 = 2 * eps * 1.0;
        scalar_type rho2 = 0.0;
        crs_residual();
        scalar_type res_norm = cuda_norm(d_r,n_);
        size_t iter = 0;
        for(; iter < prm.maxiter && res_norm / norm_rhs > eps; ++iter) {
            P.apply(d_r, d_s,n_);// s = z_iter

            rho2 = rho1;
            rho1 = gpu_dot(d_r,d_s);

            if (iter) // get direction
                axpby(1.0,d_s,rho1/rho2,d_p);
            else
                copyDeviceToDevice(d_s,d_p,n_);

            csr_aAxby(1.0,d_p,0.0,d_q);
            scalar_type alpha = rho1 /gpu_dot(d_p,d_q);

            axpby(alpha,d_p,1.0,d_x);
            axpby(-alpha,d_q,1.0,d_r);

            res_norm = cuda_norm(d_r,n_);
            if (prm.verbose && iter % 5 == 0)
                std::cout << iter << "\t" << res_norm / norm_rhs << std::endl;
        }
        std::cout << iter << "\t" <<  res_norm / norm_rhs << std::endl;
        copyDeviceToHost(d_x,x,n_);
        return std::make_tuple(iter, res_norm / norm_rhs);
    }

    void build(const Eigen::SparseMatrixR& A,size_t max_iter,GlobalReal rel_tol,bool verbose) ;

    int nrows_ = 0;
    int ncols_ = 0;
    int nnz_   = 0;
    int cuda_block;
    int cuda_grid;
    size_t cuda_shmem;
    int*    d_row_ptr_ = nullptr;
    int*    d_col_idx_ = nullptr;
    GlobalReal* d_val_     = nullptr;
    int n_;
    GlobalReal* d_r;
    GlobalReal* d_s;
    GlobalReal* d_p;
    GlobalReal* d_q;
    GlobalReal* d_rhs;
    GlobalReal* d_x;
    params prm;
    GlobalReal gpu_dot(const GlobalReal* d_x, const GlobalReal* d_y) const;
    scalar_type cuda_norm(const GlobalReal* d_x,const int n) const;
    void crs_residual()const;
    void csr_aAxby(GlobalReal alpha,GlobalReal* d_x,GlobalReal beta,GlobalReal *d_y)const;
    void axpby(GlobalReal alpha,GlobalReal* d_x,GlobalReal beta,GlobalReal *d_y)const;
private:
};



