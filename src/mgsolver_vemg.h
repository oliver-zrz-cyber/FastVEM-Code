#pragma once
// #include "smoother_cpu.h"
#include "amgcl_smoother.h"
#include "Visualization.h"
#include "vem3d.h"
#include "vex_pcg.h"
// 最粗糙层次用 Ab_solver
class VEM_GMGSolver {
public:
    using Backend   = VEX_BACKEND;
    typename Backend::params bprm_;
    using Matrix       = Backend::matrix;
    std::vector<std::shared_ptr<Matrix>> As_gpu;
    bool direct_coarse = true;
    // Configure before init_for_pcg(); fixed during all subsequent GMRES iterations.
    int coarse_smoothing_iterations = 30;
    int pre_smoothing_iterations = 2;
    int post_smoothing_iterations = 2;
    bool diagnostic_checks = false;
    std::shared_ptr< typename Backend::direct_solver > coarse_solver;
    int vem_k;
    Real dt=0.0;
    std::vector<VEM_Domain_3d> mGrids;
    VEM_GMGSolver(int k):vem_k(k){}

    void load_mGrids(std::vector<std::string> domains);

    void load_mGrids(MiniLcc& mlcc,std::vector<std::vector<int>>&lods) ;

    void load_mGrids(std::vector<MiniLcc>&mlcc,std::vector<std::vector<int>> lods);
    void build_hierarchy() ;
    void checkDD() ;
    void gmg_solver_k1(Real dt);
    void gmg_solver_k2(Real dt);

    void apply(const GlobalReal* d_rhs,GlobalReal* d_x,int n);

    void apply(const VEX_VECTOR& d_rhs,VEX_VECTOR& d_x,int n)const;
    // AMGCL iterative-solver preconditioner interface.
    void apply(const VEX_VECTOR& d_rhs, VEX_VECTOR& d_x) const {
        apply(d_rhs, d_x, static_cast<int>(d_rhs.size()));
    }

    void build_hierarchy_k1();

    void build_hierarchy_k2();



    void set_rhs(const Eigen::VectorXr& b_fine);

    struct TimingStats {
        Real smooth_time = 0.0;
        Real other_time  = 0.0;
        void accumulate(const TimingStats& sub) {
            smooth_time += sub.smooth_time;
            other_time  += sub.other_time;
        }
    };


    TimingStats V_cycle(int lev, Eigen::VectorXr& x, const Eigen::VectorXr& b,
                 int nu1 = 5, int nu2 = 5) const;

    void VC(Eigen::VectorXr& xH,int nu1 = 3, int nu2 = 3, int cycles = 5) ;

    void VC(const Eigen::VectorXr& input_b,
        Eigen::VectorXr& xH,int nu1 = 2, int nu2 = 2, int cycles = 1) const;

    void init_for_pcg();

    const Eigen::SparseMatrixR& finest_A() const { return As.back(); }

    void cycle(int lev, VEX_VECTOR& d_x, const VEX_VECTOR& d_rhs,
               int nu1 = 2, int nu2 = 2) const ;

    void VC(const VEX_VECTOR& d_rhs,VEX_VECTOR& d_x,int nu1 = 2, int nu2 = 2, int cycles = 1) const{
        for (int t = 0; t < cycles; ++t) {
            cycle(L - 1, d_x, d_rhs, nu1, nu2);
        }
    }

    std::vector<Eigen::SparseMatrixR> Ps,Rs;
    std::vector<std::shared_ptr<Matrix>> Ps_gpu,Rs_gpu;
private:
    // Per-instance scratch buffers: solve/apply on one instance is not reentrant.
    struct LevelWorkspace {
        std::shared_ptr<VEX_VECTOR> residual, coarse_rhs, coarse_error;
    };
    std::vector<LevelWorkspace> workspace_;
    bool gpu_direct_coarse_ = false;
    int gpu_coarse_iterations_ = 30;
    int L = 0;
    GlobalReal* pcg_x;
    GlobalReal* pcg_rhs;

    std::vector<Eigen::SparseMatrixR> As;
    // std::vector<AMGCLJacobiSmoother> gs_smoothers;// not used
    std::vector<AMGCLChebyshevSmoother_GPU> chev_smoothers;
    // std::vector<AMGCLSpai0Smoother_GPU> spai_smoothers;
    // std::vector<AMGCLChebyshevBlockSmoother> chev_block_smoothers;
    // std::vector<AMGCLJacobiSmoother> jacobi_smoothers;
    // std::vector<BlockJacobiSmoother> block_jacobi_smoothers;
    AMGCLContext l0_solver;
    Eigen::VectorXr b;
    std::vector<VEM_Monomial_3d> monomials_3d;
    Eigen::SparseMatrixR
               restriction_matrix_k2;

    void restriction_k2(int ll,int hl,const Eigen::VectorXr& fine_p,Eigen::VectorXr& rH) ;

    Eigen::SparseMatrixR prolongation_k1(int ll,int hl);


    Eigen::SparseMatrixR prolongation_k2(int ll,int hl);

    Eigen::SparseMatrixR prolongation_k2_sparse(int ll,int hl);


     Eigen::SparseMatrixR prolongation_k2_sparse_fast(int ll,int hl);



    Eigen::SparseMatrixR k1_prolongation_k2(int ll,int hl);


    void assign_points_to_elements(std::vector<std::vector<int>>& contained_indices,
        std::vector<_3d_vec>&points,const VEM_Domain_3d& vem_domain);

    bool isDiagonalDominant(const Eigen::SparseMatrixR& A);
    void this_save_points_to_xyz(const std::vector<_3d_vec>& points,
        const std::string& filename) const{
        std::ofstream xyz_file(filename);
        if (!xyz_file.is_open()) {
            RZ_FATAL("Failed to open file for writing.");
        }

        // 写入每个点的坐标
        for (const auto& point : points) {
            xyz_file << point.vx << " " << point.vy << " " << point.vz << "\n";
        }

        xyz_file.close();
    }
    bool point_in_convex_hull(const _2d_vec &p, const std::vector<_2d_vec> &points) const{
        const Real eps = 1e-12;
        // 判断点是否在凸包的边的右侧
        // 计算叉积的方向
        for (size_t i = 0; i < points.size(); ++i) {
            const _2d_vec &a = points[i];
            const _2d_vec &b = points[(i + 1) % points.size()]; // 通过取模实现循环连接
            // 计算向量 ab 和 ap 的叉积
            Real cross_product = (b.vx - a.vx) * (p.vy - a.vy) - (b.vy - a.vy) * (p.vx - a.vx);

            // 如果叉积大于零，说明点 p 在边 ab 的左侧；如果小于零，说明点 p 在右侧
            if (cross_product < -eps) {
                return false; // p 在外部
            }
        }
        return true; // p 在凸包内部或边上
    }
    bool on_coarse_edge(const _3d_vec& pos,const int row,std::vector<Eigen::Triplet<GlobalReal>>& trip,
        const std::vector<int>& contained_coarse_cells,int ll,std::set<int>& active_dofs) ;

    Real VEM2d_Prolongation(const int row,const int local_poly_id,const VEM_Element_3d& element_3d,
        std::vector<Eigen::Triplet<GlobalReal>>& no_weighted_trips,const _3d_vec& pos,
        VEM_Domain_3d& coarse_domain,
        bool is_face_dof = false) ;

};
