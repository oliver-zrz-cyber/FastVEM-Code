#pragma once
#include "dfn.h"
#include <Eigen/Sparse>
#include "mgsolver_gpu.h"
#include "AbSolver.h"
// #include "smoother_cpu.h"
#define MGDEBUG 1
// 最粗糙层次用 Ab_solver
class VEM_PMGSolver {
public:
    // ---------- 对称性检查 ----------
    bool isSymmetric(const Eigen::SparseMatrixR& A) {
        for (int k = 0; k < A.outerSize(); ++k)
            for (Eigen::SparseMatrixR::InnerIterator it(A, k); it; ++it) {
                int i = it.row(), j = it.col();
                if (i != j && std::abs(A.coeff(i, j) - A.coeff(j, i)) > 1e-12)
                    return false;
            }
        return true;
    }
    void build_hierarchy(const Eigen::SparseMatrixR& A_k2
    ,const Eigen::SparseMatrixR& A_k1,int V_num, int E_num, int F_num, int P_num,
        const std::vector<std::pair<int,int>>& edges) {
        L = 2;
        As.resize(L);
        As[L-1] = A_k2;
        smoother.build(A_k2);
        P = prolongation(V_num,E_num,F_num,P_num,edges);
        R = P.transpose();
        As[0] = A_k1;
        amgclInit(k1_solver,As[0],1e-3,1000);
        RZ_ASSERT(P.rows()==As[L-1].rows() && P.cols()==As[0].rows(), "P size mismatch");
        // =====================================
        // ⬇️ 新增：检查 Galerkin 一致性
        // =====================================
        Eigen::SparseMatrixR A_galerkin = R * A_k2 * P;

        Eigen::MatrixXr diff = (A_galerkin - A_k1).toDense().template cast<Real>();
        Real norm_diff = diff.norm();
        Real norm_ref  = A_k1.norm();
        Real rel_error = norm_diff / std::max<Real>(1e-16, norm_ref);

        std::cout << "[Check] ‖R*A_f*P - A_c‖ / ‖A_c‖ = " << rel_error << std::endl;

#if MGDEBUG
        if (rel_error > 1e-3)
            std::cerr << "⚠️ Warning: coarse and fine operators differ significantly! "
                         "(rel_error = " << rel_error << ")\n";
        else
            std::cout << "✅ Galerkin consistency OK.\n";
#endif
    }


    // ---------- 右端项限制 ----------
    void set_rhs(const Eigen::VectorXr& b_k2,
        const Eigen::VectorXr& b_k1) {
        FUNC_TIMER();
        bs.resize(L);
        bs[L-1] = b_k2;
        bs[L-2] = b_k1;
    }

    struct TimingStats {
        Real smooth_time = 0.0;
        Real other_time  = 0.0;
        void accumulate(const TimingStats& sub) {
            smooth_time += sub.smooth_time;
            other_time  += sub.other_time;
        }
    };


    TimingStats V_cycle(int lev, Eigen::VectorXr& x, const Eigen::VectorXr& b,
                 int nu1 = 2, int nu2 = 2) {
        TimingStats stats;
        if (lev == 0) {// 用 amgcl
            x= amgclSolve(k1_solver,b);
            if (MGDEBUG) {
                // ---------- 收敛报告 ----------
                Eigen::VectorXr r = (As[0] * x.template cast<GlobalReal>()).template cast<Real>() - b;
                Real rel_res = r.norm() / std::max<Real>(1e-30, b.norm());
                std::cout << "[Coarse level solve] residual = " << rel_res << std::endl;
                if (rel_res < 1e-6)
                    std::cout << "✅ Coarse level converged." << std::endl;
                else
                    std::cout << "⚠️ Coarse level not fully converged." << std::endl;
            }
            return stats;
        }
        auto tic = std::chrono::high_resolution_clock::now();
        auto s1 = std::chrono::high_resolution_clock::now();
        smoother.smooth(b, x, nu1);
        auto s2 = std::chrono::high_resolution_clock::now();
        stats.smooth_time += std::chrono::duration<Real>(s2 - s1).count();

        // // 计算残差 r = b - A*x
        // Eigen::VectorXr r = b - As[lev] * x;
        // Eigen::VectorXr rH = R * r;
        //
        // Eigen::VectorXr eH = Eigen::VectorXr::Zero(As[lev - 1].rows());
        // TimingStats sub = V_cycle(lev - 1, eH, rH, nu1, nu2);
        // stats.accumulate(sub);
        // // 校正并后平滑
        // x += P * eH;

        auto s5 = std::chrono::high_resolution_clock::now();
        smoother.smooth(b, x, nu2);
        auto s6 = std::chrono::high_resolution_clock::now();
        stats.smooth_time += std::chrono::duration<Real>(s6 - s5).count();
        auto toc = std::chrono::high_resolution_clock::now();
        stats.other_time += std::chrono::duration<Real>(toc - tic).count() - stats.smooth_time;
        // if (MGDEBUG) {
        //     r = As[lev] * x - b;
        //     Real rel_res = r.norm() / std::max(1e-30, b.norm());
        //     std::cout << "level: "<<lev<<" res: " << rel_res << std::endl;
        // }
        return stats;
    }

    void VC(Eigen::VectorXr& xH,int nu1 = 3000, int nu2 = 0, int cycles = 1) {
        FUNC_TIMER();
        if (As.empty() || bs.empty())
            throw std::runtime_error("Call build_hierarchy()+set_rhs() before FMG().");
        if (xH.size()==0)
            xH = Eigen::VectorXr::Zero(As[L-1].rows());
        Real total_smooth = 0.0, total_other = 0.0;
        for (int t = 0; t < cycles; ++t) {
            RZ_INFO("Vcycle: ",t,"-------------------------");
            TimingStats stats = V_cycle(L - 1, xH, bs[L - 1], nu1, nu2);
            total_smooth += stats.smooth_time;
            total_other  += stats.other_time;
        }

        Real total = total_smooth + total_other;
        Real p1 = 100.0 * total_smooth / total;
        Real p2 = 100.0 * total_other / total;

        std::cout << "\n===== ⏱️ FMG Timing Report =====\n";
        std::cout << "Smooth time : " << total_smooth << " s (" << p1 << "%)\n";
        std::cout << "Other parts : " << total_other  << " s (" << p2 << "%)\n";
        std::cout << "Total time  : " << total        << " s\n";
        std::cout << "================================\n";
        // ============================================================
        // 新增：相对残差计算（在最细层 L-1）
        // ============================================================
        const auto& A_fine = As[L - 1];
        const auto& b_fine = bs[L - 1];
        Eigen::VectorXr r = (A_fine * xH.template cast<GlobalReal>()).template cast<Real>() - b_fine;
        Real rel_err = r.norm() / b_fine.norm();

        std::cout << "Relative residual: " << rel_err << "\n";
        std::cout << "================================\n";
    }
    Eigen::SparseMatrixR P,R;
private:
    int L = 0;
    std::vector<int> nx, ny, nz;
    std::vector<Eigen::SparseMatrixR> As;

    DampedJacobiSmootherGPU smoother;
    AMGCLContext k1_solver;
    std::vector<Eigen::VectorXr> bs;

    static inline int idx3(int i, int j, int k, int nx, int ny, int nz) {
        return (k * ny + j) * nx + i;
    }

    Eigen::SparseMatrixR prolongation(int V_num, int E_num, int F_num, int P_num,
    const std::vector<std::pair<int,int>>& edges)
    {
        const int Nc = V_num;                // 粗层 DOF 数
        const int Nf = V_num + E_num + F_num + P_num;        // 细层 DOF 数
        std::vector<Eigen::Triplet<GlobalReal>> trip;
        trip.reserve(V_num + 2*E_num); // 顶点拷贝 + 每边中点插值

        // 1️⃣ 顶点 DOF：直接拷贝
        for (int v = 0; v < V_num; ++v)
            trip.emplace_back(v, v, 1.0);

        // 2️⃣ 边 DOF：每条边新增一个中点 = 两端点的平均
        for (int e = 0; e < E_num; ++e) {
            const int va = edges[e].first;
            const int vb = edges[e].second;
            const int row = V_num + e; // 细层边中点 DOF 的全局行号
            trip.emplace_back(row, va, 0.5);
            trip.emplace_back(row, vb, 0.5);
        }

        Eigen::SparseMatrixR ans(Nf, Nc);
        ans.setFromTriplets(trip.begin(), trip.end());
        ans.makeCompressed();
        return ans;
    }
};
