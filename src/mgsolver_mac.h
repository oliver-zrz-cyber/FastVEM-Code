#pragma once
#include"dfn.h"
#include <Eigen/Sparse>
#include "mgsolver_gpu.h"
#include "smoother_cpu.h"
#define MGDEBUG 1

class MAC_MGSolver {
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

    // ---------- 多重网格层级构建 ----------
    void build_hierarchy(const Eigen::SparseMatrixR& A_fine,
                         int nx_fine, int ny_fine, int nz_fine,
                         int levels) {
        if (levels < 1) throw std::runtime_error("levels must >= 1");

        L = levels;
        nx.resize(L); ny.resize(L); nz.resize(L);
        As.resize(L); P.resize(L - 1); R.resize(L - 1);
        smoothers.resize(L);
        nx[L-1] = nx_fine; ny[L-1] = ny_fine; nz[L-1] = nz_fine;
        for (int lev = L - 2; lev >= 0; --lev) {
            nx[lev] = std::max(1, (nx[lev + 1] + 1) / 2);
            ny[lev] = std::max(1, (ny[lev + 1] + 1) / 2);
            nz[lev] = std::max(1, (nz[lev + 1] + 1) / 2);
        }

        // 最细层矩阵
        As[L-1] = A_fine;

        // 逐层构建 P,R,A
        for (int lev = L - 2; lev >= 0; --lev) {
            P[lev] = prolongation(lev, lev + 1);
            R[lev] = P[lev].transpose(); // ★ 确保对称一致性
            As[lev] = R[lev] * As[lev + 1] * P[lev];
            As[lev].makeCompressed();
        }
        for (int lev = L - 1; lev >= 0; --lev) {
            smoothers[lev].build(As[lev]);
        }

    }

    // ---------- 右端项限制 ----------
    void set_rhs(const Eigen::VectorXr& b_fine) {
        FUNC_TIMER();
        if (As.empty()) throw std::runtime_error("Call build_hierarchy() first.");
        bs.resize(L);
        bs[L-1] = b_fine;
        for (int lev = L - 2; lev >= 0; --lev)
            bs[lev] = (R[lev] * bs[lev + 1].template cast<GlobalReal>()).template cast<Real>();
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
        auto tic = std::chrono::high_resolution_clock::now();
        auto s1 = std::chrono::high_resolution_clock::now();
        smoothers[lev].smooth(b, x, nu1);
        auto s2 = std::chrono::high_resolution_clock::now();
        stats.smooth_time += std::chrono::duration<Real>(s2 - s1).count();

        // V_cycle 里最粗层
        if (lev == 0) {
            smoothers[lev].smooth(b, x, 100);
            // x = amgclSolveCPU(As[0],b,1e-3,-1,1000);
            // Eigen::SimplicialLDLT<Eigen::SparseMatrixR> ldlt;
            // ldlt.compute(As[0]);
            // x = ldlt.solve(b);
            if (MGDEBUG) {
                // ---------- 收敛报告 ----------
                Eigen::VectorXr r = (As[0] * x.template cast<GlobalReal>()).template cast<Real>() - b;
                Real rel_res = r.norm() / std::max(1e-30, b.norm());
                std::cout << "[Coarse level solve] residual = " << rel_res << std::endl;
                if (rel_res < 1e-6)
                    std::cout << "✅ Coarse level converged." << std::endl;
                else
                    std::cout << "⚠️ Coarse level not fully converged." << std::endl;
            }
            return stats;
        }



        // 计算残差 r = b - A*x
        Eigen::VectorXr r = b - (As[lev] * x.template cast<GlobalReal>()).template cast<Real>();
        // r.array() -= r.mean(); // ??
        // 限制到粗层
        Eigen::VectorXr rH = (R[lev - 1] * r.template cast<GlobalReal>()).template cast<Real>();
        Eigen::VectorXr eH = Eigen::VectorXr::Zero(As[lev - 1].rows());

        TimingStats sub = V_cycle(lev - 1, eH, rH, nu1, nu2);
        stats.accumulate(sub);
        // 校正并后平滑
        x += (P[lev - 1] * eH.template cast<GlobalReal>()).template cast<Real>();

        auto s5 = std::chrono::high_resolution_clock::now();
        smoothers[lev].smooth(b, x, nu2);
        auto s6 = std::chrono::high_resolution_clock::now();
        stats.smooth_time += std::chrono::duration<Real>(s6 - s5).count();
        auto toc = std::chrono::high_resolution_clock::now();
        stats.other_time += std::chrono::duration<Real>(toc - tic).count() - stats.smooth_time;
        if (MGDEBUG) {
            r = (As[lev] * x.template cast<GlobalReal>()).template cast<Real>() - b;
            Real rel_res = r.norm() / std::max(1e-30, b.norm());
            std::cout << "level: "<<lev<<" res: " << rel_res << std::endl;
        }
        return stats;
    }

    void FMG(Eigen::VectorXr& xH,int nu1 = 5, int nu2 = 5, int gamma = 1, int vcycles_each_level = 3) {
        FUNC_TIMER();
        if (As.empty() || bs.empty())
            throw std::runtime_error("Call build_hierarchy()+set_rhs() before FMG().");
        if (xH.size()==0)
            xH = Eigen::VectorXr::Zero(As[0].rows());
        smoothers[0].smooth(bs[0], xH, 100); // 初始粗解
        Real total_smooth = 0.0, total_other = 0.0;

        for (int lev = 1; lev < L; ++lev) {
            Eigen::VectorXr xFine = (P[lev - 1] * xH.template cast<GlobalReal>()).template cast<Real>();
            for (int t = 0; t < gamma * vcycles_each_level; ++t) {
                TimingStats stats = V_cycle(lev, xFine, bs[lev], nu1, nu2);
                total_smooth += stats.smooth_time;
                total_other  += stats.other_time;
            }
            xH = std::move(xFine);
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

    void VC(Eigen::VectorXr& xH,int nu1 = 5, int nu2 =5, int cycles = 4) {
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
private:
    int L = 0;
    std::vector<int> nx, ny, nz;
    std::vector<Eigen::SparseMatrixR> As, P, R;
    std::vector<DampedJacobiSmoother> smoothers;
    std::vector<Eigen::VectorXr> bs;

    static inline int idx3(int i, int j, int k, int nx, int ny, int nz) {
        return (k * ny + j) * nx + i;
    }

    // ---------- 限制算子 ----------
    Eigen::SparseMatrixR
    restriction(int fine_level, int coarse_level) {
        const int nxf = nx[fine_level], nyf = ny[fine_level], nzf = nz[fine_level];
        const int nxc = nx[coarse_level], nyc = ny[coarse_level], nzc = nz[coarse_level];
        const int Nc = nxc * nyc * nzc;

        std::vector<Eigen::Triplet<GlobalReal>> trip;
        trip.reserve(Nc * 8);

        for (int K = 0; K < nzc; ++K)
            for (int J = 0; J < nyc; ++J)
                for (int I = 0; I < nxc; ++I) {
                    int row = idx3(I, J, K, nxc, nyc, nzc);
                    int i0 = 2 * I, j0 = 2 * J, k0 = 2 * K;
                    for (int dk = 0; dk < 2; ++dk)
                        for (int dj = 0; dj < 2; ++dj)
                            for (int di = 0; di < 2; ++di) {
                                int iF = i0 + di, jF = j0 + dj, kF = k0 + dk;
                                if (iF < nxf && jF < nyf && kF < nzf)
                                    trip.emplace_back(row, idx3(iF, jF, kF, nxf, nyf, nzf), 1.0);
                            }
                }

        Eigen::SparseMatrixR Rm(nxc * nyc * nzc, nxf * nyf * nzf);
        Rm.setFromTriplets(trip.begin(), trip.end());
        Rm.makeCompressed();
        return Rm;
    }

    Eigen::SparseMatrixR
    prolongation(int coarse_level, int fine_level) {
        const int nxc = nx[coarse_level], nyc = ny[coarse_level], nzc = nz[coarse_level];
        const int nxf = nx[fine_level],   nyf = ny[fine_level],   nzf = nz[fine_level];
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
                    // Real x = (i + 0.5) * rx;
                    // Real y = (j + 0.5) * ry;
                    // Real z = (k + 0.5) * rz;
                    //
                    // int I = static_cast<int>(std::floor(x - 0.5));
                    // int J = static_cast<int>(std::floor(y - 0.5));
                    // int K = static_cast<int>(std::floor(z - 0.5));
                    //
                    // Real a = x - I - 0.5, b = y - J - 0.5, c = z - K - 0.5;

                    // Real la = (1-a)*(1-a);
                    // Real lb = (1-b)*(1-b);
                    // Real lc = (1-c)*(1-c);
                    // Real ra = a*a;
                    // Real rb = b*b;
                    // Real rc = c*c;
                    // // 边界 clamp + a,b,c 修正，避免 I+1 越界导致漏权
                    // if (I < 0) { I = 0; la = a*a;ra =0.0; }
                    // if (J < 0) { J = 0; lb = b*b;rb=0.0; }
                    // if (K < 0) { K = 0; lc = c*c;rc=0.0; }
                    // if (I >= nxc - 1) { I = nxc - 1; ra = 0.0; } // 只有 I
                    // if (J >= nyc - 1) { J = nyc - 1; rb = 0.0; }
                    // if (K >= nzc - 1) { K = nzc - 1; rc = 0.0; }

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
                                // Real wx = (dI == 0) ? la : ra;
                                // Real wy = (dJ == 0) ? lb : rb;
                                // Real wz = (dK == 0) ? lc : rc;
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

};
// ---------- V-cycle ----------
// void V_cycle(int lev, Eigen::VectorXr& x, const Eigen::VectorXr& b,
//              int nu1 = 2, int nu2 = 2) {
//     FUNC_TIMER();
//     smoothers[lev].smooth(b, x, nu1);
//
//     if (lev == 0) { // 最粗层多平滑几次
//         smoothers[lev].smooth( b, x, 50);
//         return;
//     }
//
//     // 计算残差 r = b - A*x
//     Eigen::VectorXr r = b - As[lev] * x;
//
//     // 限制到粗层
//     Eigen::VectorXr rH = R[lev - 1] * r;
//     Eigen::VectorXr eH = Eigen::VectorXr::Zero(As[lev - 1].rows());
//
//     V_cycle(lev - 1, eH, rH, nu1, nu2);
//
//     // 校正并后平滑
//     x += P[lev - 1] * eH;
//     smoothers[lev].smooth(b, x, nu2);
// }
