#include <stdexcept>
#include"mgsolver_vemg.h"
#include"mini_lcc.h"
#include "config.h"
#include "PCG.h"
#include "vex_pcg.h"
#include <fstream>
#include <sstream>

namespace {
size_t current_rss_kb() {
    std::ifstream status_file("/proc/self/status");
    std::string line;
    while (std::getline(status_file, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            std::istringstream iss(line);
            std::string key, unit;
            size_t value = 0;
            iss >> key >> value >> unit;
            return value;
        }
    }
    return 0;
}

void print_load_mGrids_memory(const char *stage, size_t rss_kb, size_t before_kb) {
    const Real rss_mb = static_cast<Real>(rss_kb) / 1024.0;
    const Real delta_mb = static_cast<Real>(rss_kb) / 1024.0
                          - static_cast<Real>(before_kb) / 1024.0;
    std::cout << "[load_mGrids memory] " << stage
              << " RSS = " << rss_mb << " MB"
              << ", delta = " << delta_mb << " MB\n";
}
}

void VEM_GMGSolver::load_mGrids(std::vector<std::string> domains) {
    const size_t rss_before_kb = current_rss_kb();
    print_load_mGrids_memory("before", rss_before_kb, rss_before_kb);
     {
        mGrids.resize(domains.size());
        for (int k=0;k<domains.size();++k) {
            mGrids[k].read_from_file(domains[k]);
            if (k == domains.size()-1) {
                mGrids[k].save_boundary_to_obj();
                mGrids[k].build_AABB();
            }
            mGrids[k].init();
            mGrids[k].preprocess_polygons();
            mGrids[k].generate_elements();
            mGrids[k].assemble_global_stiffness_matrix_k2();
            mGrids[k].assemble_global_stiffness_matrix_k1();
            // estimate_condition_number(mGrids[k].global_stiffness_matrix_k2.
            //     topLeftCorner(mGrids[k].points.size()+mGrids[k].edges.size()+mGrids[k].polygons.size(),
            //         mGrids[k].points.size()+mGrids[k].edges.size()+mGrids[k].polygons.size()));
            // estimate_condition_number(mGrids[k].global_stiffness_matrix_k2.
            //     bottomRightCorner(mGrids[k].polyhedrons.size(),mGrids[k].polyhedrons.size()));
            // estimate_condition_number(mGrids[k].global_stiffness_matrix_k1);

            //-------------------------------
            mGrids[k].grid_res = 100; // 可按场景调整
            mGrids[k].grid_to_elems.clear();
            mGrids[k].build_element_bboxes();
            mGrids[k].link_ele_with_uniform_grid_3d();
            //-------------------------------
            // k2 max dofs
            // Real min_dis = 1.0;
            // for (int i=0;i<mGrids[k].vem_elements.size();++i) {
            //     for (int j=i+1;j<mGrids[k].vem_elements.size();++j) {
            //         min_dis = std::min(min_dis,(mGrids[k].vem_elements[i].centroid-mGrids[k].vem_elements[j].centroid).norm());
            //     }
            // }
            // RZ_INFO(min_dis);
            // exit(0);
            int dofs = 0;
            for (const auto&element:mGrids[k].vem_elements) {
                int element_dof = 1+element.vertex_count()+element.edge_count()+element.polygons.size();
                dofs+=element_dof*element_dof;
            }
            RZ_INFO("LOD :",k," max dofs: ",dofs);
        }

        monomials_3d = {
            VEM_Monomial_3d(0, 0, 0),  // 1
            VEM_Monomial_3d(1, 0, 0),  // x
            VEM_Monomial_3d(0, 1, 0),  // y
            VEM_Monomial_3d(0, 0, 1),  // z
            VEM_Monomial_3d(2, 0, 0),  // x^2
            VEM_Monomial_3d(1, 1, 0),  // xy
            VEM_Monomial_3d(1, 0, 1),  // xz
            VEM_Monomial_3d(0, 2, 0),  // y^2
            VEM_Monomial_3d(0, 1, 1),  // yz
            VEM_Monomial_3d(0, 0, 2)   // z^2
        };

    }
    print_load_mGrids_memory("after", current_rss_kb(), rss_before_kb);
}

void VEM_GMGSolver::load_mGrids(MiniLcc &mlcc, std::vector<std::vector<int>> &lods) {
        FUNC_MEMORY_TIMER();
        FUNC_TIMER();
        std::vector<Trancer>tracers;
        // if (mGrids.size())
            // tracers = mGrids.back().tracers;
        // mGrids.clear();
        if (mGrids.size() == 0)
            mGrids.resize(lods.size());
        mlcc.copy_finest();
        for (int k=0;k<lods.size();++k) {
            mlcc.cvxs_to_finest_v2();
            mlcc.lod_repair_v2(lods[k]);
            // mlcc.save_lcc_compact_points(std::to_string(k)+"lod.vtu",lods[k]);
            mGrids[k].load_from_Minilcc_lod(mlcc,lods[k]);
            mGrids[k].init();
            mGrids[k].preprocess_polygons();
            if (k == lods.size() - 1)
                mGrids[k].generate_elements(false);
            else{
                mGrids[k].generate_elements();
            }
            if (k == lods.size() - 1)
                mGrids[k].assemble_global_stiffness_matrix_k2();
            // mGrids[k].assemble_global_stiffness_matrix_k1();
            for (auto&elem:mGrids[k].vem_elements)elem.clear_stiff_k2();
            //-------------------------------
            mGrids[k].grid_res = 100; // 可按场景调整
            mGrids[k].grid_to_elems.clear();
            mGrids[k].build_element_bboxes();
            mGrids[k].link_ele_with_uniform_grid_3d();
        }
        monomials_3d = {
            VEM_Monomial_3d(0, 0, 0),  // 1
            VEM_Monomial_3d(1, 0, 0),  // x
            VEM_Monomial_3d(0, 1, 0),  // y
            VEM_Monomial_3d(0, 0, 1),  // z
            VEM_Monomial_3d(2, 0, 0),  // x^2
            VEM_Monomial_3d(1, 1, 0),  // xy
            VEM_Monomial_3d(1, 0, 1),  // xz
            VEM_Monomial_3d(0, 2, 0),  // y^2
            VEM_Monomial_3d(0, 1, 1),  // yz
            VEM_Monomial_3d(0, 0, 2)   // z^2
        };
    mlcc.clear();
    }

void VEM_GMGSolver::load_mGrids(std::vector<MiniLcc> &mlcc, std::vector<std::vector<int>> lods) {
        const size_t rss_before_kb = current_rss_kb();
        print_load_mGrids_memory("before", rss_before_kb, rss_before_kb);
        mGrids.resize(lods.size());
        for (int k=0;k<lods.size();++k) {
            // mGrids[k].load_from_Minilcc(domains[k]);
            if (k == lods.size()-1) {
                mGrids[k].save_boundary_to_obj();
                mGrids[k].build_AABB();
            }
            mGrids[k].init();
            mGrids[k].preprocess_polygons();
            mGrids[k].generate_elements();
            mGrids[k].assemble_global_stiffness_matrix_k2();
            mGrids[k].assemble_global_stiffness_matrix_k1();
            // estimate_condition_number(mGrids[k].global_stiffness_matrix_k2.
            //     topLeftCorner(mGrids[k].points.size()+mGrids[k].edges.size()+mGrids[k].polygons.size(),
            //         mGrids[k].points.size()+mGrids[k].edges.size()+mGrids[k].polygons.size()));
            // estimate_condition_number(mGrids[k].global_stiffness_matrix_k2.
            //     bottomRightCorner(mGrids[k].polyhedrons.size(),mGrids[k].polyhedrons.size()));
            // estimate_condition_number(mGrids[k].global_stiffness_matrix_k1);

            //-------------------------------
            mGrids[k].grid_res = 100; // 可按场景调整
            mGrids[k].grid_to_elems.clear();
            mGrids[k].build_element_bboxes();
            mGrids[k].link_ele_with_uniform_grid_3d();
            //-------------------------------
            // k2 max dofs
            // Real min_dis = 1.0;
            // for (int i=0;i<mGrids[k].vem_elements.size();++i) {
            //     for (int j=i+1;j<mGrids[k].vem_elements.size();++j) {
            //         min_dis = std::min(min_dis,(mGrids[k].vem_elements[i].centroid-mGrids[k].vem_elements[j].centroid).norm());
            //     }
            // }
            // RZ_INFO(min_dis);
            // exit(0);
            int dofs = 0;
            for (const auto&element:mGrids[k].vem_elements) {
                int element_dof = 1+element.vertex_count()+element.edge_count()+element.polygons.size();
                dofs+=element_dof*element_dof;
            }
            RZ_INFO("LOD :",k," max dofs: ",dofs);
        }

        monomials_3d = {
            VEM_Monomial_3d(0, 0, 0),  // 1
            VEM_Monomial_3d(1, 0, 0),  // x
            VEM_Monomial_3d(0, 1, 0),  // y
            VEM_Monomial_3d(0, 0, 1),  // z
            VEM_Monomial_3d(2, 0, 0),  // x^2
            VEM_Monomial_3d(1, 1, 0),  // xy
            VEM_Monomial_3d(1, 0, 1),  // xz
            VEM_Monomial_3d(0, 2, 0),  // y^2
            VEM_Monomial_3d(0, 1, 1),  // yz
            VEM_Monomial_3d(0, 0, 2)   // z^2
        };

    }

void VEM_GMGSolver::build_hierarchy() {
    // build prolongations restrictions and As
    if (vem_k == 1)
        build_hierarchy_k1();
    else
        build_hierarchy_k2();
    // checkDD();
}

void VEM_GMGSolver::checkDD() {
    for (int l=0;l<L;++l) {
        if (isDiagonalDominant(As[l])) RZ_INFO(l,": DD");
        else RZ_INFO(l,": NDD");
    }
}

void VEM_GMGSolver::gmg_solver_k1(Real dt)  {
    // Eigen::VectorXr x;
    VC(mGrids.back().global_p_k1);
    // mGrids.back().global_p_k1 = x;
}

void VEM_GMGSolver::gmg_solver_k2(Real dt){
    // Eigen::VectorXr x;
    VC(mGrids.back().global_p_k2);
    // mGrids.back().global_p_k2 = x;
}

void VEM_GMGSolver::apply(const GlobalReal *d_rhs, GlobalReal *d_x, int n) {
    RZ_ASSERT(b.rows()==n,"");
    copyDeviceToHost(d_rhs,pcg_rhs,n);
    // copyDeviceToHost(d_x,pcg_x,n);
    Eigen::VectorXr pcg_eigen_x,pcg_eigen_rhs;
    pcg_eigen_x.resize(mGrids.back().global_stiffness_matrix_k2.rows());
    pcg_eigen_rhs.resize(mGrids.back().global_stiffness_matrix_k2.rows());
    for (int i=0;i<n;++i)pcg_eigen_rhs[i] = pcg_rhs[i],pcg_eigen_x[i] = 0.0;
    VC(pcg_eigen_rhs,pcg_eigen_x);
    for (int i=0;i<n;++i)pcg_x[i] = pcg_eigen_x[i];
    copyHostToDevice(pcg_x,d_x,n);
}

void VEM_GMGSolver::apply(const VEX_VECTOR &d_rhs, VEX_VECTOR &d_x, int n) const{//here
    RZ_ASSERT(b.rows()==n,"");
    amgcl::backend::clear(d_x);
    VC(d_rhs, d_x, pre_smoothing_iterations, post_smoothing_iterations);
}

void VEM_GMGSolver::build_hierarchy_k1(){
    L = mGrids.size();
    As.resize(L);
    Ps.resize(L - 1);
    Rs.resize(L - 1);
    chev_smoothers.resize(L);
    b.resize(mGrids.back().global_stiffness_matrix_k2.rows());

    pcg_x = (GlobalReal*)malloc(mGrids.back().global_stiffness_matrix_k2.rows() * sizeof(GlobalReal));
    pcg_rhs = (GlobalReal*)malloc(mGrids.back().global_stiffness_matrix_k2.rows() * sizeof(GlobalReal));
    As[L - 1] = mGrids.back().global_stiffness_matrix_k1;
    As[L-1].makeCompressed();
    visualize_A_sparsity(As[L-1],std::to_string(L-1)+"L_k1.png");
    for (int lev=L-2;lev>=0;--lev) {
        Ps[lev] = prolongation_k1(lev,lev+1);

        Rs[lev] = Ps[lev].transpose();
        As[lev] = Rs[lev] * As[lev+1] * Ps[lev];
        As[lev].makeCompressed();
        visualize_A_sparsity(As[lev],std::to_string(lev)+"L_k1.png");
        // Rs[lev] = prolongation_k1(lev+1,lev);
        // As[lev] = mGrids[lev].global_stiffness_matrix_k1;
    }
}
#include <malloc.h>
void VEM_GMGSolver::build_hierarchy_k2() {
        FUNC_TIMER();
        FUNC_MEMORY_TIMER();
        L = mGrids.size();
        As.resize(L);
        Ps.resize(L - 1);
        Rs.resize(L - 1);
        b.resize(mGrids.back().global_stiffness_matrix_k2.rows());
        // if (pcg_x)free(pcg_x);
        // if (pcg_rhs)free(pcg_rhs);
        pcg_x = (GlobalReal*)malloc(mGrids.back().global_stiffness_matrix_k2.rows() * sizeof(GlobalReal));
        pcg_rhs = (GlobalReal*)malloc(mGrids.back().global_stiffness_matrix_k2.rows() * sizeof(GlobalReal));

        chev_smoothers.clear();
        chev_smoothers.resize(L);

        As[L - 1].swap(mGrids.back().global_stiffness_matrix_k2);
        As[L-1].makeCompressed();
        mGrids.back().clear_data_for_build();
        // visualize_A_sparsity(As[L-1],std::to_string(L-1)+"L_k2.png");
        for (int lev=L-2;lev>=0;--lev) {
            // Ps Rs
            Ps[lev] = prolongation_k2_sparse(lev,lev+1);
            Rs[lev] = Ps[lev].transpose();
            Rs[lev].makeCompressed();
            Ps[lev].makeCompressed();
            As[lev] = Rs[lev] * As[lev+1] * Ps[lev];
            As[lev].makeCompressed();
            if (lev + 1 < L - 1) {
                mGrids[lev + 1].clear_hierarchy_build_cache();
            }
            // Real min_dia = 1e9;
            // for (int row=0;row<As[lev].rows();++row) {
            //     min_dia = std::fmin(std::fabs(As[lev].coeff(row,row)),min_dia);
            // }
            // visualize_A_sparsity(As[lev],std::to_string(lev)+"L_k2.png");
            // visualize_A_sparsity(mGrids[lev].global_stiffness_matrix_k2,std::to_string(lev)+"L_origin_k2.png");
        }

        for (int lev = 0; lev < L - 1; ++lev) {
            VEM_Domain_3d empty_grid;
            mGrids[lev] = std::move(empty_grid);
        }
        malloc_trim(0);

    }

void VEM_GMGSolver::set_rhs(const Eigen::VectorXr &b_fine)  {
    FUNC_TIMER();
    if (As.empty()) throw std::runtime_error("Call build_hierarchy() first.");
    b = b_fine;
}

VEM_GMGSolver::TimingStats VEM_GMGSolver::V_cycle(int lev, Eigen::VectorXr &x, const Eigen::VectorXr &b, int nu1,
    int nu2) const {
        TimingStats stats;
        if (lev == 0) {// 用 amgcl
            // Eigen::VectorXr zero_init =Eigen::VectorXr::Zero(b.rows());
            // x = amgclSolve(l0_solver,b,&zero_init);
            auto s1 = std::chrono::high_resolution_clock::now();
            chev_smoothers[lev].smooth(b, x, coarse_smoothing_iterations);
            auto s2 = std::chrono::high_resolution_clock::now();
            stats.smooth_time += std::chrono::duration<Real>(s2 - s1).count();
            // if (MGDEBUG) {
            //     // ---------- 收敛报告 ----------
            //     Eigen::VectorXr r = As[0] * x - b;
            //     Real rel_res = r.norm() / std::max(1e-30, b.norm());
            //     std::cout << "[Coarse level solve] residual = " << rel_res << std::endl;
            //     if (rel_res < 1e-6)
            //         std::cout << "✅ Coarse level converged." << std::endl;
            //     else
            //         std::cout << "⚠️ Coarse level not fully converged." << std::endl;
            // }
            return stats;
        }

        auto tic = std::chrono::high_resolution_clock::now();
        auto s1 = std::chrono::high_resolution_clock::now();
        // gs_smoothers[lev].cell_smooth(b, x, nu1);
        chev_smoothers[lev].smooth(b, x, nu1);
        // jacobi_smoothers[lev].smooth(b, x, nu1);
        auto s2 = std::chrono::high_resolution_clock::now();
        stats.smooth_time += std::chrono::duration<Real>(s2 - s1).count();
        // 计算残差 r = b - A*x
        Eigen::VectorXr rH;
        if (vem_k == 2) {
            Eigen::VectorXr r = b - (As[lev] * x.template cast<GlobalReal>()).template cast<Real>();
            rH = (Rs[lev-1] * r.template cast<GlobalReal>()).template cast<Real>();
            // restriction_k2(lev-1,lev,x,rH);
            // Eigen::VectorXr rH = Rs[lev-1] * r / 3.1818;// todo: 检验 restriction
        }else {
            Eigen::VectorXr r = b - (As[lev] * x.template cast<GlobalReal>()).template cast<Real>();
            rH = (Rs[lev-1] * r.template cast<GlobalReal>()).template cast<Real>();
        }
        Eigen::VectorXr eH = Eigen::VectorXr::Zero(As[lev - 1].rows());
        TimingStats sub = V_cycle(lev - 1, eH, rH, nu1, nu2);
        stats.accumulate(sub);
        // 校正并后平滑
        // x += eH;
        x += (Ps[lev-1] * eH.template cast<GlobalReal>()).template cast<Real>();// todo: 检验 prolongation

        auto s5 = std::chrono::high_resolution_clock::now();
        // gs_smoothers[lev].cell_smooth(b, x, nu2);
        // jacobi_smoothers[lev].smooth(b, x, nu2);
        // jacobi_smoothers[lev].smooth(b, x, nu2);
        chev_smoothers[lev].smooth(b, x, nu2);
        auto s6 = std::chrono::high_resolution_clock::now();
        stats.smooth_time += std::chrono::duration<Real>(s6 - s5).count();
        auto toc = std::chrono::high_resolution_clock::now();
        stats.other_time += std::chrono::duration<Real>(toc - tic).count() - stats.smooth_time;
        return stats;
    }

void VEM_GMGSolver::VC(Eigen::VectorXr &xH, int nu1, int nu2, int cycles) {
        FUNC_TIMER();
        if (As.empty())
            throw std::runtime_error("Call build_hierarchy()+set_rhs() before FMG().");
        if (xH.size()==0)
            xH = Eigen::VectorXr::Zero(As[L-1].rows());
        // ============================================================
        // 计算 r0（初始残差）
        // ============================================================
        const auto& A_fine = As[L - 1];
        const auto& b_fine = b;
        Eigen::VectorXr r0 = (A_fine * xH.template cast<GlobalReal>()).template cast<Real>() - b_fine;
        Real r0_norm2 = r0.squaredNorm();
        //----------------------------
        Real total_smooth = 0.0, total_other = 0.0;
        for (int t = 0; t < cycles; ++t) {
            RZ_INFO("cycle: ",t,"================================");
            RZ_INFO("before rel:",((A_fine * xH.template cast<GlobalReal>()).template cast<Real>() - b_fine).norm()/(b_fine.norm()));
            Real bn2 = ((A_fine * xH.template cast<GlobalReal>()).template cast<Real>() - b_fine).squaredNorm();
            TimingStats stats = V_cycle(L - 1, xH, b, nu1, nu2);
            RZ_INFO("after rel:",((A_fine * xH.template cast<GlobalReal>()).template cast<Real>() - b_fine).norm()/(b_fine.norm()));
            Real an2 = ((A_fine * xH.template cast<GlobalReal>()).template cast<Real>() - b_fine).squaredNorm();
            Real rho = std::exp(std::log(an2 / bn2));
            RZ_INFO("rho: ",rho);
            RZ_INFO("cycle: ",t,"================================");
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

        Eigen::VectorXr rN = (A_fine * xH.template cast<GlobalReal>()).template cast<Real>() - b_fine;
        Real rel_err = rN.norm() / b_fine.norm();
        Real rN_norm2 = rN.squaredNorm();

        std::cout << "Relative residual: " << rel_err << "\n";
        std::cout << "================================\n";

        Real rho = std::exp((1.0 / cycles) * std::log(rN_norm2 / r0_norm2));
        std::cout << "Smoothing rate ρ: " << rho << "\n";
        std::cout << "================================\n";
    }

void VEM_GMGSolver::VC(const Eigen::VectorXr &input_b,
Eigen::VectorXr &xH, int nu1, int nu2, int cycles) const {
    FUNC_TIMER();
    if (As.empty())
        throw std::runtime_error("Call build_hierarchy()+set_rhs() before FMG().");
    if (xH.size()==0)
        xH = Eigen::VectorXr::Zero(As[L-1].rows());

    for (int t = 0; t < cycles; ++t) {
        TimingStats stats = V_cycle(L - 1, xH, input_b, nu1, nu2);
    }

}

void VEM_GMGSolver::init_for_pcg() {
    FUNC_TIMER();
    FUNC_MEMORY_TIMER();
    if (L <= 0 || As.size() != static_cast<size_t>(L) ||
        Rs.size() != static_cast<size_t>(L - 1) || Ps.size() != Rs.size())
        throw std::logic_error("GMG hierarchy must be built before init_for_pcg");
    if (!direct_coarse && coarse_smoothing_iterations < 1)
        throw std::invalid_argument("coarse_smoothing_iterations must be positive");
    if (pre_smoothing_iterations < 1 || post_smoothing_iterations < 1)
        throw std::invalid_argument("GMG pre/post smoothing iterations must be positive");

    workspace_.clear();
    chev_smoothers.clear();
    chev_smoothers.resize(L);
    As_gpu.clear();
    Rs_gpu.clear();
    Ps_gpu.clear();
    coarse_solver.reset();
    gpu_direct_coarse_ = direct_coarse;
    gpu_coarse_iterations_ = coarse_smoothing_iterations;

    if (bprm_.q.empty()) {
        vex::Context ctx(vex::Filter::Env);
        if (ctx.empty()) throw std::runtime_error("No OpenCL device for GMG");
        bprm_.q = ctx;
    }
    bprm_.fast_matrix_setup = false;
    for (const auto &A : As) As_gpu.push_back(eigentovex(A, bprm_));

    for (int lev = L - 1; lev >= 0; --lev) {
        chev_smoothers[lev].bprm_ = bprm_;
        chev_smoothers[lev].build(As[lev], As_gpu[lev]);
    }
    if (gpu_direct_coarse_)
        coarse_solver = Backend::create_solver(eigentobuildin(As[0], bprm_), bprm_);
    for (const auto &R : Rs) Rs_gpu.push_back(eigentovex(R, bprm_));
    for (const auto &P : Ps) Ps_gpu.push_back(eigentovex(P, bprm_));

    workspace_.resize(L);
    for (int lev = 1; lev < L; ++lev) {
        auto &w = workspace_[lev];
        w.residual = std::make_shared<VEX_VECTOR>(bprm_.q, As[lev].rows());
        w.coarse_rhs = std::make_shared<VEX_VECTOR>(bprm_.q, As[lev - 1].rows());
        w.coarse_error = std::make_shared<VEX_VECTOR>(bprm_.q, As[lev - 1].rows());
    }
}

void VEM_GMGSolver::cycle(int lev, VEX_VECTOR &d_x, const VEX_VECTOR &d_rhs,
                         int nu1, int nu2) const {
    if (lev < 0 || lev >= L || workspace_.size() != static_cast<size_t>(L))
        throw std::logic_error("GMG GPU workspace is not initialized");
    if (d_x.size() != static_cast<size_t>(As[lev].cols()) ||
        d_rhs.size() != static_cast<size_t>(As[lev].rows()))
        throw std::invalid_argument("GMG cycle vector size mismatch");

    auto check = [&](const VEX_VECTOR &v, const char *stage) {
        if (diagnostic_checks && !std::isfinite(amgcl::backend::inner_product(v, v)))
            throw std::runtime_error(std::string("GMG nonfinite vector: ") + stage);
    };
    check(d_x, "input x");
    check(d_rhs, "input rhs");
    if (lev == 0) {
        if (gpu_direct_coarse_) (*coarse_solver)(d_rhs, d_x);
        else chev_smoothers[0].smooth(d_rhs, d_x, gpu_coarse_iterations_);
        check(d_x, "coarse solve");
        return;
    }

    chev_smoothers[lev].smooth(d_rhs, d_x, nu1);
    check(d_x, "pre-smoothing");
    const auto &w = workspace_[lev];
    auto &res = *w.residual;
    auto &rH = *w.coarse_rhs;
    auto &eH = *w.coarse_error;
    // Both operations overwrite their destination; no separate clear is needed.
    amgcl::backend::residual(d_rhs, *As_gpu[lev], d_x, res);
    check(res, "residual");
    amgcl::backend::spmv(1.0, *Rs_gpu[lev - 1], res, 0.0, rH);
    check(rH, "restriction");
    // A fixed zero initial correction preserves the preconditioner semantics.
    amgcl::backend::clear(eH);
    cycle(lev - 1, eH, rH, nu1, nu2);
    check(eH, "coarse correction");
    amgcl::backend::spmv(1.0, *Ps_gpu[lev - 1], eH, 1.0, d_x);
    check(d_x, "prolongation");
    chev_smoothers[lev].smooth(d_rhs, d_x, nu2);
    check(d_x, "post-smoothing");
}

void VEM_GMGSolver::restriction_k2(int ll, int hl, const Eigen::VectorXr &fine_p, Eigen::VectorXr &rH) {
{
        const auto&coarse_domain = mGrids[ll];
        const auto&fine_domain = mGrids[hl];
        const auto&vem_elements = fine_domain.vem_elements;
        // virtual_apply_p
        int element_id = 0;
        int max_per_element_dof_num_k2 = 0;
        for (const auto &element : vem_elements) {
            int ndof = (int)element.vertex_local_to_global_dof.size()
                      + (int)element.edge_count()
                      + (int)element.polygons.size()
                      + 1;
            max_per_element_dof_num_k2 = std::max(max_per_element_dof_num_k2, ndof);
        }
        // === 缓存常数偏移 ===
        const int base_v = 0;
        const int base_e = (int)fine_domain.points.size();
        const int base_f = base_e + (int)fine_domain.edges.size();
        const int base_c = base_f + (int)fine_domain.polygons.size();

        // === 临时向量缓存 (一次分配，多次复用) ===
        Eigen::VectorXr element_dof_values_p;
        element_dof_values_p.resize(max_per_element_dof_num_k2); // 可在函数外声明缓存
        Eigen::VectorXr du,dv,dw,weight;
        du.resize(fine_domain.k1_dof_u.size());
        dv.resize(fine_domain.k1_dof_v.size());
        dw.resize(fine_domain.k1_dof_w.size());
        weight.resize(fine_domain.k1_dof_w.size());
        du.setZero();
        dv.setZero();
        dw.setZero();
        weight.setZero();

        for (const auto &element: vem_elements) {
            Eigen::MatrixXr proj;
            if (element.is_U)proj = fine_domain.U3d_project_matrix_star_k2;
            else proj = element.project_matrix_star_k2;

            const auto &verts = element.vertex_local_to_global_dof;
            int offset = 0;
            // === 顶点自由度 ===
            const int nv = (int)verts.size();
            for (int i = 0; i < nv; ++i)
                element_dof_values_p[offset++] = fine_p[base_v + verts[i]];
            // === 2️⃣ 边自由度 ===
            for (int eid = 0; eid < (int)element.edge_count(); ++eid) {
                element_dof_values_p[offset++] =
                    fine_p[base_e + element.edge_local_to_global_dof[eid]];
            }
            // === 3️⃣ 面自由度 ===
            const int npoly = (int)element.polygons.size();
            for (int i = 0; i < npoly; ++i)
                element_dof_values_p[offset++] =
                    fine_p[base_f + element.polygon_local_to_global_dof[i]];
            // === 4️⃣ 体自由度 ===
            element_dof_values_p[offset++] = fine_p[base_c + element_id];
            // === 5️⃣ 投影 (无中间分配) ===
            const Eigen::VectorXr poly_coeffs_p = proj * element_dof_values_p.head(offset);
            for (int local_vid=0;local_vid<element.vertex_count();++local_vid) {
                const auto &pos = element.vertex_position(local_vid);
                const int global_vid = element.vertex_local_to_global_dof[local_vid];
                Real dvx = 0.0, dvy = 0.0, dvz = 0.0;
                const Real xc = element.centroid.vx;
                const Real yc = element.centroid.vy;
                const Real zc = element.centroid.vz;
                const Real h  = element.diameter;
                const int n_basis = (int)monomials_3d.size();
                for (int i = 0; i < n_basis; ++i) {
                    const auto &mono = monomials_3d[i];
                    const auto grad = mono.gradient(pos.vx, pos.vy, pos.vz, xc, yc, zc, h);
                    const Real coeff = poly_coeffs_p(i);
                    dvx += grad.vx * coeff;
                    dvy += grad.vy * coeff;
                    dvz += grad.vz * coeff;
                }

                dvx *= dt;
                dvy *= dt;
                dvz *= dt;

                du[global_vid] += dvx * element.volume;
                dv[global_vid] += dvy * element.volume;
                dw[global_vid] += dvz * element.volume;
                weight[global_vid] += element.volume;
            }
            element_id++;
        }
        for (int vid=0;vid<fine_domain.k1_dof_u.size();++vid) {
            du[vid] /= weight[vid];
            dv[vid] /= weight[vid];
            dw[vid] /= weight[vid];
        }
        Eigen::VectorXr virtual_projed_u = fine_domain.k1_dof_u - du;
        Eigen::VectorXr virtual_projed_v = fine_domain.k1_dof_v - dv;
        Eigen::VectorXr virtual_projed_w = fine_domain.k1_dof_w - dw;

        // restriction of velocity
        mGrids[ll].k1_dof_u = (Rs[ll] * virtual_projed_u.template cast<GlobalReal>()).template cast<Real>();
        mGrids[ll].k1_dof_v = (Rs[ll] * virtual_projed_v.template cast<GlobalReal>()).template cast<Real>();
        mGrids[ll].k1_dof_w = (Rs[ll] * virtual_projed_w.template cast<GlobalReal>()).template cast<Real>();
        // get the restriction of r
        rH =
            (coarse_domain.global_LX_matrix_k2 * mGrids[ll].k1_dof_u.template cast<GlobalReal>()).template cast<Real>()
            + (coarse_domain.global_LY_matrix_k2 * mGrids[ll].k1_dof_v.template cast<GlobalReal>()).template cast<Real>()
            + (coarse_domain.global_LZ_matrix_k2 * mGrids[ll].k1_dof_w.template cast<GlobalReal>()).template cast<Real>();
        rH /= dt;
    }
}

Eigen::SparseMatrixR VEM_GMGSolver::prolongation_k1(int ll, int hl) {
{
        const auto&coarse_domain = mGrids[ll];
        const auto&fine_domain = mGrids[hl];
        std::vector<Eigen::Triplet<GlobalReal>> trip;
        const auto&vem_elements = coarse_domain.vem_elements;
        std::vector<_3d_vec> fine_points;
        for (auto&p:fine_domain.points) {
            fine_points.push_back(p);
        }

        std::vector<std::vector<int>> contained_indices;
        assign_points_to_elements(contained_indices,fine_points,coarse_domain);

        // vertices dofs-----------------------------------
        for (int r=0;r<fine_domain.points.size();++r) {
            const auto pos = fine_domain.points[r];
            RZ_ASSERT(contained_indices[r].size()>0,"");
            for (auto eid:contained_indices[r]) {
                const auto& element = vem_elements[eid];
                const auto &projector_k1 = element.project_matrix_star_k1;
                const auto &vert_ids = element.vertex_local_to_global_dof;
                const int ndof = (int)vert_ids.size();
                if (ndof == 0 ) RZ_FATAL("ndof == 0");
                Eigen::VectorXr element_dof_values(ndof);
                for (int k=0;k<ndof;++k) {
                    RZ_ASSERT(k<element.vertex_local_to_global_dof.size()," ");
                    const int col = element.vertex_local_to_global_dof[k];
                    element_dof_values.setZero();
                    element_dof_values[k] = 1.0;
                    Eigen::MatrixXr poly_coeffs = projector_k1 * element_dof_values;
                    Real weight = 0.0;
                    for (int i = 0; i < poly_coeffs.size(); ++i) {
                        const auto &mono = monomials_3d[i];
                        Real val = mono.value(
                            pos.vx, pos.vy, pos.vz,
                            element.centroid.vx,
                            element.centroid.vy,
                            element.centroid.vz,
                            element.diameter);
                        weight += poly_coeffs(i) * val;
                    }
                    trip.emplace_back(r, col, weight/contained_indices[r].size());
                }
            }
        }
        Eigen::SparseMatrixR ans(fine_domain.points.size(),
            coarse_domain.points.size());
        ans.setFromTriplets(trip.begin(), trip.end());
        ans.makeCompressed();
        return ans;
    }
}

Eigen::SparseMatrixR VEM_GMGSolver::prolongation_k2(int ll, int hl) {
{
        const auto&coarse_domain = mGrids[ll];
        const auto&fine_domain = mGrids[hl];
        std::vector<Eigen::Triplet<GlobalReal>> trip;
        const auto&vem_elements = coarse_domain.vem_elements;
        // 判定高层级的每一个点 、每个边的中点 、每个面的中心 每个体的中心
        // 处于 低层级的哪一个cell中 注意只能有一个归属
        // 对于每个点 r 根据所处的cell计算出来对应到这个cell的不同自由度（全局来讲是c）的权重w
        // trip.emplace_back(r, c, w);
        std::vector<_3d_vec> fine_points;
        std::unordered_map<int,int>p_to_dof;
        std::map<int,int> p_to_fine_cid;
        std::map<int,std::pair<int,int>> p_to_fine_c_lpid;
        std::set<int>inserted_pid;
        int vdofn = fine_domain.points.size();
        int edofn = fine_domain.edges.size();
        int fdofn = fine_domain.polygons.size();
        for (auto&p:fine_domain.points) {
            p_to_dof[fine_points.size()] = fine_points.size();
            fine_points.push_back(p);
        }
        for (int eid = 0; eid < (int)fine_domain.edges.size(); ++eid) {
            const auto& edge = fine_domain.edges[eid];
            const auto& p0 = fine_domain.points[edge.v1];
            const auto& p1 = fine_domain.points[edge.v2];
            _3d_vec mid = (p0+p1)/2.0;
            p_to_dof[fine_points.size()] = vdofn + eid;
            fine_points.push_back(mid);
        }
        for (int cid = 0;cid<fine_domain.vem_elements.size();++cid) {
            const auto&vem_element = fine_domain.vem_elements[cid];
            int local_pid=0;
            for (auto&pid:vem_element.polygon_local_to_global_dof) {
                if (!inserted_pid.count(pid)) {
                    inserted_pid.insert(pid);
                    p_to_dof[fine_points.size()] = vdofn+edofn+pid;
                    p_to_fine_c_lpid[fine_points.size()] = std::make_pair(cid,local_pid);
                    fine_points.push_back(fine_domain.polygons_plane[pid]._3d_centroid);
                }
                ++local_pid;
            }
        }
        // for (int pid=0;pid<fine_domain.polygons.size();++pid) {
        //     p_to_dof[fine_points.size()] = vdofn+edofn+pid;
        //     fine_points.push_back(fine_domain.polygons_plane[pid]._3d_centroid);
        // }
        for (int cid = 0;cid<fine_domain.vem_elements.size();++cid) {
            p_to_dof[fine_points.size()] = vdofn+edofn+fdofn+cid;
            p_to_fine_cid[fine_points.size()] = cid;
            fine_points.push_back(fine_domain.vem_elements[cid].centroid);
        }



        std::vector<std::vector<int>> contained_indices;
        assign_points_to_elements(contained_indices,fine_points,coarse_domain);
        // //---------------------------------------------------
        const int base_e = (int)coarse_domain.points.size();
        const int base_f = base_e + (int)coarse_domain.edges.size();
        const int base_c = base_f + (int)coarse_domain.polygons.size();
        std::set<int> used_coarsed_dofs;
        for (int i = 0;i < fine_points.size(); ++i) {
            const int row = p_to_dof[i];
            const auto pos = fine_points[i];
            RZ_ASSERT(contained_indices[i].size()>0,"");
            // RZ_ASSERT(contained_indices[i]<vem_elements.size(), contained_indices[i]);
            for (const auto& cid:contained_indices[i]) {
                // const int cid = contained_indices[i];
                const auto& element = vem_elements[cid];
                int element_dof_n = element.vertex_count()+element.edge_count()
                    +element.polygons.size()+1;
                int element_vdof_n = element.vertex_count();
                int element_edof_n = element.edge_count();
                int element_fdof_n = element.polygons.size();
                Eigen::MatrixXr proj;
                if (element.is_U)proj = coarse_domain.U3d_project_matrix_star_k2;
                else proj = element.project_matrix_star_k2;
                Eigen::VectorXr element_dof_values_p;
                element_dof_values_p.resize(element_dof_n); // 可在函数外声明缓存
                // -- vertices
                for (int vid = 0;vid<element_vdof_n;++vid) {
                    int global_coarse_dof = element.vertex_local_to_global_dof[vid];
                    element_dof_values_p.setZero();
                    element_dof_values_p[vid] = 1.0;
                    Real weight = 0.0;
                    const Eigen::VectorXr poly_coeffs_p = proj * element_dof_values_p;
                    const Real xc = element.centroid.vx;
                    const Real yc = element.centroid.vy;
                    const Real zc = element.centroid.vz;
                    const Real h  = element.diameter;
                    const int n_basis = (int)monomials_3d.size();
                    for (int j = 0; j < n_basis; ++j) {
                        const auto &mono = monomials_3d[j];
                        Real val =
                            mono.value(pos.vx, pos.vy, pos.vz, xc, yc, zc, h);
                        const Real coeff = poly_coeffs_p(j);
                        weight += coeff * val;
                    }
                    used_coarsed_dofs.insert(global_coarse_dof);
                    trip.emplace_back(row, global_coarse_dof, weight/contained_indices[i].size());
                }

                // -- edges
                for (int eid = 0;eid<element_edof_n;++eid) {
                    int global_coarse_dof = base_e + element.edge_local_to_global_dof[eid];
                    element_dof_values_p.setZero();
                    element_dof_values_p[eid+element_vdof_n] = 1.0;
                    Real weight = 0.0;
                    const Eigen::VectorXr poly_coeffs_p = proj * element_dof_values_p;
                    const Real xc = element.centroid.vx;
                    const Real yc = element.centroid.vy;
                    const Real zc = element.centroid.vz;
                    const Real h  = element.diameter;
                    const int n_basis = (int)monomials_3d.size();
                    for (int j = 0; j < n_basis; ++j) {
                        const auto &mono = monomials_3d[j];
                        Real val =
                            mono.value(pos.vx, pos.vy, pos.vz, xc, yc, zc, h);
                        const Real coeff = poly_coeffs_p(j);
                        weight += coeff * val;
                    }
                    used_coarsed_dofs.insert(global_coarse_dof);
                    trip.emplace_back(row, global_coarse_dof, weight/contained_indices[i].size());
                }
                // faces
                for (int fid=0;fid<element_fdof_n;++fid) {
                    int global_coarse_dof = base_f + element.polygon_local_to_global_dof[fid];
                    element_dof_values_p.setZero();
                    element_dof_values_p[fid+element_edof_n+element_vdof_n] = 1.0;
                    Real weight = 0.0;
                    const Eigen::VectorXr poly_coeffs_p = proj * element_dof_values_p;
                    const Real xc = element.centroid.vx;
                    const Real yc = element.centroid.vy;
                    const Real zc = element.centroid.vz;
                    const Real h  = element.diameter;
                    const int n_basis = (int)monomials_3d.size();
                    for (int j = 0; j < n_basis; ++j) {
                        const auto &mono = monomials_3d[j];
                        auto val =
                            mono.value(pos.vx, pos.vy, pos.vz, xc, yc, zc, h);
                        // // 对于面自由度用积分求val
                        // if (p_to_fine_c_lpid.count(i)) {
                        //     const auto fine_cid = p_to_fine_c_lpid[i].first;
                        //     const auto fine_lpid = p_to_fine_c_lpid[i].second;
                        //     const auto& fine_elem_2d = fine_domain.vem_elements[fine_cid].elements_2d[fine_lpid];
                        //     val = fine_elem_2d.polygon_moment_general_3d(mono,_3d_vec(xc,yc,zc))
                        //         /std::pow(h,mono.p+mono.q+mono.r) / fine_elem_2d.area;
                        // }
                        // // 对于体自由度用积分求val
                        // if (p_to_fine_cid.count(i)) {
                        //     const auto& fine_elem = fine_domain.vem_elements[p_to_fine_cid[i]];
                        //     val = fine_elem.
                        //         integrate_centered_monomial_over_polyhedron_divergence(mono,_3d_vec(xc,yc,zc))
                        //     /std::pow(h,mono.p+mono.q+mono.r) / fine_elem.volume;
                        // }
                        const Real coeff = poly_coeffs_p(j);
                        weight += coeff * val;
                    }
                    used_coarsed_dofs.insert(global_coarse_dof);
                    trip.emplace_back(row, global_coarse_dof, weight/contained_indices[i].size());
                }
                // cell
                {
                    int cell_global_coarse_dof = base_c + cid;
                    element_dof_values_p.setZero();
                    element_dof_values_p[element_fdof_n+element_edof_n+element_vdof_n] = 1.0;
                    Real weight = 0.0;
                    const Eigen::VectorXr poly_coeffs_p = proj * element_dof_values_p;
                    const Real xc = element.centroid.vx;
                    const Real yc = element.centroid.vy;
                    const Real zc = element.centroid.vz;
                    const Real h  = element.diameter;
                    const int n_basis = (int)monomials_3d.size();
                    for (int j = 0; j < n_basis; ++j) { //
                        const auto &mono = monomials_3d[j];
                        Real val =
                            mono.value(pos.vx, pos.vy, pos.vz, xc, yc, zc, h);
                        // // 对于面自由度用积分求val
                        // if (p_to_fine_c_lpid.count(i)) {
                        //     const auto fine_cid = p_to_fine_c_lpid[i].first;
                        //     const auto fine_lpid = p_to_fine_c_lpid[i].second;
                        //     const auto& fine_elem_2d = fine_domain.vem_elements[fine_cid].elements_2d[fine_lpid];
                        //     val = fine_elem_2d.polygon_moment_general_3d(mono,_3d_vec(xc,yc,zc))
                        //         /std::pow(h,mono.p+mono.q+mono.r) / fine_elem_2d.area;
                        // }
                        // // 对于体自由度用积分求val
                        // if (p_to_fine_cid.count(i)) {
                        //     const auto& fine_elem = fine_domain.vem_elements[p_to_fine_cid[i]];
                        //     val = fine_elem.
                        //         integrate_centered_monomial_over_polyhedron_divergence(mono,_3d_vec(xc,yc,zc))
                        //     /std::pow(h,mono.p+mono.q+mono.r)/ fine_elem.volume;
                        // }
                        const Real coeff = poly_coeffs_p(j);
                        weight += coeff * val;
                    }

                    used_coarsed_dofs.insert(cell_global_coarse_dof);
                    trip.emplace_back(row, cell_global_coarse_dof, weight/contained_indices[i].size());
                }
            }
        }
        RZ_INFO("used coarse dofs: ",used_coarsed_dofs.size());
        if (!DOFSCALE) {// todo :缩放 weights

        }
        Eigen::SparseMatrixR
        ans(fine_domain.points.size() +
         fine_domain.edges.size()+
         fine_domain.polygons.size()+
         fine_domain.polyhedrons.size(),
         coarse_domain.points.size()+
         coarse_domain.edges.size()+
         coarse_domain.polygons.size()+
         coarse_domain.polyhedrons.size()
         );
        ans.setFromTriplets(trip.begin(), trip.end());
        ans.makeCompressed();
        return ans;
    }

}

Eigen::SparseMatrixR VEM_GMGSolver::prolongation_k2_sparse(int ll, int hl){
        FUNC_TIMER();
        const auto&coarse_domain = mGrids[ll];
        const auto&fine_domain = mGrids[hl];
        std::vector<Eigen::Triplet<GlobalReal>> trip;
        const auto&vem_elements = coarse_domain.vem_elements;
        bool use_volume_intergration = false;
        bool use_face_intergration = false;
        std::vector<_3d_vec> fine_points;
        std::unordered_map<int,int>p_to_dof;
        std::map<int,int> p_to_fine_cid;
        std::map<int,std::pair<int,int>> p_to_fine_c_lpid;
        std::set<int>inserted_pid;
        int vdofn = fine_domain.points.size();
        int edofn = fine_domain.edges.size();
        int fdofn = fine_domain.polygons.size();
        for (auto&p:fine_domain.points) {
            p_to_dof[fine_points.size()] = fine_points.size();
            fine_points.push_back(p);
        }
        for (int eid = 0; eid < (int)fine_domain.edges.size(); ++eid) {
            const auto& edge = fine_domain.edges[eid];
            const auto& p0 = fine_domain.points[edge.v1];
            const auto& p1 = fine_domain.points[edge.v2];
            _3d_vec mid = (p0+p1)/2.0;
            p_to_dof[fine_points.size()] = vdofn + eid;
            fine_points.push_back(mid);
        }
        for (int cid = 0;cid<fine_domain.vem_elements.size();++cid) {
            const auto&vem_element = fine_domain.vem_elements[cid];
            int local_pid=0;
            for (auto&pid:vem_element.polygon_local_to_global_dof) {
                if (!inserted_pid.count(pid)) {
                    inserted_pid.insert(pid);
                    p_to_dof[fine_points.size()] = vdofn+edofn+pid;
                    p_to_fine_c_lpid[fine_points.size()] = std::make_pair(cid,local_pid);
                    fine_points.push_back(fine_domain.polygons_plane[pid]._3d_centroid);
                }
                ++local_pid;
            }
        }
        for (int cid = 0;cid<fine_domain.vem_elements.size();++cid) {
            p_to_dof[fine_points.size()] = vdofn+edofn+fdofn+cid;
            p_to_fine_cid[fine_points.size()] = cid;
            fine_points.push_back(fine_domain.vem_elements[cid].centroid);
        }

        std::vector<std::vector<int>> contained_indices;
        assign_points_to_elements(contained_indices,fine_points,coarse_domain);
        // //---------------------------------------------------
        const int base_e = (int)coarse_domain.points.size();
        const int base_f = base_e + (int)coarse_domain.edges.size();
        const int base_c = base_f + (int)coarse_domain.polygons.size();
// #pragma omp parallel for
        for (int i = 0;i < fine_points.size(); ++i) {
            if (!contained_indices[i].size())continue;
            const int row = p_to_dof[i];
            const auto pos = fine_points[i];
            // 判断是否可以继承于粗层级的某一个节点
            int corr_coarse_id = -1;
            if (row<fine_domain.points.size()) {
                for (const auto& cid:contained_indices[i]) {
                    const auto& element = vem_elements[cid];
                    for (int local_vid=0;local_vid<element.vertex_count();++local_vid) {
                        if ((element.vertex_position(local_vid)-pos).norm()<1e-15) {
                            corr_coarse_id = element.vertex_local_to_global_dof[local_vid];
                            break;
                        }
                    }
                    if (corr_coarse_id>=0)break;
                }
            }
            if (corr_coarse_id>=0) {
#pragma omp critical
                trip.emplace_back(row, corr_coarse_id, 1.0);
                continue;
            }
            // 判断是否落在粗层级的某一个边上 若是 则用 这个fine point边相关的所有面来插值
            //如果和面一样都是用 所有的 体呢？ todo
            std::set<int> active_dofs;
            std::vector<Eigen::Triplet<GlobalReal>> edge_triplets;
            if (row<fine_domain.points.size() + fine_domain.edges.size() && on_coarse_edge(pos,row,edge_triplets,contained_indices[i],ll,active_dofs)) {
                RZ_ASSERT(edge_triplets.size()>0,"edge_triplets not initialized");
                for (auto&tri:edge_triplets) {
    #pragma omp critical
                    trip.emplace_back(tri.row(),tri.col(),tri.value());
                }
                //----------------------------------------------------------
//                 std::vector<Eigen::Triplet<GlobalReal>> no_weight_trips;
//                 for (const auto& cid:contained_indices[i]) {
//                     const auto& element_3d = vem_elements[cid];
//                     for (int local_poly_id =0;local_poly_id<element_3d.elements_2d.size();++local_poly_id) {
//                         VEM2d_Prolongation(row,local_poly_id,element_3d,no_weight_trips,pos,mGrids[ll]);
//                     }
//                 }
//                 Real weight_sum = 0.0;
//                 for (auto&tri:no_weight_trips) {
//                     if (!active_dofs.count(tri.col()))continue;
//                     weight_sum += tri.value();
//                 }
//                 for (auto&tri:no_weight_trips) {
//                     if (!active_dofs.count(tri.col()))continue;
// #pragma omp critical
//                     trip.emplace_back(tri.row(),tri.col(),tri.value()/weight_sum);
//                 }
                continue;
            }

            if (contained_indices[i].size()>1) {
                // 找到这个pos在哪一个vem_element_2上
                std::vector<Eigen::Triplet<GlobalReal>> no_weight_trips;
                bool found = false;
                for (const auto& cid:contained_indices[i]) {
                    const auto& element_3d = vem_elements[cid];
                    for (int local_poly_id =0;local_poly_id<element_3d.elements_2d.size();++local_poly_id)
                    {
                        const auto& element_2d = element_3d.elements_2d[local_poly_id];
                        CGAL_Point_3 cgal_point(pos.vx,pos.vy,pos.vz);
                        auto pos_2d = element_2d.plane._to_2d(pos);
                        if (std::sqrt(CGAL::squared_distance(element_2d.plane.cgal_plane,cgal_point))>1e-12)continue;
                        // 要判定这个点是否在这个element_2d内部
                        if (!point_in_convex_hull(pos_2d,element_2d.points))continue;
                        // 就在这个平面上 用 vem_element_2 的 自由度 prolongation 到这个 fine dof 上
                        found = true;
                        Real weight_sum = VEM2d_Prolongation(row,local_poly_id,element_3d,no_weight_trips,pos,mGrids[ll]);
                        for (auto&tri:no_weight_trips) {
                            #pragma omp critical
                            trip.emplace_back(tri.row(),tri.col(),tri.value()/weight_sum);
                        }
                        break;
                    }

                    if (found)break;
                }

                if (!found) {
                    std::vector<_3d_vec>points;
                    points.push_back(pos);
                    this_save_points_to_xyz(points,"on_plane.xyz");
                    for (const auto& tcid:contained_indices[i]) {
                        vem_elements[tcid].save_polygonmesh_to_obj(std::to_string(tcid)+".obj");
                    }
                    RZ_FATAL("not found");
                }
                continue;
            }
//---------------------------------------------------------------------------------------
            RZ_ASSERT(contained_indices[i].size() == 1,"");

            for (const auto& cid:contained_indices[i]) {
                std::vector<Eigen::Triplet<GlobalReal>> no_weight_trips;
                Real weight_sum = 0.0;
                Real deno = contained_indices[i].size();
                const auto& element = vem_elements[cid];
                int element_dof_n = element.vertex_count()+element.edge_count()
                    +element.polygons.size()+1;
                int element_vdof_n = element.vertex_count();
                int element_edof_n = element.edge_count();
                int element_fdof_n = element.polygons.size();
                Eigen::MatrixXr proj;
                if (element.is_U)proj = coarse_domain.U3d_project_matrix_star_k2;
                else proj = element.project_matrix_star_k2;
                Eigen::VectorXr element_dof_values_p;
                element_dof_values_p.resize(element_dof_n); // 可在函数外声明缓存

                // -- vertices
                for (int vid = 0;vid<element_vdof_n;++vid) {
                    int global_coarse_dof = element.vertex_local_to_global_dof[vid];
                    element_dof_values_p.setZero();
                    element_dof_values_p[vid] = 1.0;
                    Real weight = 0.0;
                    const Eigen::VectorXr poly_coeffs_p = proj * element_dof_values_p;
                    const Real xc = element.centroid.vx;
                    const Real yc = element.centroid.vy;
                    const Real zc = element.centroid.vz;
                    const Real h  = element.diameter;
                    const int n_basis = (int)monomials_3d.size();
                    for (int j = 0; j < n_basis; ++j) {
                        const auto &mono = monomials_3d[j];
                        Real val =
                            mono.value(pos.vx, pos.vy, pos.vz, xc, yc, zc, h);
                        // // 对于面自由度用积分求val
                        if (p_to_fine_c_lpid.count(i)&&use_face_intergration) {
                            const auto fine_cid = p_to_fine_c_lpid[i].first;
                            const auto fine_lpid = p_to_fine_c_lpid[i].second;
                            const auto& fine_elem_2d = fine_domain.vem_elements[fine_cid].elements_2d[fine_lpid];
                            val = fine_elem_2d.polygon_moment_general_3d(mono,_3d_vec(xc,yc,zc))
                                /std::pow(h,mono.p+mono.q+mono.r) / fine_elem_2d.area;
                        }
                        // // 对于体自由度用积分求val
                        if (p_to_fine_cid.count(i)&& use_volume_intergration) {
                            const auto& fine_elem = fine_domain.vem_elements[p_to_fine_cid[i]];
                            val = fine_elem.
                                integrate_centered_monomial_over_polyhedron_divergence(mono,_3d_vec(xc,yc,zc))
                            /std::pow(h,mono.p+mono.q+mono.r) / fine_elem.volume;
                        }
                        const Real coeff = poly_coeffs_p(j);
                        weight += coeff * val;
                    }
                    weight_sum+=weight;
                    // if (std::fabs(weight)>1e-3)
                        no_weight_trips.emplace_back(row, global_coarse_dof, weight/deno);
                }

                // -- edges
                for (int eid = 0;eid<element_edof_n;++eid) {
                    int global_coarse_dof = base_e + element.edge_local_to_global_dof[eid];
                    element_dof_values_p.setZero();
                    element_dof_values_p[eid+element_vdof_n] = 1.0;
                    Real weight = 0.0;
                    const Eigen::VectorXr poly_coeffs_p = proj * element_dof_values_p;
                    const Real xc = element.centroid.vx;
                    const Real yc = element.centroid.vy;
                    const Real zc = element.centroid.vz;
                    const Real h  = element.diameter;
                    const int n_basis = (int)monomials_3d.size();
                    for (int j = 0; j < n_basis; ++j) {
                        const auto &mono = monomials_3d[j];
                        Real val =
                            mono.value(pos.vx, pos.vy, pos.vz, xc, yc, zc, h);
                        // // 对于面自由度用积分求val
                        if (p_to_fine_c_lpid.count(i)&&use_face_intergration) {
                            const auto fine_cid = p_to_fine_c_lpid[i].first;
                            const auto fine_lpid = p_to_fine_c_lpid[i].second;
                            const auto& fine_elem_2d = fine_domain.vem_elements[fine_cid].elements_2d[fine_lpid];
                            val = fine_elem_2d.polygon_moment_general_3d(mono,_3d_vec(xc,yc,zc))
                                /std::pow(h,mono.p+mono.q+mono.r) / fine_elem_2d.area;
                        }
                        // // 对于体自由度用积分求val
                        if (p_to_fine_cid.count(i)&& use_volume_intergration) {
                            const auto& fine_elem = fine_domain.vem_elements[p_to_fine_cid[i]];
                            val = fine_elem.
                                integrate_centered_monomial_over_polyhedron_divergence(mono,_3d_vec(xc,yc,zc))
                            /std::pow(h,mono.p+mono.q+mono.r) / fine_elem.volume;
                        }
                        const Real coeff = poly_coeffs_p(j);
                        weight += coeff * val;
                    }
                    weight_sum+=weight;
                    // if (std::fabs(weight)>1e-3)
                        no_weight_trips.emplace_back(row, global_coarse_dof, weight/deno);
                }
                // faces
                for (int fid=0;fid<element_fdof_n;++fid) {
                    int global_coarse_dof = base_f + element.polygon_local_to_global_dof[fid];
                    element_dof_values_p.setZero();
                    element_dof_values_p[fid+element_edof_n+element_vdof_n] = 1.0;
                    Real weight = 0.0;
                    const Eigen::VectorXr poly_coeffs_p = proj * element_dof_values_p;
                    const Real xc = element.centroid.vx;
                    const Real yc = element.centroid.vy;
                    const Real zc = element.centroid.vz;
                    const Real h  = element.diameter;
                    const int n_basis = (int)monomials_3d.size();
                    for (int j = 0; j < n_basis; ++j) {
                        const auto &mono = monomials_3d[j];
                        auto val =
                            mono.value(pos.vx, pos.vy, pos.vz, xc, yc, zc, h);
                        // // 对于面自由度用积分求val
                        if (p_to_fine_c_lpid.count(i)&&use_face_intergration) {
                            const auto fine_cid = p_to_fine_c_lpid[i].first;
                            const auto fine_lpid = p_to_fine_c_lpid[i].second;
                            const auto& fine_elem_2d = fine_domain.vem_elements[fine_cid].elements_2d[fine_lpid];
                            val = fine_elem_2d.polygon_moment_general_3d(mono,_3d_vec(xc,yc,zc))
                                /std::pow(h,mono.p+mono.q+mono.r) / fine_elem_2d.area;
                        }
                        // // 对于体自由度用积分求val
                        if (p_to_fine_cid.count(i)&& use_volume_intergration) {
                            const auto& fine_elem = fine_domain.vem_elements[p_to_fine_cid[i]];
                            val = fine_elem.
                                integrate_centered_monomial_over_polyhedron_divergence(mono,_3d_vec(xc,yc,zc))
                            /std::pow(h,mono.p+mono.q+mono.r) / fine_elem.volume;
                        }
                        const Real coeff = poly_coeffs_p(j);
                        weight += coeff * val;
                    }
                    weight_sum+=weight;
                    // if (std::fabs(weight)>1e-3)
                        no_weight_trips.emplace_back(row, global_coarse_dof, weight/deno);
                }
                // cell
                {
                    int cell_global_coarse_dof = base_c + cid;
                    element_dof_values_p.setZero();
                    element_dof_values_p[element_fdof_n+element_edof_n+element_vdof_n] = 1.0;
                    Real weight = 0.0;
                    const Eigen::VectorXr poly_coeffs_p = proj * element_dof_values_p;
                    const Real xc = element.centroid.vx;
                    const Real yc = element.centroid.vy;
                    const Real zc = element.centroid.vz;
                    const Real h  = element.diameter;
                    const int n_basis = (int)monomials_3d.size();
                    for (int j = 0; j < n_basis; ++j) { //
                        const auto &mono = monomials_3d[j];
                        Real val =
                            mono.value(pos.vx, pos.vy, pos.vz, xc, yc, zc, h);
                        // // 对于面自由度用积分求val
                        if (p_to_fine_c_lpid.count(i)&&use_face_intergration) {
                            const auto fine_cid = p_to_fine_c_lpid[i].first;
                            const auto fine_lpid = p_to_fine_c_lpid[i].second;
                            const auto& fine_elem_2d = fine_domain.vem_elements[fine_cid].elements_2d[fine_lpid];
                            val = fine_elem_2d.polygon_moment_general_3d(mono,_3d_vec(xc,yc,zc))
                                /std::pow(h,mono.p+mono.q+mono.r) / fine_elem_2d.area;
                        }
                        // // 对于体自由度用积分求val
                        if (p_to_fine_cid.count(i)&& use_volume_intergration) {
                            const auto& fine_elem = fine_domain.vem_elements[p_to_fine_cid[i]];
                            val = fine_elem.
                                integrate_centered_monomial_over_polyhedron_divergence(mono,_3d_vec(xc,yc,zc))
                            /std::pow(h,mono.p+mono.q+mono.r)/ fine_elem.volume;
                        }
                        const Real coeff = poly_coeffs_p(j);
                        weight += coeff * val;
                    }
                    weight_sum+=weight;//对一个来说的所以不用除deno
                    // if (std::fabs(weight)>1e-3)
                        no_weight_trips.emplace_back(row, cell_global_coarse_dof, weight/deno);
                }
                RZ_ASSERT(std::fabs(weight_sum-1.0)<1e-2,"");
                for (auto&tri:no_weight_trips) {
#pragma omp critical
                    trip.emplace_back(tri.row(),tri.col(),tri.value()/weight_sum);
                }
            }
        }

        if (!DOFSCALE) {// todo :缩放 weights

        }

        Eigen::SparseMatrixR
        ans(fine_domain.points.size() +
         fine_domain.edges.size()+
         fine_domain.polygons.size()+
         fine_domain.polyhedrons.size(),
         coarse_domain.points.size()+
         coarse_domain.edges.size()+
         coarse_domain.polygons.size()+
         coarse_domain.polyhedrons.size()
         );
        ans.setFromTriplets(trip.begin(), trip.end());
        ans.makeCompressed();

        Real tol = 1e-20;
        const auto &P = ans;
        const int rows = P.rows();
        const int cols = P.cols();

        std::vector<char> has_nz(cols, 0);

        for (int k = 0; k < P.outerSize(); ++k) { // RowMajor: outer = 行
            for (Eigen::SparseMatrixR::InnerIterator it(P, k); it; ++it) {
                if (std::abs(it.value()) > tol) {
                    has_nz[it.col()] = 1;
                }
            }
        }

        std::vector<int> old_to_new(cols, -1);
        int new_col_count = 0;
        for (int c = 0; c < cols; ++c) {
            if (has_nz[c]) {
                old_to_new[c] = new_col_count++;
            }
        }

        if (new_col_count == cols) {
            return ans;
        }

        std::vector<Eigen::Triplet<GlobalReal>> new_trip;
        new_trip.reserve(ans.nonZeros());

        for (int k = 0; k < ans.outerSize(); ++k) {
            for (Eigen::SparseMatrixR::InnerIterator it(ans, k); it; ++it) {
                int new_col = old_to_new[it.col()];
                if (new_col >= 0) {
                    new_trip.emplace_back(it.row(), new_col, it.value());
                }
            }
        }

        Eigen::SparseMatrixR ans_compact(rows, new_col_count);
        ans_compact.setFromTriplets(new_trip.begin(), new_trip.end());
        ans_compact.makeCompressed();

        RZ_INFO("[prolongation_k2_sparse] remove zero columns: ",
                cols - new_col_count,
                ", new cols = ", new_col_count);

        return ans_compact;

    }

Eigen::SparseMatrixR VEM_GMGSolver::prolongation_k2_sparse_fast(int ll, int hl) {
 {
        FUNC_TIMER();
        const auto&coarse_domain = mGrids[ll];
        const auto&fine_domain = mGrids[hl];
        std::vector<Eigen::Triplet<GlobalReal>> trip;
        const auto&vem_elements = coarse_domain.vem_elements;
        bool use_volume_intergration = false;
        bool use_face_intergration = false;
        std::vector<_3d_vec> fine_points;
        std::unordered_map<int,int>p_to_dof;
        std::unordered_map<int,int> p_to_fine_cid;
        std::unordered_map<int,std::pair<int,int>> p_to_fine_c_lpid;
        std::unordered_set<int>inserted_pid;
        int vdofn = fine_domain.points.size();
        int edofn = fine_domain.edges.size();
        int fdofn = fine_domain.polygons.size();
        for (auto&p:fine_domain.points) {
            p_to_dof[fine_points.size()] = fine_points.size();
            fine_points.push_back(p);
        }
        for (int eid = 0; eid < (int)fine_domain.edges.size(); ++eid) {
            const auto& edge = fine_domain.edges[eid];
            const auto& p0 = fine_domain.points[edge.v1];
            const auto& p1 = fine_domain.points[edge.v2];
            _3d_vec mid = (p0+p1)/2.0;
            p_to_dof[fine_points.size()] = vdofn + eid;
            fine_points.push_back(mid);
        }
        for (int cid = 0;cid<fine_domain.vem_elements.size();++cid) {
            const auto&vem_element = fine_domain.vem_elements[cid];
            int local_pid=0;
            for (auto&pid:vem_element.polygon_local_to_global_dof) {
                if (!inserted_pid.count(pid)) {
                    inserted_pid.insert(pid);
                    p_to_dof[fine_points.size()] = vdofn+edofn+pid;
                    p_to_fine_c_lpid[fine_points.size()] = std::make_pair(cid,local_pid);
                    fine_points.push_back(fine_domain.polygons_plane[pid]._3d_centroid);
                }
                ++local_pid;
            }
        }

        for (int cid = 0;cid<fine_domain.vem_elements.size();++cid) {
            p_to_dof[fine_points.size()] = vdofn+edofn+fdofn+cid;
            p_to_fine_cid[fine_points.size()] = cid;
            fine_points.push_back(fine_domain.vem_elements[cid].centroid);
        }

        std::vector<std::vector<int>> contained_indices;
        assign_points_to_elements(contained_indices,fine_points,coarse_domain);
        // //---------------------------------------------------
        const int base_e = (int)coarse_domain.points.size();
        const int base_f = base_e + (int)coarse_domain.edges.size();
        const int base_c = base_f + (int)coarse_domain.polygons.size();
// #pragma omp parallel for
        for (int i = 0;i < fine_points.size(); ++i) {
            if (contained_indices[i].size()==0)continue;
            const int row = p_to_dof[i];
            const auto pos = fine_points[i];
            // 判断是否可以继承于粗层级的某一个节点
            int corr_coarse_id = -1;
            if (row<fine_domain.points.size()) {
                for (const auto& cid:contained_indices[i]) {
                    const auto& element = vem_elements[cid];
                    for (int local_vid=0;local_vid<element.vertex_count();++local_vid) {
                        if ((element.vertex_position(local_vid)-pos).norm()<1e-15) {
                            corr_coarse_id = element.vertex_local_to_global_dof[local_vid];
                            break;
                        }
                    }
                    if (corr_coarse_id>=0)break;
                }
            }
            if (corr_coarse_id>=0) {
#pragma omp critical
                trip.emplace_back(row, corr_coarse_id, 1.0);
                continue;
            }
            // 判断是否落在粗层级的某一个边上 若是 则用 这个fine point边相关的所有面来插值
            //如果和面一样都是用 所有的 体呢？ todo
            std::set<int> active_dofs;
            std::vector<Eigen::Triplet<GlobalReal>> edge_triplets;
            if (row<fine_domain.points.size() + fine_domain.edges.size()
                && on_coarse_edge(pos,row,edge_triplets,contained_indices[i],ll,active_dofs)) {
                RZ_ASSERT(edge_triplets.size()>0,"edge_triplets not initialized");
                for (auto&tri:edge_triplets) {
    #pragma omp critical
                    trip.emplace_back(tri.row(),tri.col(),tri.value());
                }
                //----------------------------------------------------------
//                 std::vector<Eigen::Triplet<GlobalReal>> no_weight_trips;
//                 for (const auto& cid:contained_indices[i]) {
//                     const auto& element_3d = vem_elements[cid];
//                     for (int local_poly_id =0;local_poly_id<element_3d.elements_2d.size();++local_poly_id) {
//                         VEM2d_Prolongation(row,local_poly_id,element_3d,no_weight_trips,pos,mGrids[ll]);
//                     }
//                 }
//                 Real weight_sum = 0.0;
//                 for (auto&tri:no_weight_trips) {
//                     if (!active_dofs.count(tri.col()))continue;
//                     weight_sum += tri.value();
//                 }
//                 for (auto&tri:no_weight_trips) {
//                     if (!active_dofs.count(tri.col()))continue;
// #pragma omp critical
//                     trip.emplace_back(tri.row(),tri.col(),tri.value()/weight_sum);
//                 }
                continue;
            }

            if (contained_indices[i].size()>1) {
                // 找到这个pos在哪一个vem_element_2上
                std::vector<Eigen::Triplet<GlobalReal>> no_weight_trips;
                bool found = false;
                for (const auto& cid:contained_indices[i]) {
                    const auto& element_3d = vem_elements[cid];
                    for (int local_poly_id =0;local_poly_id<element_3d.elements_2d.size();++local_poly_id)
                    {
                        const auto& element_2d = element_3d.elements_2d[local_poly_id];
                        CGAL_Point_3 cgal_point(pos.vx,pos.vy,pos.vz);
                        auto pos_2d = element_2d.plane._to_2d(pos);
                        if (std::sqrt(CGAL::squared_distance(element_2d.plane.cgal_plane,cgal_point))>1e-12)continue;
                        // 要判定这个点是否在这个element_2d内部
                        if (!point_in_convex_hull(pos_2d,element_2d.points))continue;
                        // 就在这个平面上 用 vem_element_2 的 自由度 prolongation 到这个 fine dof 上
                        found = true;
                        Real weight_sum = VEM2d_Prolongation(row,local_poly_id,element_3d,no_weight_trips,pos,mGrids[ll]);
                        for (auto&tri:no_weight_trips) {
                            #pragma omp critical
                            trip.emplace_back(tri.row(),tri.col(),tri.value()/weight_sum);
                        }
                        break;
                    }

                    if (found)break;
                }

                if (!found) {
                    std::vector<_3d_vec>points;
                    points.push_back(pos);
                    this_save_points_to_xyz(points,"on_plane.xyz");
                    for (const auto& tcid:contained_indices[i]) {
                        vem_elements[tcid].save_polygonmesh_to_obj(std::to_string(tcid)+".obj");
                    }
                    RZ_FATAL("not found");
                }
                continue;
            }
//---------------------------------------------------------------------------------------
            RZ_ASSERT(contained_indices[i].size() == 1,"");

            for (const auto& cid:contained_indices[i]) {
                std::vector<Eigen::Triplet<GlobalReal>> no_weight_trips;
                Real weight_sum = 0.0;
                Real deno = contained_indices[i].size();
                const auto& element = vem_elements[cid];
                int element_dof_n = element.vertex_count()+element.edge_count()
                    +element.polygons.size()+1;
                int element_vdof_n = element.vertex_count();
                int element_edof_n = element.edge_count();
                int element_fdof_n = element.polygons.size();
                Eigen::MatrixXr proj;
                if (element.is_U)proj = coarse_domain.U3d_project_matrix_star_k2;
                else proj = element.project_matrix_star_k2;
                Eigen::VectorXr element_dof_values_p;
                element_dof_values_p.resize(element_dof_n); // 可在函数外声明缓存

                // -- vertices
                for (int vid = 0;vid<element_vdof_n;++vid) {
                    int global_coarse_dof = element.vertex_local_to_global_dof[vid];
                    element_dof_values_p.setZero();
                    element_dof_values_p[vid] = 1.0;
                    Real weight = 0.0;
                    const Eigen::VectorXr poly_coeffs_p = proj * element_dof_values_p;
                    const Real xc = element.centroid.vx;
                    const Real yc = element.centroid.vy;
                    const Real zc = element.centroid.vz;
                    const Real h  = element.diameter;
                    const int n_basis = (int)monomials_3d.size();
                    for (int j = 0; j < n_basis; ++j) {
                        const auto &mono = monomials_3d[j];
                        Real val =
                            mono.value(pos.vx, pos.vy, pos.vz, xc, yc, zc, h);
                        // // 对于面自由度用积分求val
                        if (p_to_fine_c_lpid.count(i)&&use_face_intergration) {
                            const auto fine_cid = p_to_fine_c_lpid[i].first;
                            const auto fine_lpid = p_to_fine_c_lpid[i].second;
                            const auto& fine_elem_2d = fine_domain.vem_elements[fine_cid].elements_2d[fine_lpid];
                            val = fine_elem_2d.polygon_moment_general_3d(mono,_3d_vec(xc,yc,zc))
                                /std::pow(h,mono.p+mono.q+mono.r) / fine_elem_2d.area;
                        }
                        // // 对于体自由度用积分求val
                        if (p_to_fine_cid.count(i)&& use_volume_intergration) {
                            const auto& fine_elem = fine_domain.vem_elements[p_to_fine_cid[i]];
                            val = fine_elem.
                                integrate_centered_monomial_over_polyhedron_divergence(mono,_3d_vec(xc,yc,zc))
                            /std::pow(h,mono.p+mono.q+mono.r) / fine_elem.volume;
                        }
                        const Real coeff = poly_coeffs_p(j);
                        weight += coeff * val;
                    }
                    weight_sum+=weight;
                    // if (std::fabs(weight)>1e-3)
                        no_weight_trips.emplace_back(row, global_coarse_dof, weight/deno);
                }

                // -- edges
                for (int eid = 0;eid<element_edof_n;++eid) {
                    int global_coarse_dof = base_e + element.edge_local_to_global_dof[eid];
                    element_dof_values_p.setZero();
                    element_dof_values_p[eid+element_vdof_n] = 1.0;
                    Real weight = 0.0;
                    const Eigen::VectorXr poly_coeffs_p = proj * element_dof_values_p;
                    const Real xc = element.centroid.vx;
                    const Real yc = element.centroid.vy;
                    const Real zc = element.centroid.vz;
                    const Real h  = element.diameter;
                    const int n_basis = (int)monomials_3d.size();
                    for (int j = 0; j < n_basis; ++j) {
                        const auto &mono = monomials_3d[j];
                        Real val =
                            mono.value(pos.vx, pos.vy, pos.vz, xc, yc, zc, h);
                        // // 对于面自由度用积分求val
                        if (p_to_fine_c_lpid.count(i)&&use_face_intergration) {
                            const auto fine_cid = p_to_fine_c_lpid[i].first;
                            const auto fine_lpid = p_to_fine_c_lpid[i].second;
                            const auto& fine_elem_2d = fine_domain.vem_elements[fine_cid].elements_2d[fine_lpid];
                            val = fine_elem_2d.polygon_moment_general_3d(mono,_3d_vec(xc,yc,zc))
                                /std::pow(h,mono.p+mono.q+mono.r) / fine_elem_2d.area;
                        }
                        // // 对于体自由度用积分求val
                        if (p_to_fine_cid.count(i)&& use_volume_intergration) {
                            const auto& fine_elem = fine_domain.vem_elements[p_to_fine_cid[i]];
                            val = fine_elem.
                                integrate_centered_monomial_over_polyhedron_divergence(mono,_3d_vec(xc,yc,zc))
                            /std::pow(h,mono.p+mono.q+mono.r) / fine_elem.volume;
                        }
                        const Real coeff = poly_coeffs_p(j);
                        weight += coeff * val;
                    }
                    weight_sum+=weight;
                    // if (std::fabs(weight)>1e-3)
                        no_weight_trips.emplace_back(row, global_coarse_dof, weight/deno);
                }
                // faces
                for (int fid=0;fid<element_fdof_n;++fid) {
                    int global_coarse_dof = base_f + element.polygon_local_to_global_dof[fid];
                    element_dof_values_p.setZero();
                    element_dof_values_p[fid+element_edof_n+element_vdof_n] = 1.0;
                    Real weight = 0.0;
                    const Eigen::VectorXr poly_coeffs_p = proj * element_dof_values_p;
                    const Real xc = element.centroid.vx;
                    const Real yc = element.centroid.vy;
                    const Real zc = element.centroid.vz;
                    const Real h  = element.diameter;
                    const int n_basis = (int)monomials_3d.size();
                    for (int j = 0; j < n_basis; ++j) {
                        const auto &mono = monomials_3d[j];
                        auto val =
                            mono.value(pos.vx, pos.vy, pos.vz, xc, yc, zc, h);
                        // // 对于面自由度用积分求val
                        if (p_to_fine_c_lpid.count(i)&&use_face_intergration) {
                            const auto fine_cid = p_to_fine_c_lpid[i].first;
                            const auto fine_lpid = p_to_fine_c_lpid[i].second;
                            const auto& fine_elem_2d = fine_domain.vem_elements[fine_cid].elements_2d[fine_lpid];
                            val = fine_elem_2d.polygon_moment_general_3d(mono,_3d_vec(xc,yc,zc))
                                /std::pow(h,mono.p+mono.q+mono.r) / fine_elem_2d.area;
                        }
                        // // 对于体自由度用积分求val
                        if (p_to_fine_cid.count(i)&& use_volume_intergration) {
                            const auto& fine_elem = fine_domain.vem_elements[p_to_fine_cid[i]];
                            val = fine_elem.
                                integrate_centered_monomial_over_polyhedron_divergence(mono,_3d_vec(xc,yc,zc))
                            /std::pow(h,mono.p+mono.q+mono.r) / fine_elem.volume;
                        }
                        const Real coeff = poly_coeffs_p(j);
                        weight += coeff * val;
                    }
                    weight_sum+=weight;
                    // if (std::fabs(weight)>1e-3)
                        no_weight_trips.emplace_back(row, global_coarse_dof, weight/deno);
                }
                // cell
                {
                    int cell_global_coarse_dof = base_c + cid;
                    element_dof_values_p.setZero();
                    element_dof_values_p[element_fdof_n+element_edof_n+element_vdof_n] = 1.0;
                    Real weight = 0.0;
                    const Eigen::VectorXr poly_coeffs_p = proj * element_dof_values_p;
                    const Real xc = element.centroid.vx;
                    const Real yc = element.centroid.vy;
                    const Real zc = element.centroid.vz;
                    const Real h  = element.diameter;
                    const int n_basis = (int)monomials_3d.size();
                    for (int j = 0; j < n_basis; ++j) { //
                        const auto &mono = monomials_3d[j];
                        Real val =
                            mono.value(pos.vx, pos.vy, pos.vz, xc, yc, zc, h);
                        // // 对于面自由度用积分求val
                        if (p_to_fine_c_lpid.count(i)&&use_face_intergration) {
                            const auto fine_cid = p_to_fine_c_lpid[i].first;
                            const auto fine_lpid = p_to_fine_c_lpid[i].second;
                            const auto& fine_elem_2d = fine_domain.vem_elements[fine_cid].elements_2d[fine_lpid];
                            val = fine_elem_2d.polygon_moment_general_3d(mono,_3d_vec(xc,yc,zc))
                                /std::pow(h,mono.p+mono.q+mono.r) / fine_elem_2d.area;
                        }
                        // // 对于体自由度用积分求val
                        if (p_to_fine_cid.count(i)&& use_volume_intergration) {
                            const auto& fine_elem = fine_domain.vem_elements[p_to_fine_cid[i]];
                            val = fine_elem.
                                integrate_centered_monomial_over_polyhedron_divergence(mono,_3d_vec(xc,yc,zc))
                            /std::pow(h,mono.p+mono.q+mono.r)/ fine_elem.volume;
                        }
                        const Real coeff = poly_coeffs_p(j);
                        weight += coeff * val;
                    }
                    weight_sum+=weight;//对一个来说的所以不用除deno
                    // if (std::fabs(weight)>1e-3)
                        no_weight_trips.emplace_back(row, cell_global_coarse_dof, weight/deno);
                }
                RZ_ASSERT(std::fabs(weight_sum-1.0)<1e-2,"");
                for (auto&tri:no_weight_trips) {
#pragma omp critical
                    trip.emplace_back(tri.row(),tri.col(),tri.value()/weight_sum);
                }
            }
        }

        if (!DOFSCALE) {// todo :缩放 weights

        }

        Eigen::SparseMatrixR
        ans(fine_domain.points.size() +
         fine_domain.edges.size()+
         fine_domain.polygons.size()+
         fine_domain.polyhedrons.size(),
         coarse_domain.points.size()+
         coarse_domain.edges.size()+
         coarse_domain.polygons.size()+
         coarse_domain.polyhedrons.size()
         );
        ans.setFromTriplets(trip.begin(), trip.end());
        ans.makeCompressed();

        Real tol = 1e-20;
        const auto &P = ans;
        const int rows = P.rows();
        const int cols = P.cols();

        std::vector<char> has_nz(cols, 0);

        for (int k = 0; k < P.outerSize(); ++k) { // RowMajor: outer = 行
            for (Eigen::SparseMatrixR::InnerIterator it(P, k); it; ++it) {
                if (std::abs(it.value()) > tol) {
                    has_nz[it.col()] = 1;
                }
            }
        }

        std::vector<int> old_to_new(cols, -1);
        int new_col_count = 0;
        for (int c = 0; c < cols; ++c) {
            if (has_nz[c]) {
                old_to_new[c] = new_col_count++;
            }
        }

        if (new_col_count == cols) {
            return ans;
        }

        std::vector<Eigen::Triplet<GlobalReal>> new_trip;
        new_trip.reserve(ans.nonZeros());

        for (int k = 0; k < ans.outerSize(); ++k) {
            for (Eigen::SparseMatrixR::InnerIterator it(ans, k); it; ++it) {
                int new_col = old_to_new[it.col()];
                if (new_col >= 0) {
                    new_trip.emplace_back(it.row(), new_col, it.value());
                }
            }
        }

        Eigen::SparseMatrixR ans_compact(rows, new_col_count);
        ans_compact.setFromTriplets(new_trip.begin(), new_trip.end());
        ans_compact.makeCompressed();

        RZ_INFO("[prolongation_k2_sparse] remove zero columns: ",
                cols - new_col_count,
                ", new cols = ", new_col_count);

        return ans_compact;

    }
}

Eigen::SparseMatrixR VEM_GMGSolver::k1_prolongation_k2(int ll, int hl){// 思考下 这个是否等价于 k1 - k2 PMG 如果两个网格一样的情况下 应该是等价的  所以我们可以快速的试一下
        const auto&coarse_domain = mGrids[ll];
        const auto&fine_domain = mGrids[hl];
        std::vector<Eigen::Triplet<GlobalReal>> trip;
        const auto&vem_elements = coarse_domain.vem_elements;
        std::vector<_3d_vec> fine_points;
        std::unordered_map<int,int>p_to_dof;
        std::map<int,int> p_to_fine_cid;
        std::map<int,std::pair<int,int>> p_to_fine_c_lpid;
        std::set<int>inserted_pid;
        int vdofn = fine_domain.points.size();
        int edofn = fine_domain.edges.size();
        int fdofn = fine_domain.polygons.size();
        for (auto&p:fine_domain.points) {
            p_to_dof[fine_points.size()] = fine_points.size();
            fine_points.push_back(p);
        }
        for (int eid = 0; eid < (int)fine_domain.edges.size(); ++eid) {
            const auto& edge = fine_domain.edges[eid];
            const auto& p0 = fine_domain.points[edge.v1];
            const auto& p1 = fine_domain.points[edge.v2];
            _3d_vec mid = (p0+p1)/2.0;
            p_to_dof[fine_points.size()] = vdofn + eid;
            fine_points.push_back(mid);
        }
        for (int cid = 0;cid<fine_domain.vem_elements.size();++cid) {
            const auto&vem_element = fine_domain.vem_elements[cid];
            int local_pid=0;
            for (auto&pid:vem_element.polygon_local_to_global_dof) {
                if (!inserted_pid.count(pid)) {
                    inserted_pid.insert(pid);
                    p_to_dof[fine_points.size()] = vdofn+edofn+pid;
                    p_to_fine_c_lpid[fine_points.size()] = std::make_pair(cid,local_pid);
                    fine_points.push_back(fine_domain.polygons_plane[pid]._3d_centroid);
                }
                ++local_pid;
            }
        }
        for (int cid = 0;cid<fine_domain.vem_elements.size();++cid) {
            p_to_dof[fine_points.size()] = vdofn+edofn+fdofn+cid;
            p_to_fine_cid[fine_points.size()] = cid;
            fine_points.push_back(fine_domain.vem_elements[cid].centroid);
        }

        std::vector<std::vector<int>> contained_indices;
        assign_points_to_elements(contained_indices,fine_points,coarse_domain);
        RZ_ASSERT(contained_indices.size() == fine_points.size(),
            contained_indices.size()," ",fine_points.size());

        // #pragma omp parallel for
        for (int i = 0; i < fine_points.size(); ++i) {
            const int row = p_to_dof[i];
            const auto pos = fine_points[i];

            RZ_ASSERT (!contained_indices[i].empty(),"");

            // 每个线程分别处理不同的 cid
            for (const auto& cid : contained_indices[i]) {
                const auto& element = vem_elements[cid];
                int element_vdof_n = element.vertex_count();
                const auto& proj = element.project_matrix_star_k1;
                Eigen::VectorXr element_dof_values_p;
                element_dof_values_p.resize(element_vdof_n); // 可在函数外声明缓存

                // -- vertices
                for (int vid = 0; vid < element_vdof_n; ++vid) {
                    int global_coarse_dof = element.vertex_local_to_global_dof[vid];
                    element_dof_values_p.setZero();
                    element_dof_values_p[vid] = 1.0;
                    Real weight = 0.0;
                    const Eigen::VectorXr poly_coeffs_p = proj * element_dof_values_p;
                    const Real xc = element.centroid.vx;
                    const Real yc = element.centroid.vy;
                    const Real zc = element.centroid.vz;
                    const Real h = element.diameter;
                    const int n_basis = (int)monomials_3d.size();

                    for (int j = 0; j < n_basis; ++j) {
                        const auto& mono = monomials_3d[j];
                        Real val = mono.value(pos.vx, pos.vy, pos.vz, xc, yc, zc, h);

                        if (p_to_fine_cid.count(i)) {
                            const auto& fine_elem = fine_domain.vem_elements[p_to_fine_cid[i]];
                            val = fine_elem.integrate_centered_monomial_over_polyhedron_divergence(mono, _3d_vec(xc, yc, zc))
                                / std::pow(h, mono.p + mono.q + mono.r) / fine_elem.volume;
                        }

                        const Real coeff = poly_coeffs_p(j);
                        weight += coeff * val;
                    }

                    #pragma omp critical
                    trip.emplace_back(row, global_coarse_dof, weight / contained_indices[i].size());
                }
            }
        }

        Eigen::SparseMatrixR
        ans(fine_domain.points.size() +
         fine_domain.edges.size()+
         fine_domain.polygons.size()+
         fine_domain.polyhedrons.size(),
         coarse_domain.points.size()+
         coarse_domain.edges.size()+
         coarse_domain.polygons.size()+
         coarse_domain.polyhedrons.size()
         );
        ans.setFromTriplets(trip.begin(), trip.end());
        ans.makeCompressed();
        return ans;
    }

void VEM_GMGSolver::assign_points_to_elements(std::vector<std::vector<int>> &contained_indices,
    std::vector<_3d_vec> &points, const VEM_Domain_3d &vem_domain)  {
        FUNC_TIMER();
        const auto&vem_elements = vem_domain.vem_elements;
        contained_indices.resize(points.size());
        const auto grid_res = vem_domain.grid_res;
        auto get_grid_index = [&](const _3d_vec &p) noexcept -> int {
            int gx = std::min(grid_res - 1, std::max(0, int(p.vx * grid_res)));
            int gy = std::min(grid_res - 1, std::max(0, int(p.vy * grid_res)));
            int gz = std::min(grid_res - 1, std::max(0, int(p.vz * grid_res)));
            return (gz * grid_res + gy) * grid_res + gx;
        };
        const auto& grid_to_elems = vem_domain.grid_to_elems;
        for (int k = 0;k < points.size(); ++k)
        {
            auto pos = points[k];
            const int gidx = get_grid_index(pos);
            RZ_ASSERT(gidx>=0 && gidx<grid_to_elems.size()," ",grid_to_elems.size());
            const auto& cand_elems = grid_to_elems[gidx];
            bool assigned = false;
            for (int eid : cand_elems) {
                RZ_ASSERT(eid<vem_elements.size(),eid," ",vem_elements.size());
                if (vem_elements[eid].point_in_elem(pos)) {
                    contained_indices[k].push_back(eid);
                    assigned = true;
                }
            }
            if (!assigned) {
                continue;
                RZ_FATAL("");
                std::vector<_3d_vec>save_points;
                save_points.push_back(pos);
                this_save_points_to_xyz(save_points,"on_plane.xyz");
                for (const auto& tcid:cand_elems) {
                    vem_elements[tcid].save_polygonmesh_to_obj(std::to_string(tcid)+".obj");
                }
                bool can_assign = false;
                for (auto&vem_3d:vem_elements)if (vem_3d.point_in_elem(pos))can_assign = true;
                if (can_assign)RZ_INFO("can");
                RZ_INFO(pos.vx * grid_res," ",pos.vy * grid_res," ",pos.vz * grid_res);
                RZ_FATAL("not assigned");
                // continue;
                // 特殊处理一下， 先分到最近的一个点所在元素
                Real min_dis = 2;
                int min_coarse_vid = -1;
                if (cand_elems.size()) {
                    for (int eid : cand_elems) {
                        const auto& coarse_elem = vem_elements[eid];
                        for (int vid=0;vid<coarse_elem.vertex_count();++vid) {
                            Real dis = (coarse_elem.vertex_position(vid) - pos).squared_norm();
                            if (dis < min_dis) {
                                min_dis = dis;
                                min_coarse_vid = coarse_elem.vertex_local_to_global_dof[vid];
                            }
                        }
                    }
                }
                else {
                    for (int vid=0;vid<vem_domain.points.size();++vid) {
                        Real dis = (vem_domain.points[vid] - pos).squared_norm();
                        if (dis < min_dis) {
                            min_dis = dis;
                            min_coarse_vid = vid;
                        }
                    }
                }
                RZ_ASSERT(min_coarse_vid!=-1,"");
                points[k] = vem_domain.points[min_coarse_vid];//
                if (cand_elems.size()) {
                    for (int eid : cand_elems) {
                        if (vem_elements[eid].point_in_elem(points[k])) {
                            contained_indices[k].push_back(eid);
                        }
                    }
                }else {
                    int eid =0;
                    for (const auto&elem:vem_elements) {
                        for (const auto&vid:elem.vertex_local_to_global_dof)
                            if (vid == min_coarse_vid) contained_indices[k].push_back(eid);
                        ++eid;
                    }
                }
            }
        }
        RZ_ASSERT(contained_indices.size() == points.size(),
            contained_indices.size()," ",points.size());
    }

bool VEM_GMGSolver::isDiagonalDominant(const Eigen::SparseMatrixR &A)  {
    // 获取矩阵的行数和列数
    int rows = A.rows();

    // 遍历每一行
    for (int i = 0; i < rows; ++i) {
        Real rowSum = 0.0;
        Real diagonalValue = 0.0;

        // 遍历当前行的所有非零元素
        for (Eigen::SparseMatrixR::InnerIterator it(A, i); it; ++it) {
            if (it.col() == i) {
                diagonalValue = it.value();  // 获取对角线元素
            } else {
                rowSum += std::abs(it.value());  // 累加非对角元素的绝对值
            }
        }

        // 判断当前行是否满足对角占优条件
        if (std::abs(diagonalValue) <= rowSum) {
            return false;  // 只要有一行不满足对角占优条件，就返回false
        }
    }

    return true;  // 如果所有行都满足条件，则返回true
}

bool VEM_GMGSolver::on_coarse_edge(const _3d_vec &pos, const int row, std::vector<Eigen::Triplet<GlobalReal>> &trip,
    const std::vector<int> &contained_coarse_cells, int ll, std::set<int> &active_dofs) {
        const auto& domain = mGrids[ll];
        const auto& elements = domain.vem_elements;
        const int base_e = domain.points.size();
        for (const auto& cid:contained_coarse_cells) {
            const auto& element_3d = elements[cid];
            for (int local_poly_id =0;local_poly_id<element_3d.elements_2d.size();++local_poly_id) {
                const auto& element_2d = element_3d.elements_2d[local_poly_id];
                CGAL_Point_3 cgal_point(pos.vx,pos.vy,pos.vz);
                auto pos_2d = element_2d.plane._to_2d(pos);
                if (std::sqrt(CGAL::squared_distance(element_2d.plane.cgal_plane,cgal_point))>1e-12)continue;
                if (!point_in_convex_hull(pos_2d,element_2d.points))continue;
                std::map<int,int>local_v_to_global_v;
                for (const auto&q_v:element_2d.volume_v_to_local_v) {
                    local_v_to_global_v[q_v.second] = element_3d.vertex_local_to_global_dof[q_v.first];
                }
                for (int eid = 0;eid<element_2d.points.size();++eid) {
                    CGAL::Point_2<K> p1(element_2d.points[eid].vx,element_2d.points[eid].vy);
                    CGAL::Point_2<K> p2(element_2d.points[(eid+1)%element_2d.points.size()].vx,
                        element_2d.points[(eid+1)%element_2d.points.size()].vy);
                    CGAL::Segment_2<K> seg(p1,p2);
                    CGAL::Point_2<K> qy(pos_2d.vx,pos_2d.vy);
                    if (std::sqrt(CGAL::squared_distance(seg,qy))<1e-12) {
                        int global_v1_dof = local_v_to_global_v[eid];
                        int global_v2_dof = local_v_to_global_v[(eid+1)%element_2d.points.size()];
                        int global_edge_dof = base_e + element_3d.global_edge_id_from_vertices(
                            local_v_to_global_v[eid],
                            local_v_to_global_v[(eid+1)%element_2d.points.size()]);
                        active_dofs.insert(global_edge_dof);
                        active_dofs.insert(global_v1_dof);
                        active_dofs.insert(global_v2_dof);
                        //check
                        _3d_vec p1_3d = domain.points[global_v1_dof];
                        _3d_vec p2_3d = domain.points[global_v2_dof];
                        CGAL_Point_3 cg_p1_3d(p1_3d.vx,p1_3d.vy,p1_3d.vz);
                        CGAL_Point_3 cg_p2_3d(p2_3d.vx,p2_3d.vy,p2_3d.vz);
                        CGAL::Segment_3<K> seg_3d(cg_p1_3d,cg_p2_3d);
                        RZ_ASSERT(CGAL::squared_distance(seg_3d,cgal_point)<1e-22,"");

                        // compute t
                        Real seg_length = (domain.points[global_v1_dof] - domain.points[global_v2_dof]).norm();
                        Real to_p1 = (pos-domain.points[global_v1_dof]).norm();
                        Real t = -1+to_p1/seg_length;

                        Real coff1 = t*(t-1)/2;
                        Real coffmid = (1-t*t);
                        Real coff2 = t*(t+1)/2;

                        trip.emplace_back(row,global_v1_dof,coff1);
                        trip.emplace_back(row,global_v2_dof,coff2);
                        trip.emplace_back(row,global_edge_dof,coffmid);


                        return true;
                    }
                }

            }
        }

        return false;
    }

Real VEM_GMGSolver::VEM2d_Prolongation(const int row, const int local_poly_id, const VEM_Element_3d &element_3d,
    std::vector<Eigen::Triplet<GlobalReal>> &no_weighted_trips, const _3d_vec &pos, VEM_Domain_3d &coarse_domain,
    bool is_face_dof) {
        std::vector<VEM_Monomial_2d> monomials_2d;
        monomials_2d = {
            VEM_Monomial_2d(0, 0),
            VEM_Monomial_2d(1, 0),
            VEM_Monomial_2d(0, 1),
            VEM_Monomial_2d(2, 0),
            VEM_Monomial_2d(1, 1),
            VEM_Monomial_2d(0, 2)
        };
        const int base_e = coarse_domain.points.size();
        const int base_f = base_e + coarse_domain.edges.size();
        const auto& element_2d = element_3d.elements_2d[local_poly_id];
        const int global_poly_id = element_3d.polygon_local_to_global_dof[local_poly_id];
        CGAL_Point_3 cgal_point(pos.vx,pos.vy,pos.vz);
        auto pos_2d = element_2d.plane._to_2d(pos);
        Real weight_sum = 0.0;
        if (std::sqrt(CGAL::squared_distance(element_2d.plane.cgal_plane,cgal_point))>1e-12)return weight_sum;
        if (!point_in_convex_hull(pos_2d,element_2d.points))return weight_sum;
        std::map<int,int>local_v_to_global_v;
        for (const auto&q_v:element_2d.volume_v_to_local_v) {
            local_v_to_global_v[q_v.second] = element_3d.vertex_local_to_global_dof[q_v.first];
        }
        //---------------------------------------------------
        Eigen::MatrixXr proj_2d;
        if (element_3d.is_U)proj_2d =coarse_domain.U2d_project_matrix_star;
        else    proj_2d = element_2d.project_matrix_star;
        // const auto &proj_2d = element_2d.project_matrix_star;
        Eigen::VectorXr element_dof_values_p;
        element_dof_values_p.resize(element_2d.points.size()*2 + 1);
        // vertices
        for (int vid = 0;vid<element_2d.points.size();++vid) {
            int global_coarse_dof = local_v_to_global_v[vid];
            element_dof_values_p.setZero();
            element_dof_values_p[vid] = 1.0;
            Real weight = 0.0;
            const Eigen::VectorXr poly_coeffs_p = proj_2d * element_dof_values_p;
            const Real xc = element_2d.centroid.vx;
            const Real yc = element_2d.centroid.vy;
            const Real h  = element_2d.diameter;

            const int n_basis = (int)monomials_2d.size();
            for (int j = 0; j < n_basis; ++j) {
                const auto &mono = monomials_2d[j];
                Real val =
                    mono.value(pos_2d.vx, pos_2d.vy, xc, yc,h);
                const Real coeff = poly_coeffs_p(j);
                weight += coeff * val;
            }
            if (weight>1e2) {
                std::vector<_3d_vec>points;
                points.push_back(pos);
                this_save_points_to_xyz(points,"on_plane.xyz");
                element_3d.save_polygonmesh_to_obj("on_plane.obj");
                element_3d.save_face_to_obj(element_3d.polygons[local_poly_id]);
                points.clear();
                for (auto pii:local_v_to_global_v)points.push_back(coarse_domain.points[pii.second]);
                this_save_points_to_xyz(points,"dof_cehck.xyz");
                exit(0);
            }
            weight_sum+=weight;
            RZ_ASSERT(weight<10.0,"");
            // if (std::fabs(weight)>1e-3)
            no_weighted_trips.emplace_back(row, global_coarse_dof, weight);
        }
        // edges
        for (int eid = 0;eid<element_2d.points.size();++eid) {
            int global_coarse_dof = base_e + element_3d.global_edge_id_from_vertices(
                local_v_to_global_v[eid],
                local_v_to_global_v[(eid+1)%element_2d.points.size()]);
            element_dof_values_p.setZero();
            element_dof_values_p[eid+element_2d.points.size()] = 1.0;
            Real weight = 0.0;
            const Eigen::VectorXr poly_coeffs_p = proj_2d * element_dof_values_p;
            const Real xc = element_2d.centroid.vx;
            const Real yc = element_2d.centroid.vy;
            const Real h  = element_2d.diameter;
            const int n_basis = (int)monomials_2d.size();
            for (int j = 0; j < n_basis; ++j) {
                const auto &mono = monomials_2d[j];
                Real val =
                    mono.value(pos_2d.vx, pos_2d.vy, xc, yc,h);
                const Real coeff = poly_coeffs_p(j);
                weight += coeff * val;
            }
            weight_sum+=weight;
            RZ_ASSERT(weight<10.0,"");
            // if (std::fabs(weight)>1e-3)
            no_weighted_trips.emplace_back(row, global_coarse_dof, weight);
        }
        // face

        int global_coarse_dof = base_f + global_poly_id;
        element_dof_values_p.setZero();
        element_dof_values_p[element_2d.points.size()*2] = 1.0;
        Real weight = 0.0;
        const Eigen::VectorXr poly_coeffs_p = proj_2d * element_dof_values_p;
        const Real xc = element_2d.centroid.vx;
        const Real yc = element_2d.centroid.vy;
        const Real h  = element_2d.diameter;
        const int n_basis = (int)monomials_2d.size();
        for (int j = 0; j < n_basis; ++j) {
            const auto &mono = monomials_2d[j];
            Real val =
                mono.value(pos_2d.vx, pos_2d.vy, xc, yc,h);
            const Real coeff = poly_coeffs_p(j);
            weight += coeff * val;
        }
        RZ_ASSERT(weight<10.0,"");
        weight_sum+=weight;
        RZ_ASSERT(std::fabs(weight_sum-1.0)<1e-2,"");
        no_weighted_trips.emplace_back(row, global_coarse_dof, weight);
        return weight_sum;
    }
