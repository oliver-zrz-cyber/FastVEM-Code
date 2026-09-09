#pragma once
#include "dfn.h"
// #include "vem3d.h"
class GSSmoother {
public:
    struct RowEntry { int j; GlobalReal val; };
    std::vector<std::vector<RowEntry>> rows; // 每行的稀疏项
    std::vector<GlobalReal> diag;                // 对角项缓存
    std::vector<std::vector<int>> cell_3s;
    // 预处理一次（矩阵结构不变，可多次调用 GS）
    void build(const Eigen::SparseMatrixR& A) {
        const int n = A.rows();
        rows.assign(n, {});
        diag.assign(n, 0.0);
        rows.reserve(n);

        for (int i = 0; i < n; ++i) {
            for (Eigen::SparseMatrixR::InnerIterator it(A, i); it; ++it) {
                if (it.col() == i)
                    diag[i] = it.value();
                else
                    rows[i].push_back({static_cast<int>(it.col()), it.value()});
            }
        }
    }


    void build(const Eigen::SparseMatrixR& A,VEM_Domain_3d& domain) {
        const int n = A.rows();
        rows.assign(n, {});
        diag.assign(n, 0.0);
        rows.reserve(n);

        for (int i = 0; i < n; ++i) {
            for (Eigen::SparseMatrixR::InnerIterator it(A, i); it; ++it) {
                if (it.col() == i)
                    diag[i] = it.value();
                else
                    rows[i].push_back({static_cast<int>(it.col()), it.value()});
            }
        }
        const int e_base = domain.points.size();
        const int f_base = e_base+domain.edges.size();
        const int c_base = f_base+domain.polygons.size();
        for (int cell_id=0;cell_id<domain.polyhedrons.size();++cell_id) {
            std::vector<int> cell3_rows;

            const auto& cell_polyhedrons = domain.polyhedrons[cell_id];

            for (auto&poly_id:cell_polyhedrons) {
                for (int v=0;v<domain.polygons[poly_id].size();++v) {
                    const int vid = domain.polygons[poly_id][v];
                    cell3_rows.push_back(vid);
                }
            }
            const auto& elem = domain.vem_elements[cell_id];
            for (int edge_id : elem.edge_local_to_global_dof) {
                cell3_rows.push_back(edge_id + e_base);
            }
            for (auto&poly_id:cell_polyhedrons) {
                cell3_rows.push_back(f_base+poly_id);
            }
            cell3_rows.push_back(c_base+cell_id);
            cell_3s.push_back(cell3_rows);
        }

    }
    // 高速 Gauss–Seidel 平滑器（前向 sweep）
    inline void smooth(const Eigen::VectorXr &b,
                       Eigen::VectorXr &x,
                       int iterations = 3,
                       Real omega = 1.0) const {
        FUNC_TIMER();

        const int n = (int)rows.size();
        if (x.size() != n) x = Eigen::VectorXr::Zero(n);

        for (int it = 0; it < iterations; ++it) {
            for (int i = 0; i < n; ++i) {
                GlobalReal sum = b[i];
                const auto &ri = rows[i];
                for (const auto &e : ri)
                    sum -= e.val * x[e.j];

                // update (加松弛系数 ω)
                GlobalReal aii = diag[i];
                if (std::abs(aii) > 1e-30)
                    x[i] += omega * ((sum / aii) - x[i]);
            }
        }
    }
    inline void cell_smooth(const Eigen::VectorXr &b,
                       Eigen::VectorXr &x,
                       int iterations = 3,
                       Real omega = 1.0) const {
        FUNC_TIMER();

        const int n = (int)rows.size();
        if (x.size() != n) x = Eigen::VectorXr::Zero(n);

        for (int it = 0; it < iterations; ++it) {
            for (const auto&cell:cell_3s) {
                for (auto i:cell) {
                    GlobalReal sum = b[i];
                    const auto &ri = rows[i];
                    for (const auto &e : ri)
                        sum -= e.val * x[e.j];

                    // update (加松弛系数 ω)
                    GlobalReal aii = diag[i];
                    if (std::abs(aii) > 1e-30)
                        x[i] += omega * ((sum / aii) - x[i]);
                }
            }
        }
    }

};



class BlockMultiColorGSSmoother {
public:
    struct RowEntry { int j; GlobalReal val; };
    std::vector<std::vector<RowEntry>> rows;   // 每行的稀疏项（不含对角）
    std::vector<GlobalReal> diag;                  // 对角项缓存
    std::vector<int> color;                    // 每个节点的颜色（多色GS）
    std::vector<std::vector<int>> color_groups; // 每种颜色对应的节点集合
    int block_size = 1;                        // 每块未知量的大小（默认标量）

    // ---------- 构建 ----------
    void build(const Eigen::SparseMatrixR& A,
               int block = 1)
    {
        block_size = block;
        const int n = A.rows();
        rows.assign(n, {});
        diag.assign(n, 0.0);
        rows.reserve(n);

        // 提取矩阵结构
        for (int i = 0; i < n; ++i) {
            for (Eigen::SparseMatrixR::InnerIterator it(A, i); it; ++it) {
                int j = it.col();
                if (i == j)
                    diag[i] = it.value();
                else
                    rows[i].push_back({j, it.value()});
            }
        }

        // -------- 自动染色 (greedy graph coloring) --------
        color.assign(n, -1);
        int max_color = 0;
        for (int i = 0; i < n; ++i) {
            std::vector<bool> used(max_color + 1, false);
            for (const auto &e : rows[i]) {
                int cj = color[e.j];
                if (cj >= 0 && cj <= max_color)
                    used[cj] = true;
            }
            int c = 0;
            while (c <= max_color && used[c]) ++c;
            if (c > max_color) max_color = c;
            color[i] = c;
        }

        // 建立颜色组
        color_groups.assign(max_color + 1, {});
        for (int i = 0; i < n; ++i)
            color_groups[color[i]].push_back(i);

        std::cout << "[BlockMultiColorGS] total colors = "
                  << max_color + 1 << std::endl;
    }

    // ---------- 平滑 ----------
    inline void smooth(const Eigen::VectorXr &b,
                       Eigen::VectorXr &x,
                       int iterations = 3,
                       Real omega = 1.0) const
    {
        const int n = (int)rows.size();
        if (x.size() != n) x = Eigen::VectorXr::Zero(n);
        if (block_size <= 0) return;

        for (int it = 0; it < iterations; ++it) {
            // 多色循环
            for (size_t c = 0; c < color_groups.size(); ++c) {
                const auto &nodes = color_groups[c];

#pragma omp parallel for schedule(static)
                for (size_t idx = 0; idx < nodes.size(); ++idx) {
                    int i = nodes[idx];
                    GlobalReal sum = b[i];
                    const auto &ri = rows[i];
                    for (const auto &e : ri)
                        sum -= e.val * x[e.j];

                    GlobalReal aii = diag[i];
                    if (std::abs(aii) > 1e-30)
                        x[i] += omega * ((sum / aii) - x[i]);
                }
            }
        }
    }
};


class DampedJacobiSmoother {
public:
    using SpMat = Eigen::SparseMatrixR;
    using Vec   = Eigen::VectorXr;

    // ---------- 预处理：抽取 CSR，分离对角 ----------
    void build(const SpMat& A) {
        A_ = &A;
        const int n = A.rows();
        n_ = n;

        row_ptr_.assign(n + 1, 0);
        // 统计每行非零，为 row_ptr_ 填前缀和
        int nnz = 0;
        for (int i = 0; i < n; ++i) {
            for (SpMat::InnerIterator it(A, i); it; ++it) ++nnz;
            row_ptr_[i + 1] = nnz;
        }
        col_idx_.resize(nnz);
        val_.resize(nnz);

        diag_.assign(n, 0.0);
        inv_diag_.assign(n, 0.0);

        // 填充 CSR，同时分离对角
        int k = 0;
        for (int i = 0; i < n; ++i) {
            int start = k;
            for (SpMat::InnerIterator it(A, i); it; ++it) {
                const int j = it.col();
                const GlobalReal aij = it.value();
                col_idx_[k] = j;
                val_[k]     = aij;
                if (i != j) diag_[i] += std::fabs(aij);
                ++k;
            }
            // 预计算对角倒数
            inv_diag_[i] = (std::abs(diag_[i]) > 1e-30) ? (1.0 / diag_[i]) : 0.0;
        }

        // 为了更快的内层：构建“非对角”掩码（可选）
        // 我们直接在迭代里判断 i!=j，避免额外内存（更 cache 友好）。
    }

    // ---------- 平滑：无显式 A*x / 无 r 向量 ----------
    // 完全并行版本：每次迭代复制一个快照 x_old，按显式 Jacobi 公式更新
    void smooth(const Vec& b, Vec& x, int iterations = 3, Real omega = 0.666) const {
        if (!A_) {
            RZ_FATAL("");
            return;
        }
        if (x.size() != n_) {
            RZ_FATAL("");
            x = Vec::Zero(n_);
        }
        Vec x_old(n_);

        for (int it = 0; it < iterations; ++it) {
            x_old = x;  // Jacobi 需要读旧解、写新解

#pragma omp parallel for schedule(static)
            for (int i = 0; i < n_; ++i) {
                GlobalReal sum_off = 0.0;
                const int rs = row_ptr_[i];
                const int re = row_ptr_[i + 1];

                // 只累加非对角项：sum_{j != i} a_ij * x_old[j]
                // 用 SIMD 提示器帮助编译器向量化
#pragma omp simd reduction(+:sum_off)
                for (int k = rs; k < re; ++k) {
                    const int j = col_idx_[k];
                    if (j != i) sum_off += val_[k] * x_old[j];
                }

                // 显式 Jacobi 更新（融合了残差与对角除法）
                // x_new = (1-ω)*x_old + ω*(b - sum_off)/a_ii
                const GlobalReal aii = diag_[i];
                const GlobalReal invd = inv_diag_[i];
                const GlobalReal x_new =
                    (std::abs(aii) > 1e-30)
                    ? ( (1.0 - omega) * x_old[i] + omega * (b[i] - sum_off) * invd )
                    : x_old[i]; // 退化保护

                x[i] = x_new;
            }
        }
    }

private:
    const SpMat* A_ = nullptr;
    int n_ = 0;

    // CSR
    std::vector<int>    row_ptr_;
    std::vector<int>    col_idx_;
    std::vector<GlobalReal> val_;

    // 对角与对角倒数
    std::vector<GlobalReal> diag_;
    std::vector<GlobalReal> inv_diag_;
};
