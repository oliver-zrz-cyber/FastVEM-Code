#include "vem3d.h"
#include "mini_lcc.h"
#include "bsp_mesh.h"
#include "bsp_naive.h"
#include <tinyxml2.h>
#include <Eigen/Dense>
#include "mgsolver_vemp.h"
#include "Visualization.h"
#include <execution>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/orientation.h>
#include <CGAL/convex_hull_3.h>

#include <Exceptions.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/measure.h>
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
#include "config.h"
#include <CGAL/Polygon_2.h>
#include "vex_pcg.h"


template<typename Range>
static std::vector<Dart_handle> collect(Range range)
{
    std::vector<Dart_handle> dhs;
    for (auto dh = range.begin(); dh != range.end(); dh++)
        dhs.push_back(dh);
    return dhs;
}
std::vector<VEM_Monomial_3d> monomials_3d;
// static BlockMultiColorGSSmoother mg_solver;
// static AMGCLJacobiSmoother amg_jcobi;
// static int domain_vertices,domain_edges,domain_polygons;
// static std::vector<Eigen::Triplet<GlobalReal> > triplets_k2;
// static std::vector<Eigen::Triplet<GlobalReal> > triplets_LX_k2;
// static std::vector<Eigen::Triplet<GlobalReal> > triplets_LY_k2;
// static std::vector<Eigen::Triplet<GlobalReal> > triplets_LZ_k2;
//
// static std::vector<Eigen::Triplet<GlobalReal> > triplets_k1;
// static std::vector<Eigen::Triplet<GlobalReal> > triplets_LX_k1;
// static std::vector<Eigen::Triplet<GlobalReal> > triplets_LY_k1;
// static std::vector<Eigen::Triplet<GlobalReal> > triplets_LZ_k1;

static void save_points_to_xyz(const std::vector<_3d_vec>& points, const std::string& filename) {
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


Real VEM_Element_3d::integrate_centered_monomial_over_polyhedron
(const VEM_Monomial_3d &mono, const _3d_vec &C) const{
    const int p = mono.p, q = mono.q, r = mono.r;
    const Real coef = mono.c;

    Real total = 0.0;

    // 对每个面做三角扇剖分： (C, v0, vi, v_{i+1})
    for (const auto& face : polygons) {
        if ((int)face.size() < 3) continue;
        const int m = (int)face.size();
        const _3d_vec v0 = vertex_position(face[0]);
        for (int i = 1; i + 1 < m; ++i) {
            _3d_vec v1 = vertex_position(face[i]);
            _3d_vec v2 = vertex_position(face[i + 1]);
            // ---- 法向外向判断 ----
            _3d_vec n = (v1 - v0).cross(v2 - v0);       // 面法向（右手系）
            Real dot_c = (C - v0).dot(n);             // 质心相对法向的方向
            if (dot_c > 0.0) std::swap(v1, v2);       // 若质心在法向同侧 → 翻转顺序使法向外向

            total += integrate_centered_monomial_over_tet(C, v0, v1, v2, p, q, r, C);
        }
    }
    return coef * total;
}
// static void makeSymmetric(Eigen::SparseMatrixR& A) {
//     // 遍历矩阵的上三角部分（包括对角线）
//     for (int k = 0; k < A.outerSize(); ++k) {
//         for (Eigen::SparseMatrixR::InnerIterator it(A, k); it; ++it) {
//             int i = it.row();
//             int j = it.col();
//
//             // 只处理上三角部分 (i < j)，避免重复处理
//             if (i < j) {
//                 // 获取 (i, j) 和 (j, i) 元素的平均值
//                 Real avg = (it.value() + A.coeff(j, i)) / 2.0;
//                 // 更新 (i, j) 和 (j, i) 位置的元素
//                 A.insert(i, j) = avg;
//                 A.insert(j, i) = avg;
//             }
//         }
//     }
// }
static void makeSymmetric(Eigen::MatrixXr& A) {
    int rows = A.rows();
    int cols = A.cols();

    // 遍历矩阵的上三角部分（包括对角线）
    for (int i = 0; i < rows; ++i) {
        for (int j = i + 1; j < cols; ++j) {
            // 计算 (i, j) 和 (j, i) 元素的平均值
            Real avg = (A(i, j) + A(j, i)) / 2.0;
            // 更新 (i, j) 和 (j, i) 位置的元素
            A(i, j) = avg;
            A(j, i) = avg;
        }
    }
}

// Real VEM_Element_3d::integrate_centered_monomial_over_polyhedron_divergence(const VEM_Monomial_3d &mono,
//     const _3d_vec &C)const {
//     Real ans = 0.0;
//     for (auto&_3d_polygon:elements_2d) {
//         if (_3d_polygon.degenerated) continue;
//         VEM_Monomial_3d origin_mono(mono.p + 1,mono.q,mono.r,mono.c / (mono.p+1));
//         origin_mono.c = origin_mono.c * (_3d_polygon.plane.normal.vx);
//         ans += _3d_polygon.polygon_moment_general_3d(origin_mono,C);
//     }
//     return ans;
// }

Real VEM_Element_3d::integrate_centered_monomial_over_polyhedron_divergence(
    const VEM_Monomial_3d &mono,
    const _3d_vec &C) const
{
    Real ans = 0.0;

    const int p = mono.p;
    const int q = mono.q;
    const int r = mono.r;
    const Real c = mono.c;

    RZ_ASSERT(p >= 0 && q >= 0 && r >= 0, "exponent < 0");
    RZ_ASSERT(p + q + r + 1 <= 3, "face monomial degree not supported");

    // ---------------------------
    // 1) 全局选轴（而不是 per-face）
    // ---------------------------
    Real Sx = 0.0, Sy = 0.0, Sz = 0.0;
    for (const auto &f : elements_2d) {
        if (f.degenerated) continue;
        const auto &n = f.plane.normal;
        const Real Af = (f.area > 0.0) ? f.area : 1.0;
        Sx += std::abs(n.vx) * Af;
        Sy += std::abs(n.vy) * Af;
        Sz += std::abs(n.vz) * Af;
    }

    int axis = 0; // 0:x, 1:y, 2:z
    if (Sy >= Sx && Sy >= Sz) axis = 1;
    else if (Sz >= Sx && Sz >= Sy) axis = 2;

    // ---------------------------
    // 2) 用选定轴，对所有面统一使用同一个散度构造
    // ---------------------------
    for (const auto &f : elements_2d) {
        if (f.degenerated) continue;
        const auto &n = f.plane.normal;

        if (axis == 0) {
            VEM_Monomial_3d m_face(p + 1, q, r, c / Real(p + 1));
            m_face.c *= n.vx;
            ans += f.polygon_moment_general_3d(m_face, C);
        } else if (axis == 1) {
            VEM_Monomial_3d m_face(p, q + 1, r, c / Real(q + 1));
            m_face.c *= n.vy;
            ans += f.polygon_moment_general_3d(m_face, C);
        } else {
            VEM_Monomial_3d m_face(p, q, r + 1, c / Real(r + 1));
            m_face.c *= n.vz;
            ans += f.polygon_moment_general_3d(m_face, C);
        }
    }

    return ans;
}

static int save_cnt = 0;

void VEM_Element_3d::build_up() {
    int n_polynomials_k1 = 4;
    // FUNC_TIMER();
    Eigen::LocalMatrixXr B_k2,D_k2,G_k2,B_k1,D_k1,G_k1,
        H_X_k2,H_Y_k2,H_Z_k2,H_X_k1,H_Y_k1,H_Z_k1;

    // 加速部分 在这里对于每个面 先计算最多到3阶的polygon_moment_general_2d 后面直接查表
    std::vector<VEM_Monomial_2d> precompute_monomials_2d;
    std::map<std::pair<int,int>,int>pq_to_id;
    for (int p=0;p<=3;++p) {
        for (int q=0;q<=3;++q) {
            if (p+q<=3) {
                pq_to_id[{p,q}] = precompute_monomials_2d.size();
                precompute_monomials_2d.push_back(VEM_Monomial_2d(p, q));
            }
        }
    }
    std::vector<std::vector<Real>> table;
    for (int face_id=0;face_id<elements_2d.size();++face_id) {
        std::vector<Real> temp;
        const auto&face = elements_2d[face_id];
        for (auto&mono:precompute_monomials_2d) {
            temp.push_back(face.polygon_moment_general_2d_advance(mono,{0,0},false));
        }
        table.push_back(temp);
    }
    //-------------------------------------------------------------

    {// k = 1 for velocity
        int per_element_dof_num_k1 = vertex_count();
        D_k1.resize(per_element_dof_num_k1, n_polynomials_k1);
        D_k1.setZero();
        B_k1.resize(n_polynomials_k1, per_element_dof_num_k1);
        B_k1.setZero();
        //计算D
        for (int vid = 0; vid < vertex_count(); ++vid) {
            const _3d_vec &vert = vertex_position(vid);
            for (int poly_id = 0; poly_id < n_polynomials_k1; ++poly_id) {
                const VEM_Monomial_3d &poly = monomials_3d[poly_id];
                D_k1(vid, poly_id) = poly.value(vert.vx,vert.vy,vert.vz,
                    this->centroid.vx,this->centroid.vy,this->centroid.vz,this->diameter);
            }
        }

        // 计算B
        std::vector<VEM_Monomial_2d> monomials_2d_k1;
        monomials_2d_k1 = {
            VEM_Monomial_2d(0, 0),
            VEM_Monomial_2d(1, 0),
            VEM_Monomial_2d(0, 1),
        };
        for (int col=0;col<per_element_dof_num_k1;++col)
            B_k1(0,col) = 1.0/(Real)per_element_dof_num_k1;
        for (int row = 1;row<n_polynomials_k1;++row) {
            for (int col = 0;col<per_element_dof_num_k1;++col) {
                // 我们这里需要能从体自由度对应到一个面内的对应自由度
                Real first_term = 0.0;
                const VEM_Monomial_3d &poly = monomials_3d[row];
                Real second_term = 0.0;

                for (int face_id=0;face_id<this->elements_2d.size();++face_id) {
                    auto&face = this->elements_2d[face_id];
                    if (face.degenerated) continue;
                    if (!face.volume_dof_to_local_dof.count(col))continue;

                    int local_dof_id = face.volume_dof_to_local_dof[col];

                    auto normal_gradient_x = poly.derivative_x();
                    normal_gradient_x.c = normal_gradient_x.c * face.plane.normal.vx;
                    auto normal_gradient_y = poly.derivative_y();
                    normal_gradient_y.c = normal_gradient_y.c * face.plane.normal.vy;
                    auto normal_gradient_z = poly.derivative_z();
                    normal_gradient_z.c = normal_gradient_z.c * face.plane.normal.vz;

                    std::vector<VEM_Monomial_2d> projected_poly_2ds_x = face.project_monomial_to_plane(normal_gradient_x,this->centroid);
                    std::vector<VEM_Monomial_2d> projected_poly_2ds_y = face.project_monomial_to_plane(normal_gradient_y,this->centroid);
                    std::vector<VEM_Monomial_2d> projected_poly_2ds_z = face.project_monomial_to_plane(normal_gradient_z,this->centroid);
                    std::vector<VEM_Monomial_2d> projected_poly_2ds;
                    projected_poly_2ds.reserve(
                        projected_poly_2ds_x.size() +
                        projected_poly_2ds_y.size() +
                        projected_poly_2ds_z.size());

                    projected_poly_2ds.insert(projected_poly_2ds.end(),
                                              projected_poly_2ds_x.begin(), projected_poly_2ds_x.end());
                    projected_poly_2ds.insert(projected_poly_2ds.end(),
                                              projected_poly_2ds_y.begin(), projected_poly_2ds_y.end());
                    projected_poly_2ds.insert(projected_poly_2ds.end(),
                                              projected_poly_2ds_z.begin(), projected_poly_2ds_z.end());

                    // RZ_INFO(face.project_matrix_star);
                    for (auto&projected_poly2d:projected_poly_2ds) {// 这些应该都是1阶的
                        // 可否改成不依赖低维度的
                        Real deno1 = std::pow(this->diameter,poly.p+poly.q+poly.r);;
                        for (int alpha = 0;alpha<monomials_2d_k1.size();++alpha) {
                            // RZ_INFO(face.project_matrix_star.rows()," ",face.project_matrix_star.cols());
                            Real weight = face.project_matrix_star_k1(alpha,local_dof_id);
                            auto no_centered_poly2ds = face.expand_centered_monomial(monomials_2d_k1[alpha]);
                            for (auto& no_centered_poly2d:no_centered_poly2ds) {
                                auto mul_mono = no_centered_poly2d * projected_poly2d;
                                // Real this_coff = weight * face.polygon_moment_general_2d(mul_mono,{0,0},false);
                                Real this_coff = weight * table[face_id][pq_to_id[{mul_mono.p,mul_mono.q}]] * mul_mono.c;
                                this_coff /= deno1;
                                this_coff /= std::pow(face.diameter,monomials_2d_k1[alpha].p+monomials_2d_k1[alpha].q);
                                second_term += this_coff;
                            }
                        }
                    }
                }
                B_k1(row,col) = first_term + second_term;
            }
        }

        // //计算 G
        // {// for check
        //     G_k1.resize(n_polynomials_k1,n_polynomials_k1);
        //     G_k1.setZero();
        //     for (int col =0;col<n_polynomials_k1;++col) {
        //         const auto& poly = monomials_3d[col];
        //         Real ans = 0.0;
        //         for (int vid = 0; vid < vertex_count(); ++vid) {
        //             const _3d_vec &vert = vertex_position(vid);
        //             ans+=poly.value(vert.vx,vert.vy,vert.vz,
        //                     this->centroid.vx,this->centroid.vy,this->centroid.vz,this->diameter);
        //         }
        //         G_k1(0,col) =  ans/(Real)per_element_dof_num_k1;
        //     }
        //     for (int row = 1;row<n_polynomials_k1;++row) {
        //         for (int col = 1;col<n_polynomials_k1;++col) {
        //             Real ans = 0.0;
        //             const auto& polyrow = monomials_3d[row];
        //             const auto& polycol = monomials_3d[col];
        //             const auto& poly_dv_x = polyrow.derivative_x()*polycol.derivative_x();
        //             const auto& poly_dv_y = polyrow.derivative_y()*polycol.derivative_y();
        //             const auto& poly_dv_z = polyrow.derivative_z()*polycol.derivative_z();
        //             Real denominator =  std::pow(this->diameter,polyrow.p+polyrow.q+polyrow.r + polycol.p + polycol.q + polycol.r);
        //             ans += integrate_centered_monomial_over_polyhedron_divergence(poly_dv_x,this->centroid) /denominator;
        //             ans += integrate_centered_monomial_over_polyhedron_divergence(poly_dv_y,this->centroid) /denominator;
        //             ans += integrate_centered_monomial_over_polyhedron_divergence(poly_dv_z,this->centroid) /denominator;
        //             G_k1(row,col) = ans;
        //         }
        //     }
        //     Eigen::LocalMatrixXr multed_G_k1= B_k1 * D_k1;
        //     // if ((multed_G_k1 - G_k1).cwiseAbs().maxCoeff()>1e-3){
        //     //     this->save_polygonmesh_to_obj("check.obj");
        //     //     RZ_FATAL((multed_G_k1 - G_k1).cwiseAbs().maxCoeff());
        //     // }
        // }
        G_k1.setZero();
        G_k1 = B_k1 * D_k1;
        // Eigen::JacobiSVD<Eigen::LocalMatrixXr> svd_k1(G_k1, Eigen::ComputeFullU|Eigen::ComputeFullV);
        // auto s = svd_k1.singularValues();
        // Real tau = 1e-10 * s(0);   // 可调：1e-12 ~ 1e-10
        // Eigen::LocalVectorXr inv = s;
        // for(int i=0;i<s.size();++i) inv(i) = (s(i) > tau) ? (1.0/s(i)) : 0.0;
        // Eigen::LocalMatrixXr Gpinv_k1 = svd_k1.matrixV() * inv.asDiagonal() * svd_k1.matrixU().transpose();
        this->project_matrix_star_k1 = G_k1.inverse() * B_k1;

        // Eigen::LocalMatrixXr I = Eigen::LocalMatrixXr::Identity(per_element_dof_num_k1, per_element_dof_num_k1);
        // Eigen::LocalMatrixXr residual = I - D_k1 * this->project_matrix_star_k1; // (per_element_dof x per_element_dof)
        // Eigen::LocalMatrixXr stabilising_term = residual.transpose() * residual * this->diameter;//
        // G_k1.row(0).setZero();
        // Eigen::MatrixXr consis_term = this->project_matrix_star_k1.transpose() * G_k1 * this->project_matrix_star_k1 ;
        // RZ_INFO(consis_term.norm()," ",residual.norm()," ",G_k1.norm()," ",stabilising_term.norm());
        // RZ_INFO(residual);
        // this->stiffness_matrix_k1 = this->project_matrix_star_k1.transpose() *
        //     G_k1 * this->project_matrix_star_k1 + stabilising_term;

    }

    int n_polynomials_k2 = monomials_3d.size();
    int per_element_dof_num_k2 = vertex_count() + edge_count() + polygons.size() + 1;
    D_k2.resize(per_element_dof_num_k2, n_polynomials_k2);
    D_k2.setZero();
    B_k2.resize(n_polynomials_k2, per_element_dof_num_k2);
    B_k2.setZero();
    // dof : 顶点 -> 边 -> 面 -> 体
    // 计算 D
    for (int vid = 0; vid < vertex_count(); ++vid) {
        const _3d_vec &vert = vertex_position(vid);
        for (int poly_id = 0; poly_id < n_polynomials_k2; ++poly_id) {
            const VEM_Monomial_3d &poly = monomials_3d[poly_id];
            D_k2(vid, poly_id) = poly.value(vert.vx,vert.vy,vert.vz,
                this->centroid.vx,this->centroid.vy,this->centroid.vz,this->diameter);
        }
    }
    for (int eid=0;eid<edge_count();++eid) {
        const Edge local_edge = edge(eid);
        const _3d_vec vert = (vertex_position(local_edge.v1)+vertex_position(local_edge.v2))/2;
        for (int poly_id = 0; poly_id < n_polynomials_k2; ++poly_id) {
            const VEM_Monomial_3d &poly = monomials_3d[poly_id];
            D_k2(vertex_count() + eid, poly_id) = poly.value(vert.vx,vert.vy,vert.vz,
                this->centroid.vx,this->centroid.vy,this->centroid.vz,this->diameter);
        }
    }

    for (int fid=0;fid<this->elements_2d.size();++fid) {
        for (int poly_id = 0; poly_id < n_polynomials_k2; ++poly_id) {
            const VEM_Monomial_3d &poly = monomials_3d[poly_id];
            // = 在这个3d面上对于poly进行积分 那么我们把poly投影过去 变成多个2d的积分
            D_k2(vertex_count() + edge_count() + fid, poly_id) =
                elements_2d[fid].polygon_moment_general_3d(poly,this->centroid)
            / std::pow(this->diameter,poly.p+poly.q+poly.r) ;
            if (DOFSCALE)
                D_k2(vertex_count() + edge_count() + fid, poly_id) =
                    D_k2(vertex_count() + edge_count() + fid, poly_id) / elements_2d[fid].area;
        }
    }

    for (int poly_id = 0; poly_id < n_polynomials_k2; ++poly_id) {
        const VEM_Monomial_3d &poly = monomials_3d[poly_id];
        // RZ_INFO("tet: ",integrate_centered_monomial_over_polyhedron(poly,this->centroid) / std::pow(this->diameter,
        //         poly.p+poly.q+poly.r));
        // RZ_INFO("div: ",integrate_centered_monomial_over_polyhedron_divergence(poly,this->centroid) / std::pow(this->diameter,
        //         poly.p + poly.q + poly.r ));
        // RZ_INFO(poly.p + poly.q + poly.r);
        // RZ_INFO(this->volume);
        // exit(0);
        D_k2(per_element_dof_num_k2 - 1, poly_id) =
            integrate_centered_monomial_over_polyhedron_divergence(poly,this->centroid) / std::pow(this->diameter,
                poly.p+poly.q+poly.r) ;// 体积分
        if (DOFSCALE)
            D_k2(per_element_dof_num_k2 - 1, poly_id) =
                D_k2(per_element_dof_num_k2 - 1, poly_id)/ this->volume;
    }
    //
    // 计算 B
    std::vector<VEM_Monomial_2d> monomials_2d;
    monomials_2d = {
        VEM_Monomial_2d(0, 0),
        VEM_Monomial_2d(1, 0),
        VEM_Monomial_2d(0, 1),
        VEM_Monomial_2d(2, 0),
        VEM_Monomial_2d(1, 1),
        VEM_Monomial_2d(0, 2)
    };
    // B_k2(0,per_element_dof_num_k2 - 1) = this->volume;
    B_k2(0,per_element_dof_num_k2 - 1) = 1;
    for (int row = 1;row<n_polynomials_k2;++row) {
        for (int col = 0;col<per_element_dof_num_k2;++col) {
            // 我们这里需要能从体自由度对应到一个面内的对应自由度
            Real first_term = 0.0;
            const VEM_Monomial_3d &poly = monomials_3d[row];//如果这里是p == 2 or q == 2 or r == 2
            if ((poly.p==2||poly.q==2||poly.r==2) && col == (per_element_dof_num_k2-1)) {
                first_term -= 2.0 / this->diameter / this->diameter;
                if (DOFSCALE) first_term *= this->volume;
            }
            Real deno1 = std::pow(this->diameter,poly.p+poly.q+poly.r);
            Real second_term = 0.0;
            for (int face_id=0;face_id<this->elements_2d.size();++face_id) {
                auto&face = this->elements_2d[face_id];
                if (face.degenerated)continue;
                if (!face.volume_dof_to_local_dof.count(col))continue;
                int local_dof_id = face.volume_dof_to_local_dof[col];

                auto normal_gradient_x = poly.derivative_x();
                normal_gradient_x.c = normal_gradient_x.c * face.plane.normal.vx;
                auto normal_gradient_y = poly.derivative_y();
                normal_gradient_y.c = normal_gradient_y.c * face.plane.normal.vy;
                auto normal_gradient_z = poly.derivative_z();
                normal_gradient_z.c = normal_gradient_z.c * face.plane.normal.vz;

                std::vector<VEM_Monomial_2d> projected_poly_2ds_x = face.project_monomial_to_plane(normal_gradient_x,this->centroid);
                std::vector<VEM_Monomial_2d> projected_poly_2ds_y = face.project_monomial_to_plane(normal_gradient_y,this->centroid);
                std::vector<VEM_Monomial_2d> projected_poly_2ds_z = face.project_monomial_to_plane(normal_gradient_z,this->centroid);
                std::vector<VEM_Monomial_2d> projected_poly_2ds;
                projected_poly_2ds.reserve(
                    projected_poly_2ds_x.size() +
                    projected_poly_2ds_y.size() +
                    projected_poly_2ds_z.size());

                projected_poly_2ds.insert(projected_poly_2ds.end(),
                                          projected_poly_2ds_x.begin(), projected_poly_2ds_x.end());
                projected_poly_2ds.insert(projected_poly_2ds.end(),
                                          projected_poly_2ds_y.begin(), projected_poly_2ds_y.end());
                projected_poly_2ds.insert(projected_poly_2ds.end(),
                                          projected_poly_2ds_z.begin(), projected_poly_2ds_z.end());

                for (auto&projected_poly2d:projected_poly_2ds) {// 这些应该都是1阶的
                    for (int alpha = 0;alpha<monomials_2d.size();++alpha) {
                        Real weight = face.project_matrix_star(alpha,local_dof_id);
                        auto no_centered_poly2ds = face.expand_centered_monomial(monomials_2d[alpha]);
                        for (auto& no_centered_poly2d:no_centered_poly2ds) {
                            //既然不center化那么是不是我们可以先预先计算 然后只要乘c就行了 最大好像到3次
                            auto mul_mono = no_centered_poly2d * projected_poly2d;
                            // Real this_coff = weight * face.polygon_moment_general_2d(mul_mono,{0,0},false);
                            Real this_coff = weight * table[face_id][pq_to_id[{mul_mono.p,mul_mono.q}]]
                                * mul_mono.c;
                            this_coff /= deno1;
                            this_coff /= std::pow(face.diameter,monomials_2d[alpha].p+monomials_2d[alpha].q);
                            second_term += this_coff;
                        }
                    }
                }
            }
            B_k2(row,col) = first_term + second_term;
        }
    }
    // 计算 G
    {
        G_k2.resize(n_polynomials_k2,n_polynomials_k2);
        G_k2.setZero();

        for (int col =0;col<n_polynomials_k2;++col) {
            const auto& poly = monomials_3d[col];
            G_k2(0,col) = integrate_centered_monomial_over_polyhedron_divergence(poly,this->centroid) / std::pow(this->diameter,
                    poly.p+poly.q+poly.r);
            if (DOFSCALE) {
            G_k2(0,col) /= this->volume;
            }
        }

        for (int row = 1;row<n_polynomials_k2;++row) {
            for (int col = 1;col<n_polynomials_k2;++col) {
                const auto& polyrow = monomials_3d[row];
                const auto& polycol = monomials_3d[col];
                const auto& poly_dv_x = polyrow.derivative_x()*polycol.derivative_x();
                const auto& poly_dv_y = polyrow.derivative_y()*polycol.derivative_y();
                const auto& poly_dv_z = polyrow.derivative_z()*polycol.derivative_z();
                Real denominator =  std::pow(this->diameter,polyrow.p+polyrow.q+polyrow.r + polycol.p + polycol.q + polycol.r);
                G_k2(row,col) += integrate_centered_monomial_over_polyhedron_divergence(poly_dv_x,this->centroid) / denominator;
                G_k2(row,col) += integrate_centered_monomial_over_polyhedron_divergence(poly_dv_y,this->centroid) / denominator;
                G_k2(row,col) += integrate_centered_monomial_over_polyhedron_divergence(poly_dv_z,this->centroid) / denominator;
            }
        }
        Eigen::LocalMatrixXr multed_G= B_k2 * D_k2;

        if ((multed_G - G_k2).cwiseAbs().maxCoeff()>1e-3) {
            this->save_polygonmesh_to_obj("check.obj");
            RZ_FATAL("maxCoeff()>1e-3: ",(multed_G - G_k2).cwiseAbs().maxCoeff());
        }
    }
    //----------------B and D constructed----------------------
    G_k2.setZero();
    G_k2 = B_k2 * D_k2;
    Eigen::JacobiSVD<Eigen::LocalMatrixXr> svd(G_k2, Eigen::ComputeFullU|Eigen::ComputeFullV);
    auto s = svd.singularValues();
    Real tau = 1e-10 * s(0);   // 可调：1e-12 ~ 1e-10
    Eigen::LocalVectorXr inv = s;
    for(int i=0;i<s.size();++i) inv(i) = (s(i) > tau) ? (1.0/s(i)) : 0.0;
    Eigen::LocalMatrixXr Gpinv = svd.matrixV() * inv.asDiagonal() * svd.matrixU().transpose();
    Eigen::LocalMatrixXr projector = Gpinv * B_k2;
    // Eigen::MatrixXr projector = G_k2.inverse() * B_k2; // (n_polys x per_element_dof)
    // 保存 projector
    this->project_matrix_star_k2 = projector;

    Eigen::LocalMatrixXr I = Eigen::LocalMatrixXr::Identity(per_element_dof_num_k2, per_element_dof_num_k2);
    Eigen::LocalMatrixXr residual = I - D_k2 * projector; // (per_element_dof x per_element_dof)
    G_k2.row(0).setZero();

    Eigen::LocalMatrixXr stabilising_term = residual.transpose() * residual * this->diameter;
    this->stiffness_matrix_k2 = projector.transpose() * G_k2 * projector + stabilising_term;
    RZ_ASSERT(stiffness_matrix_k2.rows()>1, vertex_count());
    // RZ_INFO(this->project_matrix_star_k2);
    // RZ_INFO("----------------");
    // RZ_INFO(this->stiffness_matrix_k2);
    // H :H_X,H_Y,H_Z for RHS
    // H_X_k1.resize(n_polynomials_k1, n_polynomials_k1);
    // H_Y_k1.resize(n_polynomials_k1, n_polynomials_k1);
    // H_Z_k1.resize(n_polynomials_k1, n_polynomials_k1);
    // H_X_k1.setZero();
    // H_Y_k1.setZero();
    // H_Z_k1.setZero();
    // //-----------------------------k1
    // for (int i = 0; i < n_polynomials_k1; ++i) {
    //     for (int j = 0; j < n_polynomials_k1; ++j) {
    //         // 由于一定有一个求导 所以最后一定要多除一个 max_diameter
    //         VEM_Monomial_3d temp_x = monomials_3d[i].derivative_x() * monomials_3d[j];
    //         H_X_k1(i,j) = integrate_centered_monomial_over_polyhedron_divergence(temp_x,this->centroid)/
    //             std::pow(this->diameter, std::max(temp_x.p + temp_x.q + temp_x.r + 1, 0));

    //         VEM_Monomial_3d temp_y = monomials_3d[i].derivative_y() * monomials_3d[j];
    //         H_Y_k1(i,j) = integrate_centered_monomial_over_polyhedron_divergence(temp_y,this->centroid)/
    //             std::pow(this->diameter, std::max(temp_y.p + temp_y.q + temp_y.r + 1, 0));

    //         VEM_Monomial_3d temp_z = monomials_3d[i].derivative_z() * monomials_3d[j];
    //         H_Z_k1(i,j) = integrate_centered_monomial_over_polyhedron_divergence(temp_z,this->centroid)/
    //             std::pow(this->diameter, std::max(temp_z.p + temp_z.q + temp_z.r + 1, 0));
    //     }
    // }

    // this->local_LX_k1 = this->project_matrix_star_k1.transpose() * H_X_k1 * this->project_matrix_star_k1;
    // this->local_LY_k1 = this->project_matrix_star_k1.transpose() * H_Y_k1 * this->project_matrix_star_k1;
    // this->local_LZ_k1 = this->project_matrix_star_k1.transpose() * H_Z_k1 * this->project_matrix_star_k1;

    //-----------------------------k2
    H_X_k2.resize(n_polynomials_k2, n_polynomials_k1);
    H_Y_k2.resize(n_polynomials_k2, n_polynomials_k1);
    H_Z_k2.resize(n_polynomials_k2, n_polynomials_k1);
    H_X_k2.setZero();
    H_Y_k2.setZero();
    H_Z_k2.setZero();
    for (int i = 0; i < n_polynomials_k2; ++i) {
        for (int j = 0; j < n_polynomials_k1; ++j) {
            // 由于一定有一个求导 所以最后一定要多除一个 max_diameter
            VEM_Monomial_3d temp_x = monomials_3d[i].derivative_x() * monomials_3d[j];
            H_X_k2(i,j) = integrate_centered_monomial_over_polyhedron_divergence(temp_x,this->centroid)/
                std::pow(this->diameter, std::max(temp_x.p + temp_x.q + temp_x.r + 1, 0));

            VEM_Monomial_3d temp_y = monomials_3d[i].derivative_y() * monomials_3d[j];
            H_Y_k2(i,j) = integrate_centered_monomial_over_polyhedron_divergence(temp_y,this->centroid)/
                std::pow(this->diameter, std::max(temp_y.p + temp_y.q + temp_y.r + 1, 0));

            VEM_Monomial_3d temp_z = monomials_3d[i].derivative_z() * monomials_3d[j];
            H_Z_k2(i,j) = integrate_centered_monomial_over_polyhedron_divergence(temp_z,this->centroid)/
                std::pow(this->diameter, std::max(temp_z.p + temp_z.q + temp_z.r + 1, 0));

        }
    }
    // RZ_INFO((H_X_k2*H_X_k2.transpose()+H_Y_k2*H_Y_k2.transpose()+H_Z_k2*H_Z_k2.transpose())/volume-G_k2);
    // RZ_INFO(G_k2);
    // RZ_INFO(diameter);
    // RZ_INFO(1/this->diameter * this->volume);
    // RZ_INFO("==========================================");
    const Eigen::LocalMatrixXr& project_k1_for_k2 = this->project_matrix_star_k1;
    this->local_LX_k2 = projector.transpose() * H_X_k2 * project_k1_for_k2;
    this->local_LY_k2 = projector.transpose() * H_Y_k2 * project_k1_for_k2;
    this->local_LZ_k2 = projector.transpose() * H_Z_k2 * project_k1_for_k2;
    // RZ_INFO("----------------");
    // RZ_INFO(this->local_LX_k2);
    // Eigen::MatrixXr consis = (this->local_LX_k1*this->local_LX_k1.transpose()+
    //     this->local_LY_k1*Mass.inverse()*this->local_LY_k1.transpose()+
    //         this->local_LZ_k1*Mass.inverse()*local_LZ_k1.transpose());
    // Eigen::MatrixXr consis = this->project_matrix_star_k1.transpose() *
    //     (H_X_k1 * H_X_k1.transpose() +H_Y_k1 * H_Y_k1.transpose()+H_Z_k1 * H_Z_k1.transpose())
    // / volume * this->project_matrix_star_k1;
    //
    // RZ_INFO((consis - stiffness_matrix_k1).norm()/stiffness_matrix_k1.norm());

    // RZ_INFO((H_X_k2.norm()),"normHX");
    // RZ_INFO((this->local_LX_k2*this->local_LX_k2.transpose()).norm(),"norm");
    // RZ_INFO((this->local_LY_k2*this->local_LY_k2.transpose()).norm(),"norm");
    // RZ_INFO((this->local_LY_k2*this->local_LY_k2.transpose()).norm(),"norm");
    // RZ_INFO(consis.norm(),"normc");
    // RZ_INFO(G_k2.norm(),"normg");
    // RZ_INFO(stiffness_matrix_k2.norm(),"normstif");
    // RZ_INFO((this->D_k1 - this->D_k2.topLeftCorner(this->D_k1.rows(),this->D_k1.cols())).maxCoeff());
    // RZ_INFO((this->B_k1 - this->B_k2.topLeftCorner(this->B_k1.rows(),this->B_k1.cols())).maxCoeff());
}

void VEM_Element_3d::build_up_k1() {
int n_polynomials_k1 = 4;
    // FUNC_TIMER();
    Eigen::LocalMatrixXr B_k2,D_k2,G_k2,B_k1,D_k1,G_k1,H_X_k1,H_Y_k1,H_Z_k1;

    // 加速部分 在这里对于每个面 先计算最多到3阶的polygon_moment_general_2d 后面直接查表
    std::vector<VEM_Monomial_2d> precompute_monomials_2d;
    std::map<std::pair<int,int>,int>pq_to_id;
    for (int p=0;p<=3;++p) {
        for (int q=0;q<=3;++q) {
            if (p+q<=3) {
                pq_to_id[{p,q}] = precompute_monomials_2d.size();
                precompute_monomials_2d.push_back(VEM_Monomial_2d(p, q));
            }
        }
    }
    std::vector<std::vector<Real>> table;
    for (int face_id=0;face_id<elements_2d.size();++face_id) {
        std::vector<Real> temp;
        const auto&face = elements_2d[face_id];
        for (auto&mono:precompute_monomials_2d) {
            temp.push_back(face.polygon_moment_general_2d_advance(mono,{0,0},false));
        }
        table.push_back(temp);
    }
    //-------------------------------------------------------------

    {// k = 1 for velocity
        int per_element_dof_num_k1 = vertex_count();
        D_k1.resize(per_element_dof_num_k1, n_polynomials_k1);
        D_k1.setZero();
        B_k1.resize(n_polynomials_k1, per_element_dof_num_k1);
        B_k1.setZero();
        //计算D
        for (int vid = 0; vid < vertex_count(); ++vid) {
            const _3d_vec &vert = vertex_position(vid);
            for (int poly_id = 0; poly_id < n_polynomials_k1; ++poly_id) {
                const VEM_Monomial_3d &poly = monomials_3d[poly_id];
                D_k1(vid, poly_id) = poly.value(vert.vx,vert.vy,vert.vz,
                    this->centroid.vx,this->centroid.vy,this->centroid.vz,this->diameter);
            }
        }

        // 计算B
        std::vector<VEM_Monomial_2d> monomials_2d_k1;
        monomials_2d_k1 = {
            VEM_Monomial_2d(0, 0),
            VEM_Monomial_2d(1, 0),
            VEM_Monomial_2d(0, 1),
        };
        for (int col=0;col<per_element_dof_num_k1;++col)
            B_k1(0,col) = 1.0/(Real)per_element_dof_num_k1;
        for (int row = 1;row<n_polynomials_k1;++row) {
            for (int col = 0;col<per_element_dof_num_k1;++col) {
                // 我们这里需要能从体自由度对应到一个面内的对应自由度
                Real first_term = 0.0;
                const VEM_Monomial_3d &poly = monomials_3d[row];
                Real second_term = 0.0;

                for (int face_id=0;face_id<this->elements_2d.size();++face_id) {
                    auto&face = this->elements_2d[face_id];
                    if (face.degenerated) continue;
                    if (!face.volume_dof_to_local_dof.count(col))continue;

                    int local_dof_id = face.volume_dof_to_local_dof[col];

                    auto normal_gradient_x = poly.derivative_x();
                    normal_gradient_x.c = normal_gradient_x.c * face.plane.normal.vx;
                    auto normal_gradient_y = poly.derivative_y();
                    normal_gradient_y.c = normal_gradient_y.c * face.plane.normal.vy;
                    auto normal_gradient_z = poly.derivative_z();
                    normal_gradient_z.c = normal_gradient_z.c * face.plane.normal.vz;

                    std::vector<VEM_Monomial_2d> projected_poly_2ds_x = face.project_monomial_to_plane(normal_gradient_x,this->centroid);
                    std::vector<VEM_Monomial_2d> projected_poly_2ds_y = face.project_monomial_to_plane(normal_gradient_y,this->centroid);
                    std::vector<VEM_Monomial_2d> projected_poly_2ds_z = face.project_monomial_to_plane(normal_gradient_z,this->centroid);
                    std::vector<VEM_Monomial_2d> projected_poly_2ds;
                    projected_poly_2ds.reserve(
                        projected_poly_2ds_x.size() +
                        projected_poly_2ds_y.size() +
                        projected_poly_2ds_z.size());

                    projected_poly_2ds.insert(projected_poly_2ds.end(),
                                              projected_poly_2ds_x.begin(), projected_poly_2ds_x.end());
                    projected_poly_2ds.insert(projected_poly_2ds.end(),
                                              projected_poly_2ds_y.begin(), projected_poly_2ds_y.end());
                    projected_poly_2ds.insert(projected_poly_2ds.end(),
                                              projected_poly_2ds_z.begin(), projected_poly_2ds_z.end());

                    // RZ_INFO(face.project_matrix_star);
                    for (auto&projected_poly2d:projected_poly_2ds) {// 这些应该都是1阶的
                        // 可否改成不依赖低维度的
                        Real deno1 = std::pow(this->diameter,poly.p+poly.q+poly.r);;
                        for (int alpha = 0;alpha<monomials_2d_k1.size();++alpha) {
                            // RZ_INFO(face.project_matrix_star.rows()," ",face.project_matrix_star.cols());
                            Real weight = face.project_matrix_star_k1(alpha,local_dof_id);
                            auto no_centered_poly2ds = face.expand_centered_monomial(monomials_2d_k1[alpha]);
                            for (auto& no_centered_poly2d:no_centered_poly2ds) {
                                auto mul_mono = no_centered_poly2d * projected_poly2d;
                                // Real this_coff = weight * face.polygon_moment_general_2d(mul_mono,{0,0},false);
                                Real this_coff = weight * table[face_id][pq_to_id[{mul_mono.p,mul_mono.q}]] * mul_mono.c;
                                this_coff /= deno1;
                                this_coff /= std::pow(face.diameter,monomials_2d_k1[alpha].p+monomials_2d_k1[alpha].q);
                                second_term += this_coff;
                            }
                        }
                    }
                }
                B_k1(row,col) = first_term + second_term;
            }
        }

        //计算 G
        {// for check
            G_k1.resize(n_polynomials_k1,n_polynomials_k1);
            G_k1.setZero();
            for (int col =0;col<n_polynomials_k1;++col) {
                const auto& poly = monomials_3d[col];
                Real ans = 0.0;
                for (int vid = 0; vid < vertex_count(); ++vid) {
                    const _3d_vec &vert = vertex_position(vid);
                    ans+=poly.value(vert.vx,vert.vy,vert.vz,
                            this->centroid.vx,this->centroid.vy,this->centroid.vz,this->diameter);
                }
                G_k1(0,col) =  ans/(Real)per_element_dof_num_k1;
            }
            for (int row = 1;row<n_polynomials_k1;++row) {
                for (int col = 1;col<n_polynomials_k1;++col) {
                    Real ans = 0.0;
                    const auto& polyrow = monomials_3d[row];
                    const auto& polycol = monomials_3d[col];
                    const auto& poly_dv_x = polyrow.derivative_x()*polycol.derivative_x();
                    const auto& poly_dv_y = polyrow.derivative_y()*polycol.derivative_y();
                    const auto& poly_dv_z = polyrow.derivative_z()*polycol.derivative_z();
                    Real denominator =  std::pow(this->diameter,polyrow.p+polyrow.q+polyrow.r + polycol.p + polycol.q + polycol.r);
                    ans += integrate_centered_monomial_over_polyhedron_divergence(poly_dv_x,this->centroid) /denominator;
                    ans += integrate_centered_monomial_over_polyhedron_divergence(poly_dv_y,this->centroid) /denominator;
                    ans += integrate_centered_monomial_over_polyhedron_divergence(poly_dv_z,this->centroid) /denominator;
                    G_k1(row,col) = ans;
                }
            }
            Eigen::LocalMatrixXr multed_G_k1= B_k1 * D_k1;
            // if ((multed_G_k1 - G_k1).cwiseAbs().maxCoeff()>1e-3){
            //     this->save_polygonmesh_to_obj("check.obj");
            //     RZ_FATAL((multed_G_k1 - G_k1).cwiseAbs().maxCoeff());
            // }
        }
        G_k1.setZero();
        G_k1 = B_k1 * D_k1;
        // Eigen::JacobiSVD<Eigen::LocalMatrixXr> svd_k1(G_k1, Eigen::ComputeFullU|Eigen::ComputeFullV);
        // auto s = svd_k1.singularValues();
        // Real tau = 1e-10 * s(0);   // 可调：1e-12 ~ 1e-10
        // Eigen::LocalVectorXr inv = s;
        // for(int i=0;i<s.size();++i) inv(i) = (s(i) > tau) ? (1.0/s(i)) : 0.0;
        // Eigen::LocalMatrixXr Gpinv_k1 = svd_k1.matrixV() * inv.asDiagonal() * svd_k1.matrixU().transpose();
        Eigen::LocalMatrixXr project_k1 = G_k1.inverse() * B_k1;
        this->project_matrix_star_k1 = project_k1;

        Eigen::LocalMatrixXr I = Eigen::LocalMatrixXr::Identity(per_element_dof_num_k1, per_element_dof_num_k1);
        Eigen::LocalMatrixXr residual = I - D_k1 * project_k1; // (per_element_dof x per_element_dof)
        Eigen::LocalMatrixXr stabilising_term = residual.transpose() * residual * this->diameter;//
        G_k1.row(0).setZero();
        this->stiffness_matrix_k1 = project_k1.transpose() *
            G_k1 * project_k1 + stabilising_term;

        H_X_k1.resize(n_polynomials_k1, n_polynomials_k1);
        H_Y_k1.resize(n_polynomials_k1, n_polynomials_k1);
        H_Z_k1.resize(n_polynomials_k1, n_polynomials_k1);
        H_X_k1.setZero();
        H_Y_k1.setZero();
        H_Z_k1.setZero();
        //-----------------------------k1
        for (int i = 0; i < n_polynomials_k1; ++i) {
            for (int j = 0; j < n_polynomials_k1; ++j) {
                // 由于一定有一个求导 所以最后一定要多除一个 max_diameter
                VEM_Monomial_3d temp_x = monomials_3d[i].derivative_x() * monomials_3d[j];
                H_X_k1(i,j) = integrate_centered_monomial_over_polyhedron_divergence(temp_x,this->centroid)/
                    std::pow(this->diameter, std::max(temp_x.p + temp_x.q + temp_x.r + 1, 0));

                VEM_Monomial_3d temp_y = monomials_3d[i].derivative_y() * monomials_3d[j];
                H_Y_k1(i,j) = integrate_centered_monomial_over_polyhedron_divergence(temp_y,this->centroid)/
                    std::pow(this->diameter, std::max(temp_y.p + temp_y.q + temp_y.r + 1, 0));

                VEM_Monomial_3d temp_z = monomials_3d[i].derivative_z() * monomials_3d[j];
                H_Z_k1(i,j) = integrate_centered_monomial_over_polyhedron_divergence(temp_z,this->centroid)/
                    std::pow(this->diameter, std::max(temp_z.p + temp_z.q + temp_z.r + 1, 0));
            }
        }

        this->local_LX_k1 = project_k1.transpose() * H_X_k1 * project_k1;
        this->local_LY_k1 = project_k1.transpose() * H_Y_k1 * project_k1;
        this->local_LZ_k1 = project_k1.transpose() * H_Z_k1 * project_k1;

    }

}


void VEM_Element_3d::save_polygonmesh_to_obj(const std::string &filename) const{
    {
        std::map<Edge,int> edge_cnt_map;
        for (auto& face:polygons) {
            int n = (int)face.size();
            for (int i = 0; i < n; ++i) {
                int v1 = face[i];
                int v2 = face[(i + 1) % n]; // 闭合
                RZ_ASSERT(v1>=0&&v2>=0,"v1>=0&&v2>=0");
                edge_cnt_map[Edge(v1,v2)]++;
                if (edge_cnt_map[Edge(v1,v2)]>2)
                    RZ_FATAL("no manifold");
            }
        }
        for (auto& edge: edge_cnt_map) {
            RZ_ASSERT(edge.second == 2,edge.second);
        }
        std::ofstream file(filename.c_str());
        if (!file.is_open()) {
            std::cerr << "无法打开文件: " << filename << std::endl;
            return;
        }

        // 写入顶点
        for (int vid = 0; vid < vertex_count(); ++vid) {
            file << vertex_position(vid).to_obj_format() << std::endl;
        }

        // 写入面
        for (const auto& polygon : polygons) {
            file << "f";
            for (int idx : polygon) {
                // .obj 文件中索引从 1 开始，所以我们加 1
                file << " " << idx + 1;
            }
            file << std::endl;
        }

        file.close();
        std::cout << "文件保存成功: " << filename << std::endl;
    }
}

void VEM_Element_3d::save_face_to_obj(const std::vector<int>& polygon,const std::string &filename) const{

    std::ofstream file(filename.c_str());
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return;
    }

    // 写入顶点
    for (const auto& pid : polygon) {
        file << vertex_position(pid).to_obj_format() << std::endl;
    }

    // 写入面
    file << "f";
    for (int idx =0;idx<polygon.size();++idx) {
        file << " " << idx + 1;
    }

    file.close();
    std::cout << "文件保存成功: " << filename << std::endl;
}

bool VEM_Element_3d::point_in_elem(const _3d_vec &p,const Real eps) const{
    // if (is_convex)return point_in_elem_fast(p,eps);
    // else return point_in_elem_nonconvex(p);
    CGAL_Point_3 cg_p(p.vx,p.vy,p.vz);
    for (int fid=0;fid<polygons.size();++fid) {
        auto cg_plane = poly_planes[fid];
        if (cg_plane.has_on_positive_side(cg_p) && CGAL::squared_distance(cg_plane,cg_p)>1e-24)return false;
    }
    return true;
}

inline bool VEM_Element_3d::point_in_elem_fast(const _3d_vec& p,const Real eps) const
{
    for (int fid = 0; fid < polygons.size(); ++fid) {
        const auto& plane = poly_planes[fid];

        // 直接用平面方程值
        Real s =
            CGAL::to_double(plane.a()) * p.vx +
            CGAL::to_double(plane.b()) * p.vy +
            CGAL::to_double(plane.c()) * p.vz +
            CGAL::to_double(plane.d());

        if (s > eps)
            return false;
    }
    return true;
}

static Real triangle_solid_angle(
    const Eigen::Vector3d& a,
    const Eigen::Vector3d& b,
    const Eigen::Vector3d& c,
    const Eigen::Vector3d& p)
{
    Eigen::Vector3d A = a - p;
    Eigen::Vector3d B = b - p;
    Eigen::Vector3d C = c - p;

    Real la = A.norm();
    Real lb = B.norm();
    Real lc = C.norm();

    constexpr Real eps = 1e-12;
    if (la < eps || lb < eps || lc < eps)
        return 0.0; // 点在顶点上，交给上层处理

    Real numerator = A.dot(B.cross(C));
    Real denominator =
        la * lb * lc +
        A.dot(B) * lc +
        B.dot(C) * la +
        C.dot(A) * lb;

    return 2.0 * std::atan2(numerator, denominator);
}


bool VEM_Element_3d::point_in_elem_nonconvex(const _3d_vec &p) const
{
    Eigen::Vector3d P(p.vx, p.vy, p.vz);
    Real omega_sum = 0.0;

    // ---------- 数值容忍参数（可根据网格尺度调整） ----------
    const Real area_eps = 1e-14;          // 过滤极小三角形
    const Real near_eps = 1e-12;          // 点靠近面/顶点的宽容阈值
    const Real inside_threshold = 3.0 * M_PI; // 比 2π 更稳健

    const int F = static_cast<int>(polygons.size());
    for (int fid = 0; fid < F; ++fid) {
        const auto& poly = polygons[fid];
        const int n = static_cast<int>(poly.size());
        if (n < 3) continue;

        // ---------- CGAL 平面法向（已统一方向） ----------
        const auto& plane = poly_planes[fid];
        Eigen::Vector3d n_plane(
            CGAL::to_double(plane.a()),
            CGAL::to_double(plane.b()),
            CGAL::to_double(plane.c())
        );

        // ---------- fan triangulation 的锚点 ----------
        const _3d_vec& v0_ = vertex_position(poly[0]);
        Eigen::Vector3d v0(v0_.vx, v0_.vy, v0_.vz);

        for (int i = 1; i + 1 < n; ++i) {
            const _3d_vec& v1_ = vertex_position(poly[i]);
            const _3d_vec& v2_ = vertex_position(poly[i + 1]);

            Eigen::Vector3d v1(v1_.vx, v1_.vy, v1_.vz);
            Eigen::Vector3d v2(v2_.vx, v2_.vy, v2_.vz);

            // ---------- 近表面宽容：tracer 场景很有用 ----------
            if ((v0 - P).squaredNorm() < near_eps ||
                (v1 - P).squaredNorm() < near_eps ||
                (v2 - P).squaredNorm() < near_eps) {
                return true;
            }

            // ---------- 三角形法向（未归一化） ----------
            Eigen::Vector3d n_tri = (v1 - v0).cross(v2 - v0);
            Real area2 = n_tri.norm(); // = 2 * area

            // 极小 / 退化三角形直接跳过（法向极不稳定）
            if (area2 < area_eps) {
                continue;
            }

            // ---------- 用 CGAL plane 法向纠正三角形朝向 ----------
            // 不再依赖 polygon 顶点顺序
            if (n_tri.dot(n_plane) < 0.0) {
                std::swap(v1, v2);
            }

            // ---------- 累加有向 solid angle ----------
            omega_sum += triangle_solid_angle(v0, v1, v2, P);
        }
    }

    // inside ≈ ±4π, outside ≈ 0
    return std::abs(omega_sum) > inside_threshold;
}

void VEM_Element_3d::bind_geometry(const std::vector<_3d_vec>& domain_points,
                                   const std::vector<Edge>& domain_edges) {
    domain_points_data_ = domain_points.data();
    domain_points_count_ = (int)domain_points.size();
    domain_edges_data_ = domain_edges.data();
    domain_edges_count_ = (int)domain_edges.size();
}

int VEM_Element_3d::vertex_count() const {
    return (int)vertex_local_to_global_dof.size();
}

int VEM_Element_3d::edge_count() const {
    return (int)edge_local_to_global_dof.size();
}

const _3d_vec& VEM_Element_3d::vertex_position(int local_vertex_id) const {
    const int global_id = vertex_local_to_global_dof[local_vertex_id];
    RZ_ASSERT(domain_points_data_ != nullptr, "");
    RZ_ASSERT(global_id >= 0 && global_id < domain_points_count_, global_id);
    return domain_points_data_[global_id];
}

Edge VEM_Element_3d::edge(int local_edge_id) const {
    const int global_edge_id = edge_local_to_global_dof[local_edge_id];
    RZ_ASSERT(domain_edges_data_ != nullptr, "");
    RZ_ASSERT(global_edge_id >= 0 && global_edge_id < domain_edges_count_, global_edge_id);
    const Edge global_edge = domain_edges_data_[global_edge_id];

    int lv1 = -1;
    int lv2 = -1;
    for (int i = 0; i < (int)vertex_local_to_global_dof.size(); ++i) {
        if (vertex_local_to_global_dof[i] == global_edge.v1) lv1 = i;
        if (vertex_local_to_global_dof[i] == global_edge.v2) lv2 = i;
    }
    RZ_ASSERT(lv1 >= 0 && lv2 >= 0, global_edge.v1, " ", global_edge.v2);
    return Edge(lv1, lv2);
}


int VEM_Element_3d::global_edge_id_from_vertices(int global_v1, int global_v2) const {
    const Edge query(global_v1, global_v2);
    for (int eid = 0; eid < edge_count(); ++eid) {
        const Edge local_edge = edge(eid);
        const Edge global_edge(vertex_local_to_global_dof[local_edge.v1],
                               vertex_local_to_global_dof[local_edge.v2]);
        if (global_edge == query) {
            return edge_local_to_global_dof[eid];
        }
    }
    RZ_FATAL("missing local edge for global vertices ", global_v1, " ", global_v2);
    return -1;
}



void VEM_Element_3d::clear_stiff_k2() {
    Eigen::LocalMatrixXr().swap(stiffness_matrix_k2);
}

void VEM_Element_3d::clear_stiff_k1() {
    Eigen::LocalMatrixXr().swap(stiffness_matrix_k1);
}

void VEM_Element_3d::clear_assembly_cache() {
    std::vector<VEM_Element_2d>().swap(elements_2d);
    clear_stiff_k2();
    clear_stiff_k1();
}

void VEM_Element_3d::clear_generate_elements_cache() {
    clear_assembly_cache();
}

void VEM_Element_3d::clear_hierarchy_cache() {
    clear_assembly_cache();
    Eigen::MatrixXr().swap(project_matrix_star_k2);
    Eigen::MatrixXr().swap(project_matrix_star_k1);
    Eigen::MatrixXr().swap(local_LX_k2);
    Eigen::MatrixXr().swap(local_LY_k2);
    Eigen::MatrixXr().swap(local_LZ_k2);
    Eigen::MatrixXr().swap(local_LX_k1);
    Eigen::MatrixXr().swap(local_LY_k1);
    Eigen::MatrixXr().swap(local_LZ_k1);
    Eigen::VectorXr().swap(poly_coeffs_u_k1);
    Eigen::VectorXr().swap(poly_coeffs_v_k1);
    Eigen::VectorXr().swap(poly_coeffs_w_k1);
    Eigen::VectorXr().swap(poly_coeffs_rho_k1);
    Eigen::VectorXr().swap(poly_coeffs_oldu_k1);
    Eigen::VectorXr().swap(poly_coeffs_oldv_k1);
    Eigen::VectorXr().swap(poly_coeffs_oldw_k1);
    Eigen::VectorXr().swap(poly_coeffs_du_k1);
    Eigen::VectorXr().swap(poly_coeffs_dv_k1);
    Eigen::VectorXr().swap(poly_coeffs_dw_k1);
}
void VEM_Domain_3d::build_boundary_handle_grid() {
    FUNC_TIMER();

    const int nx = boundary_handle_grid.nx;
    const int ny = boundary_handle_grid.ny;
    const int nz = boundary_handle_grid.nz;
    const Real dx = 1.0 / (Real)nx;

    // 1. resize & clear
    boundary_handle_grid.resize(nx);

    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
                boundary_handle_grid.grid[i][j][k] = false;

    // 2. triangle bbox overmark
    for (const auto& tri : tris) {

        Real xmin = std::min({ tri.vertex(0).x(), tri.vertex(1).x(), tri.vertex(2).x() });
        Real ymin = std::min({ tri.vertex(0).y(), tri.vertex(1).y(), tri.vertex(2).y() });
        Real zmin = std::min({ tri.vertex(0).z(), tri.vertex(1).z(), tri.vertex(2).z() });

        Real xmax = std::max({ tri.vertex(0).x(), tri.vertex(1).x(), tri.vertex(2).x() });
        Real ymax = std::max({ tri.vertex(0).y(), tri.vertex(1).y(), tri.vertex(2).y() });
        Real zmax = std::max({ tri.vertex(0).z(), tri.vertex(1).z(), tri.vertex(2).z() });

        int i0 = (int)std::floor(xmin / dx);
        int j0 = (int)std::floor(ymin / dx);
        int k0 = (int)std::floor(zmin / dx);

        int i1 = (int)std::floor(xmax / dx);
        int j1 = (int)std::floor(ymax / dx);
        int k1 = (int)std::floor(zmax / dx);

        i0 = std::max(0, std::min(nx - 1, i0));
        j0 = std::max(0, std::min(ny - 1, j0));
        k0 = std::max(0, std::min(nz - 1, k0));

        i1 = std::max(0, std::min(nx - 1, i1));
        j1 = std::max(0, std::min(ny - 1, j1));
        k1 = std::max(0, std::min(nz - 1, k1));

        for (int i = i0; i <= i1; ++i)
            for (int j = j0; j <= j1; ++j)
                for (int k = k0; k <= k1; ++k)
                    boundary_handle_grid.grid[i][j][k] = true;
    }

    // 3. snapshot 原始 boundary cells
    std::vector<UGID> seeds;
    seeds.reserve(1024);

    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
                if (boundary_handle_grid.grid[i][j][k])
                    seeds.emplace_back(i, j, k);
    RZ_INFO("number of grids contain boundary: ",seeds.size());
    // 4. 1-ring 扩散（仅 seeds 参与）
    for (const auto& id : seeds) {
        for (int di = -1; di <= 1; ++di)
            for (int dj = -1; dj <= 1; ++dj)
                for (int dk = -1; dk <= 1; ++dk) {

                    if (di == 0 && dj == 0 && dk == 0) continue;

                    int ni = id.x + di;
                    int nj = id.y + dj;
                    int nk = id.z + dk;

                    if (ni < 0 || ni >= nx ||
                        nj < 0 || nj >= ny ||
                        nk < 0 || nk >= nz)
                        continue;

                    boundary_handle_grid.grid[ni][nj][nk] = true;
                }
    }

    RZ_INFO("Boundary handle grid built (bool grid + 1-ring buffer).");
}



bool VEM_Domain_3d::read_from_file(std::string filename) {
        using namespace tinyxml2;

    XMLDocument doc;
    if (doc.LoadFile(filename.c_str()) != XML_SUCCESS) {
        std::cerr << "❌ 无法打开VTU文件: " << filename << std::endl;
        return false;
    }

    XMLElement* root = doc.FirstChildElement("VTKFile");
    if (!root) { std::cerr << "❌ 无效的VTU文件结构" << std::endl; return false; }

    XMLElement* grid = root->FirstChildElement("UnstructuredGrid");
    if (!grid) return false;
    XMLElement* piece = grid->FirstChildElement("Piece");
    if (!piece) return false;

    // ---------- 读取 Points ----------
    XMLElement* points_xml = piece->FirstChildElement("Points");
    XMLElement* da_pts = points_xml ? points_xml->FirstChildElement("DataArray") : nullptr;
    if (!da_pts) return false;

    std::istringstream iss_pts(da_pts->GetText() ? da_pts->GetText() : "");
    Real x, y, z;
    while (iss_pts >> x >> y >> z) points.emplace_back(x, y, z);

    // ---------- 读取 Cells ----------
    XMLElement* cells_xml = piece->FirstChildElement("Cells");
    if (!cells_xml) return false;

    std::vector<int> faces_data;
    std::vector<int> faceoffsets;

    for (XMLElement* da = cells_xml->FirstChildElement("DataArray"); da; da = da->NextSiblingElement("DataArray")) {
        std::string name = da->Attribute("Name") ? da->Attribute("Name") : "";
        std::istringstream iss(da->GetText() ? da->GetText() : "");
        if (name == "faces") {
            int v; while (iss >> v) faces_data.push_back(v);
        } else if (name == "faceoffsets") {
            int v; while (iss >> v) faceoffsets.push_back(v);
        }
    }

    if (faces_data.empty() || faceoffsets.empty()) {
        std::cerr << "❌ 未找到 faces / faceoffsets 数组" << std::endl;
        return false;
    }
    std::vector<int> polyhedrons_end;
    // ---------- 解析 faces_flat / faceoffsets ----------
    size_t cursor = 0;
    for (size_t c = 0; c < faceoffsets.size(); ++c) {
        size_t end = faceoffsets[c];
        if (cursor >= end) break;

        int n_faces = faces_data[cursor++];
        std::vector<std::vector<int>> cell_faces;
        for (int fi = 0; fi < n_faces; ++fi) {
            int n_v = faces_data[cursor++];
            std::vector<int> face;
            for (int j = 0; j < n_v; ++j)
                face.push_back(faces_data[cursor++]);
            polygons.push_back(face);
        }
        polyhedrons_end.push_back(polygons.size());  // 记录多面体结束位置
        cursor = end;
    }

    // ==================== 去重 polygons + 构建 polyhedrons ====================
    {
            std::vector<std::vector<int>> unique_polygons;
            std::vector<int> remap(polygons.size(), -1);
            std::vector<std::vector<int>> new_polyhedrons;  // ✅ 每个多面体 → 面索引集合

            auto canonical_key = [](std::vector<int> face) {
                if (face.empty()) return std::string();
                std::sort(face.begin(), face.end()); // 忽略顺序方向
                std::ostringstream oss;
                for (size_t i = 0; i < face.size(); ++i) {
                    oss << face[i];
                    if (i + 1 < face.size()) oss << ",";
                }
                return oss.str();
            };

            std::unordered_map<std::string, int> face_map;
            face_map.reserve(polygons.size() * 2);

            // 首先建立唯一面集合及 remap
            for (size_t i = 0; i < polygons.size(); ++i) {
                std::string key = canonical_key(polygons[i]);
                auto it = face_map.find(key);
                if (it == face_map.end()) {
                    int new_idx = (int)unique_polygons.size();
                    face_map[key] = new_idx;
                    remap[i] = new_idx;
                    unique_polygons.push_back(polygons[i]);
                    is_polygon_boundary.push_back(true);//todo:
                } else {
                    remap[i] = it->second; // 相同集合 → 复用同一面
                    is_polygon_boundary[it->second] = false;
                }
            }

            // ✅ 重新构建 polyhedrons
            size_t start = 0;
            for (size_t i = 0; i < polyhedrons_end.size(); ++i) {
                size_t end = polyhedrons_end[i];
                std::vector<int> poly;
                poly.reserve(end - start);
                for (size_t j = start; j < end; ++j)
                    poly.push_back(remap[j]);  // remap 到唯一面索引
                new_polyhedrons.push_back(poly);
                start = end;
            }

            polygons.swap(unique_polygons);
            polyhedrons.swap(new_polyhedrons); // ✅ 保存结构化的多面体信息

            std::cout << "面去重完成: 原面数 = " << remap.size()
                      << " → 新面数 = " << polygons.size()
                      << "，多面体数 = " << polyhedrons.size() << std::endl;
    }

    // ==================== 生成唯一边 ====================
    {
            std::set<Edge> temp_unique_edges;
            // 遍历每个面
            for (const auto& face : polygons) {
                int n = (int)face.size();
                RZ_ASSERT(n>2,"");

                // 循环取边 (v[i], v[i+1])
                for (int i = 0; i < n; ++i) {
                    int v1 = face[i];
                    int v2 = face[(i + 1) % n]; // 闭合
                    temp_unique_edges.insert(Edge(v1, v2));
                }
            }

            for (auto &edge: temp_unique_edges) {
                edges.push_back(edge);
            }
            std::cout << "唯一边构建完成: 共 " << edges.size() << " 条边" << std::endl;

    }
    // // check no manifold edges
    // std::set<int> used_points;
    // std::set<int> used_edges;
    // for (auto&polyherdon:polyhedrons) {
    //     std::map<Edge,int> edge_cnt_map;
    //     for (auto& polyid:polyherdon) {
    //         auto face= polygons[polyid];
    //         int n = (int)face.size();
    //         for (int i = 0; i < n; ++i) {
    //             int v1 = face[i];
    //             int v2 = face[(i + 1) % n]; // 闭合
    //             used_points.insert(v1);
    //             used_points.insert(v2);
    //             edge_cnt_map[Edge(v1,v2)]++;
    //             if (edge_cnt_map[Edge(v1,v2)]>2)
    //                 RZ_FATAL("no manifold");
    //         }
    //     }
    //     for (auto& edge: edge_cnt_map) {
    //         RZ_ASSERT(edge.second == 2,edge.second);
    //     }
    // }
    // RZ_ASSERT(used_points.size() == points.size(),"use: ",used_points.size());
    // RZ_ASSERT(used_edges.size() == edges.size(),"");
    // // 检查是否 所有的 polygons都被用到了  每个边 都被用到了 每个点都被用到了
    // std::set<int> used_polygons;
    // for (auto&polyherdon:polyhedrons) {
    //     for (auto&poly:polyherdon) {
    //         used_polygons.insert(poly);
    //     }
    // }
    // RZ_ASSERT(used_polygons.size() == polygons.size()," ");

    std::cout << "✅ 成功读取VTU: " << filename << std::endl;
    std::cout << "   顶点数: " << points.size()
              << "  面数: " << polygons.size()
              << "  多面体数: " << polyhedrons.size() << std::endl;
    return true;
}

void VEM_Domain_3d::load_from_lcc(Split_lcc& split_lcc)
{
    // =========================
    // 0) clear / reset
    // =========================
    EK_to_IK to_ik;
    auto& lcc = split_lcc.lcc;
    points.clear();
    polygons.clear();
    polyhedrons.clear();
    edges.clear();
    is_polygon_boundary.clear();
    is_obj_boundary.clear();
    is_vertex_boundary.clear();
    vertex_normals.clear();
    vertex_neumann_vels.clear();
    face_neumann_vels.clear();
    vem_elements.clear();
    edges.clear();
    // 若你还有其它缓存（邻接、cell->faces、face->edges、cell markers 等），也建议这里清掉
    // clear_adjacency(); ...

    // =========================
    // 1) 收集顶点并给每个 LCC 0-cell 分配唯一 id
    //    这段逻辑与“lcc->vtu”一致：只遍历非 ghost 且非 inside 的 3-cell，收集其 incident 顶点
    // =========================
    {
        // 先把所有 0-cell 的 info.id 置 -1（跟你 vtu 导出代码一致）:contentReference[oaicite:2]{index=2}
        auto v0dhs = collect(lcc.one_dart_per_cell<0>());
        for (auto dh : v0dhs) lcc.info<0>(dh).id = -1;
        int vertex_index = 0;
        auto cdhs = collect(lcc.one_dart_per_cell<3>());
        for (auto cdh : cdhs) {
            if (lcc.info<3>(cdh).is_ghost)  continue;
            if (lcc.info<3>(cdh).is_inside) continue;

            for (auto vdh : collect(lcc.one_dart_per_incident_cell<0, 3>(cdh))) {
                if (lcc.info<0>(vdh).id == -1) {
                    lcc.info<0>(vdh).id = vertex_index++;

                    auto p = to_ik(lcc.point(vdh));
                    // 这里 points 的类型按你工程来（下面假设是 std::vector<Vec3d> 或 EK::Point_3）
                    points.emplace_back(p.x(), p.y(), p.z());
                }
            }
        }
    }

    // =========================
    // 2) 遍历每个 3-cell，提取其所有 2-cell 作为 faces
    //    每个 face 沿 beta<1> 环走一圈得到顶点 id 序列（与你 vtu 导出一致）:contentReference[oaicite:3]{index=3}
    //    注意：这里先把每个 cell 的 faces “扁平 push 到 polygons”，同时记录每个 cell 的 end 偏移
    // =========================
    std::vector<int> polyhedrons_end; // 每个 polyhedron 在 polygons（临时面列表）里的结束下标

    {
        auto cdhs = collect(lcc.one_dart_per_cell<3>());
        for (auto cdh : cdhs) {
            if (lcc.info<3>(cdh).is_ghost)  continue;
            if (lcc.info<3>(cdh).is_inside) continue;

            for (auto fdh : collect(lcc.one_dart_per_incident_cell<2, 3>(cdh))) {
                auto face_start = fdh;
                auto face_dart  = face_start;

                std::vector<int> face_vids;
                // 沿面边界走一圈：beta<1>
                do {
                    const int vid = lcc.info<0>(face_dart).id;
                    // 防御：理论上不会出现 -1，除非前面顶点分配漏掉了
                    RZ_ASSERT(vid >= 0, "vertex id not assigned (id = -1)");
                    face_vids.push_back(vid);
                    face_dart = lcc.beta<1>(face_dart);
                } while (face_dart != face_start);

                // face 顶点数至少 3
                RZ_ASSERT((int)face_vids.size() >= 3, "degenerate face");
                // static thread_local std::mt19937 gen(std::random_device{}());
                // static thread_local std::bernoulli_distribution dist(0.5);
                //
                // // 50% 概率 reverse
                // if (dist(gen)) {
                //     std::reverse(face_vids.begin(), face_vids.end());
                // }
                polygons.push_back(std::move(face_vids));
            }

            polyhedrons_end.push_back((int)polygons.size());
        }
    }

    std::vector<std::vector<int>> obj_surface_polygons;
    {
        auto cdhs = collect(lcc.one_dart_per_cell<3>());
        for (auto cdh : cdhs) {
            if (!lcc.info<3>(cdh).is_inside) continue;

            for (auto fdh : collect(lcc.one_dart_per_incident_cell<2, 3>(cdh))) {
                auto face_start = fdh;
                auto face_dart  = face_start;

                std::vector<int> face_vids;
                // 沿面边界走一圈：beta<1>
                bool flag = true;
                do {
                    const int vid = lcc.info<0>(face_dart).id;
                    if (vid<0)flag = false;
                    face_vids.push_back(vid);
                    face_dart = lcc.beta<1>(face_dart);
                } while (face_dart != face_start);

                // face 顶点数至少 3
                RZ_ASSERT((int)face_vids.size() >= 3, "degenerate face");
                if (flag)
                    obj_surface_polygons.push_back(std::move(face_vids));
            }
        }
    }
    // =========================
    // 3) 面去重 + 构建 polyhedrons（cell->unique face ids）
    //    这段逻辑按你 VTU 读取代码的 canonical_key 去重策略复刻：忽略顺序方向，用“顶点集合排序后字符串”作为 key:contentReference[oaicite:4]{index=4}
    //    注意：这会把同一组顶点但反向/循环位移的面认为相同（适合无向面唯一化）
    // =========================
    {
        std::vector<std::vector<int>> unique_polygons;
        unique_polygons.reserve(polygons.size());

        std::vector<int> remap(polygons.size(), -1);              // old face id -> new unique face id
        std::vector<std::vector<int>> new_polyhedrons;            // 每个 cell: unique face ids

        auto canonical_key = [](std::vector<int> face) -> std::string {
            if (face.empty()) return std::string();
            std::sort(face.begin(), face.end()); // 忽略顺序/方向：集合 key
            std::ostringstream oss;
            for (size_t i = 0; i < face.size(); ++i) {
                oss << face[i];
                if (i + 1 < face.size()) oss << ",";
            }
            return oss.str();
        };

        std::unordered_map<std::string, int> face_map;
        std::set<std::string> obj_surfaces;
        for (auto&polygon:obj_surface_polygons) {
            obj_surfaces.insert(canonical_key(polygon));
        }

        face_map.reserve(polygons.size() * 2);

        // boundary 标志：先默认 true，若发现重复（被两个 cell 共用），置 false
        // 这与 VTU loader 里的处理一致:contentReference[oaicite:5]{index=5}
        int cnt= 0;
        for (size_t i = 0; i < polygons.size(); ++i) {
            std::string key = canonical_key(polygons[i]);
            auto it = face_map.find(key);
            if (it == face_map.end()) {
                int new_idx = (int)unique_polygons.size();
                face_map.emplace(std::move(key), new_idx);
                remap[i] = new_idx;
                unique_polygons.push_back(polygons[i]);
                is_polygon_boundary.push_back(true); // 初见认为是边界面
                if (obj_surfaces.count(canonical_key(polygons[i]))) {
                    is_obj_boundary.push_back(true);
                    ++cnt;
                }
                else is_obj_boundary.push_back(false);
            } else {
                remap[i] = it->second;
                // 重复面 => 内部共享面
                is_polygon_boundary[it->second] = false;
            }
        }
        // 重建每个 polyhedron 的 face 列表（用 remap 后的 unique face id）
        size_t start = 0;
        for (size_t ci = 0; ci < polyhedrons_end.size(); ++ci) {
            size_t end = (size_t)polyhedrons_end[ci];

            std::vector<int> poly;
            poly.reserve(end - start);
            for (size_t fj = start; fj < end; ++fj) poly.push_back(remap[fj]);

            new_polyhedrons.push_back(std::move(poly));
            start = end;
        }

        polygons.swap(unique_polygons);
        polyhedrons.swap(new_polyhedrons);
    }


    // =========================
    // 4) 生成唯一边
    //    完全照搬你 VTU loader 的“遍历每个面，循环取边(v[i], v[i+1])，用 set 去重”:contentReference[oaicite:6]{index=6}
    // =========================
    {
        std::set<Edge> temp_unique_edges;

        for (const auto& face : polygons) {
            const int n = (int)face.size();
            RZ_ASSERT(n >= 3, "degenerate face");

            for (int i = 0; i < n; ++i) {
                int v1 = face[i];
                int v2 = face[(i + 1) % n];
                temp_unique_edges.insert(Edge(v1, v2));
            }
        }

        edges.reserve(temp_unique_edges.size());
        for (const auto& e : temp_unique_edges) {
            edges.push_back(e);
        }
    }

    // =========================
    // 5) 可选：一致性检查（建议你保留）
    // =========================
    {
        // 检查 polyhedrons face id 合法
        for (const auto& cell : polyhedrons) {
            for (int fid : cell) {
                RZ_ASSERT(fid >= 0 && fid < (int)polygons.size(), "invalid face id in polyhedron");
            }
        }
    }

    std::cout << "✅ load_from_lcc done.\n"
              << "   points      : " << points.size() << "\n"
              << "   polygons    : " << polygons.size() << "\n"
              << "   polyhedrons : " << polyhedrons.size() << "\n"
              << "   edges       : " << edges.size() << std::endl;
}

void VEM_Domain_3d::load_from_Minilcc(MiniLcc &lcc) {
    FUNC_TIMER();
    points.clear();
    polygons.clear();
    polyhedrons.clear();
    edges.clear();
    is_polygon_boundary.clear();
    is_obj_boundary.clear();
    is_vertex_boundary.clear();
    vertex_normals.clear();
    vertex_neumann_vels.clear();
    face_neumann_vels.clear();
    const int point_count = (int)lcc.points.size();
    const int face_count = (int)lcc.faces.size();
    const int cvx_count = (int)lcc.cvxs.size();

    auto canonicalize_face = [](const std::vector<int> &face) -> std::vector<int> {
        std::vector<int> key = face;
        std::sort(key.begin(), key.end());
        return key;
    };

    auto edge_key = [](int a, int b) -> uint64_t {
        const uint32_t lo = (uint32_t)std::min(a, b);
        const uint32_t hi = (uint32_t)std::max(a, b);
        return (uint64_t(hi) << 32) | lo;
    };

    struct VecHash {
        size_t operator()(const std::vector<int> &v) const noexcept {
            size_t h = v.size();
            for (int x : v) {
                h ^= std::hash<int>{}(x) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            }
            return h;
        }
    };

    struct SmallAdj {
        int nbrs[2] = {-1, -1};
        uint8_t degree = 0;

        void add(int v) {
            RZ_ASSERT(degree < 2, "face is not a valid simple polygon (vertex degree > 2)");
            nbrs[degree++] = v;
        }
    };

    std::vector<int> vid_map(point_count, -1);
    std::vector<char> raw_vertex_used(point_count, 0);
    std::vector<char> is_outside_cvx(cvx_count, 0);

    size_t outside_face_refs = 0;
    size_t inside_face_refs = 0;
    int outside_cvx_count = 0;

    for (int cid = 0; cid < cvx_count; ++cid) {
        const bool outside = !lcc.cvx_is_inside[cid];
        is_outside_cvx[cid] = outside ? 1 : 0;
        const auto &cvx = lcc.cvxs[cid];
        if (outside) {
            ++outside_cvx_count;
            outside_face_refs += cvx.size();
            for (int fid : cvx) {
                RZ_ASSERT(fid >= 0 && fid < face_count, "face id out of range in cvx");
                for (int eid : lcc.faces[fid]) {
                    RZ_ASSERT(eid >= 0 && eid < (int)lcc.edges.size(), "edge id out of range in face");
                    const auto &e = lcc.edges[eid];
                    raw_vertex_used[e.v1] = 1;
                    raw_vertex_used[e.v2] = 1;
                }
            }
        } else {
            inside_face_refs += cvx.size();
        }
    }

    points.reserve(std::count(raw_vertex_used.begin(), raw_vertex_used.end(), char(1)));
    polygons.reserve(outside_face_refs);
    polyhedrons.reserve(outside_cvx_count);
    is_polygon_boundary.reserve(outside_face_refs);
    is_obj_boundary.reserve(outside_face_refs);

    for (int raw_vid = 0; raw_vid < point_count; ++raw_vid) {
        if (!raw_vertex_used[raw_vid]) continue;
        vid_map[raw_vid] = (int)points.size();
        const auto &p = lcc.points[raw_vid];
        points.emplace_back(CGAL::to_double(p.x()), CGAL::to_double(p.y()), CGAL::to_double(p.z()));
    }

    std::vector<std::vector<int>> raw_face_loops(face_count);
    #pragma omp parallel for schedule(dynamic)
    for (int fid = 0; fid < face_count; ++fid) {
        const auto &eids = lcc.faces[fid];
        RZ_ASSERT(eids.size() >= 3, "face has fewer than 3 edges");

        std::unordered_map<int, SmallAdj> adj;
        adj.reserve(eids.size() * 2);

        for (int eid : eids) {
            RZ_ASSERT(eid >= 0 && eid < (int)lcc.edges.size(), "edge id out of range");
            const auto &e = lcc.edges[eid];
            RZ_ASSERT(e.v1 >= 0 && e.v1 < point_count, "edge.v1 out of range");
            RZ_ASSERT(e.v2 >= 0 && e.v2 < point_count, "edge.v2 out of range");
            adj[e.v1].add(e.v2);
            adj[e.v2].add(e.v1);
        }

        for (const auto &kv : adj) {
            RZ_ASSERT(kv.second.degree == 2, "face is not a valid simple polygon (vertex degree != 2)");
        }

        auto &loop = raw_face_loops[fid];
        loop.reserve(adj.size());

        int start = eids.empty() ? -1 : lcc.edges[eids[0]].v1;
        int prev = -1;
        int cur = start;

        while (true) {
            loop.push_back(cur);
            const SmallAdj &nbrs = adj[cur];
            const int nxt = (nbrs.nbrs[0] == prev ? nbrs.nbrs[1] : nbrs.nbrs[0]);
            prev = cur;
            cur = nxt;
            if (cur == start) break;
            RZ_ASSERT((int)loop.size() <= (int)adj.size() + 1,
                      "loop reconstruction failed (likely non-manifold or broken face)");
        }

        RZ_ASSERT((int)loop.size() >= 3, "reconstructed face loop has fewer than 3 vertices");
    }

    std::vector<std::vector<int>> mapped_face_loops(face_count);
    std::vector<char> mapped_face_valid(face_count, 0);
    #pragma omp parallel for schedule(dynamic)
    for (int fid = 0; fid < face_count; ++fid) {
        const auto &raw_loop = raw_face_loops[fid];
        auto &mapped = mapped_face_loops[fid];
        mapped.reserve(raw_loop.size());

        bool all_mapped = true;
        for (int raw_vid : raw_loop) {
            RZ_ASSERT(raw_vid >= 0 && raw_vid < point_count, "raw vertex id out of range");
            const int new_vid = vid_map[raw_vid];
            if (new_vid < 0) all_mapped = false;
            mapped.push_back(new_vid);
        }
        mapped_face_valid[fid] = all_mapped ? 1 : 0;
    }

    std::vector<int> polyhedrons_end;
    polyhedrons_end.reserve(outside_cvx_count);
    std::vector<std::vector<std::vector<int>>> outside_cvx_polygons(cvx_count);
    #pragma omp parallel for schedule(dynamic)
    for (int cid = 0; cid < cvx_count; ++cid) {
        if (!is_outside_cvx[cid]) continue;
        const auto &cvx = lcc.cvxs[cid];
        auto &cvx_polys = outside_cvx_polygons[cid];
        cvx_polys.reserve(cvx.size());
        for (int fid : cvx) {
            RZ_ASSERT(fid >= 0 && fid < face_count, "face id out of range in cvx");
            const auto &face_vids = mapped_face_loops[fid];
            RZ_ASSERT(mapped_face_valid[fid] != 0, "outside face contains vertex with unassigned id");
            RZ_ASSERT((int)face_vids.size() >= 3, "degenerate face");
            cvx_polys.push_back(face_vids);
        }
    }
    for (int cid = 0; cid < cvx_count; ++cid) {
        if (!is_outside_cvx[cid]) continue;
        auto &cvx_polys = outside_cvx_polygons[cid];
        for (auto &poly : cvx_polys) polygons.push_back(std::move(poly));
        polyhedrons_end.push_back((int)polygons.size());
    }

    std::vector<std::vector<int>> obj_surface_polygons;
    obj_surface_polygons.reserve(inside_face_refs);
    std::vector<std::vector<std::vector<int>>> inside_obj_surface_polygons(cvx_count);
    #pragma omp parallel for schedule(dynamic)
    for (int cid = 0; cid < cvx_count; ++cid) {
        if (is_outside_cvx[cid]) continue;
        const auto &cvx = lcc.cvxs[cid];
        auto &cvx_polys = inside_obj_surface_polygons[cid];
        cvx_polys.reserve(cvx.size());
        for (int fid : cvx) {
            RZ_ASSERT(fid >= 0 && fid < face_count, "face id out of range in cvx");
            const auto &face_vids = mapped_face_loops[fid];
            RZ_ASSERT((int)face_vids.size() >= 3, "degenerate face");
            if (mapped_face_valid[fid] != 0) cvx_polys.push_back(face_vids);
        }
    }
    for (int cid = 0; cid < cvx_count; ++cid) {
        if (is_outside_cvx[cid]) continue;
        auto &cvx_polys = inside_obj_surface_polygons[cid];
        for (auto &poly : cvx_polys) obj_surface_polygons.push_back(std::move(poly));
    }

    {
        std::vector<std::vector<int>> unique_polygons;
        unique_polygons.reserve(polygons.size());

        std::vector<int> remap(polygons.size(), -1);
        std::vector<std::vector<int>> new_polyhedrons;
        new_polyhedrons.reserve(polyhedrons_end.size());

        std::unordered_set<std::vector<int>, VecHash> obj_surfaces;
        obj_surfaces.reserve(obj_surface_polygons.size() * 2 + 1);
        for (const auto &polygon : obj_surface_polygons) {
            obj_surfaces.insert(canonicalize_face(polygon));
        }

        std::unordered_map<std::vector<int>, int, VecHash> face_map;
        face_map.reserve(polygons.size() * 2 + 1);

        for (size_t i = 0; i < polygons.size(); ++i) {
            std::vector<int> key = canonicalize_face(polygons[i]);
            auto it = face_map.find(key);

            if (it == face_map.end()) {
                const int new_idx = (int)unique_polygons.size();
                face_map.emplace(std::move(key), new_idx);
                remap[i] = new_idx;
                unique_polygons.push_back(std::move(polygons[i]));
                is_polygon_boundary.push_back(true);
                is_obj_boundary.push_back(obj_surfaces.count(canonicalize_face(unique_polygons.back())) > 0);
            } else {
                remap[i] = it->second;
                is_polygon_boundary[it->second] = false;
            }
        }

        size_t start = 0;
        for (size_t ci = 0; ci < polyhedrons_end.size(); ++ci) {
            const size_t end = (size_t)polyhedrons_end[ci];
            std::vector<int> poly;
            poly.reserve(end - start);
            for (size_t fj = start; fj < end; ++fj) {
                RZ_ASSERT(remap[fj] >= 0, "remap face id invalid");
                poly.push_back(remap[fj]);
            }
            new_polyhedrons.push_back(std::move(poly));
            start = end;
        }

        polygons.swap(unique_polygons);
        polyhedrons.swap(new_polyhedrons);
    }

    {
        size_t edge_ref_count = 0;
        for (const auto &face : polygons) edge_ref_count += face.size();

        std::unordered_set<uint64_t> unique_edge_keys;
        unique_edge_keys.reserve(edge_ref_count * 2 + 1);
        edges.reserve(edge_ref_count / 2 + 1);

        for (const auto &face : polygons) {
            const int n = (int)face.size();
            RZ_ASSERT(n >= 3, "degenerate face");
            for (int i = 0; i < n; ++i) {
                const int v1 = face[i];
                const int v2 = face[(i + 1) % n];
                const uint64_t key = edge_key(v1, v2);
                if (!unique_edge_keys.insert(key).second) continue;

                const Edge e(v1, v2);
                edges.push_back(e);
            }
        }
    }

    is_vertex_boundary.assign(points.size(), false);
    for (int fid = 0; fid < (int)polygons.size(); ++fid) {
        if (!is_polygon_boundary[fid]) continue;

        for (int vid : polygons[fid]) {
            RZ_ASSERT(vid >= 0 && vid < (int)points.size(), "vertex id out of range");
            is_vertex_boundary[vid] = true;
        }
    }

    // =========================
    // 7) 一致性检查
    // =========================
    {
        for (const auto& cell : polyhedrons) {
            for (int fid : cell) {
                RZ_ASSERT(fid >= 0 && fid < (int)polygons.size(),
                          "invalid face id in polyhedron");
            }
        }

        for (int fid = 0; fid < (int)polygons.size(); ++fid) {
            const auto& poly = polygons[fid];
            RZ_ASSERT((int)poly.size() >= 3, "polygon has fewer than 3 vertices");

            for (int vid : poly) {
                RZ_ASSERT(vid >= 0 && vid < (int)points.size(),
                          "polygon vertex id out of range");
            }
        }

        for (int i = 0; i < (int)is_obj_boundary.size(); ++i) {
            RZ_ASSERT(i < (int)is_polygon_boundary.size(), "boundary flag size mismatch");
            if (is_obj_boundary[i]) {
                // 在你旧逻辑里，obj boundary 一定首先是 boundary polygon
                RZ_ASSERT(is_polygon_boundary[i], "obj boundary must also be polygon boundary");
            }
        }
    }
    {
        std::vector<int> owner_polyhedron(polygons.size(), -1);
        std::vector<_3d_vec> polyhedron_centers(polyhedrons.size(), _3d_vec(0, 0, 0));

        for (int pid = 0; pid < (int)polyhedrons.size(); ++pid) {
            _3d_vec cc(0, 0, 0);
            Real cnt = 0.0;

            for (int fid : polyhedrons[pid]) {
                RZ_ASSERT(fid >= 0 && fid < (int)polygons.size(), "face id out of range");
                if (is_obj_boundary[fid]) {
                    if (owner_polyhedron[fid] == -1) owner_polyhedron[fid] = pid;
                    else RZ_ASSERT(false, "obj boundary polygon belongs to more than one polyhedron");
                }

                for (int vid : polygons[fid]) {
                    cc = cc + points[vid];
                    cnt += 1.0;
                }
            }

            RZ_ASSERT(cnt > 0.0, "polyhedron centroid accumulation failed");
            polyhedron_centers[pid] = cc / cnt;
        }

        #pragma omp parallel for schedule(dynamic)
        for (int fid = 0; fid < (int)polygons.size(); ++fid) {
            if (!is_obj_boundary[fid]) continue;

            int pid = owner_polyhedron[fid];
            RZ_ASSERT(pid >= 0 && pid < (int)polyhedrons.size(),
                      "obj boundary polygon does not belong to any polyhedron");

            auto& poly = polygons[fid];
            RZ_ASSERT((int)poly.size() >= 3, "degenerate polygon");

            std::vector<K::Point_3> cgal_points;
            cgal_points.reserve(poly.size());
            for (int vid : poly) {
                const auto &p = points[vid];
                cgal_points.push_back({p.vx, p.vy, p.vz});
            }
            Real max_tri_area = 0.0;
            K::Vector_3 cgal_normal;
            for (int k = 2; k < (int)poly.size(); ++k) {
                K::Triangle_3 tri(cgal_points[0], cgal_points[1], cgal_points[k]);
                const Real area = tri.squared_area();
                if (area > max_tri_area) {
                    max_tri_area = area;
                    auto plane = tri.supporting_plane();
                    cgal_normal = plane.orthogonal_vector() / std::sqrt(plane.orthogonal_vector().squared_length());
                }
            }
            auto normal = _3d_vec(cgal_normal.x(),
                cgal_normal.y(),
                cgal_normal.z());

            const _3d_vec &cc = polyhedron_centers[pid];
            _3d_vec cctoface = points[poly[0]] - cc;
            if (normal.dot(cctoface) > 0.0) {
                std::reverse(poly.begin(), poly.end());
            }
        }
    }

    // for (auto&polyhedron:polyhedrons) {
    //     for (auto&pid:polyhedron) {
    //         for (auto&vid:polygons[pid])
    //             RZ_INFO(vid);
    //     }
    // }
    // exit(0);

    std::cout << "✅ load_from_Minilcc done.\n"
              << "   points      : " << points.size() << "\n"
              << "   polygons    : " << polygons.size() << "\n"
              << "   polyhedrons : " << polyhedrons.size() << "\n"
              << "   edges       : " << edges.size() << std::endl;
}

void VEM_Domain_3d::load_from_Minilcc_lod(MiniLcc &lcc, const std::vector<int> &lod) {
    FUNC_TIMER();
    // =========================
    // 0) clear / reset
    // =========================
    points.clear();
    polygons.clear();
    polyhedrons.clear();
    edges.clear();
    is_polygon_boundary.clear();
    is_obj_boundary.clear();
    is_vertex_boundary.clear();
    vertex_normals.clear();
    vertex_neumann_vels.clear();
    face_neumann_vels.clear();

    std::unordered_set<int> lod_set;
    lod_set.reserve(lod.size() * 2);
    for (int cid : lod) {
        RZ_ASSERT(cid >= 0 && cid < (int)lcc.cvxs.size(), "lod cid out of range");
        lod_set.insert(cid);
    }

    // =========================================================
    // 局部工具 1：
    // 从 MiniLcc 的一个 face(edge loop) 恢复“原始顶点环”
    // =========================================================
    auto ordered_raw_vertices_of_face = [&](int fid) -> std::vector<int> {
        RZ_ASSERT(fid >= 0 && fid < (int)lcc.faces.size(), "face id out of range");

        const auto& eids = lcc.faces[fid];
        RZ_ASSERT(eids.size() >= 3, "face has fewer than 3 edges");

        std::map<int, std::vector<int>> adj;

        for (int eid : eids) {
            RZ_ASSERT(eid >= 0 && eid < (int)lcc.edges.size(), "edge id out of range");

            const auto& e = lcc.edges[eid];
            RZ_ASSERT(e.v1 >= 0 && e.v1 < (int)lcc.points.size(), "edge.v1 out of range");
            RZ_ASSERT(e.v2 >= 0 && e.v2 < (int)lcc.points.size(), "edge.v2 out of range");

            adj[e.v1].push_back(e.v2);
            adj[e.v2].push_back(e.v1);
        }

        for (const auto& kv : adj) {
            RZ_ASSERT(kv.second.size() == 2,
                      "face is not a valid simple polygon (vertex degree != 2)");
        }

        std::vector<int> loop;
        loop.reserve(adj.size());

        int start = adj.begin()->first;
        int prev  = -1;
        int cur   = start;

        while (true) {
            loop.push_back(cur);

            const auto& nbrs = adj[cur];
            int nxt = (nbrs[0] == prev ? nbrs[1] : nbrs[0]);

            prev = cur;
            cur  = nxt;

            if (cur == start) break;

            RZ_ASSERT((int)loop.size() <= (int)adj.size() + 1,
                      "loop reconstruction failed (likely non-manifold or broken face)");
        }

        RZ_ASSERT((int)loop.size() >= 3, "reconstructed face loop has fewer than 3 vertices");
        return loop;
    };

    // =========================================================
    // 局部工具 2：
    // 把 raw vertex loop 映射到“新 domain vertex id”
    // =========================================================
    auto mapped_vertices_of_face = [&](int fid,
                                       const std::vector<int>& vid_map,
                                       bool& valid) -> std::vector<int> {
        auto raw_loop = ordered_raw_vertices_of_face(fid);

        std::vector<int> mapped;
        mapped.reserve(raw_loop.size());

        valid = true;
        for (int raw_vid : raw_loop) {
            RZ_ASSERT(raw_vid >= 0 && raw_vid < (int)vid_map.size(), "raw vertex id out of range");
            int new_vid = vid_map[raw_vid];
            if (new_vid < 0) valid = false;
            mapped.push_back(new_vid);
        }

        return mapped;
    };

    // =========================================================
    // 局部工具 3：
    // canonical key：忽略顺序 / 方向，只按顶点集合去重
    // =========================================================
    auto canonical_key = [](std::vector<int> face) -> std::string {
        if (face.empty()) return std::string();
        std::sort(face.begin(), face.end());

        std::ostringstream oss;
        for (size_t i = 0; i < face.size(); ++i) {
            oss << face[i];
            if (i + 1 < face.size()) oss << ",";
        }
        return oss.str();
    };

    // =========================
    // 1) 只给 lod 中、且非 inside 的 cvx incident 顶点分配唯一 id
    // =========================
    std::vector<int> vid_map(lcc.points.size(), -1);

    for (int cid : lod) {
        if (lcc.cvx_is_inside[cid]) continue;

        const auto& cvx = lcc.cvxs[cid];
        for (int fid : cvx) {
            RZ_ASSERT(fid >= 0 && fid < (int)lcc.faces.size(), "face id out of range in cvx");

            for (int eid : lcc.faces[fid]) {
                RZ_ASSERT(eid >= 0 && eid < (int)lcc.edges.size(), "edge id out of range in face");

                const auto& e = lcc.edges[eid];

                if (vid_map[e.v1] == -1) {
                    vid_map[e.v1] = (int)points.size();
                    const auto& p = lcc.points[e.v1];
                    points.emplace_back(
                        CGAL::to_double(p.x()),
                        CGAL::to_double(p.y()),
                        CGAL::to_double(p.z())
                    );
                }

                if (vid_map[e.v2] == -1) {
                    vid_map[e.v2] = (int)points.size();
                    const auto& p = lcc.points[e.v2];
                    points.emplace_back(
                        CGAL::to_double(p.x()),
                        CGAL::to_double(p.y()),
                        CGAL::to_double(p.z())
                    );
                }
            }
        }
    }

    // =========================
    // 2) 只遍历 lod 中、且非 inside 的 cvx
    // =========================
    std::vector<int> polyhedrons_end;

    for (int cid : lod) {
        if (lcc.cvx_is_inside[cid]) continue;

        const auto& cvx = lcc.cvxs[cid];
        for (int fid : cvx) {
            bool valid = false;
            auto face_vids = mapped_vertices_of_face(fid, vid_map, valid);

            RZ_ASSERT(valid, "outside face contains vertex with unassigned id");
            RZ_ASSERT((int)face_vids.size() >= 3, "degenerate face");

            polygons.push_back(std::move(face_vids));
        }

        polyhedrons_end.push_back((int)polygons.size());
    }

    // =========================
    // 3) 只遍历 lod 中、且 inside 的 cvx，提取 obj_surface_polygons
    // =========================
    std::vector<std::vector<int>> obj_surface_polygons;

    for (int cid : lod) {
        if (!lcc.cvx_is_inside[cid]) continue;

        const auto& cvx = lcc.cvxs[cid];
        for (int fid : cvx) {
            bool valid = false;
            auto face_vids = mapped_vertices_of_face(fid, vid_map, valid);

            RZ_ASSERT((int)face_vids.size() >= 3, "degenerate face");

            if (valid) {
                obj_surface_polygons.push_back(std::move(face_vids));
            }
        }
    }

    // =========================
    // 4) 面去重 + 构建 polyhedrons
    // =========================
    {
        std::vector<std::vector<int>> unique_polygons;
        unique_polygons.reserve(polygons.size());

        std::vector<int> remap(polygons.size(), -1);
        std::vector<std::vector<int>> new_polyhedrons;

        std::unordered_map<std::string, int> face_map;
        face_map.reserve(polygons.size() * 2);

        std::set<std::string> obj_surfaces;
        for (auto& polygon : obj_surface_polygons) {
            obj_surfaces.insert(canonical_key(polygon));
        }

        for (size_t i = 0; i < polygons.size(); ++i) {
            std::string key = canonical_key(polygons[i]);
            auto it = face_map.find(key);

            if (it == face_map.end()) {
                int new_idx = (int)unique_polygons.size();
                face_map.emplace(key, new_idx);
                remap[i] = new_idx;

                unique_polygons.push_back(polygons[i]);

                is_polygon_boundary.push_back(true);
            } else {
                remap[i] = it->second;
                is_polygon_boundary[it->second] = false;
            }
        }


        size_t start = 0;
        for (size_t ci = 0; ci < polyhedrons_end.size(); ++ci) {
            size_t end = (size_t)polyhedrons_end[ci];

            std::vector<int> poly;
            poly.reserve(end - start);

            for (size_t fj = start; fj < end; ++fj) {
                RZ_ASSERT(remap[fj] >= 0, "remap face id invalid");
                poly.push_back(remap[fj]);
            }

            new_polyhedrons.push_back(std::move(poly));
            start = end;
        }

        polygons.swap(unique_polygons);
        polyhedrons.swap(new_polyhedrons);
    }
    for (int pid=0;pid<is_polygon_boundary.size();++pid) {
        if (is_polygon_boundary[pid]) {
            _3d_vec centroid(0,0,0);
            for (auto&vid:polygons[pid])centroid = centroid+points[vid];
            centroid = centroid/polygons[pid].size();
            if (centroid.vx>1e-5
                &&centroid.vy>1e-5
                &&centroid.vz>1e-5
                &&centroid.vx<1-1e-5
                &&centroid.vy<1-1e-5
                &&centroid.vz<1-1e-5)
                is_obj_boundary.push_back(true);
            else
                is_obj_boundary.push_back(false);

        }else {
            is_obj_boundary.push_back(false);
        }
    }
    // =========================
    // 5) 用最终 polygons 重建唯一边
    // =========================
    {
        std::set<Edge> temp_unique_edges;

        for (const auto& face : polygons) {
            const int n = (int)face.size();
            RZ_ASSERT(n >= 3, "degenerate face");

            for (int i = 0; i < n; ++i) {
                int v1 = face[i];
                int v2 = face[(i + 1) % n];
                temp_unique_edges.insert(Edge(v1, v2));
            }
        }

        edges.reserve(temp_unique_edges.size());
        for (const auto& e : temp_unique_edges) {
            edges.push_back(e);
        }
    }

    // =========================
    // 6) is_vertex_boundary
    // =========================
    is_vertex_boundary.assign(points.size(), false);
    for (int fid = 0; fid < (int)polygons.size(); ++fid) {
        if (!is_polygon_boundary[fid]) continue;

        for (int vid : polygons[fid]) {
            RZ_ASSERT(vid >= 0 && vid < (int)points.size(), "vertex id out of range");
            is_vertex_boundary[vid] = true;
        }
    }

    // =========================
    // 7) 一致性检查 + obj boundary 朝向
    // =========================
    {
        for (const auto& cell : polyhedrons) {
            for (int fid : cell) {
                RZ_ASSERT(fid >= 0 && fid < (int)polygons.size(),
                          "invalid face id in polyhedron");
            }
        }

        for (int fid = 0; fid < (int)polygons.size(); ++fid) {
            const auto& poly = polygons[fid];
            RZ_ASSERT((int)poly.size() >= 3, "polygon has fewer than 3 vertices");

            for (int vid : poly) {
                RZ_ASSERT(vid >= 0 && vid < (int)points.size(),
                          "polygon vertex id out of range");
            }
        }

        for (int i = 0; i < (int)is_obj_boundary.size(); ++i) {
            RZ_ASSERT(i < (int)is_polygon_boundary.size(), "boundary flag size mismatch");
            if (is_obj_boundary[i]) {
                RZ_ASSERT(is_polygon_boundary[i], "obj boundary must also be polygon boundary");
            }
        }
    }

    {
        std::vector<int> owner_polyhedron(polygons.size(), -1);

        for (int pid = 0; pid < (int)polyhedrons.size(); ++pid) {
            for (int fid : polyhedrons[pid]) {
                RZ_ASSERT(fid >= 0 && fid < (int)polygons.size(), "face id out of range");
                if (!is_polygon_boundary[fid]) continue;

                if (owner_polyhedron[fid] == -1) {
                    owner_polyhedron[fid] = pid;
                } else {
                    RZ_ASSERT(false, "obj boundary polygon belongs to more than one polyhedron");
                }
            }
        }

        for (int fid = 0; fid < (int)polygons.size(); ++fid) {
            if (!is_polygon_boundary[fid]) continue;

            int pid = owner_polyhedron[fid];
            RZ_ASSERT(pid >= 0 && pid < (int)polyhedrons.size(),
                      "obj boundary polygon does not belong to any polyhedron");

            auto& poly = polygons[fid];
            RZ_ASSERT((int)poly.size() >= 3, "degenerate polygon");

            std::vector<K::Point_3> cgal_points;
            cgal_points.reserve(poly.size());
            for (int vid : poly) {
                const auto& p = points[vid];
                cgal_points.push_back({p.vx, p.vy, p.vz});
            }

            Real max_tri_area = 0.0;
            K::Vector_3 cgal_normal;
            for (int k = 2; k < (int)poly.size(); ++k) {
                K::Triangle_3 tri(cgal_points[0], cgal_points[1], cgal_points[k]);
                Real area = tri.squared_area();
                if (area > max_tri_area) {
                    max_tri_area = area;
                    auto plane = tri.supporting_plane();
                    cgal_normal = plane.orthogonal_vector() /
                                  std::sqrt(plane.orthogonal_vector().squared_length());
                }
            }

            auto normal = _3d_vec(cgal_normal.x(), cgal_normal.y(), cgal_normal.z());

            _3d_vec cc(0, 0, 0);
            Real cnt = 0;
            for (int face : polyhedrons[pid]) {
                for (int p : polygons[face]) {
                    cc = cc + points[p];
                    cnt += 1.0;
                }
            }
            cc = cc / cnt;

            _3d_vec cctoface = points[poly[0]] - cc;
            if (normal.dot(cctoface) > 0.0) {
                std::reverse(poly.begin(), poly.end());
            }
        }
    }


}

void VEM_Domain_3d::init() {
    int Ndofs = points.size() + edges.size() + polygons.size() + polyhedrons.size();
    int Ndofs_k1 = points.size();

    global_stiffness_matrix_k2.resize(Ndofs, Ndofs);
    global_stiffness_matrix_k2.setZero();
    global_stiffness_matrix_k1.resize(Ndofs_k1,Ndofs_k1);
    global_stiffness_matrix_k1.setZero();

    global_LX_matrix_k1.resize(Ndofs_k1, Ndofs_k1);
    global_LY_matrix_k1.resize(Ndofs_k1, Ndofs_k1);
    global_LZ_matrix_k1.resize(Ndofs_k1, Ndofs_k1);
    global_LX_matrix_k1.setZero();
    global_LY_matrix_k1.setZero();
    global_LZ_matrix_k1.setZero();
    global_LX_matrix_k2.resize(Ndofs, Ndofs_k1);
    global_LY_matrix_k2.resize(Ndofs, Ndofs_k1);
    global_LZ_matrix_k2.resize(Ndofs, Ndofs_k1);
    global_LX_matrix_k2.setZero();
    global_LY_matrix_k2.setZero();
    global_LZ_matrix_k2.setZero();

    global_rhs_k1.resize(Ndofs_k1);
    global_rhs_k1.setZero();
    global_rhs_k2.resize(Ndofs);
    global_rhs_k2.setZero();

    k1_dof_u.resize(Ndofs_k1);
    k1_dof_u.setZero();
    k1_dof_v.resize(Ndofs_k1);
    k1_dof_v.setZero();
    k1_dof_w.resize(Ndofs_k1);
    k1_dof_w.setZero();
    k1_dof_u_pmg.resize(Ndofs_k1);
    k1_dof_u_pmg.setZero();
    k1_dof_v_pmg.resize(Ndofs_k1);
    k1_dof_v_pmg.setZero();
    k1_dof_w_pmg.resize(Ndofs_k1);
    k1_dof_w_pmg.setZero();



    k1_dof_du.resize(Ndofs_k1);
    k1_dof_du.setZero();
    k1_dof_dv.resize(Ndofs_k1);
    k1_dof_dv.setZero();
    k1_dof_dw.resize(Ndofs_k1);
    k1_dof_dw.setZero();


    k1_dof_density.resize(Ndofs_k1);
    k1_dof_density.setZero();
    k1_dof_temp.resize(Ndofs_k1);
    k1_dof_temp.setZero();

    global_p_k2.resize(Ndofs);
    global_p_k2.setZero();

    global_p_k1.resize(Ndofs_k1);
    global_p_k1.setZero();
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

    for (auto&elem:vem_elements) {
        elem.poly_coeffs_u_k1.resize(4);
        elem.poly_coeffs_u_k1.setZero();
        elem.poly_coeffs_v_k1.resize(4);
        elem.poly_coeffs_v_k1.setZero();
        elem.poly_coeffs_w_k1.resize(4);
        elem.poly_coeffs_w_k1.setZero();

        elem.poly_coeffs_du_k1.resize(4);
        elem.poly_coeffs_du_k1.setZero();
        elem.poly_coeffs_dv_k1.resize(4);
        elem.poly_coeffs_dv_k1.setZero();
        elem.poly_coeffs_dw_k1.resize(4);
        elem.poly_coeffs_dw_k1.setZero();

        elem.poly_coeffs_oldu_k1.resize(4);
        elem.poly_coeffs_oldu_k1.setZero();
        elem.poly_coeffs_oldv_k1.resize(4);
        elem.poly_coeffs_oldv_k1.setZero();
        elem.poly_coeffs_oldw_k1.resize(4);
        elem.poly_coeffs_oldw_k1.setZero();
    }

}

void VEM_Domain_3d::preprocess_polygons() {
    FUNC_TIMER();
    FUNC_MEMORY_TIMER();
    typedef CGAL::Polygon_2<K> Polygon_2;
    polygons_plane.clear();
    polygons_plane.resize(polygons.size());
    for (int pid=0;pid<polygons.size();++pid) {
        std::vector<_3d_vec>local_points;
        auto& poly_plane = polygons_plane[pid];
        auto&polygon = polygons[pid];
        std::vector<K::Point_3>cgal_points;
        Real _3d_max_diameter = 0.0;
        for (int i=0;i<polygon.size();++i) {
            for (int j=i+1;j<polygon.size();++j) {
                _3d_max_diameter = std::max(_3d_max_diameter,
                    (points[polygon[i]]-points[polygon[j]]).norm());
            }
        }
        for (auto&vid:polygon) {
            const auto&p = points[vid];
            local_points.push_back(p);
            cgal_points.push_back({p.vx,p.vy,p.vz});
        }
        CGAL_Point_3 centroid_3 = CGAL::centroid(cgal_points.begin(),cgal_points.end());
        poly_plane._3d_centroid = _3d_vec(centroid_3.x(),centroid_3.y(),centroid_3.z());



        Real max_tri_area = 0.0;
        for (int k=2;k<cgal_points.size();++k) {
            K::Triangle_3 tri(cgal_points[0],cgal_points[1],cgal_points[k]);
            Real area = tri.squared_area();
            if (area>max_tri_area) {
                max_tri_area = area;
                poly_plane.cgal_plane = tri.supporting_plane();
            }
        }

        Real sum_area = 0.0;
        for (int k=0;k<cgal_points.size();++k) {
            K::Triangle_3 tri(cgal_points[k],cgal_points[(k+1)%cgal_points.size()],centroid_3);

            sum_area += std::sqrt(tri.squared_area());
        }

        poly_plane.origin = poly_plane._3d_centroid;
        K::Vector_3 cgal_unit_normal = poly_plane.cgal_plane.orthogonal_vector()/std::sqrt(poly_plane.cgal_plane.orthogonal_vector().squared_length());
        poly_plane.normal = _3d_vec(cgal_unit_normal.x(),
            cgal_unit_normal.y(),
            cgal_unit_normal.z());

        Polygon_2 cgal_polygon;
        // //构造局部坐标系
        _3d_vec tmp = (std::fabs(poly_plane.normal.vx) < 0.9) ? _3d_vec(1,0,0) : _3d_vec(0,1,0);
        if (std::fabs(poly_plane.normal.vx) > std::fabs(poly_plane.normal.vy))
            tmp = _3d_vec(0, 1, 0);
        else
            tmp = _3d_vec(1, 0, 0);

        poly_plane.u_axis = (poly_plane.normal.cross(tmp)).normalized();
        poly_plane.v_axis = (poly_plane.normal.cross(poly_plane.u_axis)).normalized();
        // 投影所有点
        for (auto &p : local_points) {
            _3d_vec rel = p - poly_plane.origin;
            Real u = rel.vx * poly_plane.u_axis.vx + rel.vy * poly_plane.u_axis.vy + rel.vz * poly_plane.u_axis.vz;
            Real v = rel.vx * poly_plane.v_axis.vx + rel.vy * poly_plane.v_axis.vy + rel.vz * poly_plane.v_axis.vz;

            cgal_polygon.push_back({u, v});
        }
        poly_plane.area = std::fabs(cgal_polygon.area());

        // RZ_ASSERT(std::fabs(poly_plane.area- sum_area)<1e-9,poly_plane.area," ",sum_area);
        // compute diameter
        Real max_diameter = 0.0;
        for (int i=0;i<polygon.size();++i) {
            for (int j=i+1;j<polygon.size();++j) {
                max_diameter = std::fmax(max_diameter,(local_points[i]-local_points[j]).norm());
            }
        }
        poly_plane.diameter = max_diameter;
        // RZ_ASSERT(std::fabs(poly_plane.diameter- _3d_max_diameter)<1e-9,poly_plane.diameter," ",_3d_max_diameter);
    }
    
}




void VEM_Domain_3d::ensure_polygon_normals_outward(VEM_Element_3d &element) {
    _3d_vec C_ref(0,0,0);
    for (int vid = 0; vid < element.vertex_count(); ++vid) C_ref = C_ref + element.vertex_position(vid);
    C_ref = C_ref / element.vertex_count();
    const _3d_vec &centroid = C_ref;

    for (int local_pid=0;local_pid<element.polygons.size();++local_pid) {
        if (this->polygons_plane[element.polygon_local_to_global_dof[local_pid]].is_degenerated)
            continue;
        auto&polygon = element.polygons[local_pid];
        RZ_ASSERT(polygon.size()>=3,"");
        _3d_vec v0, v1, v2;
        Real max_cross_product_norm = -1.0;
        size_t best_i = 0, best_j = 0, best_k = 0;
        for (size_t i = 0; i < polygon.size() - 2; ++i) {
            for (size_t j = i + 1; j < polygon.size() - 1; ++j) {
                for (size_t k = j + 1; k < polygon.size(); ++k) {
                    v0 = element.vertex_position(polygon[i]);
                    v1 = element.vertex_position(polygon[j]);
                    v2 = element.vertex_position(polygon[k]);
                    // 计算叉积
                    _3d_vec edge1 = v1 - v0;
                    _3d_vec edge2 = v2 - v0;
                    _3d_vec cross_product = edge1.cross(edge2);
                    // 计算叉积的模
                    Real cross_product_norm = cross_product.norm();
                    // 如果叉积模更大，更新最佳组合
                    if (cross_product_norm > max_cross_product_norm) {
                        max_cross_product_norm = cross_product_norm;
                        best_i = i;
                        best_j = j;
                        best_k = k;
                    }
                }
            }
        }
        v0 = element.vertex_position(polygon[best_i]);
        v1 = element.vertex_position(polygon[best_j]);
        v2 = element.vertex_position(polygon[best_k]);
        if (max_cross_product_norm<1e-16) {
            for (auto&pid:polygon)std::cout<<pid<<" ";
            element.save_face_to_obj(polygon);
            element.save_polygonmesh_to_obj("collinear_face.obj");
            RZ_FATAL("");
        }
        // 面法向量（右手系）
        _3d_vec normal = (v1 - v0).cross(v2 - v0);
        // 面上任一点到质心的向量
        _3d_vec to_centroid = centroid - v0;
        // 如果法向指向质心（朝内），则反转该面顶点顺序
        if (normal.dot(to_centroid) > 0) {
            std::reverse(polygon.begin(), polygon.end());
        }
    }
}

void VEM_Domain_3d::orient_mesh_and_compute_volume(VEM_Element_3d &element) {
    RZ_ASSERT(element.vertex_count() > 0, "");

    _3d_vec reference_center(0, 0, 0);
    for (int vid = 0; vid < element.vertex_count(); ++vid) {
        reference_center = reference_center + element.vertex_position(vid);
    }
    reference_center = reference_center / element.vertex_count();

    auto polygon_normal_and_centroid = [&](const std::vector<int> &polygon,
                                           _3d_vec &normal,
                                           _3d_vec &face_centroid) {
        normal = _3d_vec(0, 0, 0);
        face_centroid = _3d_vec(0, 0, 0);
        const int n = static_cast<int>(polygon.size());
        RZ_ASSERT(n >= 3, "");

        for (int i = 0; i < n; ++i) {
            const _3d_vec &a = element.vertex_position(polygon[i]);
            const _3d_vec &b = element.vertex_position(polygon[(i + 1) % n]);
            normal.vx += (a.vy - b.vy) * (a.vz + b.vz);
            normal.vy += (a.vz - b.vz) * (a.vx + b.vx);
            normal.vz += (a.vx - b.vx) * (a.vy + b.vy);
            face_centroid = face_centroid + a;
        }
        face_centroid = face_centroid / n;

        if (normal.squared_norm() < 1e-24) {
            for (int i = 1; i + 1 < n; ++i) {
                const _3d_vec &v0 = element.vertex_position(polygon[0]);
                const _3d_vec &v1 = element.vertex_position(polygon[i]);
                const _3d_vec &v2 = element.vertex_position(polygon[i + 1]);
                normal = normal + (v1 - v0).cross(v2 - v0);
            }
        }
    };

    for (int local_pid = 0; local_pid < static_cast<int>(element.polygons.size()); ++local_pid) {
        if (this->polygons_plane[element.polygon_local_to_global_dof[local_pid]].is_degenerated) {
            continue;
        }
        auto &polygon = element.polygons[local_pid];
        _3d_vec normal, face_centroid;
        polygon_normal_and_centroid(polygon, normal, face_centroid);
        if (normal.squared_norm() < 1e-24) {
            element.save_face_to_obj(polygon);
            element.save_polygonmesh_to_obj("degenerate_face.obj");
            RZ_FATAL("degenerate polygon normal");
        }
        if (normal.dot(face_centroid - reference_center) < 0.0) {
            std::reverse(polygon.begin(), polygon.end());
        }
    }

    Real total_volume = 0.0;
    _3d_vec centroid_sum(0, 0, 0);

    for (int local_pid = 0; local_pid < static_cast<int>(element.polygons.size()); ++local_pid) {
        if (this->polygons_plane[element.polygon_local_to_global_dof[local_pid]].is_degenerated) {
            continue;
        }
        const auto &polygon = element.polygons[local_pid];
        if (polygon.size() < 3) continue;

        const _3d_vec &v0 = element.vertex_position(polygon[0]);
        const _3d_vec v0_ref = v0 - reference_center;
        for (size_t i = 1; i + 1 < polygon.size(); ++i) {
            const _3d_vec &v1 = element.vertex_position(polygon[i]);
            const _3d_vec &v2 = element.vertex_position(polygon[i + 1]);
            const Real v6 = v0_ref.dot((v1 - reference_center).cross(v2 - reference_center));
            const Real vol = v6 / 6.0;
            total_volume += vol;
            centroid_sum = centroid_sum + ((v0 + v1 + v2 + reference_center) / 4.0) * vol;
        }
    }

    if (total_volume < 0.0) {
        for (auto &polygon : element.polygons) {
            std::reverse(polygon.begin(), polygon.end());
        }
        total_volume = -total_volume;
        centroid_sum = centroid_sum * (-1.0);
    }

    RZ_ASSERT(total_volume > 0.0, "non-positive mesh volume");

    Real max_diameter = 0.0;
    for (int i = 0; i < element.vertex_count(); ++i) {
        for (int j = i + 1; j < element.vertex_count(); ++j) {
            const auto& pi = element.vertex_position(i);
            const auto& pj = element.vertex_position(j);
            const Real dx = pi.vx - pj.vx;
            const Real dy = pi.vy - pj.vy;
            const Real dz = pi.vz - pj.vz;
            max_diameter = std::max(max_diameter, std::sqrt(dx * dx + dy * dy + dz * dz));
        }
    }

    element.volume = total_volume;
    element.centroid = centroid_sum * (1.0 / total_volume);
    element.diameter = max_diameter;
    element.is_convex = false;
}

void VEM_Domain_3d::cgal_ccw_orientation(VEM_Element_3d &element) {
    orient_mesh_and_compute_volume(element);
}

void VEM_Domain_3d::cgal_centroid_volume(VEM_Element_3d &element) {
    // 先粗修正
    _3d_vec C_ref(0,0,0);
    for (int vid = 0; vid < element.vertex_count(); ++vid) C_ref = C_ref + element.vertex_position(vid);
    C_ref = C_ref / element.vertex_count();
    Real total_volume = 0.0;
    _3d_vec centroid_sum(0, 0, 0);
    // === 遍历每个面 ===
    int local_pid=0;
    for (const auto &polygon : element.polygons) {
        if (this->polygons_plane[element.polygon_local_to_global_dof[local_pid++]].is_degenerated)
            continue;
        if (polygon.size() < 3) continue;
        const _3d_vec &v0 = element.vertex_position(polygon[0]);
        for (size_t i = 1; i + 1 < polygon.size(); ++i) {
            const _3d_vec &v1 = element.vertex_position(polygon[i]);
            const _3d_vec &v2 = element.vertex_position(polygon[i + 1]);
            // 使用 C_ref 作为基点构建四面体体积
            _3d_vec v0_ref = v0 - C_ref;
            _3d_vec v1_ref = v1 - C_ref;
            _3d_vec v2_ref = v2 - C_ref;
            Real v6 = v0_ref.dot(v1_ref.cross(v2_ref));
            Real vol = v6 / 6.0;
            // 体积加权质心
            _3d_vec c = (v0 + v1 + v2 + C_ref) / 4.0;
            centroid_sum = centroid_sum + c * vol;
            total_volume += vol;
        }
    }

    // === 计算体积与质心 ===
    RZ_ASSERT(total_volume>0.0,"");
    Real abs_volume = std::fabs(total_volume);
    _3d_vec centroid = centroid_sum * (1.0 / total_volume);

    // === 直径计算 ===
    Real max_diameter = 0.0;
    for (int i = 0; i < element.vertex_count(); ++i) {
        for (int j = i + 1; j < element.vertex_count(); ++j) {
            const auto& pi = element.vertex_position(i);
            const auto& pj = element.vertex_position(j);
            Real dx = pi.vx - pj.vx;
            Real dy = pi.vy - pj.vy;
            Real dz = pi.vz - pj.vz;
            Real dist = std::sqrt(dx * dx + dy * dy + dz * dz);
            max_diameter = std::max(max_diameter, dist);
        }
    }
    // === 存储到 elem ===
    element.volume = abs_volume;
    element.centroid = centroid;
    element.diameter = max_diameter;
}



size_t getCurrentRSS() {
    std::ifstream status_file("/proc/self/status");
    std::string line;
    while (std::getline(status_file, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            std::istringstream iss(line);
            std::string key, value, unit;
            iss >> key >> value >> unit;
            return std::stoul(value); // KB
        }
    }
    return 0;
}
template <class DenseMatrix>
inline void prune_relative_small_entries(DenseMatrix& A, Real rel_eps)
{
    // ---------- 1. 尺寸检查 ----------
    if (A.size() == 0) return;

    // ---------- 2. 计算 max_abs（显式遍历，避免 Eigen corner case） ----------
    Real max_abs = 0.0;
    using Scalar = typename DenseMatrix::Scalar;
    const Scalar* ptr = A.data();
    const Eigen::Index n = A.size();

    for (Eigen::Index i = 0; i < n; ++i) {
        const Real v = std::abs(ptr[i]);
        if (std::isfinite(v))
            max_abs = std::max(max_abs, v);
    }

    if (max_abs == 0.0 || !std::isfinite(max_abs))
        return;

    const Real thresh = rel_eps * max_abs;

    // ---------- 3. 就地 prune（无 expression template） ----------
    for (Eigen::Index i = 0; i < n; ++i) {
        Scalar& v = A.data()[i];
        if (!std::isfinite(v) || std::abs(v) < thresh)
            v = 0.0;
    }
}

static std::vector<Edge> sorted_polyhedron_edges_by_midpoint(const VEM_Element_3d& element) {
    std::set<Edge> unique_edges;
    for (int pid = 0; pid < (int)element.polygons.size(); ++pid) {
        const int n = (int)element.polygons[pid].size();
        for (int ii = 0; ii < n; ++ii) {
            const int a = element.polygons[pid][ii];
            const int b = element.polygons[pid][(ii + 1) % n];
            unique_edges.insert(Edge(a, b));
        }
    }

    std::vector<Edge> sorted_edges(unique_edges.begin(), unique_edges.end());
    std::sort(sorted_edges.begin(), sorted_edges.end(), [&](const Edge& a, const Edge& b) {
        const _3d_vec amid = (element.vertex_position(a.v1) + element.vertex_position(a.v2)) * Real(0.5);
        const _3d_vec bmid = (element.vertex_position(b.v1) + element.vertex_position(b.v2)) * Real(0.5);
        if (_3d_vec::sort(amid, bmid)) return true;
        if (_3d_vec::sort(bmid, amid)) return false;
        if (a.v1 != b.v1) return a.v1 < b.v1;
        return a.v2 < b.v2;
    });
    return sorted_edges;
}

static std::map<Edge, int> build_edge_to_id_cache(const std::vector<Edge>& edges) {
    std::map<Edge, int> ans;
    for (int eid = 0; eid < (int)edges.size(); ++eid) {
        ans[edges[eid]] = eid;
    }
    return ans;
}

static int global_edge_id_from_local_edge(const std::map<Edge, int>& global_edge_to_id,
                                          const VEM_Element_3d& element,
                                          const Edge& local_edge) {
    const int gv1 = element.vertex_local_to_global_dof[local_edge.v1];
    const int gv2 = element.vertex_local_to_global_dof[local_edge.v2];
    auto it = global_edge_to_id.find(Edge(gv1, gv2));
    RZ_ASSERT(it != global_edge_to_id.end(), gv1, " ", gv2);
    return it->second;
}

// 这里的一些设定导致我们目前只能处理凸的 要想处理非凸的 我们需要更好的知道面的方向然后直接拿过来 而不是在这里计算
void VEM_Domain_3d::generate_elements(bool generate_2d) {
    FUNC_TIMER();
    FUNC_MEMORY_TIMER();
    vem_elements.clear();
    vem_elements.resize(polyhedrons.size());

    const auto global_edge_to_id = build_edge_to_id_cache(edges);

    //先找到一个 is_U的 串行
    for (int i = 0; i < (int)polyhedrons.size(); ++i) {
        if (polyhedrons[i].size()!=6)continue;
        Real max_area=0,min_area=1;
        Real max_dia = 0,min_dia = 1;
        for (auto&polyid:polyhedrons[i]) {
            max_area = std::max(max_area,polygons_plane[polyid].area);
            min_area = std::min(min_area,polygons_plane[polyid].area);
            max_dia = std::max(max_dia,polygons_plane[polyid].diameter);
            min_dia = std::min(min_dia,polygons_plane[polyid].diameter);
        }
        if ((max_area - min_area)/min_area < 1e-2 &&
            (max_dia - min_dia)/min_dia<1e-2 && min_area>0.000025) {
            has_builded_u = true;
            U_area = min_area;
            const auto& polyhedron_polygons = polyhedrons[i];
            VEM_Element_3d element;
            element.bind_geometry(points, edges);
            for (auto&polygon : polyhedron_polygons) {
                if (is_polygon_boundary[polygon])
                    element.has_boundary=true;
            }
            element.polygon_local_to_global_dof = polyhedron_polygons;

            std::set<int> unique_vertices;
            for (auto&polygon_id : polyhedron_polygons) {
                const auto& polygon = polygons[polygon_id];
                for (auto&vertex: polygon) {
                    unique_vertices.insert(vertex);
                }
            }
            std::map<int,int> global_to_local_vertex_map;
            for (auto&vertex : unique_vertices) {
                global_to_local_vertex_map[vertex] = element.vertex_count();
                                element.vertex_local_to_global_dof.push_back(vertex);
            }

            {// sort points inside a 3d volume
                // ======================================================
                // 对 element.points 做字典序排序 (x, y, z)
                // 并同步更新 vertex_local_to_global_dof 与 map
                // ======================================================

                const int n = static_cast<int>(element.vertex_count());

                // 1. 构造排序索引
                std::vector<int> order(n);
                std::iota(order.begin(), order.end(), 0);

                // 2. 按 points 的几何位置排序索引
                std::sort(order.begin(), order.end(),
                    [&](int a, int b) {
                        return _3d_vec::sort(element.vertex_position(a), element.vertex_position(b));
                    });

                // 3. 重排 dof
                std::vector<int> new_vertex_local_to_global_dof;
                new_vertex_local_to_global_dof.reserve(n);

                for (int new_i = 0; new_i < n; ++new_i) {
                    int old_i = order[new_i];
                    new_vertex_local_to_global_dof.push_back(
                        element.vertex_local_to_global_dof[old_i]
                    );
                }
                element.vertex_local_to_global_dof.swap(new_vertex_local_to_global_dof);

                // 4. 重建 global_to_local_vertex_map
                global_to_local_vertex_map.clear();
                for (int local_id = 0; local_id < n; ++local_id) {
                    int global_id = element.vertex_local_to_global_dof[local_id];
                    global_to_local_vertex_map[global_id] = local_id;
                }

            }
            RZ_ASSERT(element.vertex_local_to_global_dof.size()>0,"");

            //sort each polyherdron_polygons
            for (auto&polygon_id : polyhedron_polygons) {
                const auto& polygon = polygons[polygon_id];
                std::vector<int> local_polygon;
                for (auto&vertex: polygon) {
                    local_polygon.push_back(global_to_local_vertex_map[vertex]);
                }
                element.polygons.push_back(local_polygon);
            }
            // polygons也需要被排序
            // polygons 也需要排序：按照对应 plane_data 的 _3d_centroid 做字典序排序
            {
                const int m = static_cast<int>(element.polygons.size());

                // 1. 构造排序索引
                std::vector<int> order(m);
                std::iota(order.begin(), order.end(), 0);

                // 2. 根据每个 polygon 对应的全局 plane_data._3d_centroid 排序
                std::sort(order.begin(), order.end(),
                    [&](int a, int b) {
                        int global_pid_a = element.polygon_local_to_global_dof[a];
                        int global_pid_b = element.polygon_local_to_global_dof[b];

                        const auto& plane_a = polygons_plane[global_pid_a];
                        const auto& plane_b = polygons_plane[global_pid_b];

                        return _3d_vec::sort(plane_a._3d_centroid, plane_b._3d_centroid);
                    });

                // 3. 重排 polygons 和 polygon_local_to_global_dof
                std::vector<std::vector<int>> new_polygons;
                std::vector<int> new_polygon_local_to_global_dof;

                new_polygons.reserve(m);
                new_polygon_local_to_global_dof.reserve(m);

                for (int new_i = 0; new_i < m; ++new_i) {
                    int old_i = order[new_i];
                    new_polygons.push_back(element.polygons[old_i]);
                    new_polygon_local_to_global_dof.push_back(
                        element.polygon_local_to_global_dof[old_i]
                    );
                }

                element.polygons.swap(new_polygons);
                element.polygon_local_to_global_dof.swap(new_polygon_local_to_global_dof);
            }

            {
                orient_mesh_and_compute_volume(element);
                //进一步 sort 每一个 polygon使之 左下到右上
                {
                    for (int pid = 0; pid < (int)element.polygons.size(); pid++) {
                        auto& plane_data = polygons_plane[element.polygon_local_to_global_dof[pid]];
                        auto& poly = element.polygons[pid];
                        const _3d_vec& origin = plane_data.origin;
                        _3d_vec normal = plane_data.normal;
                        _3d_vec outer_vec = element.vertex_position(element.polygons[pid][0]) - element.centroid;
                        if (normal.dot(outer_vec)<0) {
                            element.poly_planes.push_back(plane_data.cgal_plane.opposite());
                            normal.vx = -normal.vx;
                            normal.vy = -normal.vy;
                            normal.vz = -normal.vz;
                        }else {
                            element.poly_planes.push_back(plane_data.cgal_plane);
                        }
                        // -----------------------------
                        // 2. 构造平面局部正交基 (u, v)
                        // -----------------------------
                        _3d_vec tmp;
                        if (std::fabs(normal.vx) > std::fabs(normal.vy))
                            tmp = _3d_vec(0, 1, 0);
                        else
                            tmp = _3d_vec(1, 0, 0);

                        _3d_vec u_axis = (normal.cross(tmp)).normalized();
                        _3d_vec v_axis = (normal.cross(u_axis)).normalized();

                        // -----------------------------
                        // 3. 找到投影后 (u,v) 最小的顶点
                        //    作为“左下角”
                        // -----------------------------
                        int best_id = 0;
                        Real best_u = 0.0, best_v = 0.0;

                        for (int i = 0; i < (int)poly.size(); ++i) {
                            const _3d_vec& p = element.vertex_position(poly[i]);
                            _3d_vec rel = p - origin;

                            Real u = rel.dot(u_axis);
                            Real v = rel.dot(v_axis);

                            if (i == 0 ||
                                u < best_u ||
                                (u == best_u && v < best_v)) {
                                best_u = u;
                                best_v = v;
                                best_id = i;
                                }
                        }

                        // -----------------------------
                        // 4. cyclic shift，使 best_id 成为第 0 个
                        // -----------------------------
                        if (best_id != 0) {
                            std::rotate(poly.begin(),
                                        poly.begin() + best_id,
                                        poly.end());
                        }
                    }
                }


            }
            auto unique_edges_in_polyhedron = sorted_polyhedron_edges_by_midpoint(element);
            std::map<Edge,int> edge_to_volume_dof;
            for (auto &edge: unique_edges_in_polyhedron) {
                edge_to_volume_dof[edge] = element.vertex_count() + element.edge_count();
                element.edge_local_to_global_dof.push_back(global_edge_id_from_local_edge(global_edge_to_id, element, edge));            }
            //  构建这里面的VEM_Element_2d

            auto& elements_2d = element.elements_2d;
            elements_2d.resize(element.polygons.size());

            for (int pid = 0; pid < (int)element.polygons.size(); pid++) {
                auto& plane_data = polygons_plane[element.polygon_local_to_global_dof[pid]];
                VEM_Element_2d element_2d;

                std::vector<_3d_vec> points_3d;
                points_3d.reserve(element.polygons[pid].size());

                for (const auto& vertex : element.polygons[pid]) {
                    element_2d.volume_v_to_local_v[vertex] = points_3d.size();
                    points_3d.push_back(element.vertex_position(vertex));
                }
                element_2d.volume_dof_to_local_dof[element.vertex_count()
                    + element.edge_count() + pid] = points_3d.size() * 2;

                if (!element_2d.compute_plane_and_projection(points_3d,
                    plane_data,element.centroid, edge_to_volume_dof)) {
                    element.save_face_to_obj(element.polygons[pid]);
                    element.save_polygonmesh_to_obj("normal.obj");
                    RZ_FATAL("");
                    }
                element_2d.area = plane_data.area;
                element_2d.diameter = std::sqrt(element_2d.area);
                element_2d._3d_centroid = plane_data._3d_centroid;
                element_2d.build_geometric = false;
                if (!element_2d.project_only_build_up()) {
                    element.save_face_to_obj(element.polygons[pid]);
                    element.save_polygonmesh_to_obj("noccw.obj");
                }
                elements_2d[pid] = std::move(element_2d);
            }
            element.build_up();
            this->U3d_project_matrix_star_k2 = element.project_matrix_star_k2;
            prune_relative_small_entries(this->U3d_project_matrix_star_k2,1e-10);
            this->U3d_project_matrix_star_k1 = element.project_matrix_star_k1;
            prune_relative_small_entries(this->U3d_project_matrix_star_k1,1e-10);
            this->U3d_stiffness_matrix_k2 = element.stiffness_matrix_k2;
            prune_relative_small_entries(this->U3d_stiffness_matrix_k2,1e-10);
            this->U3d_stiffness_matrix_k1 = element.stiffness_matrix_k1;
            prune_relative_small_entries(this->U3d_stiffness_matrix_k1,1e-10);

            this->U3d_local_LX_k2 = element.local_LX_k2;
            prune_relative_small_entries(this->U3d_local_LX_k2,1e-10);
            this->U3d_local_LY_k2 = element.local_LY_k2;
            prune_relative_small_entries(this->U3d_local_LY_k2,1e-10);
            this->U3d_local_LZ_k2 = element.local_LZ_k2;
            prune_relative_small_entries(this->U3d_local_LZ_k2,1e-10);
            this->U3d_local_LX_k1 = element.local_LX_k1;
            prune_relative_small_entries(this->U3d_local_LX_k1,1e-10);
            this->U3d_local_LY_k1 = element.local_LY_k1;
            prune_relative_small_entries(this->U3d_local_LY_k1,1e-10);
            this->U3d_local_LZ_k1 = element.local_LZ_k1;
            prune_relative_small_entries(this->U3d_local_LZ_k1,1e-10);

            this->U2d_project_matrix_star = element.elements_2d[0].project_matrix_star;
            prune_relative_small_entries(this->U2d_project_matrix_star,1e-10);
            this->U2d_project_matrix_star_k1 = element.elements_2d[0].project_matrix_star_k1;
            prune_relative_small_entries(this->U2d_project_matrix_star_k1,1e-10);
            break;
        }
    }

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int)polyhedrons.size(); ++i) {
        const auto& polyhedron = polyhedrons[i];
        const auto& polyhedron_polygons = polyhedron;
        VEM_Element_3d element;
            element.bind_geometry(points, edges);
        element.is_U = false;
        for (auto&polygon : polyhedron_polygons) {
            if (is_polygon_boundary[polygon])
                element.has_boundary=true;
        }
        element.polygon_local_to_global_dof = polyhedron_polygons;

        //顶点唯一化 且 构建好到全局映射
        std::set<int> unique_vertices;
        for (auto&polygon_id : polyhedron_polygons) {
            const auto& polygon = polygons[polygon_id];
            for (auto&vertex: polygon) {
                unique_vertices.insert(vertex);
            }
        }
        std::map<int,int> global_to_local_vertex_map;
        for (auto&vertex : unique_vertices) {
            global_to_local_vertex_map[vertex] = element.vertex_count();
                        element.vertex_local_to_global_dof.push_back(vertex);
        }

        {// sort points inside a 3d volume
            // ======================================================
            // 对 element.points 做字典序排序 (x, y, z)
            // 并同步更新 vertex_local_to_global_dof 与 map
            // ======================================================

            const int n = static_cast<int>(element.vertex_count());

            // 1. 构造排序索引
            std::vector<int> order(n);
            std::iota(order.begin(), order.end(), 0);

            // 2. 按 points 的几何位置排序索引
            std::sort(order.begin(), order.end(),
                [&](int a, int b) {
                    return _3d_vec::sort(element.vertex_position(a), element.vertex_position(b));
                });

            // 3. 重排 dof
                std::vector<int> new_vertex_local_to_global_dof;
                new_vertex_local_to_global_dof.reserve(n);

            for (int new_i = 0; new_i < n; ++new_i) {
                int old_i = order[new_i];
                    new_vertex_local_to_global_dof.push_back(
                    element.vertex_local_to_global_dof[old_i]
                );
            }
                element.vertex_local_to_global_dof.swap(new_vertex_local_to_global_dof);

            // 4. 重建 global_to_local_vertex_map
            global_to_local_vertex_map.clear();
            for (int local_id = 0; local_id < n; ++local_id) {
                int global_id = element.vertex_local_to_global_dof[local_id];
                global_to_local_vertex_map[global_id] = local_id;
            }

        }
        RZ_ASSERT(element.vertex_local_to_global_dof.size()>0,"");

        //sort each polyherdron_polygons
        for (auto&polygon_id : polyhedron_polygons) {
            const auto& polygon = polygons[polygon_id];
            std::vector<int> local_polygon;
            for (auto&vertex: polygon) {
                local_polygon.push_back(global_to_local_vertex_map[vertex]);
            }
            element.polygons.push_back(local_polygon);
        }

        // polygons 也需要排序：按照对应 plane_data 的 _3d_centroid 做字典序排序
        {
            const int m = static_cast<int>(element.polygons.size());

            // 1. 构造排序索引
            std::vector<int> order(m);
            std::iota(order.begin(), order.end(), 0);

            // 2. 根据每个 polygon 对应的全局 plane_data._3d_centroid 排序
            std::sort(order.begin(), order.end(),
                [&](int a, int b) {
                    int global_pid_a = element.polygon_local_to_global_dof[a];
                    int global_pid_b = element.polygon_local_to_global_dof[b];

                    const auto& plane_a = polygons_plane[global_pid_a];
                    const auto& plane_b = polygons_plane[global_pid_b];

                    return _3d_vec::sort(plane_a._3d_centroid, plane_b._3d_centroid);
                });

            // 3. 重排 polygons 和 polygon_local_to_global_dof
            std::vector<std::vector<int>> new_polygons;
            std::vector<int> new_polygon_local_to_global_dof;

            new_polygons.reserve(m);
            new_polygon_local_to_global_dof.reserve(m);

            for (int new_i = 0; new_i < m; ++new_i) {
                int old_i = order[new_i];
                new_polygons.push_back(element.polygons[old_i]);
                new_polygon_local_to_global_dof.push_back(
                    element.polygon_local_to_global_dof[old_i]
                );
            }

            element.polygons.swap(new_polygons);
            element.polygon_local_to_global_dof.swap(new_polygon_local_to_global_dof);
        }


        // 计算3d基本属性
        {
            orient_mesh_and_compute_volume(element);
            //进一步 sort 每一个 polygon使之 左下到右上
            {
                for (int pid = 0; pid < (int)element.polygons.size(); pid++) {
                    auto& plane_data = polygons_plane[element.polygon_local_to_global_dof[pid]];
                    auto& poly = element.polygons[pid];
                    const _3d_vec& origin = plane_data.origin;
                    _3d_vec normal = plane_data.normal;
                    _3d_vec outer_vec = element.vertex_position(element.polygons[pid][0]) - element.centroid;
                    if (normal.dot(outer_vec)<0) {
                        element.poly_planes.push_back(plane_data.cgal_plane.opposite());
                        normal.vx = -normal.vx;
                        normal.vy = -normal.vy;
                        normal.vz = -normal.vz;
                    }else {
                        element.poly_planes.push_back(plane_data.cgal_plane);
                    }
                    // -----------------------------
                    // 2. 构造平面局部正交基 (u, v)
                    // -----------------------------
                    _3d_vec tmp;
                    if (std::fabs(normal.vx) > std::fabs(normal.vy))
                        tmp = _3d_vec(0, 1, 0);
                    else
                        tmp = _3d_vec(1, 0, 0);

                    _3d_vec u_axis = (normal.cross(tmp)).normalized();
                    _3d_vec v_axis = (normal.cross(u_axis)).normalized();

                    // -----------------------------
                    // 3. 找到投影后 (u,v) 最小的顶点
                    //    作为“左下角”
                    // -----------------------------
                    int best_id = 0;
                    Real best_u = 0.0, best_v = 0.0;

                    for (int i = 0; i < (int)poly.size(); ++i) {
                        const _3d_vec& p = element.vertex_position(poly[i]);
                        _3d_vec rel = p - origin;

                        Real u = rel.dot(u_axis);
                        Real v = rel.dot(v_axis);

                        if (i == 0 ||
                            u < best_u ||
                            (u == best_u && v < best_v)) {
                            best_u = u;
                            best_v = v;
                            best_id = i;
                            }
                    }

                    // -----------------------------
                    // 4. cyclic shift，使 best_id 成为第 0 个
                    // -----------------------------
                    if (best_id != 0) {
                        std::rotate(poly.begin(),
                                    poly.begin() + best_id,
                                    poly.end());
                    }
                }
            }
        }

        auto unique_edges_in_polyhedron = sorted_polyhedron_edges_by_midpoint(element);
        std::map<Edge,int> edge_to_volume_dof;
        for (auto &edge: unique_edges_in_polyhedron) {
            edge_to_volume_dof[edge] = element.vertex_count() + element.edge_count();
            element.edge_local_to_global_dof.push_back(global_edge_id_from_local_edge(global_edge_to_id, element, edge));        }

        if (element.polygons.size()==6&&element.vertex_count() == 8&&unique_edges_in_polyhedron.size()==12)//cube
            // if (element.polygons.size()==12)//cube
        {
            Real max_poly_area = 0.0,min_poly_area=1e9;
            for (auto& gpid:element.polygon_local_to_global_dof) {
                max_poly_area = std::max(max_poly_area,polygons_plane[gpid].area);
                min_poly_area = std::min(min_poly_area,polygons_plane[gpid].area);
            }

            if (has_builded_u && U_area > 0.0 && min_poly_area > 0.0 &&
                (max_poly_area - min_poly_area) / min_poly_area < 1e-2 &&
                std::fabs(max_poly_area - U_area) / U_area < 1e-2) {
                element.is_U = true;
            }

        }


        //  构建这里面的VEM_Element_2d
        if (generate_2d||!element.is_U){
        // if (!element.is_U) {
            auto& elements_2d = element.elements_2d;
            elements_2d.resize(element.polygons.size());

            for (int pid = 0; pid < (int)element.polygons.size(); pid++) {
                auto& plane_data = polygons_plane[element.polygon_local_to_global_dof[pid]];
                VEM_Element_2d element_2d;

                std::vector<_3d_vec> points_3d;
                points_3d.reserve(element.polygons[pid].size());

                for (const auto& vertex : element.polygons[pid]) {
                    element_2d.volume_v_to_local_v[vertex] = points_3d.size();
                    points_3d.push_back(element.vertex_position(vertex));
                }
                element_2d.volume_dof_to_local_dof[element.vertex_count()
                    + element.edge_count() + pid] = points_3d.size() * 2;

                if (!element_2d.compute_plane_and_projection(points_3d,
                    plane_data,element.centroid, edge_to_volume_dof)) {
                    element.save_face_to_obj(element.polygons[pid]);
                    element.save_polygonmesh_to_obj("normal.obj");
                    RZ_FATAL("");
                    }
                // element_2d.build_up();
                element_2d.area = plane_data.area;
                element_2d.diameter = std::sqrt(element_2d.area);
                element_2d._3d_centroid = plane_data._3d_centroid;
                element_2d.build_geometric = false;
                if (!element.is_U) {
                    if (!element_2d.project_only_build_up()) {
                        element.save_face_to_obj(element.polygons[pid]);
                        element.save_polygonmesh_to_obj("noccw.obj");
                    }
                }
                elements_2d[pid] = std::move(element_2d);
            }
        }
        if (element.is_U) {
            vem_elements[i] = std::move(element);
            continue;
        }
        element.build_up();
        // end !
        vem_elements[i] = std::move(element);
    }
    for (auto&vem:vem_elements) {
        if (!vem.is_U) {
            RZ_ASSERT(vem.stiffness_matrix_k2.rows()>1,vem.stiffness_matrix_k2.rows()," ",
                vem.is_U," ",vem.vertex_count()," ",vem.edge_count()," ",vem.polygons.size()," ",vem.volume);
        }
    }
}

void VEM_Domain_3d::generate_elements_k1_only() {
     FUNC_TIMER();
    vem_elements.clear();
    vem_elements.resize(polyhedrons.size());
    const auto global_edge_to_id = build_edge_to_id_cache(edges);

    //先找到一个 is_U的 串行
    for (int i = 0; i < (int)polyhedrons.size(); ++i) {
        if (polyhedrons[i].size()!=6)continue;
        Real max_area=0,min_area=1;
        Real max_dia = 0,min_dia = 1;
        for (auto&polyid:polyhedrons[i]) {
            max_area = std::max(max_area,polygons_plane[polyid].area);
            min_area = std::min(min_area,polygons_plane[polyid].area);
            max_dia = std::max(max_dia,polygons_plane[polyid].diameter);
            min_dia = std::min(min_dia,polygons_plane[polyid].diameter);
        }
        if ((max_area - min_area)/min_area < 1e-2 &&
            (max_dia - min_dia)/min_dia<1e-2 && min_area>0.000025) {
            has_builded_u = true;
            U_area = min_area;
            const auto& polyhedron_polygons = polyhedrons[i];
            VEM_Element_3d element;
            element.bind_geometry(points, edges);
            for (auto&polygon : polyhedron_polygons) {
                if (is_polygon_boundary[polygon])
                    element.has_boundary=true;
            }
            element.polygon_local_to_global_dof = polyhedron_polygons;

            std::set<int> unique_vertices;
            for (auto&polygon_id : polyhedron_polygons) {
                const auto& polygon = polygons[polygon_id];
                for (auto&vertex: polygon) {
                    unique_vertices.insert(vertex);
                }
            }
            std::map<int,int> global_to_local_vertex_map;
            for (auto&vertex : unique_vertices) {
                global_to_local_vertex_map[vertex] = element.vertex_count();
                                element.vertex_local_to_global_dof.push_back(vertex);
            }

            {// sort points inside a 3d volume
                // ======================================================
                // 对 element.points 做字典序排序 (x, y, z)
                // 并同步更新 vertex_local_to_global_dof 与 map
                // ======================================================

                const int n = static_cast<int>(element.vertex_count());

                // 1. 构造排序索引
                std::vector<int> order(n);
                std::iota(order.begin(), order.end(), 0);

                // 2. 按 points 的几何位置排序索引
                std::sort(order.begin(), order.end(),
                    [&](int a, int b) {
                        return _3d_vec::sort(element.vertex_position(a), element.vertex_position(b));
                    });

                // 3. 重排 dof
                std::vector<int> new_vertex_local_to_global_dof;
                new_vertex_local_to_global_dof.reserve(n);

                for (int new_i = 0; new_i < n; ++new_i) {
                    int old_i = order[new_i];
                    new_vertex_local_to_global_dof.push_back(
                        element.vertex_local_to_global_dof[old_i]
                    );
                }
                element.vertex_local_to_global_dof.swap(new_vertex_local_to_global_dof);

                // 4. 重建 global_to_local_vertex_map
                global_to_local_vertex_map.clear();
                for (int local_id = 0; local_id < n; ++local_id) {
                    int global_id = element.vertex_local_to_global_dof[local_id];
                    global_to_local_vertex_map[global_id] = local_id;
                }

            }
            RZ_ASSERT(element.vertex_local_to_global_dof.size()>0,"");

            //sort each polyherdron_polygons
            for (auto&polygon_id : polyhedron_polygons) {
                const auto& polygon = polygons[polygon_id];
                std::vector<int> local_polygon;
                for (auto&vertex: polygon) {
                    local_polygon.push_back(global_to_local_vertex_map[vertex]);
                }
                element.polygons.push_back(local_polygon);
            }
            // polygons也需要被排序
            // polygons 也需要排序：按照对应 plane_data 的 _3d_centroid 做字典序排序
            {
                const int m = static_cast<int>(element.polygons.size());

                // 1. 构造排序索引
                std::vector<int> order(m);
                std::iota(order.begin(), order.end(), 0);

                // 2. 根据每个 polygon 对应的全局 plane_data._3d_centroid 排序
                std::sort(order.begin(), order.end(),
                    [&](int a, int b) {
                        int global_pid_a = element.polygon_local_to_global_dof[a];
                        int global_pid_b = element.polygon_local_to_global_dof[b];

                        const auto& plane_a = polygons_plane[global_pid_a];
                        const auto& plane_b = polygons_plane[global_pid_b];

                        return _3d_vec::sort(plane_a._3d_centroid, plane_b._3d_centroid);
                    });

                // 3. 重排 polygons 和 polygon_local_to_global_dof
                std::vector<std::vector<int>> new_polygons;
                std::vector<int> new_polygon_local_to_global_dof;

                new_polygons.reserve(m);
                new_polygon_local_to_global_dof.reserve(m);

                for (int new_i = 0; new_i < m; ++new_i) {
                    int old_i = order[new_i];
                    new_polygons.push_back(element.polygons[old_i]);
                    new_polygon_local_to_global_dof.push_back(
                        element.polygon_local_to_global_dof[old_i]
                    );
                }

                element.polygons.swap(new_polygons);
                element.polygon_local_to_global_dof.swap(new_polygon_local_to_global_dof);
            }

            auto unique_edges_in_polyhedron = sorted_polyhedron_edges_by_midpoint(element);
            std::map<Edge,int> edge_to_volume_dof;
            for (auto &edge: unique_edges_in_polyhedron) {
                edge_to_volume_dof[edge] = element.vertex_count() + element.edge_count();
                element.edge_local_to_global_dof.push_back(global_edge_id_from_local_edge(global_edge_to_id, element, edge));            }
            {
                orient_mesh_and_compute_volume(element);
                //进一步 sort 每一个 polygon使之 左下到右上
                {
                    for (int pid = 0; pid < (int)element.polygons.size(); pid++) {
                        auto& plane_data = polygons_plane[element.polygon_local_to_global_dof[pid]];
                        auto& poly = element.polygons[pid];
                        const _3d_vec& origin = plane_data.origin;
                        _3d_vec normal = plane_data.normal;
                        _3d_vec outer_vec = element.vertex_position(element.polygons[pid][0]) - element.centroid;
                        if (normal.dot(outer_vec)<0) {
                            element.poly_planes.push_back(plane_data.cgal_plane.opposite());
                            normal.vx = -normal.vx;
                            normal.vy = -normal.vy;
                            normal.vz = -normal.vz;
                        }else {
                            element.poly_planes.push_back(plane_data.cgal_plane);
                        }
                        // -----------------------------
                        // 2. 构造平面局部正交基 (u, v)
                        // -----------------------------
                        _3d_vec tmp;
                        if (std::fabs(normal.vx) > std::fabs(normal.vy))
                            tmp = _3d_vec(0, 1, 0);
                        else
                            tmp = _3d_vec(1, 0, 0);

                        _3d_vec u_axis = (normal.cross(tmp)).normalized();
                        _3d_vec v_axis = (normal.cross(u_axis)).normalized();

                        // -----------------------------
                        // 3. 找到投影后 (u,v) 最小的顶点
                        //    作为“左下角”
                        // -----------------------------
                        int best_id = 0;
                        Real best_u = 0.0, best_v = 0.0;

                        for (int i = 0; i < (int)poly.size(); ++i) {
                            const _3d_vec& p = element.vertex_position(poly[i]);
                            _3d_vec rel = p - origin;

                            Real u = rel.dot(u_axis);
                            Real v = rel.dot(v_axis);

                            if (i == 0 ||
                                u < best_u ||
                                (u == best_u && v < best_v)) {
                                best_u = u;
                                best_v = v;
                                best_id = i;
                                }
                        }

                        // -----------------------------
                        // 4. cyclic shift，使 best_id 成为第 0 个
                        // -----------------------------
                        if (best_id != 0) {
                            std::rotate(poly.begin(),
                                        poly.begin() + best_id,
                                        poly.end());
                        }
                    }
                }
            }
            //  构建这里面的VEM_Element_2d

            auto& elements_2d = element.elements_2d;
            elements_2d.resize(element.polygons.size());

            for (int pid = 0; pid < (int)element.polygons.size(); pid++) {
                auto& plane_data = polygons_plane[element.polygon_local_to_global_dof[pid]];
                VEM_Element_2d element_2d;

                std::vector<_3d_vec> points_3d;
                points_3d.reserve(element.polygons[pid].size());

                for (const auto& vertex : element.polygons[pid]) {
                    element_2d.volume_v_to_local_v[vertex] = points_3d.size();
                    points_3d.push_back(element.vertex_position(vertex));
                }
                element_2d.volume_dof_to_local_dof[element.vertex_count()
                    + element.edge_count() + pid] = points_3d.size() * 2;

                if (!element_2d.compute_plane_and_projection(points_3d,
                    plane_data,element.centroid, edge_to_volume_dof)) {
                    element.save_face_to_obj(element.polygons[pid]);
                    element.save_polygonmesh_to_obj("normal.obj");
                    RZ_FATAL("");
                    }
                element_2d.area = plane_data.area;
                element_2d.diameter = std::sqrt(element_2d.area);
                element_2d._3d_centroid = plane_data._3d_centroid;
                element_2d.build_geometric = false;
                if (!element_2d.project_only_build_up_k1()) {
                    element.save_face_to_obj(element.polygons[pid]);
                    element.save_polygonmesh_to_obj("noccw.obj");
                }
                elements_2d[pid] = std::move(element_2d);
            }
            element.build_up_k1();
            this->U3d_project_matrix_star_k1 = element.project_matrix_star_k1;
            prune_relative_small_entries(this->U3d_project_matrix_star_k1,1e-15);
            this->U3d_stiffness_matrix_k1 = element.stiffness_matrix_k1;
            prune_relative_small_entries(this->U3d_stiffness_matrix_k1,1e-15);


            this->U3d_local_LX_k1 = element.local_LX_k1;
            prune_relative_small_entries(this->U3d_local_LX_k1,1e-15);
            this->U3d_local_LY_k1 = element.local_LY_k1;
            prune_relative_small_entries(this->U3d_local_LY_k1,1e-15);
            this->U3d_local_LZ_k1 = element.local_LZ_k1;
            prune_relative_small_entries(this->U3d_local_LZ_k1,1e-15);

            this->U2d_project_matrix_star_k1 = element.elements_2d[0].project_matrix_star_k1;
            prune_relative_small_entries(this->U2d_project_matrix_star_k1,1e-15);
            break;
        }
    }


#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < (int)polyhedrons.size(); ++i) {
        const auto& polyhedron = polyhedrons[i];
        const auto& polyhedron_polygons = polyhedron;
        VEM_Element_3d element;
            element.bind_geometry(points, edges);
        element.is_U = false;
        for (auto&polygon : polyhedron_polygons) {
            if (is_polygon_boundary[polygon])
                element.has_boundary=true;
        }
        element.polygon_local_to_global_dof = polyhedron_polygons;

        //顶点唯一化 且 构建好到全局映射
        std::vector<int> unique_vertices;

        size_t vertex_ref_count = 0;
        for (const int polygon_id : polyhedron_polygons) {
            vertex_ref_count += polygons[polygon_id].size();
        }

        unique_vertices.reserve(vertex_ref_count);

        for (const int polygon_id : polyhedron_polygons) {
            const auto& polygon = polygons[polygon_id];
            for (const int vertex : polygon) {
                unique_vertices.push_back(vertex);
            }
        }

        std::sort(unique_vertices.begin(), unique_vertices.end());

        unique_vertices.erase(
            std::unique(unique_vertices.begin(), unique_vertices.end()),
            unique_vertices.end()
        );

        // 如果 global vertex id 不连续或 points.size() 很大，用 unordered_map
        std::unordered_map<int, int> global_to_local_vertex_map;
        global_to_local_vertex_map.reserve(unique_vertices.size() * 2 + 1);        element.vertex_local_to_global_dof.reserve(
            element.vertex_local_to_global_dof.size() + unique_vertices.size()
        );

        for (int vertex : unique_vertices) {
            const int local_id = static_cast<int>(element.vertex_count());

            global_to_local_vertex_map.emplace(vertex, local_id);

                        element.vertex_local_to_global_dof.push_back(vertex);
        }

        {// sort points inside a 3d volume
            // ======================================================
            // 对 element.points 做字典序排序 (x, y, z)
            // 并同步更新 vertex_local_to_global_dof 与 map
            // ======================================================

            const int n = static_cast<int>(element.vertex_count());

            // 1. 构造排序索引
            std::vector<int> order(n);
            std::iota(order.begin(), order.end(), 0);

            // 2. 按 points 的几何位置排序索引
            std::sort(order.begin(), order.end(),
                [&](int a, int b) {
                    return _3d_vec::sort(element.vertex_position(a), element.vertex_position(b));
                });

            // 3. 重排 dof
                std::vector<int> new_vertex_local_to_global_dof;
                new_vertex_local_to_global_dof.reserve(n);

            for (int new_i = 0; new_i < n; ++new_i) {
                int old_i = order[new_i];
                    new_vertex_local_to_global_dof.push_back(
                    element.vertex_local_to_global_dof[old_i]
                );
            }
                element.vertex_local_to_global_dof.swap(new_vertex_local_to_global_dof);

            // 4. 重建 global_to_local_vertex_map
            global_to_local_vertex_map.clear();
            for (int local_id = 0; local_id < n; ++local_id) {
                int global_id = element.vertex_local_to_global_dof[local_id];
                global_to_local_vertex_map[global_id] = local_id;
            }

        }
        RZ_ASSERT(element.vertex_local_to_global_dof.size()>0,"");

        //sort each polyherdron_polygons
        for (auto&polygon_id : polyhedron_polygons) {
            const auto& polygon = polygons[polygon_id];
            std::vector<int> local_polygon;
            for (auto&vertex: polygon) {
                local_polygon.push_back(global_to_local_vertex_map[vertex]);
            }
            element.polygons.push_back(local_polygon);
        }

        // polygons 也需要排序：按照对应 plane_data 的 _3d_centroid 做字典序排序
        {
            const int m = static_cast<int>(element.polygons.size());

            // 1. 构造排序索引
            std::vector<int> order(m);
            std::iota(order.begin(), order.end(), 0);

            // 2. 根据每个 polygon 对应的全局 plane_data._3d_centroid 排序
            std::sort(order.begin(), order.end(),
                [&](int a, int b) {
                    int global_pid_a = element.polygon_local_to_global_dof[a];
                    int global_pid_b = element.polygon_local_to_global_dof[b];

                    const auto& plane_a = polygons_plane[global_pid_a];
                    const auto& plane_b = polygons_plane[global_pid_b];

                    return _3d_vec::sort(plane_a._3d_centroid, plane_b._3d_centroid);
                });

            // 3. 重排 polygons 和 polygon_local_to_global_dof
            std::vector<std::vector<int>> new_polygons;
            std::vector<int> new_polygon_local_to_global_dof;

            new_polygons.reserve(m);
            new_polygon_local_to_global_dof.reserve(m);

            for (int new_i = 0; new_i < m; ++new_i) {
                int old_i = order[new_i];
                new_polygons.push_back(element.polygons[old_i]);
                new_polygon_local_to_global_dof.push_back(
                    element.polygon_local_to_global_dof[old_i]
                );
            }

            element.polygons.swap(new_polygons);
            element.polygon_local_to_global_dof.swap(new_polygon_local_to_global_dof);
        }


        auto unique_edges_in_polyhedron = sorted_polyhedron_edges_by_midpoint(element);
        std::map<Edge,int> edge_to_volume_dof;
        for (auto &edge: unique_edges_in_polyhedron) {
            edge_to_volume_dof[edge] = element.vertex_count() + element.edge_count();
            element.edge_local_to_global_dof.push_back(global_edge_id_from_local_edge(global_edge_to_id, element, edge));        }
        // 计算3d基本属性
        {
            orient_mesh_and_compute_volume(element);
            //进一步 sort 每一个 polygon使之 左下到右上
            {
                for (int pid = 0; pid < (int)element.polygons.size(); pid++) {
                    auto& plane_data = polygons_plane[element.polygon_local_to_global_dof[pid]];
                    auto& poly = element.polygons[pid];
                    const _3d_vec& origin = plane_data.origin;
                    _3d_vec normal = plane_data.normal;
                    _3d_vec outer_vec = element.vertex_position(element.polygons[pid][0]) - element.centroid;
                    if (normal.dot(outer_vec)<0) {
                        element.poly_planes.push_back(plane_data.cgal_plane.opposite());
                        normal.vx = -normal.vx;
                        normal.vy = -normal.vy;
                        normal.vz = -normal.vz;
                    }else {
                        element.poly_planes.push_back(plane_data.cgal_plane);
                    }
                    // -----------------------------
                    // 2. 构造平面局部正交基 (u, v)
                    // -----------------------------
                    _3d_vec tmp;
                    if (std::fabs(normal.vx) > std::fabs(normal.vy))
                        tmp = _3d_vec(0, 1, 0);
                    else
                        tmp = _3d_vec(1, 0, 0);

                    _3d_vec u_axis = (normal.cross(tmp)).normalized();
                    _3d_vec v_axis = (normal.cross(u_axis)).normalized();

                    // -----------------------------
                    // 3. 找到投影后 (u,v) 最小的顶点
                    //    作为“左下角”
                    // -----------------------------
                    int best_id = 0;
                    Real best_u = 0.0, best_v = 0.0;

                    for (int i = 0; i < (int)poly.size(); ++i) {
                        const _3d_vec& p = element.vertex_position(poly[i]);
                        _3d_vec rel = p - origin;

                        Real u = rel.dot(u_axis);
                        Real v = rel.dot(v_axis);

                        if (i == 0 ||
                            u < best_u ||
                            (u == best_u && v < best_v)) {
                            best_u = u;
                            best_v = v;
                            best_id = i;
                            }
                    }

                    // -----------------------------
                    // 4. cyclic shift，使 best_id 成为第 0 个
                    // -----------------------------
                    if (best_id != 0) {
                        std::rotate(poly.begin(),
                                    poly.begin() + best_id,
                                    poly.end());
                    }
                }
            }
        }

        if (element.polygons.size()==6&&element.vertex_count() == 8&&unique_edges_in_polyhedron.size()==12)//cube
            // if (element.polygons.size()==12)//cube
        {
            Real max_poly_area = 0.0,min_poly_area=1e9;
            for (auto& gpid:element.polygon_local_to_global_dof) {
                max_poly_area = std::max(max_poly_area,polygons_plane[gpid].area);
                min_poly_area = std::min(min_poly_area,polygons_plane[gpid].area);
            }

            if (has_builded_u && U_area > 0.0 && min_poly_area > 0.0 &&
                (max_poly_area - min_poly_area) / min_poly_area < 1e-2 &&
                std::fabs(max_poly_area - U_area) / U_area < 1e-2) {
                element.is_U = true;
            }
        }


        //  构建这里面的VEM_Element_2d
        if (true){// todo: 1s
        // if (!element.is_U) {
            auto& elements_2d = element.elements_2d;
            elements_2d.resize(element.polygons.size());

            for (int pid = 0; pid < (int)element.polygons.size(); pid++) {
                auto& plane_data = polygons_plane[element.polygon_local_to_global_dof[pid]];
                VEM_Element_2d element_2d;

                std::vector<_3d_vec> points_3d;
                points_3d.reserve(element.polygons[pid].size());

                for (const auto& vertex : element.polygons[pid]) {
                    element_2d.volume_v_to_local_v[vertex] = points_3d.size();
                    points_3d.push_back(element.vertex_position(vertex));
                }
                element_2d.volume_dof_to_local_dof[element.vertex_count()
                    + element.edge_count() + pid] = points_3d.size() * 2;

                if (!element_2d.compute_plane_and_projection(points_3d,
                    plane_data,element.centroid, edge_to_volume_dof)) {
                    element.save_face_to_obj(element.polygons[pid]);
                    element.save_polygonmesh_to_obj("normal.obj");
                    RZ_FATAL("");
                    }
                // element_2d.build_up();
                element_2d.area = plane_data.area;
                element_2d.diameter = std::sqrt(element_2d.area);
                element_2d._3d_centroid = plane_data._3d_centroid;
                element_2d.build_geometric = false;
                if (!element.is_U) {
                    if (!element_2d.project_only_build_up_k1()) {
                        element.save_face_to_obj(element.polygons[pid]);
                        element.save_polygonmesh_to_obj("noccw.obj");
                    }
                }
                elements_2d[pid] = std::move(element_2d);
            }
        }
        if (element.is_U) {
            vem_elements[i] = std::move(element);
            continue;
        }
        element.build_up_k1();//todo: 0.?s
        // end !
        vem_elements[i] = std::move(element);
    }

}

void VEM_Domain_3d::compute_face_neumann_velocity() {
    face_neumann_vels.assign(polygons.size(), 0.0);

    // 1. 先计算每个边界面的 neumann velocity
    Real min_v = 1e30, max_v = -1e30;

    for (const auto& element : vem_elements) {
        for (int local_pid = 0; local_pid < element.polygons.size(); ++local_pid) {
            int gpid = element.polygon_local_to_global_dof[local_pid];
            if (is_obj_boundary[gpid]) {
                RZ_ASSERT(std::fabs(face_neumann_vels[gpid]) < 1e-10, "");

                _3d_vec face_velocity(boundary_mesh->vx, boundary_mesh->vy, boundary_mesh->vz);

                auto cgal_normal = element.poly_planes[local_pid].orthogonal_vector();
                _3d_vec normal(cgal_normal.x(), cgal_normal.y(), cgal_normal.z());
                normal = normal / normal.norm();

                Real face_neumann_vel = face_velocity.dot(normal);
                face_neumann_vels[gpid] =  face_neumann_vel;
                min_v = std::min(min_v, face_neumann_vel);
                max_v = std::max(max_v, face_neumann_vel);
            }
        }
    }
}

static const int flash_threshold = 80000;

static const std::size_t FLUSH_TRIPLETS_LIMIT = 80000000; // 控制装配峰值内存；太大时 triplet buffer 会常驻数 GB

void VEM_Domain_3d::assemble_global_stiffness_matrix_k2() {
    using namespace Eigen;
    FUNC_TIMER();
    FUNC_MEMORY_TIMER();

    const int n_vertices  = (int)points.size();
    const int n_edges     = (int)edges.size();
    const int n_polygons  = (int)polygons.size();
    const int n_elements  = (int)vem_elements.size();

    const int total_dofs = n_vertices + n_edges + n_polygons + n_elements;

    // （可选但推荐）确保矩阵尺寸正确
    global_stiffness_matrix_k2.resize(total_dofs, total_dofs);

    // Triplets：预留容量，减少扩容峰值（按你的平均 ndofs^2 估）
    std::vector<Triplet<GlobalReal>> triplets;

    triplets.reserve(FLUSH_TRIPLETS_LIMIT);

    // 复用映射数组，避免每个 element new 一个 vector 造成小分配风暴
    std::vector<int> element_to_global_dof;
    element_to_global_dof.reserve(128); // k2 下通常 < 64，给点富余

    auto flush_triplets = [&](Eigen::SparseMatrixR& A,
                              std::vector<Eigen::Triplet<GlobalReal>>& T)
    {
        if (T.empty()) return;

        // 注意：setFromTriplets 会重建 A 的稀疏结构。
        // 为了实现“累加”，这里用一个 chunk，但 chunk 的尺寸只由 T 决定（内部压缩），
        // 且不再创建 A.rows()*A.cols() 级别的大临时结构。
        // 同时不做 makeCompressed()，避免频繁压缩引发峰值和耗时。
        Eigen::SparseMatrixR chunk(A.rows(), A.cols());
        chunk.setFromTriplets(T.begin(), T.end(), std::plus<GlobalReal>()); // 重复条目求和
        A += chunk;

        // 释放 triplets 的 size，但保留 capacity 以复用，减少频繁分配
        T.clear();
    };

    // 主循环：装配
    for (int el_id = 0; el_id < n_elements; ++el_id)
    {
        const VEM_Element_3d& elem = vem_elements[el_id];

        element_to_global_dof.clear();

        // 顶点 dof
        const int nv = (int)elem.vertex_count();
        element_to_global_dof.reserve((std::size_t)nv + elem.edge_count() + elem.polygons.size() + 1);
        for (int vid = 0; vid < nv; ++vid)
            element_to_global_dof.push_back(elem.vertex_local_to_global_dof[vid]);

        // 边 dof
        for (int eid = 0; eid < (int)elem.edge_count(); ++eid)
            element_to_global_dof.push_back(n_vertices + elem.edge_local_to_global_dof[eid]);

        // 面 dof
        for (int pid = 0; pid < (int)elem.polygons.size(); ++pid)
            element_to_global_dof.push_back(n_vertices + n_edges + elem.polygon_local_to_global_dof[pid]);

        // 体 dof
        element_to_global_dof.push_back(n_vertices + n_edges + n_polygons + el_id);

        const Eigen::LocalMatrixXr& Ke = elem.is_U ? U3d_stiffness_matrix_k2 : elem.stiffness_matrix_k2;
        const int ndofs = (int)element_to_global_dof.size();

        // K
        for (int i = 0; i < ndofs; ++i) {
            const int gi = element_to_global_dof[i];
            for (int j = 0; j < ndofs; ++j) {
                const Real val = Ke(i, j);
                if (std::fabs(val) > 1e-20) {
                    const int gj = element_to_global_dof[j];
                    triplets.emplace_back(gi, gj, val);
                }
            }
        }
        vem_elements[el_id].stiffness_matrix_k2.resize(0,0);
        // ---- flush（按条目数，而不是 element id）----
        if (triplets.size() >= FLUSH_TRIPLETS_LIMIT) {
            flush_triplets(global_stiffness_matrix_k2, triplets);
        }

    }

    // 最后 flush
    flush_triplets(global_stiffness_matrix_k2, triplets);

    // 只在最后压缩一次
    global_stiffness_matrix_k2.makeCompressed();
    {
        Eigen::GlobalVectorXr c = Eigen::GlobalVectorXr::Ones(total_dofs);
        Eigen::VectorXr r = (global_stiffness_matrix_k2 * c).template cast<Real>();

        Real inf_norm = r.cwiseAbs().maxCoeff();
        Real l2_norm  = r.norm();

        RZ_INFO("[check] ||A*c||_inf = ", inf_norm );
        RZ_INFO("[check] ||A*c||_2   = ", l2_norm );
    }
}



void VEM_Domain_3d::assemble_global_stiffness_matrix_k1() {
    using namespace Eigen;
    FUNC_TIMER();

    const int n_vertices = (int)points.size();
    const int n_elements = (int)vem_elements.size();

    // 全局矩阵尺寸（k1 只有顶点 dof）
    global_stiffness_matrix_k1.resize(n_vertices, n_vertices);

    std::vector<Triplet<GlobalReal>> triplets;

    triplets.reserve(FLUSH_TRIPLETS_LIMIT);

    // 复用映射数组
    std::vector<int> element_to_global_dof;
    element_to_global_dof.reserve(16); // k1 顶点数很小

    auto flush_triplets = [&](Eigen::SparseMatrixR& A,
                              std::vector<Eigen::Triplet<GlobalReal>>& T)
    {
        if (T.empty()) return;
        Eigen::SparseMatrixR chunk(A.rows(), A.cols());
        chunk.setFromTriplets(T.begin(), T.end(), std::plus<GlobalReal>());
        A += chunk;
        T.clear();
    };

    // ================== 主装配循环 ==================
    for (int el_id = 0; el_id < n_elements; ++el_id)
    {
        const VEM_Element_3d& elem = vem_elements[el_id];

        element_to_global_dof.clear();

        // 顶点 dof
        const int nv = (int)elem.vertex_count();
        for (int vid = 0; vid < nv; ++vid)
            element_to_global_dof.push_back(elem.vertex_local_to_global_dof[vid]);

        const Eigen::LocalMatrixXr& Ke =
            elem.is_U ? U3d_stiffness_matrix_k1 : elem.stiffness_matrix_k1;

        // ---- K ----
        for (int i = 0; i < nv; ++i) {
            const int gi = element_to_global_dof[i];
            for (int j = 0; j < nv; ++j) {
                const Real val = Ke(i, j);
                if (std::fabs(val) > 1e-20) {
                    const int gj = element_to_global_dof[j];
                    triplets.emplace_back(gi, gj, val);
                }
            }
        }



        // ---- flush（按条目数，而不是 element id）----
        if (triplets.size() >= FLUSH_TRIPLETS_LIMIT) {
            flush_triplets(global_stiffness_matrix_k1, triplets);
        }

    }
    // ========== 最终 flush ==========
    flush_triplets(global_stiffness_matrix_k1, triplets);

    // ========== 一次性压缩 ==========
    global_stiffness_matrix_k1.makeCompressed();
}
