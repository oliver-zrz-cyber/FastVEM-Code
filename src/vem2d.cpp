#include"vem2d.h"
#include <matio.h>
#include <Eigen/Dense>
#include "AbSolver.h"
#include "matplotlibcpp.h"
#include <GLFW/glfw3.h>
#include "stb_image_write.h"
#include "mgsolver_vemp.h"
#include <CGAL/Polygon_2.h>
#include <CGAL/centroid.h>
#include <CGAL/linear_least_squares_fitting_3.h>
#include "config.h"
typedef CGAL::Polygon_2<K> Polygon_2;

static AMGCLContext GPU_AMG;
static bool inited_k2 = false;
Real square_domain_rhs(const _2d_vec &point) {
    Real x = point.vx;
    Real y = point.vy;
    return 15.0 * std::sin(M_PI * x) * std::sin(M_PI * y);
}

Real _2d_vec_dot(const _2d_vec &a, const _2d_vec &b) {
    return a.vx * b.vx + a.vy * b.vy;
}


Real polygon_moment(int p, int q,
                      const std::vector<_2d_vec> &vertices,
                      Real xc, Real yc, Real area) {
    int n = (int) vertices.size();
    Real moment = 0.0;

    for (int i = 0; i < n; ++i) {
        const _2d_vec &v0 = vertices[i];
        const _2d_vec &v1 = vertices[(i + 1) % n];

        Real x0 = v0.vx, y0 = v0.vy;
        Real x1 = v1.vx, y1 = v1.vy;

        Real cross = x0 * y1 - x1 * y0;
        Real term = 0.0;

        if (p == 2 && q == 0) {
            // ∫ x^2 dA
            term = (x0 * x0 + x0 * x1 + x1 * x1);
        } else if (p == 0 && q == 2) {
            // ∫ y^2 dA
            term = (y0 * y0 + y0 * y1 + y1 * y1);
        } else if (p == 1 && q == 1) {
            // ∫ xy dA
            term = (2 * x0 * y0 + x0 * y1 + x1 * y0 + 2 * x1 * y1);
        } else if (p == 1 && q == 0) {
            // ∫ x dA
            term = (x0 + x1);
        } else if (p == 0 && q == 1) {
            // ∫ y dA
            term = (y0 + y1);
        }

        moment += cross * term;
    }

    if (p == 2 && q == 0) {
        moment /= 12.0;
    } else if (p == 0 && q == 2) {
        moment /= 12.0;
    } else if (p == 1 && q == 1) {
        moment /= 24.0;
    }
    if (p == 1 && q == 0) {
        moment /= 6.0;
    } else if (p == 0 && q == 1) {
        moment /= 6.0;
    }


    // 平移到重心系
    if (p > 0 || q > 0) {
        moment -= area * std::pow(xc, p) * std::pow(yc, q);
    }

    return moment;
}


std::vector<int> toIntVector(matvar_t *var, bool minus1 = true) {
    std::vector<int> result;
    if (!var || !var->data) return result;

    size_t n = 1;
    for (size_t k = 0; k < var->rank; ++k) n *= var->dims[k];
    result.resize(n);

    if (var->class_type == MAT_C_DOUBLE) {
        Real *p = static_cast<Real *>(var->data);
        for (size_t i = 0; i < n; i++)
            result[i] = static_cast<int>(p[i]) - (minus1 ? 1 : 0);
    } else if (var->class_type == MAT_C_INT32) {
        int32_t *p = static_cast<int32_t *>(var->data);
        for (size_t i = 0; i < n; i++)
            result[i] = static_cast<int>(p[i]) - (minus1 ? 1 : 0);
    }
    return result;
}
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
bool VEM_Element_2d::compute_plane_and_projection
(std::vector<_3d_vec> &points,const PlaneData& plane_data,const _3d_vec& centroid,std::map<Edge,int>& edge_to_volume_dof) {
    //----------------------------------
    plane = plane_data;
    K::Vector_3 cgal_vec =
        K::Point_3(points[0].vx,points[0].vy,points[0].vz) - K::Point_3(centroid.vx,centroid.vy,centroid.vz);
    cgal_vec = cgal_vec / std::sqrt(cgal_vec.squared_length());
    K::Vector_3 cgal_unit_normal = this->plane.cgal_plane.orthogonal_vector()
    /std::sqrt(this->plane.cgal_plane.orthogonal_vector().squared_length());
    if (this->plane.cgal_plane.orthogonal_vector() * cgal_vec < 0.0) {
        this->plane.cgal_plane = this->plane.cgal_plane.opposite();
        cgal_unit_normal = this->plane.cgal_plane.orthogonal_vector()
            /std::sqrt(this->plane.cgal_plane.orthogonal_vector().squared_length());
        plane.normal = _3d_vec(cgal_unit_normal.x(),
            cgal_unit_normal.y(),
            cgal_unit_normal.z());
    }
    {
        // //构造局部坐标系
        _3d_vec tmp = (std::fabs(plane.normal.vx) < 0.9) ? _3d_vec(1,0,0) : _3d_vec(0,1,0);
        if (std::fabs(plane.normal.vx) > std::fabs(plane.normal.vy))
            tmp = _3d_vec(0, 1, 0);
        else
            tmp = _3d_vec(1, 0, 0);
        plane.u_axis = (plane.normal.cross(tmp)).normalized();
        plane.v_axis = (plane.normal.cross(plane.u_axis)).normalized();

        // 投影所有点
        this->points.reserve(points.size());
        Polygon_2 cgal_poly2;
        for (auto &p : points) {
            _3d_vec rel = p - plane.origin;
            Real u = rel.vx * plane.u_axis.vx + rel.vy * plane.u_axis.vy + rel.vz * plane.u_axis.vz;
            Real v = rel.vx * plane.v_axis.vx + rel.vy * plane.v_axis.vy + rel.vz * plane.v_axis.vz;
            this->points.emplace_back(u, v);
            cgal_poly2.push_back({u,v});
        }
        if (!cgal_poly2.is_counterclockwise_oriented()) {
            plane.normal = _3d_vec(0,0,0) - plane.normal;
            cgal_poly2.clear();
            _3d_vec tmp = (std::fabs(plane.normal.vx) < 0.9) ? _3d_vec(1,0,0) : _3d_vec(0,1,0);
            if (std::fabs(plane.normal.vx) > std::fabs(plane.normal.vy))
                tmp = _3d_vec(0, 1, 0);
            else
                tmp = _3d_vec(1, 0, 0);
            plane.u_axis = (plane.normal.cross(tmp)).normalized();
            this->points.clear();
            this->points.reserve(points.size());
            for (auto &p : points) {
                _3d_vec rel = p - plane.origin;
                Real u = rel.vx * plane.u_axis.vx + rel.vy * plane.u_axis.vy + rel.vz * plane.u_axis.vz;
                Real v = rel.vx * plane.v_axis.vx + rel.vy * plane.v_axis.vy + rel.vz * plane.v_axis.vz;
                this->points.emplace_back(u, v);
                cgal_poly2.push_back({u,v});
            }
        }
        // if (!cgal_poly2.is_counterclockwise_oriented()) {
        //     std::vector<_3d_vec> cen;
        //     cen.push_back(centroid);
        //     cen.push_back(points[0]);
        //     save_points_to_xyz(cen,"centroid.xyz");
        //     std::vector<_3d_vec> n_vec;
        //     n_vec.push_back(points[0]);
        //     n_vec.push_back(points[0] + plane.normal*plane_data.diameter);
        //     save_points_to_xyz(n_vec,"normal.xyz");
        //     return false;
        //     RZ_FATAL("not ccw check");
        // }
    }

    std::map<int,int>local_v_to_volume_v;
    for (const auto&q_v:volume_v_to_local_v) {
        volume_dof_to_local_dof[q_v.first] = q_v.second;
        local_v_to_volume_v[q_v.second] = q_v.first;
    }
    int nsides = points.size();
    for (int vid = 0; vid < nsides; ++vid) {
        Edge volume_edge(local_v_to_volume_v[vid],local_v_to_volume_v[(vid + 1) % nsides]);
        volume_dof_to_local_dof[edge_to_volume_dof[volume_edge]] = points.size() + vid;
    }
    return true;
}
//
// std::vector<VEM_Monomial_2d> VEM_Element_2d::project_monomial_to_plane(//这里我们考虑了3d的mono的coff
//     const VEM_Monomial_3d &m3d,
//     const _3d_vec &center3d) const // 注意：center3d 只作为常数参与计算，不用于中心化
// {
//     RZ_ASSERT(m3d.p>=0&&m3d.q>=0&&m3d.r>=0,"exponent<0");
//     std::vector<VEM_Monomial_2d> monomials;
//
//     // 与面有关的基
//
//     const Real axu = plane.u_axis.vx, axv = plane.v_axis.vx;
//     const Real ayu = plane.u_axis.vy, ayv = plane.v_axis.vy;
//     const Real azu = plane.u_axis.vz, azv = plane.v_axis.vz;
//
//     // 面上任意一点（origin）在世界坐标中的坐标
//     const _3d_vec &O = plane._3d_centroid;
//
//     // 平面上点的 3D 表达式：
//     //   P(u,v) = O + u*u_axis + v*v_axis
//     // 因此：
//     //   X = (O.vx - center3d.vx) + axu*u + axv*v
//     //   Y = (O.vy - center3d.vy) + ayu*u + ayv*v
//     //   Z = (O.vz - center3d.vz) + azu*u + azv*v
//
//     const Real a0x = O.vx - center3d.vx;
//     const Real a0y = O.vy - center3d.vy;
//     const Real a0z = O.vz - center3d.vz;
//
//     const int p = m3d.p, q = m3d.q, r = m3d.r;
//     const int deg = p + q + r;
//
//     auto add = [&](int pu, int pv, Real c) {
//         c = c * m3d.c;  // ✅ 乘上 3D monomial 的系数
//         monomials.emplace_back(pu, pv, c);
//     };
//
//     // === 0 次 ===
//     if (deg == 0) {
//         add(0, 0, 1.0);
//         return monomials;
//     }
//
//     // === 1 次 ===
//     if (deg == 1) {
//         Real c0 = 0, cu = 0, cv = 0;
//         if (p == 1) { c0 += a0x; cu += axu; cv += axv; }
//         if (q == 1) { c0 += a0y; cu += ayu; cv += ayv; }
//         if (r == 1) { c0 += a0z; cu += azu; cv += azv; }
//         add(1, 0, cu);
//         add(0, 1, cv);
//         add(0, 0, c0);
//         return monomials;
//     }
//
//     // === 2 次 ===
//     if (deg == 2) {
//         struct Aff { Real a0, au, av; };
//         auto X = Aff{a0x, axu, axv};
//         auto Y = Aff{a0y, ayu, ayv};
//         auto Z = Aff{a0z, azu, azv};
//
//         auto sq = [&](const Aff &A) {
//             add(2, 0, A.au * A.au);
//             add(1, 1, 2.0 * A.au * A.av);
//             add(0, 2, A.av * A.av);
//             add(1, 0, 2.0 * A.a0 * A.au);
//             add(0, 1, 2.0 * A.a0 * A.av);
//             add(0, 0, A.a0 * A.a0);
//         };
//
//         auto cr = [&](const Aff &A, const Aff &B) {
//             add(2, 0, A.au * B.au);
//             add(1, 1, A.au * B.av + A.av * B.au);
//             add(0, 2, A.av * B.av);
//             add(1, 0, A.a0 * B.au + B.a0 * A.au);
//             add(0, 1, A.a0 * B.av + B.a0 * A.av);
//             add(0, 0, A.a0 * B.a0);
//         };
//
//         if (p == 2) sq(X);
//         if (q == 2) sq(Y);
//         if (r == 2) sq(Z);
//         if (p == 1 && q == 1) cr(X, Y);
//         if (p == 1 && r == 1) cr(X, Z);
//         if (q == 1 && r == 1) cr(Y, Z);
//         return monomials;
//     }
//         // === 3 次 ===
//     if (deg == 3) {
//         struct Aff { Real a0, au, av; };
//         auto X = Aff{a0x, axu, axv};
//         auto Y = Aff{a0y, ayu, ayv};
//         auto Z = Aff{a0z, azu, azv};
//
//         auto cubic = [&](const Aff &A) {
//             // (a0 + au u + av v)^3
//             add(3, 0, A.au*A.au*A.au);
//             add(2, 1, 3.0*A.au*A.au*A.av);
//             add(1, 2, 3.0*A.au*A.av*A.av);
//             add(0, 3, A.av*A.av*A.av);
//
//             add(2, 0, 3.0*A.a0*A.au*A.au);
//             add(1, 1, 6.0*A.a0*A.au*A.av);
//             add(0, 2, 3.0*A.a0*A.av*A.av);
//
//             add(1, 0, 3.0*A.a0*A.a0*A.au);
//             add(0, 1, 3.0*A.a0*A.a0*A.av);
//             add(0, 0, A.a0*A.a0*A.a0);
//         };
//
//         auto cross2_1 = [&](const Aff &A, const Aff &B) {
//             // A^2 * B
//             add(3,0, A.au*A.au*B.au);
//             add(2,1, 2.0*A.au*A.av*B.au + A.au*A.au*B.av);
//             // 修正这一行：uv^2 项
//             add(1,2, A.av*A.av*B.au + 2.0*A.au*A.av*B.av);
//             add(0,3, A.av*A.av*B.av);
//
//             add(2,0, 2.0*A.a0*A.au*B.au + B.a0*A.au*A.au);
//             add(1,1, 2.0*A.a0*A.av*B.au + 2.0*A.a0*A.au*B.av + 2.0*B.a0*A.au*A.av);
//             add(0,2, 2.0*A.a0*A.av*B.av + B.a0*A.av*A.av);
//
//             add(1,0, A.a0*A.a0*B.au + 2.0*B.a0*A.a0*A.au);
//             add(0,1, A.a0*A.a0*B.av + 2.0*B.a0*A.a0*A.av);
//             add(0,0, A.a0*A.a0*B.a0);
//         };
//
//
//         auto cross1_2 = [&](const Aff &A, const Aff &B) {
//             // A * B^2
//             cross2_1(B, A); // 对称形式
//         };
//
//         auto cross_xyz = [&](const Aff &A, const Aff &B, const Aff &C) {
//             // === 三重积 (A)(B)(C) 的完整展开 ===
//             // A = a0 + au*u + av*v
//             // B = b0 + bu*u + bv*v
//             // C = c0 + cu*u + cv*v
//
//             // 方便阅读
//             Real a0=A.a0, au=A.au, av=A.av;
//             Real b0=B.a0, bu=B.au, bv=B.av;
//             Real c0=C.a0, cu=C.au, cv=C.av;
//
//             // 3次项
//             add(3,0, au*bu*cu);
//             add(2,1, au*bu*cv + au*bv*cu + av*bu*cu);
//             add(1,2, av*bv*cu + av*bu*cv + au*bv*cv);
//             add(0,3, av*bv*cv);
//
//             // 2次项（含1个常数）
//             add(2,0, a0*bu*cu + b0*au*cu + c0*au*bu);
//             add(1,1,
//                 a0*(bu*cv + bv*cu)
//               + b0*(au*cv + av*cu)
//               + c0*(au*bv + av*bu)
//             );
//             add(0,2, a0*bv*cv + b0*av*cv + c0*av*bv);
//
//             // 1次项（含2个常数）
//             add(1,0, a0*b0*cu + a0*c0*bu + b0*c0*au);
//             add(0,1, a0*b0*cv + a0*c0*bv + b0*c0*av);
//
//             // 常数项
//             add(0,0, a0*b0*c0);
//         };
//
//
//         // 单项展开
//         if (p == 3) cubic(X);
//         if (q == 3) cubic(Y);
//         if (r == 3) cubic(Z);
//         if (p == 2 && q == 1) cross2_1(X, Y);
//         if (p == 2 && r == 1) cross2_1(X, Z);
//         if (q == 2 && p == 1) cross2_1(Y, X);
//         if (q == 2 && r == 1) cross2_1(Y, Z);
//         if (r == 2 && p == 1) cross2_1(Z, X);
//         if (r == 2 && q == 1) cross2_1(Z, Y);
//         if (p == 1 && q == 1 && r == 1) cross_xyz(X, Y, Z);
//
//         return monomials;
//     }
//     RZ_FATAL("no more deg!");
//     // 更高阶可再扩展
//     return monomials;
// }

std::vector<VEM_Monomial_2d>
VEM_Element_2d::project_monomial_to_plane(
    const VEM_Monomial_3d &m3d,
    const _3d_vec &center3d) const
{
    RZ_ASSERT(m3d.p >= 0 && m3d.q >= 0 && m3d.r >= 0, "exponent < 0");

    const int p = m3d.p;
    const int q = m3d.q;
    const int r = m3d.r;
    const int deg = p + q + r;

    // 本实现仅支持到三次（与原代码一致）
    if (deg > 3) {
        RZ_FATAL("degree > 3 not supported");
    }

    // --------------------------------------------------
    // 1. 几何常量（假设 u/v 已正交归一）
    // --------------------------------------------------
    const Real axu = plane.u_axis.vx, axv = plane.v_axis.vx;
    const Real ayu = plane.u_axis.vy, ayv = plane.v_axis.vy;
    const Real azu = plane.u_axis.vz, azv = plane.v_axis.vz;

    const _3d_vec &O = plane._3d_centroid;

    const Real a0x = O.vx - center3d.vx;
    const Real a0y = O.vy - center3d.vy;
    const Real a0z = O.vz - center3d.vz;

    struct Aff { Real a0, au, av; };
    const Aff X{a0x, axu, axv};
    const Aff Y{a0y, ayu, ayv};
    const Aff Z{a0z, azu, azv};

    // --------------------------------------------------
    // 2. 系数累加器（合并同类项）
    //    deg<=3 → pu,pv ∈ [0,3]
    // --------------------------------------------------
    Real acc[4][4] = {}; // 自动初始化为 0

    auto add = [&](int pu, int pv, Real c) {
        acc[pu][pv] += c;
    };

    // --------------------------------------------------
    // 3. 展开逻辑（逐类、无遗漏）
    // --------------------------------------------------
    if (deg == 0) {
        add(0, 0, 1.0);
    }
    else if (deg == 1) {
        if (p == 1) {
            add(1, 0, X.au); add(0, 1, X.av); add(0, 0, X.a0);
        }
        if (q == 1) {
            add(1, 0, Y.au); add(0, 1, Y.av); add(0, 0, Y.a0);
        }
        if (r == 1) {
            add(1, 0, Z.au); add(0, 1, Z.av); add(0, 0, Z.a0);
        }
    }
    else if (deg == 2) {
        auto sq = [&](const Aff &A) {
            add(2,0, A.au*A.au);
            add(1,1, 2.0*A.au*A.av);
            add(0,2, A.av*A.av);
            add(1,0, 2.0*A.a0*A.au);
            add(0,1, 2.0*A.a0*A.av);
            add(0,0, A.a0*A.a0);
        };
        auto cr = [&](const Aff &A, const Aff &B) {
            add(2,0, A.au*B.au);
            add(1,1, A.au*B.av + A.av*B.au);
            add(0,2, A.av*B.av);
            add(1,0, A.a0*B.au + B.a0*A.au);
            add(0,1, A.a0*B.av + B.a0*A.av);
            add(0,0, A.a0*B.a0);
        };

        if (p == 2) sq(X);
        if (q == 2) sq(Y);
        if (r == 2) sq(Z);
        if (p == 1 && q == 1) cr(X, Y);
        if (p == 1 && r == 1) cr(X, Z);
        if (q == 1 && r == 1) cr(Y, Z);
    }
    else if (deg == 3) {
        auto cubic = [&](const Aff &A) {
            add(3,0, A.au*A.au*A.au);
            add(2,1, 3.0*A.au*A.au*A.av);
            add(1,2, 3.0*A.au*A.av*A.av);
            add(0,3, A.av*A.av*A.av);

            add(2,0, 3.0*A.a0*A.au*A.au);
            add(1,1, 6.0*A.a0*A.au*A.av);
            add(0,2, 3.0*A.a0*A.av*A.av);

            add(1,0, 3.0*A.a0*A.a0*A.au);
            add(0,1, 3.0*A.a0*A.a0*A.av);
            add(0,0, A.a0*A.a0*A.a0);
        };

        auto cross2_1 = [&](const Aff &A, const Aff &B) {
            add(3,0, A.au*A.au*B.au);
            add(2,1, 2.0*A.au*A.av*B.au + A.au*A.au*B.av);
            add(1,2, A.av*A.av*B.au + 2.0*A.au*A.av*B.av);
            add(0,3, A.av*A.av*B.av);

            add(2,0, 2.0*A.a0*A.au*B.au + B.a0*A.au*A.au);
            add(1,1, 2.0*A.a0*A.av*B.au + 2.0*A.a0*A.au*B.av + 2.0*B.a0*A.au*A.av);
            add(0,2, 2.0*A.a0*A.av*B.av + B.a0*A.av*A.av);

            add(1,0, A.a0*A.a0*B.au + 2.0*B.a0*A.a0*A.au);
            add(0,1, A.a0*A.a0*B.av + 2.0*B.a0*A.a0*A.av);
            add(0,0, A.a0*A.a0*B.a0);
        };

        auto cross_xyz = [&](const Aff &A, const Aff &B, const Aff &C) {
            add(3,0, A.au*B.au*C.au);
            add(2,1, A.au*B.au*C.av + A.au*B.av*C.au + A.av*B.au*C.au);
            add(1,2, A.av*B.av*C.au + A.av*B.au*C.av + A.au*B.av*C.av);
            add(0,3, A.av*B.av*C.av);

            add(2,0, A.a0*B.au*C.au + B.a0*A.au*C.au + C.a0*A.au*B.au);
            add(1,1,
                A.a0*(B.au*C.av + B.av*C.au) +
                B.a0*(A.au*C.av + A.av*C.au) +
                C.a0*(A.au*B.av + A.av*B.au));
            add(0,2, A.a0*B.av*C.av + B.a0*A.av*C.av + C.a0*A.av*B.av);

            add(1,0, A.a0*B.a0*C.au + A.a0*C.a0*B.au + B.a0*C.a0*A.au);
            add(0,1, A.a0*B.a0*C.av + A.a0*C.a0*B.av + B.a0*C.a0*A.av);

            add(0,0, A.a0*B.a0*C.a0);
        };

        if (p == 3) cubic(X);
        if (q == 3) cubic(Y);
        if (r == 3) cubic(Z);

        if (p == 2 && q == 1) cross2_1(X, Y);
        if (p == 2 && r == 1) cross2_1(X, Z);
        if (q == 2 && p == 1) cross2_1(Y, X);
        if (q == 2 && r == 1) cross2_1(Y, Z);
        if (r == 2 && p == 1) cross2_1(Z, X);
        if (r == 2 && q == 1) cross2_1(Z, Y);

        if (p == 1 && q == 1 && r == 1) cross_xyz(X, Y, Z);
    }

    // --------------------------------------------------
    // 4. 输出（过滤 0 项，统一乘 m3d.c）
    // --------------------------------------------------
    std::vector<VEM_Monomial_2d> monomials;
    monomials.reserve(16);

    const Real C = m3d.c;
    const Real eps = 0.0; // 若你希望数值过滤，可设为 1e-15

    for (int pu = 0; pu <= 3; ++pu) {
        for (int pv = 0; pv <= 3; ++pv) {
            Real v = acc[pu][pv];
            if (std::abs(v) > eps) {
                monomials.emplace_back(pu, pv, C * v);
            }
        }
    }

    return monomials;
}



Real VEM_Element_2d::polygon_moment_general_2d(const VEM_Monomial_2d &mono, const _2d_vec& moment_centroid, bool central) const {
        const int p = mono.p;
        const int q = mono.q;
        const Real c = mono.c;
        RZ_ASSERT(p>=0&&q>=0,"exponent<0");
        // if (p==0&&q==0) central = false;
        const int n = (int)points.size();
        if (n < 3) RZ_FATAL("n<3");

        const Real xc = moment_centroid.vx;
        const Real yc = moment_centroid.vy;

        // 根据是否中心化选择坐标
        auto X = [&](int i) {
            return (central ? (points[i].vx - xc) : points[i].vx);
        };
        auto Y = [&](int i) {
            return (central ? (points[i].vy - yc) : points[i].vy);
        };

        // 预计算二项式系数
        auto binom = [](int n, int k) -> Real {
            if (k < 0 || k > n) return 0.0;
            if (k > n - k) k = n - k;
            Real r = 1.0;
            for (int i = 1; i <= k; ++i) r = r * (n - k + i) / i;
            return r;
        };

        Real moment = 0.0;

        // === 主循环 ===
        for (int i = 0; i < n; ++i) {
            int j = (i + 1) % n;

            const Real x0 = X(i), y0 = Y(i);
            const Real x1 = X(j), y1 = Y(j);
            const Real dx = x1 - x0;
            const Real dy = y1 - y0;

            // edge1 = ∫ x^{p+1} y^q dy
            Real edge1 = 0.0;
            for (int a = 0; a <= p + 1; ++a) {
                const Real Cpa = binom(p + 1, a);
                const Real x_pow = std::pow(x0, (p + 1) - a) * std::pow(dx, a);
                for (int b = 0; b <= q; ++b) {
                    const Real Cqb = binom(q, b);
                    const Real y_pow = std::pow(y0, q - b) * std::pow(dy, b + 1);
                    edge1 += (Cpa * Cqb) * x_pow * y_pow / Real(a + b + 1);
                }
            }

            // edge2 = ∫ x^p y^{q+1} dx
            Real edge2 = 0.0;
            for (int a = 0; a <= p; ++a) {
                const Real Cpa = binom(p, a);
                const Real x_pow = std::pow(x0, p - a) * std::pow(dx, a + 1);
                for (int b = 0; b <= q + 1; ++b) {
                    const Real Cqb = binom(q + 1, b);
                    const Real y_pow = std::pow(y0, (q + 1) - b) * std::pow(dy, b);
                    edge2 += (Cpa * Cqb) * x_pow * y_pow / Real(a + b + 1);
                }
            }

            moment += (edge1 - edge2) / Real(p + q + 2);
        }
        return moment * c;
}

// Real VEM_Element_2d::polygon_moment_general_2d_advance(const VEM_Monomial_2d &mono, const _2d_vec &moment_centroid,
//     bool central) const {
//     const int p = mono.p;
//     const int q = mono.q;
//     const Real c = mono.c;
//     RZ_ASSERT(p >= 0 && q >= 0, "exponent < 0");
//
//     const int n = (int)points.size();
//     if (n < 3) RZ_FATAL("n < 3");
//
//     // ----------------------------------------------------
//     // 1. 选一个“数值安全”的尺度 h（非常关键）
//     // ----------------------------------------------------
//     // 推荐：sqrt(area)，比 max-diameter 稳定得多
//     Real h = std::sqrt(std::max(area, 1e-30));
//     if (h <= 0.0) RZ_FATAL("invalid area");
//
//     const Real xc = moment_centroid.vx;
//     const Real yc = moment_centroid.vy;
//
//     // 缩放后的坐标（O(1)）
//     auto X = [&](int i) {
//         Real x = points[i].vx;
//         return central ? (x - xc) / h : x / h;
//     };
//     auto Y = [&](int i) {
//         Real y = points[i].vy;
//         return central ? (y - yc) / h : y / h;
//     };
//
//     // ----------------------------------------------------
//     // 2. 预计算二项式系数（小阶，数值安全）
//     // ----------------------------------------------------
//     auto binom = [](int n, int k) -> Real {
//         if (k < 0 || k > n) return 0.0;
//         if (k > n - k) k = n - k;
//         Real r = 1.0;
//         for (int i = 1; i <= k; ++i)
//             r = r * (n - k + i) / i;
//         return r;
//     };
//
//     Real moment_hat = 0.0; // 在“单位尺度”下的积分
//
//     // ----------------------------------------------------
//     // 3. 主循环（数值安全版本）
//     // ----------------------------------------------------
//     for (int i = 0; i < n; ++i) {
//         int j = (i + 1) % n;
//
//         const Real x0 = X(i);
//         const Real y0 = Y(i);
//         const Real x1 = X(j);
//         const Real y1 = Y(j);
//
//         const Real dx = x1 - x0;
//         const Real dy = y1 - y0;
//
//         // 预计算幂（避免 std::pow）
//         Real x0_pow[p + 2], dx_pow[p + 2];
//         Real y0_pow[q + 2], dy_pow[q + 2];
//
//         x0_pow[0] = dx_pow[0] = 1.0;
//         for (int k = 1; k <= p + 1; ++k) {
//             x0_pow[k] = x0_pow[k - 1] * x0;
//             dx_pow[k] = dx_pow[k - 1] * dx;
//         }
//
//         y0_pow[0] = dy_pow[0] = 1.0;
//         for (int k = 1; k <= q + 1; ++k) {
//             y0_pow[k] = y0_pow[k - 1] * y0;
//             dy_pow[k] = dy_pow[k - 1] * dy;
//         }
//
//         // edge1 = ∫ x^{p+1} y^q dy
//         Real edge1 = 0.0;
//         for (int a = 0; a <= p + 1; ++a) {
//             const Real Cpa = binom(p + 1, a);
//             const Real x_part = x0_pow[p + 1 - a] * dx_pow[a];
//             for (int b = 0; b <= q; ++b) {
//                 const Real Cqb = binom(q, b);
//                 const Real y_part = y0_pow[q - b] * dy_pow[b + 1];
//                 edge1 += (Cpa * Cqb) * x_part * y_part
//                          / Real(a + b + 1);
//             }
//         }
//
//         // edge2 = ∫ x^p y^{q+1} dx
//         Real edge2 = 0.0;
//         for (int a = 0; a <= p; ++a) {
//             const Real Cpa = binom(p, a);
//             const Real x_part = x0_pow[p - a] * dx_pow[a + 1];
//             for (int b = 0; b <= q + 1; ++b) {
//                 const Real Cqb = binom(q + 1, b);
//                 const Real y_part = y0_pow[q + 1 - b] * dy_pow[b];
//                 edge2 += (Cpa * Cqb) * x_part * y_part
//                          / Real(a + b + 1);
//             }
//         }
//
//         moment_hat += (edge1 - edge2) / Real(p + q + 2);
//     }
//
//     // ----------------------------------------------------
//     // 4. 把尺度乘回来（Jacobian）
//     // ----------------------------------------------------
//     // 2D 面积分：dx dy → h^2
//     // 单项式阶数：x^p y^q → h^{p+q}
//     Real moment = moment_hat * std::pow(h, p + q + 2);
//
//     return moment * c;
// }

Real VEM_Element_2d::polygon_moment_general_2d_advance(
    const VEM_Monomial_2d &mono,
    const _2d_vec &moment_centroid,
    bool central) const
{
    const int p = mono.p;
    const int q = mono.q;
    const Real c = mono.c;
    RZ_ASSERT(p >= 0 && q >= 0, "exponent < 0");

    const int n = (int)points.size();
    if (n < 3) RZ_FATAL("n < 3");

    // ----------------------------------------------------
    // 1) 数值安全尺度 h
    // ----------------------------------------------------
    const Real h = std::sqrt(std::max<Real>(area, 1e-30));
    if (!(h > 0.0)) RZ_FATAL("invalid area");

    const Real xc = moment_centroid.vx;
    const Real yc = moment_centroid.vy;

    auto X = [&](int i) noexcept {
        const Real x = points[i].vx;
        return central ? (x - xc) / h : x / h;
    };
    auto Y = [&](int i) noexcept {
        const Real y = points[i].vy;
        return central ? (y - yc) / h : y / h;
    };

    // ----------------------------------------------------
    // 2) 预计算 binomial 行（避免内层反复算）
    //    用 long double 计算再转 Real，减小舍入误差
    // ----------------------------------------------------
    auto binom_row = [](int n, Real *out) {
        // out[0..n]
        out[0] = 1.0;
        for (int k = 1; k <= n; ++k) out[k] = 0.0;

        // Pascal 递推：C(n,k) = C(n,k-1) * (n-k+1)/k
        long double v = 1.0L;
        out[0] = 1.0;
        for (int k = 1; k <= n; ++k) {
            v = v * (long double)(n - k + 1) / (long double)k;
            out[k] = (Real)v;
        }
    };

    // 为了鲁棒：栈上优先，若阶过大则回退到 vector（保证功能正确）
    constexpr int MAX_STACK = 16; // 对 VEM(k<=6) 绰绰有余；也兼顾极端情况
    std::vector<Real> Cp_vec, Cp1_vec, Cq_vec, Cq1_vec;

    Real Cp_stack[MAX_STACK], Cp1_stack[MAX_STACK], Cq_stack[MAX_STACK], Cq1_stack[MAX_STACK];
    Real *Cp  = nullptr;
    Real *Cp1 = nullptr;
    Real *Cq  = nullptr;
    Real *Cq1 = nullptr;

    auto alloc_row = [&](int len, Real *stack_buf, std::vector<Real> &heap_buf) -> Real* {
        // len = n+1
        if (len <= MAX_STACK) return stack_buf;
        heap_buf.assign(len, 0.0);
        return heap_buf.data();
    };

    Cp  = alloc_row(p + 1, Cp_stack,  Cp_vec);     // C(p, a), a=0..p
    Cp1 = alloc_row(p + 2, Cp1_stack, Cp1_vec);    // C(p+1, a), a=0..p+1
    Cq  = alloc_row(q + 1, Cq_stack,  Cq_vec);     // C(q, b), b=0..q
    Cq1 = alloc_row(q + 2, Cq1_stack, Cq1_vec);    // C(q+1, b), b=0..q+1

    binom_row(p,   Cp);
    binom_row(p+1, Cp1);
    binom_row(q,   Cq);
    binom_row(q+1, Cq1);

    // ----------------------------------------------------
    // 3) 预计算 1/(a+b+1) （减少除法）
    //    最大 a+b+1：edge1里 a<=p+1, b<=q -> p+q+2
    //               edge2里 a<=p, b<=q+1 -> p+q+2
    // ----------------------------------------------------
    const int invN = p + q + 2; // 我们需要 inv[1..invN]
    std::vector<Real> inv;
    inv.resize(invN + 1);
    inv[0] = 0.0; // 不用
    for (int k = 1; k <= invN; ++k) inv[k] = 1.0 / (Real)k;

    Real moment_hat = 0.0;

    // ----------------------------------------------------
    // 4) 主循环：对每条边
    // ----------------------------------------------------
    // 幂表也采用“栈优先，超阶回退 vector”，避免 VLA 且兼顾鲁棒
    std::vector<Real> x0_pow_v, dx_pow_v, y0_pow_v, dy_pow_v;

    // 需要长度：x 方向到 p+1（用于 edge1），dx 也到 p+1；数组长度 p+2（含0）
    // y 方向到 q+1；数组长度 q+2
    Real x0_pow_s[MAX_STACK], dx_pow_s[MAX_STACK];
    Real y0_pow_s[MAX_STACK], dy_pow_s[MAX_STACK];

    auto alloc_pow = [&](int len, Real *stack_buf, std::vector<Real> &heap_buf) -> Real* {
        if (len <= MAX_STACK) return stack_buf;
        heap_buf.assign(len, 0.0);
        return heap_buf.data();
    };

    const int xLen = p + 2; // 0..p+1
    const int yLen = q + 2; // 0..q+1

    for (int i = 0; i < n; ++i) {
        const int j = (i + 1) % n;

        const Real x0 = X(i);
        const Real y0 = Y(i);
        const Real x1 = X(j);
        const Real y1 = Y(j);

        const Real dx = x1 - x0;
        const Real dy = y1 - y0;

        Real *x0_pow = alloc_pow(xLen, x0_pow_s, x0_pow_v);
        Real *dx_pow = alloc_pow(xLen, dx_pow_s, dx_pow_v);
        Real *y0_pow = alloc_pow(yLen, y0_pow_s, y0_pow_v);
        Real *dy_pow = alloc_pow(yLen, dy_pow_s, dy_pow_v);

        // 递推幂表
        x0_pow[0] = 1.0; dx_pow[0] = 1.0;
        for (int k = 1; k <= p + 1; ++k) {
            x0_pow[k] = x0_pow[k - 1] * x0;
            dx_pow[k] = dx_pow[k - 1] * dx;
        }

        y0_pow[0] = 1.0; dy_pow[0] = 1.0;
        for (int k = 1; k <= q + 1; ++k) {
            y0_pow[k] = y0_pow[k - 1] * y0;
            dy_pow[k] = dy_pow[k - 1] * dy;
        }

        // edge1 = ∫ x^{p+1} y^q dy
        Real edge1 = 0.0;
        for (int a = 0; a <= p + 1; ++a) {
            const Real Cpa = Cp1[a]; // C(p+1,a)
            const Real x_part = x0_pow[p + 1 - a] * dx_pow[a];

            for (int b = 0; b <= q; ++b) {
                const Real Cqb = Cq[b]; // C(q,b)
                const Real y_part = y0_pow[q - b] * dy_pow[b + 1];

                // / (a+b+1)
                edge1 += (Cpa * Cqb) * x_part * y_part * inv[a + b + 1];
            }
        }

        // edge2 = ∫ x^p y^{q+1} dx
        Real edge2 = 0.0;
        for (int a = 0; a <= p; ++a) {
            const Real Cpa = Cp[a]; // C(p,a)
            const Real x_part = x0_pow[p - a] * dx_pow[a + 1];

            for (int b = 0; b <= q + 1; ++b) {
                const Real Cqb = Cq1[b]; // C(q+1,b)
                const Real y_part = y0_pow[q + 1 - b] * dy_pow[b];

                edge2 += (Cpa * Cqb) * x_part * y_part * inv[a + b + 1];
            }
        }

        moment_hat += (edge1 - edge2) / Real(p + q + 2);
    }

    // ----------------------------------------------------
    // 5) 乘回尺度：moment = moment_hat * h^{p+q+2}
    //    避免 std::pow：整数幂递推更稳更快
    // ----------------------------------------------------
    const int hp = p + q + 2;
    Real hpow = 1.0;
    for (int k = 0; k < hp; ++k) hpow *= h;

    const Real moment = moment_hat * hpow;
    return moment * c;
}


Real VEM_Element_2d::polygon_moment_2d_sub_poly(const VEM_Monomial_2d &mono, const _2d_vec &moment_centroid,
                                                  const std::vector<_2d_vec> &points, bool central) const {
    const int p = mono.p;
        const int q = mono.q;
        const Real c = mono.c;
        RZ_ASSERT(p>=0&&q>=0,"exponent<0");
        // if (p==0&&q==0) central = false;
        const int n = (int)points.size();
        if (n < 3) RZ_FATAL("n<3");

        const Real xc = moment_centroid.vx;
        const Real yc = moment_centroid.vy;

        // 根据是否中心化选择坐标
        auto X = [&](int i) {
            return (central ? (points[i].vx - xc) : points[i].vx);
        };
        auto Y = [&](int i) {
            return (central ? (points[i].vy - yc) : points[i].vy);
        };

        // 预计算二项式系数
        auto binom = [](int n, int k) -> Real {
            if (k < 0 || k > n) return 0.0;
            if (k > n - k) k = n - k;
            Real r = 1.0;
            for (int i = 1; i <= k; ++i) r = r * (n - k + i) / i;
            return r;
        };

        Real moment = 0.0;

        // === 主循环 ===
        for (int i = 0; i < n; ++i) {
            int j = (i + 1) % n;

            const Real x0 = X(i), y0 = Y(i);
            const Real x1 = X(j), y1 = Y(j);
            const Real dx = x1 - x0;
            const Real dy = y1 - y0;

            // edge1 = ∫ x^{p+1} y^q dy
            Real edge1 = 0.0;
            for (int a = 0; a <= p + 1; ++a) {
                const Real Cpa = binom(p + 1, a);
                const Real x_pow = std::pow(x0, (p + 1) - a) * std::pow(dx, a);
                for (int b = 0; b <= q; ++b) {
                    const Real Cqb = binom(q, b);
                    const Real y_pow = std::pow(y0, q - b) * std::pow(dy, b + 1);
                    edge1 += (Cpa * Cqb) * x_pow * y_pow / Real(a + b + 1);
                }
            }

            // edge2 = ∫ x^p y^{q+1} dx
            Real edge2 = 0.0;
            for (int a = 0; a <= p; ++a) {
                const Real Cpa = binom(p, a);
                const Real x_pow = std::pow(x0, p - a) * std::pow(dx, a + 1);
                for (int b = 0; b <= q + 1; ++b) {
                    const Real Cqb = binom(q + 1, b);
                    const Real y_pow = std::pow(y0, (q + 1) - b) * std::pow(dy, b);
                    edge2 += (Cpa * Cqb) * x_pow * y_pow / Real(a + b + 1);
                }
            }

            moment += (edge1 - edge2) / Real(p + q + 2);
        }
        return moment * c;
}
//注意这里没有除 diameter
Real VEM_Element_2d::polygon_moment_general_3d(const VEM_Monomial_3d &mono,const _3d_vec& momcentroid,bool central)const {
    _2d_vec centroid2d(0,0);
    auto projected_monomials2d =
        this->project_monomial_to_plane(mono,momcentroid);
    Real moment = 0.0;
    for (const auto& mono2d:projected_monomials2d) {
        moment += polygon_moment_general_2d_advance(mono2d,centroid2d,false);
    }
    return moment;
}

// std::vector<VEM_Monomial_2d> VEM_Element_2d::expand_centered_monomial(const VEM_Monomial_2d& mono) const {
//     std::vector<VEM_Monomial_2d> result;
//
//     const Real xc = centroid.vx;
//     const Real yc = centroid.vy;
//
//     // 根据二项式展开：
//     // (x - xc)^p = sum_i C(p,i) * x^i * (-xc)^(p-i)
//     // (y - yc)^q = sum_j C(q,j) * y^j * (-yc)^(q-j)
//     for (int i = 0; i <= mono.p; ++i) {
//         Real cx = std::pow(-xc, mono.p - i) *
//                     std::tgamma(mono.p + 1) /
//                     (std::tgamma(i + 1) * std::tgamma(mono.p - i + 1)); // C(p,i)
//
//         for (int j = 0; j <= mono.q; ++j) {
//             Real cy = std::pow(-yc, mono.q - j) *
//                         std::tgamma(mono.q + 1) /
//                         (std::tgamma(j + 1) * std::tgamma(mono.q - j + 1)); // C(q,j)
//
//             Real coeff = mono.c * cx * cy;
//             result.emplace_back(i, j, coeff);
//         }
//     }
//     return result;
// }
std::vector<VEM_Monomial_2d>
VEM_Element_2d::expand_centered_monomial(const VEM_Monomial_2d& mono) const
{
    std::vector<VEM_Monomial_2d> result;

    const int p = mono.p;
    const int q = mono.q;
    const Real c0 = mono.c;

    RZ_ASSERT(p >= 0 && q >= 0, "exponent < 0");

    // 你原函数用的是 centroid 成员；保持一致（不改接口/语义）
    const Real xc = centroid.vx;
    const Real yc = centroid.vy;

    // (p+1)(q+1) 项
    result.reserve(static_cast<size_t>(p + 1) * static_cast<size_t>(q + 1));

    // 小整数二项式系数：比 tgamma 稳定且快
    auto binom = [](int n, int k) -> Real {
        if (k < 0 || k > n) return 0.0;
        if (k > n - k) k = n - k;
        long double r = 1.0L;
        for (int i = 1; i <= k; ++i) {
            r = r * static_cast<long double>(n - k + i) / static_cast<long double>(i);
        }
        return static_cast<Real>(r);
    };

    // 预计算 (-xc)^k, (-yc)^k，避免 std::pow
    // 注意：这是物理坐标的幂，因此数学语义完全保持
    std::vector<Real> xpow(p + 1), ypow(q + 1);
    xpow[0] = 1.0;
    for (int k = 1; k <= p; ++k) xpow[k] = xpow[k - 1] * (-xc);

    ypow[0] = 1.0;
    for (int k = 1; k <= q; ++k) ypow[k] = ypow[k - 1] * (-yc);

    // 展开：
    // (x-xc)^p = sum_i C(p,i) x^i (-xc)^(p-i)
    // (y-yc)^q = sum_j C(q,j) y^j (-yc)^(q-j)
    for (int i = 0; i <= p; ++i) {
        const Real cx = binom(p, i) * xpow[p - i];
        for (int j = 0; j <= q; ++j) {
            const Real cy = binom(q, j) * ypow[q - j];
            const Real coeff = c0 * cx * cy;
            result.emplace_back(i, j, coeff);
        }
    }

    return result;
}


// std::vector<VEM_Monomial_2d> VEM_Element_2d::expand_centered_monomial(const VEM_Monomial_2d& mono) const {
//     static std::vector<Real> factorial_p(5, 1.0); // p 最大值为 4，因此长度为 5
//     static std::vector<Real> factorial_q(5, 1.0); // q 最大值为 4，因此长度为 5
//     static bool factorial_initialized = false;
//
//     const Real xc = centroid.vx;
//     const Real yc = centroid.vy;
//
//     // 预计算 (-xc)^i 和 (-yc)^j
//     std::vector<Real> x_powers(mono.p + 1, 1.0);
//     std::vector<Real> y_powers(mono.q + 1, 1.0);
//
//     // 计算 (-xc)^i 和 (-yc)^j
//     for (int i = 1; i <= mono.p; ++i) {
//         x_powers[i] = x_powers[i - 1] * (-xc);
//     }
//
//     for (int j = 1; j <= mono.q; ++j) {
//         y_powers[j] = y_powers[j - 1] * (-yc);
//     }
//
//     // 仅在第一次调用时初始化阶乘
//     if (!factorial_initialized) {
//         for (int i = 1; i <= 4; ++i) {
//             factorial_p[i] = factorial_p[i - 1] * i;
//             factorial_q[i] = factorial_q[i - 1] * i;
//         }
//         factorial_initialized = true;
//     }
//
//     // 预分配空间，避免多次分配
//     std::vector<VEM_Monomial_2d> result;
//     result.reserve((mono.p + 1) * (mono.q + 1));
//
//     // 计算 C(p, i) 和 C(q, j)，并且直接计算系数
//     for (int i = 0; i <= mono.p; ++i) {
//         Real cx = std::pow(-xc, mono.p - i) * factorial_p[mono.p] / (factorial_p[i] * factorial_p[mono.p - i]); // C(p,i)
//         for (int j = 0; j <= mono.q; ++j) {
//             Real cy = std::pow(-yc, mono.q - j) * factorial_q[mono.q] / (factorial_q[j] * factorial_q[mono.q - j]); // C(q,j)
//             Real coeff = mono.c * cx * cy;
//             result.emplace_back(i, j, coeff);
//         }
//     }
//
//     return result;
// }




void VEM_Domain::read_from_file(std::string filename) {
    mat_t *matfp = Mat_Open(filename.c_str(), MAT_ACC_RDONLY);
    if (!matfp) {
        std::cerr << "cannot open mat file\n";
    }
    // 读取变量 vertices
    matvar_t *matvar = Mat_VarRead(matfp, "vertices");
    if (matvar) {
        size_t nrows = matvar->dims[0];
        size_t ncols = matvar->dims[1];
        Real *data = static_cast<Real *>(matvar->data);
        std::cout << "vertices size: " << nrows << " x " << ncols << "\n";
        for (size_t i = 0; i < nrows; i++) {
            points.push_back({(Real)data[i + 0*nrows],(Real)data[i + 1*nrows]});
            // points.push_back({(Real) data[i + 1 * nrows], (Real) data[i + 0 * nrows]}); // for uniform
            std::cout << data[i + 0 * nrows] << " "
                    << data[i + 1 * nrows] << "\n"; // MATLAB 列优先
        }
        Mat_VarFree(matvar);
    }
    // --------- elements (cell array) ----------
    matvar = Mat_VarRead(matfp, "elements");
    if (matvar && matvar->class_type == MAT_C_CELL) {
        size_t n = matvar->nbytes / sizeof(matvar_t *);
        std::cout << "elements count = " << n << "\n";
        polygons.reserve(n);
        for (size_t i = 0; i < n; i++) {
            matvar_t *cell = static_cast<matvar_t **>(matvar->data)[i];
            std::vector<int> elem = toIntVector(cell, true); // -1 转成 0-based
            polygons.push_back(elem);
        }

        // 打印
        for (auto &e: polygons) {
            for (int v: e) std::cout << v << " ";
            std::cout << "\n";
        }
        Mat_VarFree(matvar);
    }
    is_border.resize(points.size(), false);
    // --------- boundary ----------
    matvar = Mat_VarRead(matfp, "boundary");
    if (matvar) {
        std::vector<int> boundary = toIntVector(matvar, true);
        std::cout << "boundary: ";
        for (int b: boundary) {
            is_border[b] = true;
            std::cout << b << " ";
        }
        std::cout << "\n";
        Mat_VarFree(matvar);
    }

    Mat_Close(matfp);

    // plot_mesh();
}

void VEM_Domain::init() {
    for (int pid = 0; pid < polygons.size(); pid++) {
        int n = polygons[pid].size();
        for (int i = 0; i < n; i++) {
            int a = polygons[pid][i];
            int b = polygons[pid][(i + 1) % n];
            Edge edge(a, b);
            temp_unique_edges.insert(edge);
        }
    }
    for (auto &edge: temp_unique_edges) {
        edgetoid_map[edge] = edges.size();
        edgetoid_map[{edge.v2, edge.v1}] = edges.size();
        edges.push_back(edge);
    }

    int Ndofs = points.size() + edges.size() + polygons.size();
    int Ndofs_k1 = points.size();
    global_stiffness_matrix.resize(Ndofs, Ndofs);
    global_stiffness_matrix.setZero();
    global_stiffness_matrix_k1.resize(Ndofs_k1, Ndofs_k1);
    global_stiffness_matrix_k1.setZero();
    global_LX_matrix.resize(Ndofs, Ndofs_k1);
    global_LY_matrix.resize(Ndofs, Ndofs_k1);
    global_LX_matrix.setZero();
    global_LY_matrix.setZero();
    global_rhs.resize(Ndofs);
    global_rhs.setZero();
    global_rhs_k1.resize(Ndofs_k1);
    global_rhs_k1.setZero();
    k1_dof_u.resize(Ndofs_k1);
    k1_dof_u.setZero();
    k1_dof_v.resize(Ndofs_k1);
    k1_dof_v.setZero();
    global_p.resize(Ndofs);
    global_p.setZero();
    monomials = {
        VEM_Monomial_2d(0, 0),
        VEM_Monomial_2d(1, 0),
        VEM_Monomial_2d(0, 1),
        VEM_Monomial_2d(2, 0),
        VEM_Monomial_2d(1, 1),
        VEM_Monomial_2d(0, 2)
    };
}

void VEM_Domain::generate_elements() {
    vem_elements.clear();
    vem_elements.reserve(polygons.size());
    for (const auto &vert_ids: polygons) {
        VEM_Element_2d elem;
        elem.local_to_global_dof= vert_ids;
        size_t n_sides = vert_ids.size();
        for (auto &global_vid: vert_ids) {
            elem.points.push_back(points[global_vid]);
        }
        if (n_sides < 3) continue; // 至少三角形
        elem.build_up();
        vem_elements.push_back(std::move(elem));
    }
}

void VEM_Domain::assemble_global_stiffness_matrix() {
    using namespace Eigen;
    RZ_DEBUG_INFO();
    int n_vertices = (int) points.size();
    int n_edges = (int) edges.size();
    int n_elements = (int) vem_elements.size();

    int total_dofs = n_vertices + n_edges + n_elements;

    // 先准备 triplet
    std::vector<Triplet<GlobalReal> > triplets;
    std::vector<Triplet<GlobalReal> > triplets_LX;
    std::vector<Triplet<GlobalReal> > triplets_LY;
    for (int el_id = 0; el_id < n_elements; ++el_id) {
        const VEM_Element_2d &elem = vem_elements[el_id];

        int n_sides = (int) elem.points.size();

        // 构造 element_to_global_dof
        std::vector<int> element_to_global_dof;

        // 顶点 dof
        for (int vid = 0; vid < n_sides; ++vid) {
            element_to_global_dof.push_back(elem.local_to_global_dof[vid]);
        }

        // 边 dof
        for (int vid = 0; vid < n_sides; ++vid) {
            int a = elem.local_to_global_dof[vid];
            int b = elem.local_to_global_dof[(vid + 1) % n_sides];

            // ⚠️ 需要有 edge_map (pair<int,int> -> edge_id)
            // 可以预先在 init() 里构建好
            int edge_id = edgetoid_map[{a, b}];

            element_to_global_dof.push_back(n_vertices + edge_id);
        }

        // 内部 dof
        element_to_global_dof.push_back(n_vertices + n_edges + el_id);

        // --------------------------
        // 将局部矩阵加到全局矩阵
        // --------------------------
        const MatrixXr &Ke = elem.stiffness_matrix;
        int ndofs = (int) element_to_global_dof.size();
        // assemble L
        for (int i = 0; i < ndofs; ++i) {
            for (int j = 0; j < n_sides; ++j) {
                Real val_Lx = elem.local_LX(i, j);
                int gi = element_to_global_dof[i];
                int gj = element_to_global_dof[j];
                if (std::fabs(val_Lx) > 1e-14) {
                    triplets_LX.emplace_back(gi, gj, val_Lx);
                }
                Real val_Ly = elem.local_LY(i, j);
                if (std::fabs(val_Ly) > 1e-14) {
                    triplets_LY.emplace_back(gi, gj, val_Ly);
                }
            }
        }
        for (int i = 0; i < ndofs; ++i) {
            for (int j = 0; j < ndofs; ++j) {
                int gi = element_to_global_dof[i];
                int gj = element_to_global_dof[j];
                Real val = Ke(i, j);
                if (std::fabs(val) > 1e-14) {
                    triplets.emplace_back(gi, gj, val);
                }
            }
        }
    }

    global_stiffness_matrix.setFromTriplets(triplets.begin(), triplets.end());
    global_LX_matrix.setFromTriplets(triplets_LX.begin(), triplets_LX.end());
    global_LY_matrix.setFromTriplets(triplets_LY.begin(), triplets_LY.end());
    RZ_DEBUG_INFO();
}

void VEM_Domain::assemble_global_stiffness_matrix_k1() {
    using namespace Eigen;
    RZ_DEBUG_INFO();
    int n_vertices = (int) points.size();
    int n_edges = (int) edges.size();
    int n_elements = (int) vem_elements.size();

    int total_dofs = n_vertices + n_edges + n_elements;

    // 先准备 triplet
    std::vector<Triplet<GlobalReal> > triplets;
    for (int el_id = 0; el_id < n_elements; ++el_id) {
        const VEM_Element_2d &elem = vem_elements[el_id];
        int n_sides = (int) elem.points.size();

        // 构造 element_to_global_dof
        std::vector<int> element_to_global_dof;

        // 顶点 dof
        for (int vid = 0; vid < n_sides; ++vid) {
            element_to_global_dof.push_back(elem.local_to_global_dof[vid]);
        }

        // --------------------------
        // 将局部矩阵加到全局矩阵
        // --------------------------
        const MatrixXr &Ke = elem.stiffness_matrix_k1;
        int ndofs = (int) element_to_global_dof.size();

        for (int i = 0; i < ndofs; ++i) {
            for (int j = 0; j < ndofs; ++j) {
                int gi = element_to_global_dof[i];
                int gj = element_to_global_dof[j];
                Real val = Ke(i, j);
                if (std::fabs(val) > 1e-14) {
                    triplets.emplace_back(gi, gj, val);
                }
            }
        }
    }

    global_stiffness_matrix_k1.setFromTriplets(triplets.begin(), triplets.end());
}

void VEM_Domain::assemble_global_rhs() {
    int n_vertices = (int) points.size();
    int n_edges = (int) edges.size();
    int n_elements = (int) vem_elements.size();

    int total_dofs = n_vertices + n_edges + n_elements;
    global_rhs = Eigen::VectorXr::Zero(total_dofs);

    for (int el_id = 0; el_id < n_elements; ++el_id) {
        const VEM_Element_2d &elem = vem_elements[el_id];
        int n_sides = (int) elem.points.size();
        Real area = elem.area;
        _2d_vec centroid = elem.centroid;

        int per_element_dof = n_sides ;
        // // ---------------------
        // // 顶点自由度
        // // ---------------------
        for (int vid = 0; vid < n_sides; ++vid) {
            int global_vid = elem.local_to_global_dof[vid];
            global_rhs[global_vid] += square_domain_rhs(centroid) * (area / per_element_dof);
        }
        //
        // // ---------------------
        // // 边自由度
        // // ---------------------
        // for (int vid = 0; vid < n_sides; ++vid) {
        //     int a = elem.local_to_global_dof[vid];
        //     int b = elem.local_to_global_dof[(vid + 1) % n_sides];
        //     Edge e(a, b);
        //
        //     int edge_id = edgetoid_map[e]; // ⚠️ 你需要在 VEM_Domain 里维护 edge_map
        //     int global_edge_dof = n_vertices + edge_id;
        //
        //     global_rhs[global_edge_dof] += square_domain_rhs(centroid) * (area / per_element_dof);
        // }

        // // ---------------------
        // // 单元内部自由度
        // // ---------------------
        // int global_elem_dof = n_vertices + n_edges + el_id;
        // global_rhs[global_elem_dof] += square_domain_rhs(centroid)*area;
    }
}

void VEM_Domain::assemble_global_rhs_k1() {
    int n_vertices = (int) points.size();
    int n_edges = (int) edges.size();
    int n_elements = (int) vem_elements.size();
    int total_dofs = n_vertices;
    global_rhs_k1 = Eigen::VectorXr::Zero(total_dofs);

    for (int el_id = 0; el_id < n_elements; ++el_id) {
        const VEM_Element_2d &elem = vem_elements[el_id];
        int n_sides = (int) elem.points.size();
        Real area = elem.area;
        _2d_vec centroid = elem.centroid;
        int per_element_dof = n_sides;
        // ---------------------
        // 顶点自由度
        // ---------------------
        for (int vid = 0; vid < n_sides; ++vid) {
            int global_vid = elem.local_to_global_dof[vid];
            global_rhs_k1[global_vid] += square_domain_rhs(centroid) * (area / per_element_dof);
        }
    }
}

void VEM_Domain::assemble_weakform_rhs(Real dt) {
    RZ_DEBUG_INFO();
    // 检查维度是否匹配
    if (global_LX_matrix.cols() != k1_dof_u.size()) {
        RZ_FATAL("Dimension mismatch: LX.cols() = ", global_LX_matrix.cols(),
                 " but k1_dof_u.size() = ", k1_dof_u.size());
    }

    if (global_LY_matrix.cols() != k1_dof_v.size()) {
        RZ_FATAL("Dimension mismatch: LY.cols() = ", global_LY_matrix.cols(),
                 " but k1_dof_v.size() = ", k1_dof_v.size());
    }

    if (global_LX_matrix.rows() != global_LY_matrix.rows()) {
        RZ_FATAL("Dimension mismatch: LX.rows() = ", global_LX_matrix.rows(),
                 " but LY.rows() = ", global_LY_matrix.rows());
    }
    global_rhs =
        (global_LX_matrix * k1_dof_u.template cast<GlobalReal>()).template cast<Real>()
        + (global_LY_matrix * k1_dof_v.template cast<GlobalReal>()).template cast<Real>();
    global_rhs /= dt;
}

Eigen::VectorXr boundary_condition(const std::vector<_2d_vec> &pts) {
    Eigen::VectorXr vals(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) {
        // 例如：u(x,y) = 0 边界条件（可换成真实函数）
        vals[i] = 0.0;
    }
    return vals;
}

void VEM_Domain::solve() {
    int n_vertices = (int) points.size();
    int n_edges = (int) edges.size();
    int n_elements = (int) vem_elements.size();
    int total_dofs = n_vertices + n_edges + n_elements;

    // -----------------------------
    // 1. 找 boundary dofs
    // -----------------------------
    std::vector<int> boundary_dofs;
    std::vector<_2d_vec> boundary_query_points;

    // 顶点边界 dofs
    for (int v = 0; v < n_vertices; ++v) {
        if (is_border[v]) {
            boundary_dofs.push_back(v);
            boundary_query_points.push_back(points[v]);
        }
    }

    // 边界边 dofs
    for (int e_id = 0; e_id < n_edges; ++e_id) {
        const Edge &e = edges[e_id];
        if (is_border[e.v1] && is_border[e.v2]) {
            boundary_dofs.push_back(n_vertices + e_id);
            _2d_vec mid((points[e.v1].vx + points[e.v2].vx) * 0.5,
                        (points[e.v1].vy + points[e.v2].vy) * 0.5);
            boundary_query_points.push_back(mid);
        }
    }

    Eigen::VectorXr boundary_vals = boundary_condition(boundary_query_points);

    // -----------------------------
    // 2. 找 internal dofs
    // -----------------------------
    std::vector<int> internal_dofs;

    // 非边界顶点
    for (int v = 0; v < n_vertices; ++v) {
        if (!is_border[v]) {
            internal_dofs.push_back(v);
        }
    }

    // 非纯边界边
    for (int e_id = 0; e_id < n_edges; ++e_id) {
        const Edge &e = edges[e_id];
        if (!(is_border[e.v1] && is_border[e.v2])) {
            internal_dofs.push_back(n_vertices + e_id);
        }
    }

    // 全部单元 dofs
    for (int el = 0; el < n_elements; ++el) {
        internal_dofs.push_back(n_vertices + n_edges + el);
    }

    int num_idof = (int) internal_dofs.size();

    // -----------------------------
    // 3. 修改 RHS: F = F - K[:, boundary_dofs] * boundary_vals
    // -----------------------------
    // Eigen::VectorXr F = global_rhs;
    // Eigen::MatrixXr Kdense = Eigen::MatrixXr(global_stiffness_matrix);
    // Eigen::MatrixXr K_bc = Kdense(Eigen::all, Eigen::VectorXi::Map(boundary_dofs.data(), boundary_dofs.size()));
    // F -= K_bc * boundary_vals;
    // 以下是以上矩阵模式的稀疏展开
    Eigen::VectorXr F = global_rhs;
    Eigen::VectorXr Fb = Eigen::VectorXr::Zero(F.size());
    for (int j = 0; j < (int) boundary_dofs.size(); ++j) {
        int col = boundary_dofs[j];
        Real val = boundary_vals[j];
        for (Eigen::SparseMatrixCr::InnerIterator it(global_stiffness_matrix, col); it; ++it) {
            Fb[it.row()] += it.value() * val;
        }
    }
    F -= Fb;

    // -----------------------------
    // 4. 提取内部子系统 K_II
    // -----------------------------
    Eigen::VectorXi idofs = Eigen::VectorXi::Map(internal_dofs.data(), num_idof);
    // Eigen::MatrixXr K_II = Kdense(idofs, idofs);
    const int nI = idofs.size();
    std::unordered_map<int, int> g2l;
    g2l.reserve(nI * 2);
    for (int i = 0; i < nI; ++i) g2l[idofs[i]] = i;
    Eigen::SparseMatrixR Krow = global_stiffness_matrix;
    Eigen::SparseMatrixCr K_II(nI, nI);
    std::vector<Eigen::Triplet<GlobalReal> > trips;
    trips.reserve(static_cast<size_t>(Krow.nonZeros() * 0.3)); // 估个上界，避免频繁扩容
    for (int ii = 0; ii < nI; ++ii) {
        int gi = idofs[ii]; // 全局行
        for (Eigen::SparseMatrixR::InnerIterator it(Krow, gi); it; ++it) {
            int gj = it.col(); // 全局列
            auto found = g2l.find(gj); // 列也必须是内部自由度
            if (found != g2l.end()) {
                int jj = found->second; // 局部列
                trips.emplace_back(ii, jj, it.value());
            }
        }
    }
    K_II.setFromTriplets(trips.begin(), trips.end());
    K_II.makeCompressed();

    Eigen::VectorXr F_I(num_idof);
    for (int i = 0; i < num_idof; ++i) F_I[i] = F[internal_dofs[i]];
    // -----------------------------
    // 5. 解 K_II u_I = F_I
    // -----------------------------
    Eigen::VectorXr u = Eigen::VectorXr::Zero(total_dofs);
    Eigen::VectorXr u_I = amgclSolve(K_II, F_I, 1e-3, -1, 10000);
    // Eigen::VectorXr u_I = K_II.llt().solve(F_I);

    for (int i = 0; i < num_idof; ++i) {
        u[internal_dofs[i]] = u_I[i];
    }
    for (size_t i = 0; i < boundary_dofs.size(); ++i) {
        u[boundary_dofs[i]] = boundary_vals[i];
    }

    // 存储到 global_p（解向量）
    global_p = u;
}

void VEM_Domain::solve_k1() {
    int n_vertices = (int) points.size();
    int total_dofs = n_vertices;

    // -----------------------------
    // 1. 找 boundary dofs
    // -----------------------------
    std::vector<int> boundary_dofs;
    std::vector<_2d_vec> boundary_query_points;

    // 顶点边界 dofs
    for (int v = 0; v < n_vertices; ++v) {
        if (is_border[v]) {
            boundary_dofs.push_back(v);
            boundary_query_points.push_back(points[v]);
        }
    }

    Eigen::VectorXr boundary_vals = boundary_condition(boundary_query_points);

    // -----------------------------
    // 2. 找 internal dofs
    // -----------------------------
    std::vector<int> internal_dofs;

    // 非边界顶点
    for (int v = 0; v < n_vertices; ++v) {
        if (!is_border[v]) {
            internal_dofs.push_back(v);
        }
    }

    int num_idof = (int) internal_dofs.size();

    // -----------------------------
    // 3. 修改 RHS: F = F - K[:, boundary_dofs] * boundary_vals
    // -----------------------------
    // Eigen::VectorXr F = global_rhs_k1;
    // Eigen::MatrixXr Kdense = Eigen::MatrixXr(global_stiffness_matrix_k1);
    // Eigen::MatrixXr K_bc = Kdense(Eigen::all, Eigen::VectorXi::Map(boundary_dofs.data(), boundary_dofs.size()));
    // F -= K_bc * boundary_vals;
    // 以下是以上矩阵模式的稀疏展开
    Eigen::VectorXr F = global_rhs_k1;
    Eigen::VectorXr Fb = Eigen::VectorXr::Zero(F.size());
    for (int j = 0; j < (int) boundary_dofs.size(); ++j) {
        int col = boundary_dofs[j];
        Real val = boundary_vals[j];
        for (Eigen::SparseMatrixCr::InnerIterator it(global_stiffness_matrix_k1, col); it; ++it) {
            Fb[it.row()] += it.value() * val;
        }
    }
    F -= Fb;

    // -----------------------------
    // 4. 提取内部子系统 K_II
    // -----------------------------
    Eigen::VectorXi idofs = Eigen::VectorXi::Map(internal_dofs.data(), num_idof);
    // Eigen::MatrixXr K_II = Kdense(idofs, idofs);
    const int nI = idofs.size();
    std::unordered_map<int, int> g2l;
    g2l.reserve(nI * 2);
    for (int i = 0; i < nI; ++i) g2l[idofs[i]] = i;
    Eigen::SparseMatrixR Krow = global_stiffness_matrix_k1;
    Eigen::SparseMatrixCr K_II(nI, nI);
    std::vector<Eigen::Triplet<GlobalReal> > trips;
    trips.reserve(static_cast<size_t>(Krow.nonZeros() * 0.3)); // 估个上界，避免频繁扩容
    for (int ii = 0; ii < nI; ++ii) {
        int gi = idofs[ii]; // 全局行
        for (Eigen::SparseMatrixR::InnerIterator it(Krow, gi); it; ++it) {
            int gj = it.col(); // 全局列
            auto found = g2l.find(gj); // 列也必须是内部自由度
            if (found != g2l.end()) {
                int jj = found->second; // 局部列
                trips.emplace_back(ii, jj, it.value());
            }
        }
    }
    K_II.setFromTriplets(trips.begin(), trips.end());
    K_II.makeCompressed();

    Eigen::VectorXr F_I(num_idof);
    for (int i = 0; i < num_idof; ++i) F_I[i] = F[internal_dofs[i]];
    // -----------------------------
    // 5. 解 K_II u_I = F_I
    // -----------------------------
    Eigen::VectorXr u = Eigen::VectorXr::Zero(total_dofs);
    Eigen::VectorXr u_I = amgclSolve(K_II, F_I, 1e-3, -1, 10000);
    // Eigen::VectorXr u_I = K_II.llt().solve(F_I);

    for (int i = 0; i < num_idof; ++i) {
        u[internal_dofs[i]] = u_I[i];
    }
    for (size_t i = 0; i < boundary_dofs.size(); ++i) {
        u[boundary_dofs[i]] = boundary_vals[i];
    }

    // 存储到 global_p（解向量）
    global_p = u;
}

void VEM_Domain::solve_pic() {
    int n_vertices = (int) points.size();
    int n_edges = (int) edges.size();
    int n_elements = (int) vem_elements.size();
    int total_dofs = n_vertices + n_edges + n_elements;

    // -----------------------------
    // 1. 找 boundary dofs
    // -----------------------------
    std::vector<int> boundary_dofs;
    std::vector<_2d_vec> boundary_query_points;


    // -----------------------------
    // 2. 找 internal dofs
    // -----------------------------
    std::vector<int> internal_dofs;

    // 非边界顶点
    for (int v = 0; v < n_vertices; ++v) {
        internal_dofs.push_back(v);
    }

    // 非纯边界边
    for (int e_id = 0; e_id < n_edges; ++e_id) {
        const Edge &e = edges[e_id];
        internal_dofs.push_back(n_vertices + e_id);
    }

    // 全部单元 dofs
    for (int el = 0; el < n_elements; ++el) {
        internal_dofs.push_back(n_vertices + n_edges + el);
    }

    int num_idof = (int) internal_dofs.size();

    // -----------------------------
    // 3. 修改 RHS: F = F - K[:, boundary_dofs] * boundary_vals
    // -----------------------------
    // Eigen::VectorXr F = global_rhs;
    // Eigen::MatrixXr Kdense = Eigen::MatrixXr(global_stiffness_matrix);
    // Eigen::MatrixXr K_bc = Kdense(Eigen::all, Eigen::VectorXi::Map(boundary_dofs.data(), boundary_dofs.size()));
    // F -= K_bc * boundary_vals;
    // 以下是以上矩阵模式的稀疏展开
    Eigen::VectorXr F = global_rhs;

    // -----------------------------
    // 4. 提取内部子系统 K_II
    // -----------------------------
    Eigen::VectorXi idofs = Eigen::VectorXi::Map(internal_dofs.data(), num_idof);
    // Eigen::MatrixXr K_II = Kdense(idofs, idofs);
    const int nI = idofs.size();
    std::unordered_map<int, int> g2l;
    g2l.reserve(nI * 2);
    for (int i = 0; i < nI; ++i) g2l[idofs[i]] = i;
    Eigen::SparseMatrixR Krow = global_stiffness_matrix;
    Eigen::SparseMatrixR K_II(nI, nI);
    std::vector<Eigen::Triplet<GlobalReal> > trips;
    trips.reserve(static_cast<size_t>(Krow.nonZeros() * 0.3)); // 估个上界，避免频繁扩容
    for (int ii = 0; ii < nI; ++ii) {
        int gi = idofs[ii]; // 全局行
        for (Eigen::SparseMatrixR::InnerIterator it(Krow, gi); it; ++it) {
            int gj = it.col(); // 全局列
            auto found = g2l.find(gj); // 列也必须是内部自由度
            if (found != g2l.end()) {
                int jj = found->second; // 局部列
                trips.emplace_back(ii, jj, it.value());
            }
        }
    }
    K_II.setFromTriplets(trips.begin(), trips.end());
    K_II.makeCompressed();

    Eigen::VectorXr F_I(num_idof);
    for (int i = 0; i < num_idof; ++i) F_I[i] = F[internal_dofs[i]];
    // -----------------------------
    // 5. 解 K_II u_I = F_I
    // -----------------------------
    // VEM_PMGSolver pmg_solver;
    // std::vector<std::pair<int,int>> temp_es;
    // for (auto&edge: edges) {
    //     temp_es.push_back({edge.v1, edge.v2});
    // }
    // pmg_solver.build_hierarchy(global_stiffness_matrix,
    // global_stiffness_matrix_k1,n_vertices,n_edges,0,n_elements,temp_es);
    // pmg_solver.set_rhs(F_I,F_I);
    // Eigen::VectorXr u_I;
    // pmg_solver.VC(u_I);
    if (!inited_k2) {
        amgclInit(GPU_AMG,K_II,1e-3,10000);
        inited_k2 = true;
    }
    Eigen::VectorXr u_I = amgclSolve(GPU_AMG, F_I);
    // Eigen::VectorXr u_I = K_II.llt().solve(F_I);
    Eigen::VectorXr u = Eigen::VectorXr::Zero(total_dofs);
    for (int i = 0; i < num_idof; ++i) {
        u[internal_dofs[i]] = u_I[i];
    }

    // 存储到 global_p（解向量）
    global_p = u;
}

void VEM_Domain::save_solution_png(const std::string &filename,
                                   int width,
                                   int height) {
    const Eigen::VectorXr &u = global_p;
    int n_vertices = (int) points.size();

    // ----------------------
    // 找边界
    // ----------------------
    Real xmin = 1e9, xmax = -1e9, ymin = 1e9, ymax = -1e9;
    for (auto &p: points) {
        xmin = std::min(xmin, p.vx);
        xmax = std::max(xmax, p.vx);
        ymin = std::min(ymin, p.vy);
        ymax = std::max(ymax, p.vy);
    }

    // ----------------------
    // 找解的范围
    // ----------------------
    Real umin = u.head(n_vertices).minCoeff();
    Real umax = u.head(n_vertices).maxCoeff();
    Real range = umax - umin;
    if (range < 1e-12) range = 1.0;

    // ----------------------
    // 分配图像缓冲区
    // ----------------------
    const int colorbar_width = 20;
    const int total_width = width + colorbar_width;
    std::vector<unsigned char> img(total_width * height * 3, 255);

    // ----------------------
    // 主图绘制（最近邻）
    // ----------------------
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            Real x = xmin + (xmax - xmin) * i / (width - 1);
            Real y = ymin + (ymax - ymin) * j / (height - 1);

            Real best_dist = 1e18;
            int best_id = 0;
            for (int v = 0; v < n_vertices; v++) {
                Real dx = x - points[v].vx;
                Real dy = y - points[v].vy;
                Real d2 = dx * dx + dy * dy;
                if (d2 < best_dist) {
                    best_dist = d2;
                    best_id = v;
                }
            }

            Real val = (u[best_id] - umin) / range;
            val = std::clamp<Real>(val, 0.0, 1.0);
            unsigned char r = (unsigned char) (val * 255);
            unsigned char g = 0;
            unsigned char b = (unsigned char) ((1.0 - val) * 255);

            int idx = ((height - 1 - j) * total_width + i) * 3;
            img[idx + 0] = r;
            img[idx + 1] = g;
            img[idx + 2] = b;
        }
    }

    // ----------------------
    // 绘制 color bar
    // ----------------------
    for (int j = 0; j < height; ++j) {
        Real val = (Real) j / (height - 1);
        unsigned char r = (unsigned char) (val * 255);
        unsigned char g = 0;
        unsigned char b = (unsigned char) ((1.0 - val) * 255);

        for (int i = width; i < total_width; ++i) {
            int idx = ((height - 1 - j) * total_width + i) * 3;
            img[idx + 0] = r;
            img[idx + 1] = g;
            img[idx + 2] = b;
        }
    }

    // ----------------------
    // 在 colorbar 上绘制简单数字刻度
    // ----------------------
    auto draw_hline = [&](int y) {
        for (int i = width; i < total_width; ++i) {
            int idx = ((height - 1 - y) * total_width + i) * 3;
            img[idx + 0] = 255;
            img[idx + 1] = 255;
            img[idx + 2] = 255;
        }
    };

    // 三个刻度线：顶部 / 中部 / 底部
    int y_top = height - 1;
    int y_mid = height / 2;
    int y_bot = 0;
    draw_hline(y_top);
    draw_hline(y_mid);
    draw_hline(y_bot);

    // ----------------------
    // 输出 PNG
    // ----------------------
    if (!stbi_write_png(filename.c_str(), total_width, height, 3, img.data(), total_width * 3))
        std::cerr << "Failed to write PNG: " << filename << "\n";
    else {
        std::cout << "Saved solution to " << filename << "\n";
        std::cout << "Color bar scale:\n";
        std::cout << "  Top (red):   " << umax << "\n";
        std::cout << "  Middle:      " << (umax + umin) * 0.5 << "\n";
        std::cout << "  Bottom (blue): " << umin << "\n";
    }
}

void VEM_Domain::plot_mesh() {
    namespace plt = matplotlibcpp;
    plt::figure_size(600, 600);

    // 遍历多边形
    for (const auto &poly: polygons) {
        std::vector<Real> xs, ys;
        for (int vid: poly) {
            xs.push_back(points[vid].vx);
            ys.push_back(points[vid].vy);
        }
        // 闭合
        xs.push_back(points[poly[0]].vx);
        ys.push_back(points[poly[0]].vy);

        plt::plot(xs, ys, "k-"); // 黑色边框
    }

    // 边界点高亮
    std::vector<Real> bx, by;
    for (size_t i = 0; i < points.size(); ++i) {
        if (is_border[i]) {
            bx.push_back(points[i].vx);
            by.push_back(points[i].vy);
        }
    }
    plt::scatter(bx, by, 30.0, {{"color", "red"}});

    plt::xlim(0, 1);
    plt::ylim(0, 1);
    plt::title("Verify");
    plt::show();
}

void VEM_Domain::init_flip_particles() {
    flip_particles.clear();
    int N = 300;
    flip_particles.reserve(N * N);
    Real step = 1.0f / (N); // 均匀网格间距

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            Real x = i * step + step / 2;
            Real y = j * step + step / 2;
            _2d_vec pos(x, y);
            _2d_vec vel(0.0f, 0.0f);
            // vel.vy = -0.5;
            if (i > 90 && i < 210 && j > 270 && j < 300) {
                vel.vy = -3.0;
            }
            flip_particles.emplace_back(pos, vel);
        }
    }
}

// 判断点 (px,py) 是否在多边形 poly 内
bool point_in_polygon(const _2d_vec &p, const std::vector<_2d_vec> &poly) {
    bool inside = false;
    int n = poly.size();
    for (int i = 0, j = n - 1; i < n; j = i++) {
        Real xi = poly[i].vx, yi = poly[i].vy;
        Real xj = poly[j].vx, yj = poly[j].vy;
        bool intersect = ((yi > p.vy) != (yj > p.vy)) &&
                         (p.vx < (xj - xi) * (p.vy - yi) / (yj - yi + 1e-15) + xi);
        if (intersect) inside = !inside;
    }
    return inside;
}

// 首先计算每一个elements的包围盒 然后构建一个相对比较细分的 均匀分割 把flip particles 放进去 然后根据
// elements的包围盒可以比较方便的找到离得比较近的particles,然后从这些particles中在筛选完全在内部的
// remark: 话说这样会不会导致速度的不连续 比如说两个相邻的elements 中相接很近的两个点？
// 要注意check 如果是 (0,1)之内的particles一定要被分到一个element
bool out_of_domain(const _2d_vec &p) {
    bool out = false;
    if (p.vx < 0 || p.vx > 1 || p.vy < 0 || p.vy > 1)out = true;
    return out;
}

void VEM_Domain::assign_flip_particles_to_elements() {
    int grid_res = 50;
    std::vector<std::vector<int> > grid_to_elems;

    // 清空旧的分配
    RZ_INFO("vem elements:", vem_elements.size());
    for (auto &elem: vem_elements) {
        elem.contained_flip_particles.clear();
    }
    build_element_bboxes();
    // 构建一下 均匀格点和哪些 elem有关系 后面可以判断
    link_ele_with_uniform_grid(grid_to_elems, grid_res);

    // 假定每一帧 elements 都会变化 所以需要重新计算 todo：加速
    for (int pid = 0; pid < (int) flip_particles.size(); ++pid) {
        const auto &p = flip_particles[pid].pos;
        if (out_of_domain(p)) {
            flip_particles[pid].deleted = true;
            continue;
        }
        int gx = std::min(grid_res - 1, int(p.vx * grid_res));
        int gy = std::min(grid_res - 1, int(p.vy * grid_res));
        auto &cand_elems = grid_to_elems[gy * grid_res + gx];
        bool assigned = false;
        for (int eid: cand_elems) {
            if (point_in_polygon(p, vem_elements[eid].points)) {
                vem_elements[eid].contained_flip_particles.push_back(pid);
                assigned = true;
                break;
            }
        }
        if (!assigned)flip_particles[pid].deleted = true;
        // RZ_ASSERT(assigned,"not assigned flip particles");
    }
}

void VEM_Domain::build_element_bboxes() {
    elem_bboxes.resize(vem_elements.size());
    for (int eid = 0; eid < (int) vem_elements.size(); ++eid) {
        const auto &poly = vem_elements[eid].points;
        Real xmin = 1e9, xmax = -1e9, ymin = 1e9, ymax = -1e9;
        for (auto &v: poly) {
            xmin = std::min(xmin, (Real) v.vx);
            xmax = std::max(xmax, (Real) v.vx);
            ymin = std::min(ymin, (Real) v.vy);
            ymax = std::max(ymax, (Real) v.vy);
        }
        elem_bboxes[eid] = {xmin, xmax, ymin, ymax};
    }
}

void VEM_Domain::link_ele_with_uniform_grid(std::vector<std::vector<int> > &grid_to_elems, int grid_res) {
    grid_to_elems.clear();
    grid_to_elems.resize(grid_res * grid_res);
    for (int eid = 0; eid < (int) vem_elements.size(); ++eid) {
        const auto &bb = elem_bboxes[eid];
        int x0 = std::max(0, int(bb.xmin * grid_res));
        int x1 = std::min(grid_res - 1, int(bb.xmax * grid_res));
        int y0 = std::max(0, int(bb.ymin * grid_res));
        int y1 = std::min(grid_res - 1, int(bb.ymax * grid_res));

        for (int i = x0; i <= x1; ++i)
            for (int j = y0; j <= y1; ++j) {
                grid_to_elems[j * grid_res + i].push_back(eid);
            }
    }
}

Real rbf_gaussian(Real r, Real epsilon) {
    return std::exp(-(r * r) / (epsilon * epsilon));
}

void VEM_Domain::PtoG() {
    // get rhs u
    k1_dof_u.setZero();
    k1_dof_v.setZero();
    assign_flip_particles_to_elements();
    Eigen::VectorXr k1_dof_weights;
    k1_dof_weights.resize(k1_dof_u.size());
    k1_dof_weights.setZero();
    Real epsilon = 0.005;
    for (auto &elem: vem_elements) {
        // 修改一下 不止用cp 而是对于每一个element顶点都做一个加权
        std::vector<_2d_vec> vertices_vels;
        for (auto &cp: elem.points) {
            Real wsum = 0.0;
            _2d_vec cp_vel(0, 0);
            for (auto &flip_particle_id: elem.contained_flip_particles) {
                const auto &flip_particle = flip_particles[flip_particle_id];
                if (flip_particle.deleted) {
                    RZ_FATAL("flip_particle.deleted");
                }
                const auto &pos_flip = flip_particle.pos;
                const auto &vel_flip = flip_particle.vec;
                Real dx = cp.vx - pos_flip.vx;
                Real dy = cp.vy - pos_flip.vy;
                Real r = std::sqrt(dx * dx + dy * dy);
                Real w = rbf_gaussian(r, epsilon);
                cp_vel.vx += w * vel_flip.vx;
                cp_vel.vy += w * vel_flip.vy;
                wsum += w;
            }
            if (wsum > 1e-12) {
                cp_vel.vx /= wsum;
                cp_vel.vy /= wsum;
            }
            vertices_vels.push_back(cp_vel);
        }
        //--------------------
        int n_sides = (int) elem.points.size();
        Real area = elem.area;
        int per_element_dof = n_sides; // k = 1
        for (int vid = 0; vid < n_sides; ++vid) {
            int global_vid = elem.local_to_global_dof[vid];
            Real temp = area / n_sides;
            k1_dof_weights[global_vid] += temp;
            k1_dof_u[global_vid] += vertices_vels[vid].vx * temp;
            k1_dof_v[global_vid] += vertices_vels[vid].vy * temp;
            // k1_dof_u[global_vid] += cp_vel.vx * temp;
            // k1_dof_v[global_vid] += cp_vel.vy * temp;
        }
    }
    for (int i = 0; i < k1_dof_weights.size(); ++i) {
        if (k1_dof_weights[i] > 1e-12) {
            k1_dof_u[i] /= k1_dof_weights[i];
            k1_dof_v[i] /= k1_dof_weights[i];
        }
    }
}

void VEM_Domain::GtoP(Real flip_alpha, Real dt) {
    RZ_DEBUG_INFO();
    bool re_interpolation = true;

    // get dear old velocity through 插值 。那么用哪一个old velocity更合适呢？粒子上的还是重新插值的
    if (re_interpolation) {
        for (const auto &element: vem_elements) {
            const auto &projector_k1 = element.project_matrix_star_k1;
            Eigen::VectorXr element_dof_values_u, element_dof_values_v;
            const auto &vert_ids = element.local_to_global_dof;
            element_dof_values_u.resize(vert_ids.size());
            element_dof_values_u.setZero();
            element_dof_values_v.resize(vert_ids.size());
            element_dof_values_v.setZero();
            int id = 0;
            for (auto &vid: vert_ids) {
                element_dof_values_u[id] = k1_dof_u[vid];
                element_dof_values_v[id++] = k1_dof_v[vid];
            }
            RZ_ASSERT(projector_k1.cols()==element_dof_values_u.size(),
                      "projector_k1.cols()==element_dof_values.size()");
            auto poly_coeffs_u = projector_k1 * element_dof_values_u;
            auto poly_coeffs_v = projector_k1 * element_dof_values_v;
            // RZ_INFO(element_dof_values_v);

            for (auto &particle_id: element.contained_flip_particles) {
                auto &particle = flip_particles[particle_id];
                const auto &pos = particle.pos;
                Real vx = 0.0, vy = 0.0;
                for (int i = 0; i < 3; ++i) {
                    const auto &monomial = monomials[i];
                    vx += poly_coeffs_u(i) *
                            monomial.value(pos.vx, pos.vy, element.centroid.vx,
                                           element.centroid.vy, element.diameter);
                    vy += poly_coeffs_v(i) *
                            monomial.value(pos.vx, pos.vy, element.centroid.vx,
                                           element.centroid.vy, element.diameter);
                }
                // RZ_INFO("old vy: ",particle.vec.vy," new vy: ",vy);
                particle.pic_vec.vx = vx;
                particle.pic_vec.vy = vy;
            }
        }
    }

    // 插值每个粒子处压强的梯度 并用之更新速度
    int element_id = 0;
    for (const auto &element: vem_elements) {
        const auto &projector_k2 = element.project_matrix_star;
        std::vector<Real> temp_element_dof_values_p;
        Eigen::VectorXr element_dof_values_p;
        const auto &vert_ids = element.local_to_global_dof;
        for (const auto &vid: vert_ids) {
            temp_element_dof_values_p.push_back(global_p[vid]);
        }
        for (int i = 0; i < vert_ids.size(); ++i) {
            int va_id = vert_ids[i];
            int vb_id = vert_ids[(i + 1) % vert_ids.size()];
            Edge edge(va_id, vb_id);
            int edge_id = edgetoid_map[edge];
            temp_element_dof_values_p.push_back(global_p[points.size() + edge_id]);
        }
        temp_element_dof_values_p.push_back(global_p[points.size() + edges.size() + element_id]);
        element_dof_values_p =
                Eigen::Map<Eigen::VectorXr>(temp_element_dof_values_p.data(),
                                            temp_element_dof_values_p.size());
        RZ_ASSERT(element_dof_values_p.size()==temp_element_dof_values_p.size(),
                  "element_dof_values_p.size()==temp_element_dof_values_p.size()");
        RZ_ASSERT(projector_k2.cols()==element_dof_values_p.size(), "projector_k2.cols()==element_dof_values_p.size()");
        auto poly_coeffs_p = projector_k2 * element_dof_values_p;
        RZ_ASSERT(projector_k2.rows()==6, "projector_k2.rows()==6");
        for (auto &particle_id: element.contained_flip_particles) {
            auto &particle = flip_particles[particle_id];
            const auto &pos = particle.pos;
            Real dvx = 0.0, dvy = 0.0;
            for (int i = 0; i < 6; ++i) {
                const auto &monomial = monomials[i];
                auto dv = monomial.gradient(pos.vx, pos.vy, element.centroid.vx,
                                            element.centroid.vy, element.diameter) * poly_coeffs_p(i);
                dvx += dv.vx;
                dvy += dv.vy;
            }
            particle.pic_vec.vx -= dvx * dt;
            particle.pic_vec.vy -= dvy * dt;

            _2d_vec delta_vec(-dvx * dt, -dvy * dt);
            particle.vec = particle.pic_vec * flip_alpha + (particle.vec + delta_vec) * (1 - flip_alpha);
            // todo : flip maybe wrong 应该用自由度上的值的差值 做插值?
        }
        element_id++;
    }
}

void VEM_Domain::flip_advection(Real dt) {
    for (auto &p: flip_particles) {
        if (p.deleted)continue;
        p.pos.vx += dt * p.vec.vx;
        p.pos.vy += dt * p.vec.vy;
    }
}

void VEM_Domain::uniform_construct(int nx,int ny) {
    points.clear();
    polygons.clear();


    const Real dx = 1.0 / nx;
    const Real dy = 1.0 / ny;

    // --------------------------------------------------
    // 1. 构造顶点
    // 顶点编号: id(i,j) = j*(nx+1) + i
    // i: x方向, j: y方向
    // --------------------------------------------------
    points.reserve((nx + 1) * (ny + 1));
    for (int j = 0; j <= ny; ++j) {
        for (int i = 0; i <= nx; ++i) {
            Real x = i * dx;
            Real y = j * dy;
            points.push_back({x, y});
        }
    }

    auto vid = [nx](int i, int j) {
        return j * (nx + 1) + i;
    };

    // --------------------------------------------------
    // 2. 构造四边形单元
    // 每个cell按逆时针:
    // 左下 -> 右下 -> 右上 -> 左上
    // --------------------------------------------------
    polygons.reserve(nx * ny);
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            int v0 = vid(i,     j);     // 左下
            int v1 = vid(i + 1, j);     // 右下
            int v2 = vid(i + 1, j + 1); // 右上
            int v3 = vid(i,     j + 1); // 左上

            polygons.push_back({v0, v1, v2, v3});
        }
    }

    std::cout << "uniform grid constructed:\n";
    std::cout << "points size = " << points.size() << "\n";
    std::cout << "polygons size = " << polygons.size() << "\n";
}
//
// bool VEM_Element_2d::build_up() {
//     // FUNC_TIMER();
//     std::vector<VEM_Monomial_2d> monomials;
//     monomials = {
//         VEM_Monomial_2d(0, 0),
//         VEM_Monomial_2d(1, 0),
//         VEM_Monomial_2d(0, 1),
//         VEM_Monomial_2d(2, 0),
//         VEM_Monomial_2d(1, 1),
//         VEM_Monomial_2d(0, 2)
//     };
//     // Eigen::MatrixXr project_matrix_star,B,D,G,project_matrix,
//     // project_matrix_k1,project_matrix_star_k1,B_k1,D_k1,G_k1,stiffness_matrix_k1,
//     // stiffness_matrix,local_LX,local_LY,H_X,H_Y;
//     size_t n_sides = this->points.size();
//
//     if (n_sides < 3) RZ_FATAL("n_sides < 3");
//     if (build_geometric) {
//         Real area = 0.0;
//         Real max_diameter = 0.0;
//         for (size_t i = 0; i < n_sides; i++) {
//             const _2d_vec &v0 = this->points[i];
//             for (size_t j = i + 1; j < n_sides; j++) {  // 计算所有点对之间的距离
//                 const _2d_vec &v1 = this->points[j];
//                 Real dx = v1.vx - v0.vx;
//                 Real dy = v1.vy - v0.vy;
//                 Real len = std::sqrt(dx * dx + dy * dy);
//                 max_diameter = std::max(max_diameter, len);
//             }
//         }
//         Polygon_2 cgal_polygon;
//         for (auto&p:points)cgal_polygon.push_back({p.vx,p.vy});
//         if (!cgal_polygon.is_convex()) {
//             // return false;
//             RZ_INFO("no cvx");
//         }
//         if (!cgal_polygon.is_counterclockwise_oriented()) {
//             RZ_INFO("no ccw");
//             // return false;
//         }
//         // RZ_ASSERT(cgal_polygon.is_counterclockwise_oriented(),"");
//         area = cgal_polygon.area();
//         if (area<0) {
//             RZ_FATAL("");
//             this->degenerated = true;
//         }
//         K::Point_2 cgal_centroid =CGAL::centroid(cgal_polygon.begin(),cgal_polygon.end());
//         // 存储到 elem
//         this->diameter = max_diameter;
//         this->area = area;
//         this->volume = area; // 2D 情况下把面积当作体积
//         this->centroid = _2d_vec(cgal_centroid.x(),cgal_centroid.y());
//     }else {
//         this->centroid = this->plane._to_2d(this->_3d_centroid);
//     }
//     //
//     int n_polynomials = 6; //k = 2
//     int n_polynomials_k1 = 3;
//     int per_element_dof_num = n_sides * 2 + 1;
//     int per_element_dof_num_k1 = n_sides;
//     {
//         // k = 1
//         this->D_k1.resize(per_element_dof_num_k1, n_polynomials_k1);
//         this->D_k1.setZero();
//
//         this->B_k1.resize(n_polynomials_k1, per_element_dof_num_k1);
//         this->B_k1.setZero();
//         // ------------------------
//         // 初始条件
//         // ------------------------
//         // D[:,0] = 1
//         this->D_k1.col(0).setOnes();
//         // B[0,:] = 1 / n_sides
//         for (int j = 0; j < n_sides; ++j)
//             this->B_k1(0, j) = 1.0/ static_cast<Real>(n_sides);
//         // ------------------------
//         // 填充线性项部分
//         // ------------------------
//         for (int vid = 0; vid < n_sides; ++vid) {
//             const _2d_vec &vert = this->points[vid];
//             const _2d_vec &prevv = this->points[(vid - 1 + n_sides) % n_sides];
//             const _2d_vec &nextv = this->points[(vid + 1) % n_sides];
//
//             // 顶点法向量 (包含长度)
//             _2d_vec vertex_normal(nextv.vy - prevv.vy,
//                                   prevv.vx - nextv.vx);
//
//             // 遍历所有非零多项式（跳过常数项）
//             for (int poly_id = 1; poly_id < n_polynomials_k1; ++poly_id) {
//                 const VEM_Monomial_2d &poly = monomials[poly_id];
//                 int p = poly.p;
//                 int q = poly.q;
//
//                 // 计算 (p,q) 方向梯度，k=1时 grad = [p/hd, q/hd]
//                 _2d_vec grad((Real) (p / this->diameter), (Real) (q / this->diameter));
//
//                 // D[vid, poly_id] = dot(vert - centroid, [p,q]) / hd
//                 _2d_vec delta(vert.vx - centroid.vx, vert.vy - centroid.vy);
//                 this->D_k1(vid, poly_id) =
//                     poly.value(vert.vx,vert.vy,centroid.vx,centroid.vy,this->diameter);
//
//                 // B[poly_id, vid] = dot(grad, normal) / 2
//                 this->B_k1(poly_id, vid) = _2d_vec_dot(grad, vertex_normal) / 2.0;
//             }
//         }
//         // check G and BD
//         this->G_k1.resize(n_polynomials_k1,n_polynomials_k1);
//         this->G_k1.setZero();
//         this->G_k1 = this->B_k1 * this->D_k1;
//         Eigen::MatrixXr projector_k1 = this->G_k1.inverse() * this->B_k1; // (n_polys x per_element_dof)
//         // 保存 projector
//         this->project_matrix_star_k1 = projector_k1;
//         this->project_matrix_k1 = this->D_k1 * projector_k1;
//         Eigen::MatrixXr I_k1 = Eigen::MatrixXr::Identity(per_element_dof_num_k1, per_element_dof_num_k1);
//         Eigen::MatrixXr residual_k1 = I_k1 - this->D_k1 * projector_k1; // (per_element_dof x per_element_dof)
//         Eigen::MatrixXr stabilising_term_k1 = residual_k1.transpose() * residual_k1;
//
//         // G = B*D
//         this->G_k1.row(0).setZero(); // G[0,:] = 0
//         Eigen::MatrixXr local_stiffness_k1 = projector_k1.transpose() * this->G_k1 * projector_k1 + stabilising_term_k1;
//         this->stiffness_matrix_k1 = local_stiffness_k1;
//     }
//     // k = 2
//     this->D.resize(per_element_dof_num, n_polynomials);
//     this->D.setZero();
//
//     this->B.resize(n_polynomials, per_element_dof_num);
//     this->B.setZero();
//     // ------------------------
//     // 初始条件
//     // ------------------------
//     this->D.col(0).setOnes(); // D[:,0] = 1
//     if (!DOFSCALE)
//         this->D(per_element_dof_num - 1,0) *= area;
//     else this->D(per_element_dof_num - 1,0) = 1.0;
//     this->B.row(0).setZero();
//     this->B(0, per_element_dof_num - 1) = 1.0; // B[0,-1] = 1
//     this->B(0, per_element_dof_num - 1) = area; // B[0,-1] = 1
//
//     for (int vid = 0; vid < n_sides; ++vid) {
//         const _2d_vec &vert = this->points[vid];
//         const _2d_vec &prevv = this->points[(vid - 1 + n_sides) % n_sides];
//         const _2d_vec &nextv = this->points[(vid + 1) % n_sides];
//
//         // vertex normal (包含长度)
//         _2d_vec vertex_normal(nextv.vy - prevv.vy,
//                               prevv.vx - nextv.vx);
//
//         // 遍历所有非常数单项式
//         for (int poly_id = 1; poly_id < n_polynomials; ++poly_id) {
//             const VEM_Monomial_2d &poly = monomials[poly_id];
//
//             // D[vertex_id, poly_id] = monomial_value(...)
//             this->D(vid, poly_id) = poly.value(vert.vx, vert.vy,
//                                               centroid.vx, centroid.vy, this->diameter);
//
//             // 梯度
//             _2d_vec grad = poly.gradient(vert.vx, vert.vy,
//                                          centroid.vx, centroid.vy, this->diameter);
//
//             // B[poly_id, vertex_id] = dot(grad, normal) / 6
//             this->B(poly_id, vid) = _2d_vec_dot(grad, vertex_normal) / 6.0;
//         }
//     }
//
//     for (int vid = 0; vid < n_sides; ++vid) {
//         const _2d_vec &vert = points[vid];
//         const _2d_vec &nextv = points[(vid + 1) % n_sides];
//
//         // 边法向量 (包含长度)
//         _2d_vec vertex_normal(nextv.vy - vert.vy,
//                               vert.vx - nextv.vx);
//
//         // 边中点
//         _2d_vec edge_center(0.5 * (vert.vx + nextv.vx),
//                             0.5 * (vert.vy + nextv.vy));
//
//         // 遍历所有非常数单项式
//         for (int poly_id = 1; poly_id < n_polynomials; ++poly_id) {
//             const VEM_Monomial_2d &poly = monomials[poly_id];
//
//             // D[vid + n_sides, poly_id] = monomial_value(...)
//             this->D(vid + n_sides, poly_id) =
//                     poly.value(edge_center.vx, edge_center.vy,
//                                centroid.vx, centroid.vy, this->diameter);
//
//             // 梯度
//             _2d_vec grad = poly.gradient(edge_center.vx, edge_center.vy,
//                                          centroid.vx, centroid.vy, this->diameter);
//
//             // B[poly_id, vid+n_sides] = dot(grad, normal) / 3 * 2
//             this->B(poly_id, vid + n_sides) =
//                     _2d_vec_dot(grad, vertex_normal) * (2.0 / 3.0);
//         }
//     }
//
//     // --- 更新 B 最后一列 ---
//     for (int poly_id = 1; poly_id < n_polynomials; ++poly_id) {
//         int p = monomials[poly_id].p;
//         int q = monomials[poly_id].q;
//
//         if (p == 2 || q == 2) {
//             this->B(poly_id, per_element_dof_num - 1) = -2.0  / this->diameter / this->diameter;
//             if (DOFSCALE)
//                 this->B(poly_id, per_element_dof_num - 1) *= area;
//         }
//     }
//
//     // --- 更新 D 最后一行 ---
//     for (int poly_id = 1; poly_id < n_polynomials; ++poly_id) {
//         Real mom = this->polygon_moment_general_2d(monomials[poly_id],centroid, true);
//         this->D(per_element_dof_num - 1, poly_id) = mom
//             / std::pow(this->diameter,monomials[poly_id].p + monomials[poly_id].q);
//         if (DOFSCALE)
//             this->D(per_element_dof_num - 1, poly_id) /= area;
//     }
//
//     // 算一下G 做一个 check
//     this->G.resize(n_polynomials,n_polynomials);
//     this->G.setZero();
//     {
//         for (int col =0;col<n_polynomials;++col) {
//             const auto& poly = monomials[col];
//             this->G(0,col) = this->polygon_moment_general_2d(poly,this->centroid,true) / std::pow(this->diameter,
//                     poly.p+poly.q);
//             // if (DOFSCALE) this->G(0,col) /= area;
//
//         }
//         for (int row = 1;row<n_polynomials;++row) {
//             for (int col = 1;col<n_polynomials;++col) {
//                 const auto& polyrow = monomials[row];
//                 const auto& polycol = monomials[col];
//                 const auto& poly_dv_x = polyrow.derivative_x() * polycol.derivative_x();
//                 const auto& poly_dv_y = polyrow.derivative_y() * polycol.derivative_y();
//                 Real denominator =  std::pow(this->diameter,polyrow.p+polyrow.q+ polycol.p + polycol.q );
//                 this->G(row,col) += this->polygon_moment_general_2d(poly_dv_x,this->centroid,true) / denominator;
//                 this->G(row,col) += this->polygon_moment_general_2d(poly_dv_y,this->centroid,true) / denominator;
//             }
//         }
//         auto multed_G= this->B * this->D;
//         if (std::fabs((multed_G - this->G).maxCoeff())>1e-3) {
//             RZ_INFO(this->polygon_moment_general_2d(monomials[0],this->centroid,true));
//             RZ_INFO("area: ",area);
//             RZ_INFO(multed_G);
//             RZ_INFO("-------------------------------");
//             RZ_INFO(this->G);
//             RZ_INFO("-------------------------------");
//             RZ_INFO(this->G - multed_G);
//             RZ_INFO("-------------------------------");
//             RZ_INFO(this->B);
//             RZ_INFO("-------------------------------");
//             RZ_INFO(this->D);
//             RZ_FATAL("maxCoeff()>1e-2: ",std::fabs((multed_G - this->G).maxCoeff()));
//         }
//     }
//     // G = B*D
//     // this->G.setZero();
//     // this->G = this->B * this->D;
//     //----------------B and D constructed----------------------
//     Eigen::MatrixXr projector =  this->G.inverse() * B; // (n_polys x per_element_dof)
//
//     // 保存 projector
//     this->project_matrix_star = projector;
//     this->project_matrix = D * projector;
//     // stabilising_term = (I - D*projector).T * (I - D*projector)
//     Eigen::MatrixXr I = Eigen::MatrixXr::Identity(per_element_dof_num, per_element_dof_num);
//     Eigen::MatrixXr residual = I - D * projector; // (per_element_dof x per_element_dof)
//     Eigen::MatrixXr stabilising_term = residual.transpose() * residual;
//     if (!DOFSCALE) stabilising_term *= 0;
//     this->G.row(0).setZero(); // G[0,:] = 0
//
//     // H :H_X,H_Y
//     this->H_X.resize(n_polynomials, n_polynomials_k1);
//     this->H_Y.resize(n_polynomials, n_polynomials_k1);
//     this->H_X.setZero();
//     this->H_Y.setZero();
//     for (int i = 0; i < n_polynomials; ++i) {
//         for (int j = 0; j < n_polynomials_k1; ++j) {
//             // 由于一定有一个求导 所以最后一定要多除一个 max_diameter
//             VEM_Monomial_2d temp_x = monomials[i].derivative_x() * monomials[j];
//             this->H_X(i, j) = polygon_moment_general_2d(temp_x, centroid,true)
//                              / std::pow(this->diameter, std::max(temp_x.p + temp_x.q + 1, 0));
//             VEM_Monomial_2d temp_y = monomials[i].derivative_y() * monomials[j];
//             this->H_Y(i, j) = polygon_moment_general_2d(temp_y, centroid,true)
//                              / std::pow(this->diameter, std::max(temp_y.p + temp_y.q + 1, 0));
//         }
//     }
//
//     this->local_LX = this->project_matrix_star.transpose() * this->H_X * this->project_matrix_star_k1;
//     this->local_LY = this->project_matrix_star.transpose() * this->H_Y * this->project_matrix_star_k1;
//
//     Eigen::MatrixXr local_stiffness = projector.transpose() * this->G * projector + stabilising_term;
//
//     this->stiffness_matrix = local_stiffness;
//     return true;
// }
bool VEM_Element_2d::build_up() {
    // FUNC_TIMER();
    Eigen::MatrixXr B,D,G,B_k1,D_k1,G_k1,H_X,H_Y;
    std::vector<VEM_Monomial_2d> monomials;
    monomials = {
        VEM_Monomial_2d(0, 0),
        VEM_Monomial_2d(1, 0),
        VEM_Monomial_2d(0, 1),
        VEM_Monomial_2d(2, 0),
        VEM_Monomial_2d(1, 1),
        VEM_Monomial_2d(0, 2)
    };

    size_t n_sides = this->points.size();

    if (n_sides < 3) RZ_FATAL("n_sides < 3");
    if (build_geometric) {
        Real area = 0.0;
        Real max_diameter = 0.0;
        for (size_t i = 0; i < n_sides; i++) {
            const _2d_vec &v0 = this->points[i];
            for (size_t j = i + 1; j < n_sides; j++) {  // 计算所有点对之间的距离
                const _2d_vec &v1 = this->points[j];
                Real dx = v1.vx - v0.vx;
                Real dy = v1.vy - v0.vy;
                Real len = std::sqrt(dx * dx + dy * dy);
                max_diameter = std::max(max_diameter, len);
            }
        }
        Polygon_2 cgal_polygon;
        for (auto&p:points)cgal_polygon.push_back({p.vx,p.vy});
        if (!cgal_polygon.is_convex()) {
            // return false;
            RZ_INFO("no cvx");
        }
        if (!cgal_polygon.is_counterclockwise_oriented()) {
            RZ_INFO("no ccw");
            // return false;
        }
        // RZ_ASSERT(cgal_polygon.is_counterclockwise_oriented(),"");
        area = cgal_polygon.area();
        if (area<0) {
            RZ_FATAL("");
            this->degenerated = true;
        }
        K::Point_2 cgal_centroid =CGAL::centroid(cgal_polygon.begin(),cgal_polygon.end());
        // 存储到 elem
        this->diameter = max_diameter;
        this->area = area;
        this->centroid = _2d_vec(cgal_centroid.x(),cgal_centroid.y());
    }else {
        Polygon_2 cgal_polygon;
        for (auto&p:points)cgal_polygon.push_back({p.vx,p.vy});
        if (!cgal_polygon.is_convex()) {
            // return false;
            RZ_INFO("no cvx");
        }
        if (!cgal_polygon.is_counterclockwise_oriented()) {
            RZ_INFO("no ccw");
            // return false;
        }
        // RZ_ASSERT(cgal_polygon.is_counterclockwise_oriented(),"");
        area = cgal_polygon.area();
        if (area<0) {
            RZ_FATAL("");
            this->degenerated = true;
        }
        K::Point_2 cgal_centroid =CGAL::centroid(cgal_polygon.begin(),cgal_polygon.end());
        auto check_c = _2d_vec(cgal_centroid.x(),cgal_centroid.y());
        this->centroid = this->plane._to_2d(this->_3d_centroid);
        RZ_ASSERT((this->centroid-check_c).norm() < 1e-6, "");
    }
    //
    int n_polynomials = 6; //k = 2
    int n_polynomials_k1 = 3;
    int per_element_dof_num = n_sides * 2 + 1;
    int per_element_dof_num_k1 = n_sides;
    {
        // k = 1
        D_k1.resize(per_element_dof_num_k1, n_polynomials_k1);
        D_k1.setZero();

        B_k1.resize(n_polynomials_k1, per_element_dof_num_k1);
        B_k1.setZero();
        // ------------------------
        // 初始条件
        // ------------------------
        // D[:,0] = 1
        D_k1.col(0).setOnes();
        // B[0,:] = 1 / n_sides
        for (int j = 0; j < n_sides; ++j)
            B_k1(0, j) = 1.0/ static_cast<Real>(n_sides);
        // ------------------------
        // 填充线性项部分
        // ------------------------
        for (int vid = 0; vid < n_sides; ++vid) {
            const _2d_vec &vert = this->points[vid];
            const _2d_vec &prevv = this->points[(vid - 1 + n_sides) % n_sides];
            const _2d_vec &nextv = this->points[(vid + 1) % n_sides];

            // 顶点法向量 (包含长度)
            _2d_vec vertex_normal(nextv.vy - prevv.vy,
                                  prevv.vx - nextv.vx);

            // 遍历所有非零多项式（跳过常数项）
            for (int poly_id = 1; poly_id < n_polynomials_k1; ++poly_id) {
                const VEM_Monomial_2d &poly = monomials[poly_id];
                int p = poly.p;
                int q = poly.q;

                // 计算 (p,q) 方向梯度，k=1时 grad = [p/hd, q/hd]
                _2d_vec grad((Real) (p / this->diameter), (Real) (q / this->diameter));

                // D[vid, poly_id] = dot(vert - centroid, [p,q]) / hd
                _2d_vec delta(vert.vx - centroid.vx, vert.vy - centroid.vy);
                D_k1(vid, poly_id) =
                    poly.value(vert.vx,vert.vy,centroid.vx,centroid.vy,this->diameter);

                // B[poly_id, vid] = dot(grad, normal) / 2
                B_k1(poly_id, vid) = _2d_vec_dot(grad, vertex_normal) / 2.0;
            }
        }
        // check G and BD
        G_k1.resize(n_polynomials_k1,n_polynomials_k1);
        G_k1.setZero();
        G_k1 = B_k1 * D_k1;
        // Eigen::LDLT<Eigen::MatrixXr> ldlt(G_k1);
        // if (ldlt.info() != Eigen::Success) {
        //     RZ_FATAL("G_k2 is singular or ill-conditioned");
        // }
        // Eigen::MatrixXr projector_k1 = ldlt.solve(B_k1);
        Eigen::MatrixXr projector_k1 = G_k1.inverse() * B_k1; // (n_polys x per_element_dof)
        // 保存 projector
        this->project_matrix_star_k1 = projector_k1;
        // this->project_matrix_k1 = D_k1 * projector_k1;
        Eigen::MatrixXr I_k1 = Eigen::MatrixXr::Identity(per_element_dof_num_k1, per_element_dof_num_k1);
        Eigen::MatrixXr residual_k1 = I_k1 - D_k1 * projector_k1; // (per_element_dof x per_element_dof)
        Eigen::MatrixXr stabilising_term_k1 = residual_k1.transpose() * residual_k1;

        // G = B*D
        G_k1.row(0).setZero(); // G[0,:] = 0
        Eigen::MatrixXr local_stiffness_k1 = projector_k1.transpose() * G_k1 * projector_k1 + stabilising_term_k1;
        this->stiffness_matrix_k1 = local_stiffness_k1;
    }
    // k = 2
    D.resize(per_element_dof_num, n_polynomials);
    D.setZero();

    B.resize(n_polynomials, per_element_dof_num);
    B.setZero();
    // ------------------------
    // 初始条件
    // ------------------------
    D.col(0).setOnes(); // D[:,0] = 1
    if (!DOFSCALE)
        D(per_element_dof_num - 1,0) *= area;
    else D(per_element_dof_num - 1,0) = 1.0;
    B.row(0).setZero();
    B(0, per_element_dof_num - 1) = 1.0; // B[0,-1] = 1
    // B(0, per_element_dof_num - 1) = area; // B[0,-1] = 1

    for (int vid = 0; vid < n_sides; ++vid) {
        const _2d_vec &vert = this->points[vid];
        const _2d_vec &prevv = this->points[(vid - 1 + n_sides) % n_sides];
        const _2d_vec &nextv = this->points[(vid + 1) % n_sides];

        // vertex normal (包含长度)
        _2d_vec vertex_normal(nextv.vy - prevv.vy,
                              prevv.vx - nextv.vx);

        // 遍历所有非常数单项式
        for (int poly_id = 1; poly_id < n_polynomials; ++poly_id) {
            const VEM_Monomial_2d &poly = monomials[poly_id];

            // D[vertex_id, poly_id] = monomial_value(...)
            D(vid, poly_id) = poly.value(vert.vx, vert.vy,
                                              centroid.vx, centroid.vy, this->diameter);

            // 梯度
            _2d_vec grad = poly.gradient(vert.vx, vert.vy,
                                         centroid.vx, centroid.vy, this->diameter);

            // B[poly_id, vertex_id] = dot(grad, normal) / 6
            B(poly_id, vid) = _2d_vec_dot(grad, vertex_normal) / 6.0;
        }
    }

    for (int vid = 0; vid < n_sides; ++vid) {
        const _2d_vec &vert = points[vid];
        const _2d_vec &nextv = points[(vid + 1) % n_sides];

        // 边法向量 (包含长度)
        _2d_vec vertex_normal(nextv.vy - vert.vy,
                              vert.vx - nextv.vx);

        // 边中点
        _2d_vec edge_center(0.5 * (vert.vx + nextv.vx),
                            0.5 * (vert.vy + nextv.vy));

        // 遍历所有非常数单项式
        for (int poly_id = 1; poly_id < n_polynomials; ++poly_id) {
            const VEM_Monomial_2d &poly = monomials[poly_id];

            // D[vid + n_sides, poly_id] = monomial_value(...)
            D(vid + n_sides, poly_id) =
                    poly.value(edge_center.vx, edge_center.vy,
                               centroid.vx, centroid.vy, this->diameter);

            // 梯度
            _2d_vec grad = poly.gradient(edge_center.vx, edge_center.vy,
                                         centroid.vx, centroid.vy, this->diameter);

            // B[poly_id, vid+n_sides] = dot(grad, normal) / 3 * 2
            B(poly_id, vid + n_sides) =
                    _2d_vec_dot(grad, vertex_normal) * (2.0 / 3.0);
        }
    }

    // --- 更新 B 最后一列 ---
    for (int poly_id = 1; poly_id < n_polynomials; ++poly_id) {
        int p = monomials[poly_id].p;
        int q = monomials[poly_id].q;

        if (p == 2 || q == 2) {
            B(poly_id, per_element_dof_num - 1) = -2.0  / this->diameter / this->diameter;
            if (DOFSCALE)
                B(poly_id, per_element_dof_num - 1) *= area;
        }
    }

    // --- 更新 D 最后一行 ---
    for (int poly_id = 1; poly_id < n_polynomials; ++poly_id) {
        Real mom = this->polygon_moment_general_2d(monomials[poly_id],centroid, true);
        D(per_element_dof_num - 1, poly_id) = mom
            / std::pow(this->diameter,monomials[poly_id].p + monomials[poly_id].q);
        if (DOFSCALE)
            D(per_element_dof_num - 1, poly_id) /= area;
    }

    // 算一下G 做一个 check
    G.resize(n_polynomials,n_polynomials);
    G.setZero();
    {
        for (int col =0;col<n_polynomials;++col) {
            const auto& poly = monomials[col];
            G(0,col) = this->polygon_moment_general_2d(poly,this->centroid,true) / std::pow(this->diameter,
                    poly.p+poly.q);
            if (DOFSCALE) G(0,col) /= area;

        }
        for (int row = 1;row<n_polynomials;++row) {
            for (int col = 1;col<n_polynomials;++col) {
                const auto& polyrow = monomials[row];
                const auto& polycol = monomials[col];
                const auto& poly_dv_x = polyrow.derivative_x() * polycol.derivative_x();
                const auto& poly_dv_y = polyrow.derivative_y() * polycol.derivative_y();
                Real denominator =  std::pow(this->diameter,polyrow.p+polyrow.q+ polycol.p + polycol.q );
                G(row,col) += this->polygon_moment_general_2d(poly_dv_x,this->centroid,true) / denominator;
                G(row,col) += this->polygon_moment_general_2d(poly_dv_y,this->centroid,true) / denominator;
            }
        }
        Eigen::MatrixXr multed_G= B * D;
        // if (std::fabs((multed_G - G).maxCoeff())>1e-9) {
        //     RZ_INFO(this->polygon_moment_general_2d(monomials[0],this->centroid,true));
        //     RZ_INFO("area: ",area);
        //     RZ_INFO(multed_G);
        //     RZ_INFO("-------------------------------");
        //     RZ_INFO(G);
        //     RZ_INFO("-------------------------------");
        //     RZ_INFO(G - multed_G);
        //     RZ_INFO("-------------------------------");
        //     RZ_INFO(B);
        //     RZ_INFO("-------------------------------");
        //     RZ_INFO(D);
        //     RZ_FATAL("maxCoeff()>1e-2: ",std::fabs((multed_G - G).maxCoeff()));
        // }
    }
    // G = B*D
    // this->G.setZero();
    // this->G = this->B * this->D;
    //----------------B and D constructed----------------------
    // Eigen::LDLT<Eigen::MatrixXr> ldlt(G);
    // if (ldlt.info() != Eigen::Success) {
    //     RZ_FATAL("G_k2 is singular or ill-conditioned");
    // }
    // Eigen::MatrixXr projector = ldlt.solve(B);
    Eigen::MatrixXr projector =  G.inverse() * B; // (n_polys x per_element_dof)

    // 保存 projector
    this->project_matrix_star = projector;

    // stabilising_term = (I - D*projector).T * (I - D*projector)
    Eigen::MatrixXr I = Eigen::MatrixXr::Identity(per_element_dof_num, per_element_dof_num);
    Eigen::MatrixXr residual = I - D * projector; // (per_element_dof x per_element_dof)
    Eigen::MatrixXr stabilising_term = residual.transpose() * residual;
    if (!DOFSCALE) stabilising_term *= 0;
    G.row(0).setZero(); // G[0,:] = 0
    // H :H_X,H_Y
    H_X.resize(n_polynomials, n_polynomials_k1);
    H_Y.resize(n_polynomials, n_polynomials_k1);
    H_X.setZero();
    H_Y.setZero();
    for (int i = 0; i < n_polynomials; ++i) {
        for (int j = 0; j < n_polynomials_k1; ++j) {
            // 由于一定有一个求导 所以最后一定要多除一个 max_diameter
            VEM_Monomial_2d temp_x = monomials[i].derivative_x() * monomials[j];
            H_X(i, j) = polygon_moment_general_2d(temp_x, centroid,true)
                             / std::pow(this->diameter, std::max(temp_x.p + temp_x.q + 1, 0));
            VEM_Monomial_2d temp_y = monomials[i].derivative_y() * monomials[j];
            H_Y(i, j) = polygon_moment_general_2d(temp_y, centroid,true)
                             / std::pow(this->diameter, std::max(temp_y.p + temp_y.q + 1, 0));
        }
    }

    this->local_LX = this->project_matrix_star.transpose() * H_X * this->project_matrix_star_k1;
    this->local_LY = this->project_matrix_star.transpose() * H_Y * this->project_matrix_star_k1;

    Eigen::MatrixXr local_stiffness = projector.transpose() * G * projector + stabilising_term;

    this->stiffness_matrix = local_stiffness;
    return true;
}

bool VEM_Element_2d::project_only_build_up() {
    Eigen::MatrixXr B,D,G,B_k1,D_k1,G_k1,H_X,H_Y;
    std::vector<VEM_Monomial_2d> monomials;
    monomials = {
        VEM_Monomial_2d(0, 0),
        VEM_Monomial_2d(1, 0),
        VEM_Monomial_2d(0, 1),
        VEM_Monomial_2d(2, 0),
        VEM_Monomial_2d(1, 1),
        VEM_Monomial_2d(0, 2)
    };

    size_t n_sides = this->points.size();

    if (n_sides < 3) RZ_FATAL("n_sides < 3");
    if (build_geometric) {
        Real area = 0.0;
        Real max_diameter = 0.0;
        for (size_t i = 0; i < n_sides; i++) {
            const _2d_vec &v0 = this->points[i];
            for (size_t j = i + 1; j < n_sides; j++) {  // 计算所有点对之间的距离
                const _2d_vec &v1 = this->points[j];
                Real dx = v1.vx - v0.vx;
                Real dy = v1.vy - v0.vy;
                Real len = std::sqrt(dx * dx + dy * dy);
                max_diameter = std::max(max_diameter, len);
            }
        }
        Polygon_2 cgal_polygon;
        for (auto&p:points)cgal_polygon.push_back({p.vx,p.vy});
        if (!cgal_polygon.is_convex()) {
            // return false;
            RZ_INFO("no cvx");
        }
        if (!cgal_polygon.is_counterclockwise_oriented()) {
            RZ_INFO("no ccw");
            // return false;
        }
        // RZ_ASSERT(cgal_polygon.is_counterclockwise_oriented(),"");
        area = cgal_polygon.area();
        if (area<0) {
            RZ_FATAL("");
            this->degenerated = true;
        }
        K::Point_2 cgal_centroid =CGAL::centroid(cgal_polygon.begin(),cgal_polygon.end());
        // 存储到 elem
        this->diameter = max_diameter;
        this->area = area;
        this->centroid = _2d_vec(cgal_centroid.x(),cgal_centroid.y());
    }else {
        Polygon_2 cgal_polygon;
        for (auto&p:points)cgal_polygon.push_back({p.vx,p.vy});

        if (!cgal_polygon.is_counterclockwise_oriented()) {
            RZ_INFO("no ccw");
            // return false;
        }
        // RZ_ASSERT(cgal_polygon.is_counterclockwise_oriented(),"");
        if (area<0) {
            RZ_FATAL("");
            this->degenerated = true;
        }
        K::Point_2 cgal_centroid =CGAL::centroid(cgal_polygon.begin(),cgal_polygon.end());
        auto check_c = _2d_vec(cgal_centroid.x(),cgal_centroid.y());
        this->centroid = this->plane._to_2d(this->_3d_centroid);
        // RZ_ASSERT((this->centroid-check_c).norm()/this->diameter < 1e-6, "");
    }
    //
    int n_polynomials = 6; //k = 2
    int n_polynomials_k1 = 3;
    int per_element_dof_num = n_sides * 2 + 1;
    int per_element_dof_num_k1 = n_sides;
    {
        // k = 1
        D_k1.resize(per_element_dof_num_k1, n_polynomials_k1);
        D_k1.setZero();

        B_k1.resize(n_polynomials_k1, per_element_dof_num_k1);
        B_k1.setZero();
        // ------------------------
        // 初始条件
        // ------------------------
        // D[:,0] = 1
        D_k1.col(0).setOnes();
        // B[0,:] = 1 / n_sides
        for (int j = 0; j < n_sides; ++j)
            B_k1(0, j) = 1.0/ static_cast<Real>(n_sides);
        // ------------------------
        // 填充线性项部分
        // ------------------------
        for (int vid = 0; vid < n_sides; ++vid) {
            const _2d_vec &vert = this->points[vid];
            const _2d_vec &prevv = this->points[(vid - 1 + n_sides) % n_sides];
            const _2d_vec &nextv = this->points[(vid + 1) % n_sides];

            // 顶点法向量 (包含长度)
            _2d_vec vertex_normal(nextv.vy - prevv.vy,
                                  prevv.vx - nextv.vx);

            // 遍历所有非零多项式（跳过常数项）
            for (int poly_id = 1; poly_id < n_polynomials_k1; ++poly_id) {
                const VEM_Monomial_2d &poly = monomials[poly_id];
                int p = poly.p;
                int q = poly.q;

                // 计算 (p,q) 方向梯度，k=1时 grad = [p/hd, q/hd]
                _2d_vec grad((Real) (p / this->diameter), (Real) (q / this->diameter));

                // D[vid, poly_id] = dot(vert - centroid, [p,q]) / hd
                _2d_vec delta(vert.vx - centroid.vx, vert.vy - centroid.vy);
                D_k1(vid, poly_id) =
                    poly.value(vert.vx,vert.vy,centroid.vx,centroid.vy,this->diameter);

                // B[poly_id, vid] = dot(grad, normal) / 2
                B_k1(poly_id, vid) = _2d_vec_dot(grad, vertex_normal) / 2.0;
            }
        }
        // check G and BD
        G_k1.resize(n_polynomials_k1,n_polynomials_k1);
        G_k1.setZero();
        G_k1 = B_k1 * D_k1;
        // Eigen::LDLT<Eigen::MatrixXr> ldlt(G_k1);
        // if (ldlt.info() != Eigen::Success) {
        //     RZ_FATAL("G_k2 is singular or ill-conditioned");
        // }
        // Eigen::MatrixXr projector_k1 = ldlt.solve(B_k1);
        Eigen::MatrixXr projector_k1 = G_k1.inverse() * B_k1; // (n_polys x per_element_dof)
        // 保存 projector
        this->project_matrix_star_k1 = projector_k1;
        // Eigen::MatrixXr I = Eigen::MatrixXr::Identity(per_element_dof_num_k1, per_element_dof_num_k1);
        // Eigen::MatrixXr residual = I - D_k1 * this->project_matrix_star_k1;
        // G_k1.row(0).setZero();
        // Eigen::MatrixXr consis_term = this->project_matrix_star_k1.transpose() * G_k1 * this->project_matrix_star_k1 ;
        // RZ_INFO(consis_term.norm()," ",residual.norm());
    }
    // k = 2
    D.resize(per_element_dof_num, n_polynomials);
    D.setZero();

    B.resize(n_polynomials, per_element_dof_num);
    B.setZero();
    // ------------------------
    // 初始条件
    // ------------------------
    D.col(0).setOnes(); // D[:,0] = 1
    if (!DOFSCALE)
        D(per_element_dof_num - 1,0) *= area;
    else D(per_element_dof_num - 1,0) = 1.0;
    B.row(0).setZero();
    B(0, per_element_dof_num - 1) = 1.0; // B[0,-1] = 1
    // B(0, per_element_dof_num - 1) = area; // B[0,-1] = 1

    for (int vid = 0; vid < n_sides; ++vid) {
        const _2d_vec &vert = this->points[vid];
        const _2d_vec &prevv = this->points[(vid - 1 + n_sides) % n_sides];
        const _2d_vec &nextv = this->points[(vid + 1) % n_sides];

        // vertex normal (包含长度)
        _2d_vec vertex_normal(nextv.vy - prevv.vy,
                              prevv.vx - nextv.vx);

        // 遍历所有非常数单项式
        for (int poly_id = 1; poly_id < n_polynomials; ++poly_id) {
            const VEM_Monomial_2d &poly = monomials[poly_id];

            // D[vertex_id, poly_id] = monomial_value(...)
            D(vid, poly_id) = poly.value(vert.vx, vert.vy,
                                              centroid.vx, centroid.vy, this->diameter);

            // 梯度
            _2d_vec grad = poly.gradient(vert.vx, vert.vy,
                                         centroid.vx, centroid.vy, this->diameter);

            // B[poly_id, vertex_id] = dot(grad, normal) / 6
            B(poly_id, vid) = _2d_vec_dot(grad, vertex_normal) / 6.0;
        }
    }

    for (int vid = 0; vid < n_sides; ++vid) {
        const _2d_vec &vert = points[vid];
        const _2d_vec &nextv = points[(vid + 1) % n_sides];

        // 边法向量 (包含长度)
        _2d_vec vertex_normal(nextv.vy - vert.vy,
                              vert.vx - nextv.vx);

        // 边中点
        _2d_vec edge_center(0.5 * (vert.vx + nextv.vx),
                            0.5 * (vert.vy + nextv.vy));

        // 遍历所有非常数单项式
        for (int poly_id = 1; poly_id < n_polynomials; ++poly_id) {
            const VEM_Monomial_2d &poly = monomials[poly_id];

            // D[vid + n_sides, poly_id] = monomial_value(...)
            D(vid + n_sides, poly_id) =
                    poly.value(edge_center.vx, edge_center.vy,
                               centroid.vx, centroid.vy, this->diameter);

            // 梯度
            _2d_vec grad = poly.gradient(edge_center.vx, edge_center.vy,
                                         centroid.vx, centroid.vy, this->diameter);

            // B[poly_id, vid+n_sides] = dot(grad, normal) / 3 * 2
            B(poly_id, vid + n_sides) =
                    _2d_vec_dot(grad, vertex_normal) * (2.0 / 3.0);
        }
    }

    // --- 更新 B 最后一列 ---
    for (int poly_id = 1; poly_id < n_polynomials; ++poly_id) {
        int p = monomials[poly_id].p;
        int q = monomials[poly_id].q;

        if (p == 2 || q == 2) {
            B(poly_id, per_element_dof_num - 1) = -2.0  / this->diameter / this->diameter;
            if (DOFSCALE)
                B(poly_id, per_element_dof_num - 1) *= area;
        }
    }

    // --- 更新 D 最后一行 ---
    for (int poly_id = 1; poly_id < n_polynomials; ++poly_id) {
        Real mom = this->polygon_moment_general_2d_advance(monomials[poly_id],centroid, true);
        D(per_element_dof_num - 1, poly_id) = mom
            / std::pow(this->diameter,monomials[poly_id].p + monomials[poly_id].q);
        if (DOFSCALE)
            D(per_element_dof_num - 1, poly_id) /= area;
    }

    // 算一下G 做一个 check
    G.resize(n_polynomials,n_polynomials);
    G.setZero();
    {
        for (int col =0;col<n_polynomials;++col) {
            const auto& poly = monomials[col];
            G(0,col) = this->polygon_moment_general_2d_advance(poly,this->centroid,true) / std::pow(this->diameter,
                    poly.p+poly.q);
            if (DOFSCALE) G(0,col) /= area;
        }
        for (int row = 1;row<n_polynomials;++row) {
            for (int col = 1;col<n_polynomials;++col) {
                const auto& polyrow = monomials[row];
                const auto& polycol = monomials[col];
                const auto& poly_dv_x = polyrow.derivative_x() * polycol.derivative_x();
                const auto& poly_dv_y = polyrow.derivative_y() * polycol.derivative_y();
                Real denominator =  std::pow(this->diameter,polyrow.p+polyrow.q+ polycol.p + polycol.q );
                G(row,col) += this->polygon_moment_general_2d_advance(poly_dv_x,this->centroid,true) / denominator;
                G(row,col) += this->polygon_moment_general_2d_advance(poly_dv_y,this->centroid,true) / denominator;
            }
        }
        Eigen::MatrixXr multed_G= B * D;
        // if (std::fabs((multed_G - G).norm()/multed_G.norm())>1e-3) {
        //     RZ_INFO(this->polygon_moment_general_2d(monomials[0],this->centroid,true));
        //     RZ_INFO("area: ",area);
        //     RZ_INFO(multed_G);
        //     RZ_INFO("-------------------------------");
        //     RZ_INFO(G);
        //     RZ_INFO("-------------------------------");
        //     RZ_INFO(G - multed_G);
        //     RZ_INFO("-------------------------------");
        //     RZ_INFO(B);
        //     RZ_INFO("-------------------------------");
        //     RZ_INFO(D);
        //     RZ_FATAL("maxCoeff()>1e-2: ",std::fabs((multed_G - G).maxCoeff()));
        // }
    }
    // G = B*D
    G.setZero();
    G = B * D;

    //----------------B and D constructed----------------------
    // Eigen::MatrixXr projector =  G.inverse() * B; // (n_polys x per_element_dof)
    Eigen::JacobiSVD<Eigen::MatrixXr> svd(G, Eigen::ComputeFullU|Eigen::ComputeFullV);
    auto s = svd.singularValues();
    Real tau = 1e-10 * s(0);   // 可调：1e-12 ~ 1e-10
    Eigen::VectorXr inv = s;
    for(int i=0;i<s.size();++i) inv(i) = (s(i) > tau) ? (1.0/s(i)) : 0.0;
    Eigen::MatrixXr Gpinv = svd.matrixV() * inv.asDiagonal() * svd.matrixU().transpose();
    Eigen::MatrixXr  projector = Gpinv * B;

    // 保存 projector
    this->project_matrix_star = projector;

    return true;
}

bool VEM_Element_2d::project_only_build_up_k1() {
    Eigen::MatrixXr B,D,G,B_k1,D_k1,G_k1,H_X,H_Y;
    std::vector<VEM_Monomial_2d> monomials;
    monomials = {
        VEM_Monomial_2d(0, 0),
        VEM_Monomial_2d(1, 0),
        VEM_Monomial_2d(0, 1),
        VEM_Monomial_2d(2, 0),
        VEM_Monomial_2d(1, 1),
        VEM_Monomial_2d(0, 2)
    };

    size_t n_sides = this->points.size();

    if (n_sides < 3) RZ_FATAL("n_sides < 3");
    if (build_geometric) {
        Real area = 0.0;
        Real max_diameter = 0.0;
        for (size_t i = 0; i < n_sides; i++) {
            const _2d_vec &v0 = this->points[i];
            for (size_t j = i + 1; j < n_sides; j++) {  // 计算所有点对之间的距离
                const _2d_vec &v1 = this->points[j];
                Real dx = v1.vx - v0.vx;
                Real dy = v1.vy - v0.vy;
                Real len = std::sqrt(dx * dx + dy * dy);
                max_diameter = std::max(max_diameter, len);
            }
        }
        Polygon_2 cgal_polygon;
        for (auto&p:points)cgal_polygon.push_back({p.vx,p.vy});
        if (!cgal_polygon.is_convex()) {
            // return false;
            RZ_INFO("no cvx");
        }
        if (!cgal_polygon.is_counterclockwise_oriented()) {
            RZ_INFO("no ccw");
            // return false;
        }
        // RZ_ASSERT(cgal_polygon.is_counterclockwise_oriented(),"");
        area = cgal_polygon.area();
        if (area<0) {
            RZ_FATAL("");
            this->degenerated = true;
        }
        K::Point_2 cgal_centroid =CGAL::centroid(cgal_polygon.begin(),cgal_polygon.end());
        // 存储到 elem
        this->diameter = max_diameter;
        this->area = area;
        this->centroid = _2d_vec(cgal_centroid.x(),cgal_centroid.y());
    }else {
        Polygon_2 cgal_polygon;
        for (auto&p:points)cgal_polygon.push_back({p.vx,p.vy});

        if (!cgal_polygon.is_counterclockwise_oriented()) {
            RZ_INFO("no ccw");
            // return false;
        }
        // RZ_ASSERT(cgal_polygon.is_counterclockwise_oriented(),"");
        if (area<0) {
            RZ_FATAL("");
            this->degenerated = true;
        }
        K::Point_2 cgal_centroid =CGAL::centroid(cgal_polygon.begin(),cgal_polygon.end());
        this->centroid = this->plane._to_2d(this->_3d_centroid);
    }
    //
    int n_polynomials_k1 = 3;
    int per_element_dof_num_k1 = n_sides;
    {
        // k = 1
        D_k1.resize(per_element_dof_num_k1, n_polynomials_k1);
        D_k1.setZero();

        B_k1.resize(n_polynomials_k1, per_element_dof_num_k1);
        B_k1.setZero();
        // ------------------------
        // 初始条件
        // ------------------------
        // D[:,0] = 1
        D_k1.col(0).setOnes();
        // B[0,:] = 1 / n_sides
        for (int j = 0; j < n_sides; ++j)
            B_k1(0, j) = 1.0/ static_cast<Real>(n_sides);
        // ------------------------
        // 填充线性项部分
        // ------------------------
        for (int vid = 0; vid < n_sides; ++vid) {
            const _2d_vec &vert = this->points[vid];
            const _2d_vec &prevv = this->points[(vid - 1 + n_sides) % n_sides];
            const _2d_vec &nextv = this->points[(vid + 1) % n_sides];

            // 顶点法向量 (包含长度)
            _2d_vec vertex_normal(nextv.vy - prevv.vy,
                                  prevv.vx - nextv.vx);

            // 遍历所有非零多项式（跳过常数项）
            for (int poly_id = 1; poly_id < n_polynomials_k1; ++poly_id) {
                const VEM_Monomial_2d &poly = monomials[poly_id];
                int p = poly.p;
                int q = poly.q;

                // 计算 (p,q) 方向梯度，k=1时 grad = [p/hd, q/hd]
                _2d_vec grad((Real) (p / this->diameter), (Real) (q / this->diameter));

                // D[vid, poly_id] = dot(vert - centroid, [p,q]) / hd
                _2d_vec delta(vert.vx - centroid.vx, vert.vy - centroid.vy);
                D_k1(vid, poly_id) =
                    poly.value(vert.vx,vert.vy,centroid.vx,centroid.vy,this->diameter);

                // B[poly_id, vid] = dot(grad, normal) / 2
                B_k1(poly_id, vid) = _2d_vec_dot(grad, vertex_normal) / 2.0;
            }
        }
        // check G and BD
        G_k1.resize(n_polynomials_k1,n_polynomials_k1);
        G_k1.setZero();
        G_k1 = B_k1 * D_k1;
        // Eigen::LDLT<Eigen::MatrixXr> ldlt(G_k1);
        // if (ldlt.info() != Eigen::Success) {
        //     RZ_FATAL("G_k2 is singular or ill-conditioned");
        // }
        // Eigen::MatrixXr projector_k1 = ldlt.solve(B_k1);
        Eigen::MatrixXr projector_k1 = G_k1.inverse() * B_k1; // (n_polys x per_element_dof)
        // 保存 projector
        this->project_matrix_star_k1 = projector_k1;
        // Eigen::MatrixXr I = Eigen::MatrixXr::Identity(per_element_dof_num_k1, per_element_dof_num_k1);
        // Eigen::MatrixXr residual = I - D_k1 * this->project_matrix_star_k1;
        // G_k1.row(0).setZero();
        // Eigen::MatrixXr consis_term = this->project_matrix_star_k1.transpose() * G_k1 * this->project_matrix_star_k1 ;
        // RZ_INFO(consis_term.norm()," ",residual.norm());
    }
    return true;
}
