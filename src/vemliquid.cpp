#include"vem3d.h"
#include "vem3d_internal.h"
#include <execution>
#include <array>
#include <fstream>

namespace {
bool append_level_set_triangle(std::vector<std::array<_3d_vec, 3>> &triangles,
                               const _3d_vec &a,
                               const _3d_vec &b,
                               const _3d_vec &c) {
    const Real area2 = (b - a).cross(c - a).squared_norm();
    if (area2 < 1e-24) return false;
    triangles.push_back({a, b, c});
    return true;
}

_3d_vec level_set_interp(const _3d_vec &a, const _3d_vec &b, Real fa, Real fb) {
    const Real denom = fa - fb;
    if (std::abs(denom) < 1e-14) return (a + b) * 0.5;
    Real t = fa / denom;
    t = std::max<Real>(0.0, std::min<Real>(1.0, t));
    return a + (b - a) * t;
}

void polygonize_level_set_tet(const std::array<_3d_vec, 4> &p,
                              const std::array<Real, 4> &phi,
                              std::vector<std::array<_3d_vec, 3>> &triangles) {
    std::array<int, 4> inside{};
    std::array<int, 4> outside{};
    int ni = 0, no = 0;
    for (int i = 0; i < 4; ++i) {
        if (phi[i] <= 0.0) inside[ni++] = i;
        else outside[no++] = i;
    }

    if (ni == 0 || ni == 4) return;

    if (ni == 1 || ni == 3) {
        const bool one_inside = (ni == 1);
        const int a = one_inside ? inside[0] : outside[0];
        const int b = one_inside ? outside[0] : inside[0];
        const int c = one_inside ? outside[1] : inside[1];
        const int d = one_inside ? outside[2] : inside[2];

        const _3d_vec p0 = level_set_interp(p[a], p[b], phi[a], phi[b]);
        const _3d_vec p1 = level_set_interp(p[a], p[c], phi[a], phi[c]);
        const _3d_vec p2 = level_set_interp(p[a], p[d], phi[a], phi[d]);

        if (one_inside) append_level_set_triangle(triangles, p0, p1, p2);
        else append_level_set_triangle(triangles, p0, p2, p1);
        return;
    }

    const int i0 = inside[0], i1 = inside[1];
    const int o0 = outside[0], o1 = outside[1];
    const _3d_vec a = level_set_interp(p[i0], p[o0], phi[i0], phi[o0]);
    const _3d_vec b = level_set_interp(p[i0], p[o1], phi[i0], phi[o1]);
    const _3d_vec c = level_set_interp(p[i1], p[o0], phi[i1], phi[o0]);
    const _3d_vec d = level_set_interp(p[i1], p[o1], phi[i1], phi[o1]);
    append_level_set_triangle(triangles, a, b, c);
    append_level_set_triangle(triangles, b, d, c);
}
}

void VEM_Domain_3d::simple_label_Dirichlet_dofs(const std::vector<FLIP_Particles_3d> &particles) {
    is_Dirichlet_dof.assign(points.size(), false);
    is_internal_dof.assign(points.size(), false);
    liquid_internal_dofs.clear();
    liquid_dofs_within_liquid_grids.clear();
    liquid_global_to_internal_dof.assign(points.size(), -1);

    element_has_liquid.resize(vem_elements.size(), false);
    for (const auto &particle : particles) {
        if (particle.deleted) continue;
        const int eid = particle.contained_in_element_id;
        if (eid >= 0 && eid < static_cast<int>(element_has_liquid.size())) {
            element_has_liquid[eid] = true;
        }
    }

    for (int eid = 0; eid < static_cast<int>(vem_elements.size()); ++eid) {
        const auto &element = vem_elements[eid];
        const bool has_liquid =
            element_has_liquid[eid];
        if (has_liquid) {
            for (int global_vid : element.vertex_local_to_global_dof) {
                liquid_dofs_within_liquid_grids.push_back(global_vid);
            }
            continue;
        }

        for (int global_vid : element.vertex_local_to_global_dof) {
            if (global_vid < 0 || global_vid >= static_cast<int>(is_Dirichlet_dof.size())) {
                continue;
            }
            is_Dirichlet_dof[global_vid] = true;
        }
    }

    std::sort(liquid_dofs_within_liquid_grids.begin(), liquid_dofs_within_liquid_grids.end());
    liquid_dofs_within_liquid_grids.erase(
        std::unique(liquid_dofs_within_liquid_grids.begin(), liquid_dofs_within_liquid_grids.end()),
        liquid_dofs_within_liquid_grids.end());
    std::sort(liquid_dofs_within_liquid_grids.begin(), liquid_dofs_within_liquid_grids.end());
    liquid_dofs_within_liquid_grids.erase(
        std::unique(liquid_dofs_within_liquid_grids.begin(), liquid_dofs_within_liquid_grids.end()),
        liquid_dofs_within_liquid_grids.end());
    liquid_internal_dofs.reserve(points.size());
    for (int v = 0; v < static_cast<int>(points.size()); ++v) {
        if (is_Dirichlet_dof[v]) continue;
        liquid_global_to_internal_dof[v] = static_cast<int>(liquid_internal_dofs.size());
        is_internal_dof[v] = true;
        liquid_internal_dofs.push_back(v);
    }

}

void VEM_Domain_3d::liquid_weakform_rhs_k1() {
    if (liquid_internal_dofs.empty()) {
        simple_label_Dirichlet_dofs({});
    }

    const int nI = static_cast<int>(liquid_internal_dofs.size());

    Eigen::VectorXr rhs_I(nI);
    rhs_I.setZero();

    for (int i = 0; i < nI; ++i) {
        const int gi = liquid_internal_dofs[i];
        rhs_I[i] = global_rhs_k1[gi];
    }

    global_rhs_k1.swap(rhs_I);
}
void VEM_Domain_3d::assemble_liquid_weakform_rhs_k1(Real dt) {
    FUNC_TIMER();
    //-----------------------------------------------------------------------------------------------------------------------------------
    global_rhs_k1.resize(points.size());
    global_rhs_k1.setZero();

    for (int el_id = 0; el_id < (int)vem_elements.size(); ++el_id) {
        if (!element_has_liquid[el_id])continue;
        const auto& elem = vem_elements[el_id];
        const int nv = elem.vertex_count();
        std::vector<int>element_to_global_dof;
        element_to_global_dof.reserve((std::size_t)nv);
        for (int vid = 0; vid < nv; ++vid)
            element_to_global_dof.push_back(elem.vertex_local_to_global_dof[vid]);
        const int ndofs    = (int)element_to_global_dof.size(); // k1 pressure dofs
        const int ndofs_k1 = nv;

        for (int i = 0; i < ndofs; ++i) {
            const int gi = element_to_global_dof[i];

            Real acc = 0.0;

            for (int j = 0; j < ndofs_k1; ++j) {
                const int gj = element_to_global_dof[j];

                const Real vx = elem.is_U
                    ? U3d_local_LX_k1(i, j)
                    : elem.local_LX_k1(i, j);

                const Real vy = elem.is_U
                    ? U3d_local_LY_k1(i, j)
                    : elem.local_LY_k1(i, j);

                const Real vz = elem.is_U
                    ? U3d_local_LZ_k1(i, j)
                    : elem.local_LZ_k1(i, j);

                acc += vx * k1_dof_u[gj]
                     + vy * k1_dof_v[gj]
                     + vz * k1_dof_w[gj];
            }

            global_rhs_k1[gi] += acc;
        }
    }
    RZ_DEBUG_INFO();
    global_rhs_k1 /= dt;
}

void VEM_Domain_3d::init_liquid(std::vector<FLIP_Particles_3d> &flip_particles,int n) {
    FUNC_TIMER();
    flip_particles.clear();

    for (auto &elem : vem_elements) {
        elem.contained_flip_particles.clear();
    }

    auto out_of_domain = [&](const _3d_vec &p) noexcept -> bool {
        bool out = false;
        if (p.vx < 0 || p.vx > 1 || p.vy < 0 || p.vy > 1 || p.vz < 0 || p.vz > 1)out = true;
        return out;
    };

    const Real sx = 0.3;
    const Real sy = 0.7;
    const Real sz = 0.3;
    const Real dx = 1.0 / static_cast<Real>(n);

    flip_particles.reserve(n * n * n);

    auto find_containing_element = [&](const _3d_vec &p) -> int {
        if (out_of_domain(p)) return -1;

        if (grid_res > 0 && !grid_to_elems.empty()) {
            const int gx = std::min(grid_res - 1, std::max(0, int(p.vx * grid_res)));
            const int gy = std::min(grid_res - 1, std::max(0, int(p.vy * grid_res)));
            const int gz = std::min(grid_res - 1, std::max(0, int(p.vz * grid_res)));
            const int gidx = (gz * grid_res + gy) * grid_res + gx;

            if (0 <= gidx && gidx < static_cast<int>(grid_to_elems.size())) {
                for (int eid : grid_to_elems[gidx]) {
                    if (vem_elements[eid].point_in_elem(p)) return eid;
                }
            }
        }

        for (int eid = 0; eid < static_cast<int>(vem_elements.size()); ++eid) {
            if (vem_elements[eid].point_in_elem(p)) return eid;
        }
        return -1;
    };

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                _3d_vec cell_center(
                    (i + 0.5) * dx,
                    (j + 0.5) * dx,
                    (k + 0.5) * dx
                );

                if (cell_center.vx > sx ||
                    cell_center.vy > sy ||
                    cell_center.vz > sz) {
                    continue;
                }

                for (int a = 0; a < 2; ++a) {
                    for (int b = 0; b < 2; ++b) {
                        for (int c = 0; c < 2; ++c) {
                            _3d_vec pos(
                                (i + 0.25 + 0.5 * a) * dx,
                                (j + 0.25 + 0.5 * b) * dx,
                                (k + 0.25 + 0.5 * c) * dx
                            );

                            const int eid = find_containing_element(pos);
                            if (eid < 0) continue;

                            FLIP_Particles_3d particle(pos, _3d_vec(0.0, 0.0, 0.0));
                            particle.contained_in_element_id = eid;
                            flip_particles.push_back(particle);
                            vem_elements[eid].contained_flip_particles.push_back(
                                static_cast<int>(flip_particles.size()) - 1
                            );
                        }
                    }
                }
            }
        }
    }
}



void VEM_Domain_3d::add_liquid_external_forces(const Real &dt) {
    FUNC_TIMER();

    const int ndof = (int)k1_dof_v.size();

    for (auto vid:liquid_dofs_within_liquid_grids) {
        k1_dof_v[vid] -= dt*9.8;
    }

#pragma omp parallel for schedule(dynamic)
    for (auto &element : vem_elements) {
        const auto &projector_k1 = element.is_U?
        U3d_project_matrix_star_k1 : element.project_matrix_star_k1;

        // === 从全局 DOF 提取当前元素的速度自由度 ===
        Eigen::VectorXr element_dof_values_u, element_dof_values_v, element_dof_values_w;
        const auto &vert_ids = element.vertex_local_to_global_dof;
        const int ndof = (int)vert_ids.size();

        element_dof_values_u.resize(ndof);
        element_dof_values_v.resize(ndof);
        element_dof_values_w.resize(ndof);
        element_dof_values_u.setZero();
        element_dof_values_v.setZero();
        element_dof_values_w.setZero();

        for (int i = 0; i < ndof; ++i) {
            int vid = vert_ids[i];
            element_dof_values_u[i] = k1_dof_u[vid];
            element_dof_values_v[i] = k1_dof_v[vid];
            element_dof_values_w[i] = k1_dof_w[vid];
        }
        RZ_ASSERT(projector_k1.cols() == ndof, "projector_k1.cols()==element_dof_values.size()");
        // === 计算该元素的多项式系数 ===
        element.poly_coeffs_u_k1 = projector_k1 * element_dof_values_u;
        element.poly_coeffs_v_k1 = projector_k1 * element_dof_values_v;
        element.poly_coeffs_w_k1 = projector_k1 * element_dof_values_w;
    }

}
void VEM_Domain_3d::RK_3_advection_liquid(std::vector<FLIP_Particles_3d> &flip_particles, const Real &dt) {
    FUNC_TIMER();
    std::vector<_3d_vec> rk_pos;
    for (auto&p:flip_particles)rk_pos.push_back(p.pos);
    std::vector<_3d_vec> old_pos = rk_pos;
    std::vector<Real>u1(rk_pos.size(),0.0),v1(rk_pos.size(),0.0),w1(rk_pos.size(),0.0),
        u2(rk_pos.size(),0.0),v2(rk_pos.size(),0.0),w2(rk_pos.size(),0.0),
        u3(rk_pos.size(),0.0),v3(rk_pos.size(),0.0),w3(rk_pos.size(),0.0);
    //分配到 elem并插值速度 更新位置
    std::vector<int> elemids = assign_pts_to_elements_3d(rk_pos);
#pragma omp parallel for schedule(static)
    for (int k=0;k<rk_pos.size();++k) {
        if (elemids[k] == -1) {
            continue;
        }
        const auto&element = vem_elements[elemids[k]];
        Eigen::VectorXr& poly_coeffs_u = vem_elements[elemids[k]].poly_coeffs_u_k1 ;
        Eigen::VectorXr& poly_coeffs_v = vem_elements[elemids[k]].poly_coeffs_v_k1 ;
        Eigen::VectorXr& poly_coeffs_w = vem_elements[elemids[k]].poly_coeffs_w_k1 ;

        const auto &pos =rk_pos[k];

        Real vx = 0.0, vy = 0.0, vz = 0.0;

        // === 插值多项式 ===
        const Real xc = element.centroid.vx;
        const Real yc = element.centroid.vy;
        const Real zc = element.centroid.vz;
        const Real h  = element.diameter;

        for (int i = 0; i < poly_coeffs_u.size(); ++i) {
            const auto &mono = monomials_3d[i];
            Real val = mono.value(pos.vx, pos.vy, pos.vz, xc, yc, zc, h);
            vx += poly_coeffs_u(i) * val;
            vy += poly_coeffs_v(i) * val;
            vz += poly_coeffs_w(i) * val;
        }
        u1[k] = vx;
        v1[k] = vy;
        w1[k] = vz;

        rk_pos[k].vx = flip_particles[k].pos.vx + dt*0.5*vx;
        rk_pos[k].vy = flip_particles[k].pos.vy + dt*0.5*vy;
        rk_pos[k].vz = flip_particles[k].pos.vz + dt*0.5*vz;
    }
    //分配到 elem并插值速度 更新位置
    boundary_aware_pos_fix(old_pos,rk_pos,dt*0.5);
    elemids = assign_pts_to_elements_3d(rk_pos);
#pragma omp parallel for schedule(static)
    for (int k=0;k<rk_pos.size();++k) {
        if (elemids[k] == -1) {
            u2[k] = u1[k];
            v2[k] = v1[k];
            w2[k] = w1[k];
            continue;
        }
        const auto&element = vem_elements[elemids[k]];
        Eigen::VectorXr& poly_coeffs_u = vem_elements[elemids[k]].poly_coeffs_u_k1 ;
        Eigen::VectorXr& poly_coeffs_v = vem_elements[elemids[k]].poly_coeffs_v_k1 ;
        Eigen::VectorXr& poly_coeffs_w = vem_elements[elemids[k]].poly_coeffs_w_k1 ;

        const auto &pos =rk_pos[k];

        Real vx = 0.0, vy = 0.0, vz = 0.0;

        // === 插值多项式 ===
        const Real xc = element.centroid.vx;
        const Real yc = element.centroid.vy;
        const Real zc = element.centroid.vz;
        const Real h  = element.diameter;

        for (int i = 0; i < poly_coeffs_u.size(); ++i) {
            const auto &mono = monomials_3d[i];
            Real val = mono.value(pos.vx, pos.vy, pos.vz, xc, yc, zc, h);
            vx += poly_coeffs_u(i) * val;
            vy += poly_coeffs_v(i) * val;
            vz += poly_coeffs_w(i) * val;
        }
        u2[k] = vx;
        v2[k] = vy;
        w2[k] = vz;

        rk_pos[k].vx = flip_particles[k].pos.vx + dt*0.75*vx;
        rk_pos[k].vy = flip_particles[k].pos.vy + dt*0.75*vy;
        rk_pos[k].vz = flip_particles[k].pos.vz + dt*0.75*vz;
    }
    //分配到 elem并插值速度
    boundary_aware_pos_fix(old_pos,rk_pos,dt*0.75);
    elemids = assign_pts_to_elements_3d(rk_pos);
#pragma omp parallel for schedule(static)
    for (int k=0;k<rk_pos.size();++k) {
        if (elemids[k] == -1) {
            u3[k] = u2[k];
            v3[k] = v2[k];
            w3[k] = w2[k];
            continue;
        }
        const auto&element = vem_elements[elemids[k]];
        Eigen::VectorXr& poly_coeffs_u = vem_elements[elemids[k]].poly_coeffs_u_k1 ;
        Eigen::VectorXr& poly_coeffs_v = vem_elements[elemids[k]].poly_coeffs_v_k1 ;
        Eigen::VectorXr& poly_coeffs_w = vem_elements[elemids[k]].poly_coeffs_w_k1 ;

        const auto &pos =rk_pos[k];

        Real vx = 0.0, vy = 0.0, vz = 0.0;

        // === 插值多项式 ===
        const Real xc = element.centroid.vx;
        const Real yc = element.centroid.vy;
        const Real zc = element.centroid.vz;
        const Real h  = element.diameter;

        for (int i = 0; i < poly_coeffs_u.size(); ++i) {
            const auto &mono = monomials_3d[i];
            Real val = mono.value(pos.vx, pos.vy, pos.vz, xc, yc, zc, h);
            vx += poly_coeffs_u(i) * val;
            vy += poly_coeffs_v(i) * val;
            vz += poly_coeffs_w(i) * val;
        }
        u3[k] = vx;
        v3[k] = vy;
        w3[k] = vz;
    }
    //最后更新每个flip粒子的位置
    Real c1 = 2.0/9.0*dt,c2 =3.0/9.0*dt,c3 = 4.0/9.0*dt;
// #pragma omp parallel for schedule(dynamic)
    for (int i=0;i<flip_particles.size();++i) {
        flip_particles[i].old_pos = flip_particles[i].pos;
        flip_particles[i].pos.vx += c1*u1[i]+c2*u2[i]+c3*u3[i];
        flip_particles[i].pos.vy += c1*v1[i]+c2*v2[i]+c3*v3[i];
        flip_particles[i].pos.vz += c1*w1[i]+c2*w2[i]+c3*w3[i];
    }
    std::vector<_3d_vec> temp_new_poses;
    for (auto&p:flip_particles)
        temp_new_poses.push_back(_3d_vec(p.pos.vx, p.pos.vy, p.pos.vz));
    liquid_boundary_aware_pos_fix(flip_particles,old_pos,temp_new_poses,dt);
    for (int i=0;i<flip_particles.size();++i) {
        flip_particles[i].pos = temp_new_poses[i];
    }

    assign_flip_smoke_particles_to_elements_3d(flip_particles);
    std::vector<FLIP_Particles_3d>temp;
    for (auto&p:flip_particles)if (!p.deleted)temp.push_back(p);
    flip_particles.swap(temp);
}

void VEM_Domain_3d::Boundary_Handle_vel_change(
    FLIP_Particles_3d &flip,
    _3d_vec old_pos,
    _3d_vec &new_pos,
    AABBTree &aabb_tree,
    const Real &dt)
{
    const Real eps = 1e-4;

    // 反射衰减系数
    // 1.0: 完全弹性反射
    // 0.0: 不反弹，只消除法向速度
    // 0.2~0.5: 比较适合液体边界
    const Real restitution = 0.3;

    auto dot = [](const _3d_vec &a, const _3d_vec &b) -> Real {
        return a.vx * b.vx + a.vy * b.vy + a.vz * b.vz;
    };


    if ((old_pos - new_pos).norm() < 1e-25) {
        new_pos = old_pos;
        return;
    }

    CGAL_Point_3 pos_pre(old_pos.vx, old_pos.vy, old_pos.vz);
    CGAL_Point_3 pos_now(new_pos.vx, new_pos.vy, new_pos.vz);

    K::Segment_3 seg(pos_pre, pos_now);

    if (!aabb_tree.do_intersect(seg)) {
        return;
    }
    K::Ray_3 ray(pos_pre,pos_now-pos_pre);
    auto intersection = aabb_tree.first_intersection(ray);

    if (!intersection) {
        return;
    }

    _3d_vec hit_pos;
    bool valid_hit = false;

    // 情况 1：交点是 Point
    if (const CGAL_Point_3 *p = boost::get<CGAL_Point_3>(&(intersection->first))) {
        hit_pos = _3d_vec(p->x(), p->y(), p->z());
        valid_hit = true;
    }
    // 情况 2：交集是 Segment，例如粒子路径刚好贴着三角面
    else if (const K::Segment_3 *s = boost::get<K::Segment_3>(&(intersection->first))) {
        const CGAL_Point_3 &p = s->source();
        hit_pos = _3d_vec(p.x(), p.y(), p.z());
        valid_hit = true;
    }

    if (!valid_hit) {
        new_pos = old_pos;
        return;
    }

    // -----------------------------
    // 1. 获取碰撞 primitive 的法向量
    // -----------------------------
    auto primitive_id = intersection->second;

    // 假设 primitive_id 是 triangle iterator
    const K::Triangle_3 &tri = *primitive_id;

    K::Vector_3 e1 = tri[1] - tri[0];
    K::Vector_3 e2 = tri[2] - tri[0];
    K::Vector_3 n_cgal = CGAL::cross_product(e1, e2);

    Real n_len = std::sqrt(n_cgal.squared_length());
    if (n_len < 1e-12) {
        new_pos = old_pos;
        return;
    }

    _3d_vec n(
        n_cgal.x() / n_len,
        n_cgal.y() / n_len,
        n_cgal.z() / n_len
    );

    // -----------------------------
    // 2. 调整法向方向
    // -----------------------------
    _3d_vec move_dir = new_pos - old_pos;

    // 希望 n 指向 old_pos 所在的一侧，也就是与运动方向相反
    if (dot(move_dir, n) > 0.0) {
        n = n * (-1.0);
    }

    // -----------------------------
    // 3. 位置修正：把粒子推回边界外侧
    // -----------------------------
    new_pos = hit_pos + n * eps;

    // -----------------------------
    // 4. 速度反射
    // -----------------------------
    Real vn = dot(flip.vec, n);

    // 只有速度朝向边界内部时才反射
    if (vn < 0.0) {
        flip.vec = flip.vec - n * ((1.0 + restitution) * vn);
    }

}

void VEM_Domain_3d::liquid_boundary_aware_pos_fix(std::vector<FLIP_Particles_3d>&flips,const std::vector<_3d_vec> &old_poses, std::vector<_3d_vec> &new_poses,const Real dt) {

    std::vector<int> to_handle;
    for (int k=0;k<old_poses.size();++k) {
        if (boundary_handle_grid(old_poses[k])) {
            to_handle.push_back(k);
        }
    }
    RZ_INFO("tohandle: ",to_handle.size()," all: ",old_poses.size());
    std::for_each(std::execution::par, to_handle.begin(), to_handle.end(),
       [&](int i) {
       Boundary_Handle_vel_change(flips[i],old_poses[i],new_poses[i],aabb_tree,dt);
   }
   );
}

// void VEM_Domain_3d::construct_level_set(const std::vector<FLIP_Particles_3d> &particles) {
//     FUNC_TIMER();
//     if (grid_res <= 0) {
//         level_set.assign(points.size(), 1.0);
//         std::cerr << "construct_level_set: invalid grid_res = " << grid_res << std::endl;
//         return;
//     }
//
//     const Real dx = 1.0 / static_cast<Real>(grid_res);
//     const Real particle_radius = dx * 1.01 * std::sqrt(3.0) * 0.5;
//     level_set.assign(points.size(), 3.0 * dx);
//
//     if (particles.empty() || points.empty()) {
//         return;
//     }
//
//     const int expected_grid_count = grid_res * grid_res * grid_res;
//     if (grid_to_elems.size() < static_cast<std::size_t>(expected_grid_count)) {
//         std::cerr << "construct_level_set: grid_to_elems size mismatch, grid_to_elems="
//                   << grid_to_elems.size() << ", expected at least " << expected_grid_count << std::endl;
//         return;
//     }
//
//     std::vector<int> visited(points.size(), -1);
//     int visit_tag = 0;
//
//     for (const auto &particle : particles) {
//         if (particle.deleted) continue;
//
//         const int gx = std::min(grid_res - 1, std::max(0, static_cast<int>(particle.pos.vx / dx)));
//         const int gy = std::min(grid_res - 1, std::max(0, static_cast<int>(particle.pos.vy / dx)));
//         const int gz = std::min(grid_res - 1, std::max(0, static_cast<int>(particle.pos.vz / dx)));
//         ++visit_tag;
//
//         for (int nz = std::max(0, gz - 1); nz <= std::min(grid_res - 1, gz + 1); ++nz) {
//             for (int ny = std::max(0, gy - 1); ny <= std::min(grid_res - 1, gy + 1); ++ny) {
//                 for (int nx = std::max(0, gx - 1); nx <= std::min(grid_res - 1, gx + 1); ++nx) {
//                     const int gidx = (nz * grid_res + ny) * grid_res + nx;
//                     for (int eid : grid_to_elems[gidx]) {
//                         if (eid < 0 || eid >= static_cast<int>(vem_elements.size())) continue;
//
//                         const auto &element = vem_elements[eid];
//                         for (int gid : element.vertex_local_to_global_dof) {
//                             if (gid < 0 || gid >= static_cast<int>(points.size())) continue;
//                             if (visited[gid] == visit_tag) continue;
//                             visited[gid] = visit_tag;
//
//                             const Real phi = (points[gid] - particle.pos).norm() - particle_radius;
//                             level_set[gid] = std::min(level_set[gid], phi);
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }

#include <omp.h>

void VEM_Domain_3d::construct_level_set(
    const std::vector<FLIP_Particles_3d> &particles
) {
    FUNC_TIMER();

    if (grid_res <= 0) {
        level_set.assign(points.size(), 1.0);
        std::cerr << "construct_level_set: invalid grid_res = "
                  << grid_res << std::endl;
        return;
    }

    const int n_points = static_cast<int>(points.size());
    const Real dx = 1.0 / static_cast<Real>(grid_res);
    const Real particle_radius = dx * 1.01 * std::sqrt(3.0) * 0.5;
    const Real init_phi = 3.0 * dx;

    level_set.assign(n_points, init_phi);

    if (particles.empty() || points.empty()) {
        return;
    }

    const int expected_grid_count = grid_res * grid_res * grid_res;
    if (grid_to_elems.size() < static_cast<std::size_t>(expected_grid_count)) {
        std::cerr << "construct_level_set: grid_to_elems size mismatch, grid_to_elems="
                  << grid_to_elems.size()
                  << ", expected at least "
                  << expected_grid_count << std::endl;
        return;
    }

    const int num_threads = omp_get_max_threads();

    std::vector<std::vector<Real>> local_level_sets(
        num_threads,
        std::vector<Real>(n_points, init_phi)
    );

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();

        auto &local_phi = local_level_sets[tid];

        std::vector<int> visited(n_points, -1);
        int visit_tag = 0;

#pragma omp for schedule(dynamic, 64)
        for (int p = 0; p < static_cast<int>(particles.size()); ++p) {
            const auto &particle = particles[p];

            if (particle.deleted) continue;

            const int gx = std::min(grid_res - 1, std::max(0, static_cast<int>(particle.pos.vx / dx)));
            const int gy = std::min(grid_res - 1, std::max(0, static_cast<int>(particle.pos.vy / dx)));
            const int gz = std::min(grid_res - 1, std::max(0, static_cast<int>(particle.pos.vz / dx)));

            ++visit_tag;

            for (int nz = std::max(0, gz - 1); nz <= std::min(grid_res - 1, gz + 1); ++nz) {
                for (int ny = std::max(0, gy - 1); ny <= std::min(grid_res - 1, gy + 1); ++ny) {
                    for (int nx = std::max(0, gx - 1); nx <= std::min(grid_res - 1, gx + 1); ++nx) {
                        const int gidx = (nz * grid_res + ny) * grid_res + nx;

                        for (int eid : grid_to_elems[gidx]) {
                            if (eid < 0 || eid >= static_cast<int>(vem_elements.size())) continue;

                            const auto &element = vem_elements[eid];

                            for (int gid : element.vertex_local_to_global_dof) {
                                if (gid < 0 || gid >= n_points) continue;

                                // 避免同一个 particle 重复访问同一个 gid
                                if (visited[gid] == visit_tag) continue;
                                visited[gid] = visit_tag;

                                const Real phi =
                                    (points[gid] - particle.pos).norm() - particle_radius;

                                local_phi[gid] = std::min(local_phi[gid], phi);
                            }
                        }
                    }
                }
            }
        }
    }

    // reduce: 对所有线程的 local_level_set 取 min
#pragma omp parallel for schedule(static)
    for (int gid = 0; gid < n_points; ++gid) {
        Real phi_min = init_phi;

        for (int t = 0; t < num_threads; ++t) {
            phi_min = std::min(phi_min, local_level_sets[t][gid]);
        }

        level_set[gid] = phi_min;
    }
}

void VEM_Domain_3d::save_level_set_to_mesh(const std::string filename) {
    if (level_set.size() != points.size()) {
        std::cerr << "level_set size mismatch: level_set=" << level_set.size()
                  << ", points=" << points.size() << std::endl;
        return;
    }

    std::vector<std::array<_3d_vec, 3>> triangles;

    for (const auto &element : vem_elements) {
        if (element.vertex_local_to_global_dof.empty()) continue;

        Real phi_c = 0.0;
        int phi_c_count = 0;
        for (int gid : element.vertex_local_to_global_dof) {
            if (gid < 0 || gid >= static_cast<int>(level_set.size())) continue;
            phi_c += level_set[gid];
            ++phi_c_count;
        }
        if (phi_c_count == 0) continue;
        phi_c /= static_cast<Real>(phi_c_count);

        for (const auto &polygon : element.polygons) {
            if (polygon.size() < 3) continue;
            for (int i = 1; i + 1 < static_cast<int>(polygon.size()); ++i) {
                const int lv0 = polygon[0];
                const int lv1 = polygon[i];
                const int lv2 = polygon[i + 1];
                if (lv0 < 0 || lv1 < 0 || lv2 < 0 ||
                    lv0 >= static_cast<int>(element.vertex_local_to_global_dof.size()) ||
                    lv1 >= static_cast<int>(element.vertex_local_to_global_dof.size()) ||
                    lv2 >= static_cast<int>(element.vertex_local_to_global_dof.size())) {
                    continue;
                }

                const int gv0 = element.vertex_local_to_global_dof[lv0];
                const int gv1 = element.vertex_local_to_global_dof[lv1];
                const int gv2 = element.vertex_local_to_global_dof[lv2];
                if (gv0 < 0 || gv1 < 0 || gv2 < 0 ||
                    gv0 >= static_cast<int>(points.size()) ||
                    gv1 >= static_cast<int>(points.size()) ||
                    gv2 >= static_cast<int>(points.size())) {
                    continue;
                }

                const std::array<_3d_vec, 4> tet_p{
                    element.centroid,
                    points[gv0],
                    points[gv1],
                    points[gv2]
                };
                const std::array<Real, 4> tet_phi{
                    phi_c,
                    level_set[gv0],
                    level_set[gv1],
                    level_set[gv2]
                };
                polygonize_level_set_tet(tet_p, tet_phi, triangles);
            }
        }
    }

    std::ofstream file(filename.c_str());
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return;
    }

    for (const auto &tri : triangles) {
        file << tri[0].to_obj_format() << std::endl;
        file << tri[1].to_obj_format() << std::endl;
        file << tri[2].to_obj_format() << std::endl;
    }

    int vertex_id = 1;
    for (std::size_t i = 0; i < triangles.size(); ++i) {
        file << "f " << vertex_id << " " << vertex_id + 1 << " " << vertex_id + 2 << std::endl;
        vertex_id += 3;
    }

    file.close();
    std::cout << "Saved " << triangles.size() << " level-set triangles to " << filename << std::endl;
}

void VEM_Domain_3d::ghostvem_label_Dirichlet_dofs() {
    FUNC_TIMER();
    is_Dirichlet_dof.assign(points.size(), false);
    is_internal_dof.assign(points.size(), false);
    liquid_internal_dofs.clear();
    liquid_dofs_within_liquid_grids.clear();
    liquid_global_to_internal_dof.assign(points.size(), -1);
    element_has_liquid.resize(vem_elements.size(), false);

    int eid = 0;
    for (const auto &elem : vem_elements) {
        for (auto&vid:elem.vertex_local_to_global_dof) {
            if (level_set[vid]<0)
                element_has_liquid[eid] = true;
        }
        ++eid;
    }

    for (int eid = 0; eid < static_cast<int>(vem_elements.size()); ++eid) {
        const auto &element = vem_elements[eid];
        const bool has_liquid =
            element_has_liquid[eid];
        if (!has_liquid)continue;
        for (int global_vid : element.vertex_local_to_global_dof) {
            liquid_dofs_within_liquid_grids.push_back(global_vid);
        }
        for (auto&vid:element.vertex_local_to_global_dof) {
            if (level_set[vid]>0)
                is_Dirichlet_dof[vid] = true;
        }
    }

    std::sort(liquid_dofs_within_liquid_grids.begin(), liquid_dofs_within_liquid_grids.end());
    liquid_dofs_within_liquid_grids.erase(
        std::unique(liquid_dofs_within_liquid_grids.begin(), liquid_dofs_within_liquid_grids.end()),
        liquid_dofs_within_liquid_grids.end());
    std::sort(liquid_dofs_within_liquid_grids.begin(), liquid_dofs_within_liquid_grids.end());
    liquid_dofs_within_liquid_grids.erase(
        std::unique(liquid_dofs_within_liquid_grids.begin(), liquid_dofs_within_liquid_grids.end()),
        liquid_dofs_within_liquid_grids.end());
    liquid_internal_dofs.reserve(points.size());
    for (int v = 0; v < static_cast<int>(points.size()); ++v) {
        if (is_Dirichlet_dof[v]) continue;
        if (level_set[v]>0)continue;
        liquid_global_to_internal_dof[v] = static_cast<int>(liquid_internal_dofs.size());
        is_internal_dof[v] = true;
        liquid_internal_dofs.push_back(v);
    }
}

Eigen::SparseMatrixR VEM_Domain_3d::build_ghost_Ak1() {
    FUNC_TIMER();
    Eigen::SparseMatrixR A;
    const int n_internal = liquid_internal_dofs.size();
    A.resize(n_internal, n_internal);
    std::vector<Eigen::Triplet<GlobalReal>> triplets;
    const int n_elements = static_cast<int>(vem_elements.size());

    for (int el_id = 0; el_id < n_elements; ++el_id) {
        const auto &elem = vem_elements[el_id];
        const Eigen::LocalMatrixXr &Ke  =
            elem.is_U ? U3d_stiffness_matrix_k1 : elem.stiffness_matrix_k1;
        int n_sides = elem.vertex_count();
        for (int i = 0; i < n_sides; ++i) {
            int gi = elem.vertex_local_to_global_dof[i];
            if (!is_internal_dof[gi]) continue;
            int row = liquid_global_to_internal_dof[gi];

            // stiffness
            for (int j = 0; j < n_sides; ++j) {
                int gj = elem.vertex_local_to_global_dof[j];
                Real val = Ke(i, j);

                if (std::fabs(val) < 1e-14) continue;

                if (is_internal_dof[gj]) {
                    int col = liquid_global_to_internal_dof[gj];
                    triplets.emplace_back(row, col, val);
                }
                else if (is_Dirichlet_dof[gj]) {
                    Real phi_i = level_set[gi];
                    Real phi_j = level_set[gj];
                    Real theta = phi_j/phi_i;
                    // if (std::fabs(theta)>1e3)continue;
                    triplets.emplace_back(row, row, val*theta);
                }
                else RZ_FATAL("");
            }
        }
    }
    A.setFromTriplets(triplets.begin(), triplets.end());
    return A;
}
