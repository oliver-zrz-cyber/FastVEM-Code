#include "vem3d_internal.h"
#include "Visualization.h"
#include "mgsolver_vemp.h"
#include <Eigen/Geometry>
#include "vex_pcg.h"
#include <execution>
#if defined(__GLIBC__)
#include <malloc.h>
#endif



void VEM_Domain_3d::init_flip_particles(std::vector<FLIP_Particles_3d>& flip_particles,
    std::vector<FLIP_Particles_3d>& ghost_particles,
    Real kernel_radius) {
    FUNC_TIMER();
    flip_particles.clear();
    const int N = std::max(1, int((1.0 / kernel_radius) * 2.0));
    flip_particles.reserve(N * N * N);
    Real step = 1.0f / static_cast<Real>(N);  // 均匀间距
    for (int i = -2; i < N+2; ++i) {
        for (int j = -2; j < N+2; ++j) {
            for (int k = -2; k < N+2; ++k) {
                Real x = i * step + step * 0.5f;
                Real y = j * step + step * 0.5f;
                Real z = k * step + step * 0.5f;
                _3d_vec pos(x, y, z);
                _3d_vec vel(0.0f, 0.0f, 0.0f);
                FLIP_Particles_3d particle;
                particle.pos = pos;
                particle.old_pos = pos;
                particle.vec = vel;
                particle.pic_vec =vel;
                flip_particles.push_back(particle);
            }
        }
    }
    // delete flips inside boudarys
    assign_flip_smoke_particles_to_elements_3d(flip_particles);
    std::vector<FLIP_Particles_3d> new_particles;
    for (auto &p:flip_particles) {
        if (!p.deleted) new_particles.push_back(p);
        else {
            p.deleted = false;
            ghost_particles.push_back(p);
        }
    }
    flip_particles.swap(new_particles);
    assign_flip_smoke_particles_to_elements_3d(flip_particles);
}

// void VEM_Domain_3d::element_wise_init_flip_particles(std::vector<FLIP_Particles_3d> &flip_particles) {
//     flip_particles.clear();
//     int N = 160;
//     flip_particles.reserve(N * N * N);
//     Real per_flip_volume = 1.0/N/N/N;
//     for (auto&polyhedron:vem_elements) {
//
//     }
// }
static inline Real frand()
{
    // thread-local RNG，避免 OpenMP 数据竞争
    static thread_local std::mt19937 rng(
        std::random_device{}()
    );
    static thread_local std::uniform_real_distribution<Real> dist(0.0, 1.0);
    return dist(rng);
}


void VEM_Domain_3d::build_element_bboxes() {
    FUNC_TIMER();
    elem_bboxes.clear();
    elem_bboxes.resize(vem_elements.size());
#pragma omp parallel for schedule(static)
    for (int eid = 0; eid < (int)vem_elements.size(); ++eid) {
        const auto &element = vem_elements[eid];
        Real xmin = 1e9, xmax = -1e9;
        Real ymin = 1e9, ymax = -1e9;
        Real zmin = 1e9, zmax = -1e9;

        for (int vid = 0; vid < element.vertex_count(); ++vid) {
            const _3d_vec &v = element.vertex_position(vid);
            xmin = std::min(xmin, (Real)v.vx);
            xmax = std::max(xmax, (Real)v.vx);
            ymin = std::min(ymin, (Real)v.vy);
            ymax = std::max(ymax, (Real)v.vy);
            zmin = std::min(zmin, (Real)v.vz);
            zmax = std::max(zmax, (Real)v.vz);
        }
        elem_bboxes[eid] = {xmin, xmax, ymin, ymax, zmin, zmax};
    }
}
void VEM_Domain_3d::link_ele_with_uniform_grid_3d() {
    FUNC_TIMER();
    grid_to_elems.clear();
    grid_to_elems.resize(grid_res * grid_res * grid_res);

    for (int eid = 0; eid < (int)vem_elements.size(); ++eid) {
        const auto &bb = elem_bboxes[eid];

        int x0 = std::max(0, int(bb.xmin * grid_res));
        int x1 = std::min(grid_res - 1, int(bb.xmax * grid_res));
        int y0 = std::max(0, int(bb.ymin * grid_res));
        int y1 = std::min(grid_res - 1, int(bb.ymax * grid_res));
        int z0 = std::max(0, int(bb.zmin * grid_res));
        int z1 = std::min(grid_res - 1, int(bb.zmax * grid_res));

        for (int i = x0; i <= x1; ++i)
            for (int j = y0; j <= y1; ++j)
                for (int k = z0; k <= z1; ++k) {
                    int gidx = (k * grid_res + j) * grid_res + i;
                    grid_to_elems[gidx].push_back(eid);
                }
    }
}
bool out_of_domain(const _3d_vec &p) {
    bool out = false;
    if (p.vx < 0 || p.vx > 1 || p.vy < 0 || p.vy > 1 || p.vz < 0 || p.vz > 1)out = true;
    return out;
}

bool point_in_convex_polyhedron(const _3d_vec &p, const VEM_Element_3d &elem) {
    return elem.point_in_elem(p);

}

void VEM_Domain_3d::Boundary_Handle(Smoke_Particles_3d& particle, AABBTree& aabb_tree,const Real& dt) {
    auto pos_pre = CGAL_Point_3(particle.old_pos.vx, particle.old_pos.vy, particle.old_pos.vz);
    auto pos_now = CGAL_Point_3(particle.pos.vx, particle.pos.vy, particle.pos.vz);
    auto dir = particle.pos - particle.old_pos;
    K::Segment_3 seg(pos_pre,pos_now);
    auto intersection = aabb_tree.any_intersection(seg);
    auto get_grid_index = [&](const _3d_vec &p) noexcept -> int {
        int gx = std::min(grid_res - 1, std::max(0, int(p.vx * grid_res)));
        int gy = std::min(grid_res - 1, std::max(0, int(p.vy * grid_res)));
        int gz = std::min(grid_res - 1, std::max(0, int(p.vz * grid_res)));
        return (gz * grid_res + gy) * grid_res + gx;
    };
    if(intersection)
    {
        const CGAL_Point_3* p = boost::get<CGAL_Point_3>(&(intersection->first));
        if(p) {
            particle.pos.vx = p->x();
            particle.pos.vy = p->y();
            particle.pos.vz = p->z();
            // // // 用这个点的速度...
            // const int gidx = get_grid_index(particle.pos);
            // const auto &cand_elems = grid_to_elems[gidx];
            // for (int eid : cand_elems) {
            //     if (point_in_convex_polyhedron(particle.pos, vem_elements[eid])) {
            //         const auto& element = vem_elements[eid];
            //         const auto &pos = particle.pos;
            //         Real vx = 0.0, vy = 0.0, vz = 0.0;
            //         auto poly_coeffs_u = element.poly_coeffs_u_k1;
            //         auto poly_coeffs_v = element.poly_coeffs_v_k1;
            //         auto poly_coeffs_w = element.poly_coeffs_w_k1;
            //         // 基函数插值
            //         for (int i = 0; i < poly_coeffs_u.size(); ++i) {
            //             const auto &mono = monomials_3d[i];
            //             Real val = mono.value(
            //                 pos.vx, pos.vy, pos.vz,
            //                 element.centroid.vx,
            //                 element.centroid.vy,
            //                 element.centroid.vz,
            //                 element.diameter);
            //
            //             vx += poly_coeffs_u(i) * val;
            //             vy += poly_coeffs_v(i) * val;
            //             vz += poly_coeffs_w(i) * val;
            //         }
            //         _3d_vec vel(vx,vy,vz);
            //         particle.pos = particle.old_pos + vel * dt;
            //         break;
            //     }
            // }
            // particle.pos = particle.old_pos;
            particle.deleted = false;
        }

    }
}

void VEM_Domain_3d::Boundary_Handle(FLIP_Particles_3d& particle, AABBTree& aabb_tree,const Real& dt) {
    if ((particle.old_pos-particle.pos).norm() < 1e-25) {
        particle.pos = particle.old_pos;
        return;
    }
    auto pos_pre = CGAL_Point_3(particle.old_pos.vx, particle.old_pos.vy, particle.old_pos.vz);
    auto pos_now = CGAL_Point_3(particle.pos.vx, particle.pos.vy, particle.pos.vz);
    auto dir = particle.pos - particle.old_pos;
    K::Segment_3 seg(pos_pre,pos_now);
    auto intersection = aabb_tree.any_intersection(seg);
    auto get_grid_index = [&](const _3d_vec &p) noexcept -> int {
        int gx = std::min(grid_res - 1, std::max(0, int(p.vx * grid_res)));
        int gy = std::min(grid_res - 1, std::max(0, int(p.vy * grid_res)));
        int gz = std::min(grid_res - 1, std::max(0, int(p.vz * grid_res)));
        return (gz * grid_res + gy) * grid_res + gx;
    };
    if(intersection)
    {
        const CGAL_Point_3* p = boost::get<CGAL_Point_3>(&(intersection->first));
        if(p) {
            particle.pos.vx = p->x();
            particle.pos.vy = p->y();
            particle.pos.vz = p->z();
            // // 用这个点的速度...
            Real not_used_dt_ratio = 1.0 - (particle.pos - particle.old_pos).norm()/dir.norm();
            const int gidx = get_grid_index(particle.pos);
            const auto &cand_elems = grid_to_elems[gidx];
            for (int eid : cand_elems) {
                if (point_in_convex_polyhedron(particle.pos, vem_elements[eid])) {
                    const auto& element = vem_elements[eid];
                    const auto &pos = particle.pos;
                    Real vx = 0.0, vy = 0.0, vz = 0.0;
                    auto poly_coeffs_u = element.poly_coeffs_u_k1;
                    auto poly_coeffs_v = element.poly_coeffs_v_k1;
                    auto poly_coeffs_w = element.poly_coeffs_w_k1;
                    // 基函数插值
                    for (int i = 0; i < poly_coeffs_u.size(); ++i) {
                        const auto &mono = monomials_3d[i];
                        Real val = mono.value(
                            pos.vx, pos.vy, pos.vz,
                            element.centroid.vx,
                            element.centroid.vy,
                            element.centroid.vz,
                            element.diameter);

                        vx += poly_coeffs_u(i) * val;
                        vy += poly_coeffs_v(i) * val;
                        vz += poly_coeffs_w(i) * val;
                    }
                    _3d_vec vel(vx,vy,vz);
                    particle.pos = particle.pos + vel*dt*not_used_dt_ratio;
                    break;
                }
            }
            particle.deleted = false;
        }
        // else particle.pos = particle.old_pos;

    }
}

void VEM_Domain_3d::Boundary_Handle(_3d_vec old_pos, _3d_vec& new_pos, AABBTree &aabb_tree, const Real &dt) {
 if ((old_pos-new_pos).norm() < 1e-25) {
        new_pos = old_pos;
        return;
    }

    auto pos_pre = CGAL_Point_3(old_pos.vx, old_pos.vy, old_pos.vz);
    auto pos_now = CGAL_Point_3(new_pos.vx, new_pos.vy,new_pos.vz);
    K::Segment_3 seg(pos_pre,pos_now);
    if (!aabb_tree.do_intersect(seg))return;
    K::Ray_3 ray(pos_pre,pos_now-pos_pre);

    auto dir = new_pos - old_pos;
    auto intersection = aabb_tree.first_intersection(ray);
    auto get_grid_index = [&](const _3d_vec &p) noexcept -> int {
        int gx = std::min(grid_res - 1, std::max(0, int(p.vx * grid_res)));
        int gy = std::min(grid_res - 1, std::max(0, int(p.vy * grid_res)));
        int gz = std::min(grid_res - 1, std::max(0, int(p.vz * grid_res)));
        return (gz * grid_res + gy) * grid_res + gx;
    };
    if(intersection)
    {
        if(boost::get<CGAL_Point_3>(&(intersection->first))){
            const CGAL_Point_3* p =  boost::get<CGAL_Point_3>(&(intersection->first) );
            new_pos.vx = p->x();
            new_pos.vy = p->y();
            new_pos.vz = p->z();
            // // // // 用这个点的速度...
            // Real not_used_dt_ratio = 1.0 - (new_pos - old_pos).norm()/dir.norm();
            // const int gidx = get_grid_index(new_pos);
            // const auto &cand_elems = grid_to_elems[gidx];
            // for (int eid : cand_elems) {
            //     if (point_in_convex_polyhedron(new_pos, vem_elements[eid])) {
            //         const auto& element = vem_elements[eid];
            //         const auto &pos = new_pos;
            //         Real vx = 0.0, vy = 0.0, vz = 0.0;
            //         auto poly_coeffs_u = element.poly_coeffs_u_k1;
            //         auto poly_coeffs_v = element.poly_coeffs_v_k1;
            //         auto poly_coeffs_w = element.poly_coeffs_w_k1;
            //         // 基函数插值
            //         for (int i = 0; i < poly_coeffs_u.size(); ++i) {
            //             const auto &mono = monomials_3d[i];
            //             Real val = mono.value(
            //                 pos.vx, pos.vy, pos.vz,
            //                 element.centroid.vx,
            //                 element.centroid.vy,
            //                 element.centroid.vz,
            //                 element.diameter);
            //
            //             vx += poly_coeffs_u(i) * val;
            //             vy += poly_coeffs_v(i) * val;
            //             vz += poly_coeffs_w(i) * val;
            //         }
            //         _3d_vec vel(vx,vy,vz);
            //         new_pos = new_pos + vel*dt*not_used_dt_ratio;
            //         break;
            //     }
            // }
            // pos_now = CGAL_Point_3(new_pos.vx, new_pos.vy,new_pos.vz);
            // K::Segment_3 seg_new(pos_pre,pos_now);
            // if (aabb_tree.do_intersect(seg_new)) {
            //     new_pos.vx = p->x();
            //     new_pos.vy = p->y();
            //     new_pos.vz = p->z();
            // }
            dir = new_pos - old_pos;
            new_pos = new_pos - dir*(1e-4/dir.norm());
        }
        else {
            RZ_FATAL("why?");
            new_pos.vx = -100;//deleted
            // new_pos = old_pos;
        }
    }

}


void VEM_Domain_3d::assign_flip_smoke_particles_to_elements_3d(std::vector<FLIP_Particles_3d>& flip_particles
                                                               ,std::vector<Smoke_Particles_3d>&smoke_particles) {

    // 清空旧分配
    for (auto &elem : vem_elements) {
        elem.contained_flip_particles.clear();
        elem.contained_smoke_particles.clear();
    }

   auto get_grid_index = [&](const _3d_vec &p) noexcept -> int {
        int gx = std::min(grid_res - 1, std::max(0, int(p.vx * grid_res)));
        int gy = std::min(grid_res - 1, std::max(0, int(p.vy * grid_res)));
        int gz = std::min(grid_res - 1, std::max(0, int(p.vz * grid_res)));
        return (gz * grid_res + gy) * grid_res + gx;
    };

    // ===============================
    // 1️⃣ 并行处理 FLIP 粒子
    // ===============================
#pragma omp parallel for schedule(dynamic)
    for (int pid = 0; pid < static_cast<int>(flip_particles.size()); ++pid) {
        auto &fp = flip_particles[pid];
        fp.contained_in_element_id = -1;
        if (fp.deleted) continue;

        const _3d_vec &p = fp.pos;
        if (out_of_domain(p)) {
            fp.deleted = true;
            continue;
        }

        const int gidx = get_grid_index(p);
        const auto &cand_elems = grid_to_elems[gidx];
        if (cand_elems.empty()) {
            fp.deleted = true;
            continue;
        }

        bool assigned = false;
        for (int eid : cand_elems) {
            if (vem_elements[eid].point_in_elem(p)) {//todo:vem_elements[eid].point_in_elem(p)
                fp.contained_in_element_id = eid;
                assigned = true;
                break;
            }
        }

        if (!assigned) fp.deleted = true;
    }

    // ===============================
    // 2️⃣ 并行处理 SMOKE 粒子
    // ===============================
#pragma omp parallel for schedule(dynamic)
    for (int pid = 0; pid < static_cast<int>(smoke_particles.size()); ++pid) {
        auto &fp = smoke_particles[pid];
        fp.contained_in_element_id = -1;
        if (fp.deleted) continue;

        const _3d_vec &p = fp.pos;
        if (out_of_domain(p)) {
            fp.deleted = true;
            continue;
        }

        const int gidx = get_grid_index(p);
        const auto &cand_elems = grid_to_elems[gidx];
        if (cand_elems.empty()) {
            fp.deleted = true;
            continue;
        }

        bool assigned = false;
        for (int eid : cand_elems) {
            if (vem_elements[eid].point_in_elem(p)) {
                fp.contained_in_element_id = eid;
                assigned = true;
                break;
            }
        }

        if (!assigned) fp.deleted = true;
    }

    for (int pid=0;pid<flip_particles.size();++pid) {
        const auto&flip_particle = flip_particles[pid];
        if (!flip_particle.deleted)
            vem_elements[flip_particle.contained_in_element_id].contained_flip_particles.push_back(pid);
    }
    for (int pid=0;pid<smoke_particles.size();++pid) {
        const auto&smoke_particle = smoke_particles[pid];
        if (!smoke_particle.deleted)
            vem_elements[smoke_particle.contained_in_element_id].contained_smoke_particles.push_back(pid);
    }
}

void VEM_Domain_3d::assign_flip_smoke_particles_to_elements_3d(std::vector<FLIP_Particles_3d> &flip_particles) {
    FUNC_TIMER();
    // 清空旧分配
    for (auto &elem : vem_elements) {
        elem.contained_flip_particles.clear();
    }

   auto get_grid_index = [&](const _3d_vec &p) noexcept -> int {
        int gx = std::min(grid_res - 1, std::max(0, int(p.vx * grid_res)));
        int gy = std::min(grid_res - 1, std::max(0, int(p.vy * grid_res)));
        int gz = std::min(grid_res - 1, std::max(0, int(p.vz * grid_res)));
        return (gz * grid_res + gy) * grid_res + gx;
    };

    // ===============================
    // 1️⃣ 并行处理 FLIP 粒子
    // ===============================
#pragma omp parallel for schedule(dynamic)
    for (int pid = 0; pid < static_cast<int>(flip_particles.size()); ++pid) {
        auto &fp = flip_particles[pid];
        fp.contained_in_element_id = -1;
        if (fp.deleted) continue;

        const _3d_vec &p = fp.pos;
        if (out_of_domain(p)) {
            fp.deleted = true;
            continue;
        }

        const int gidx = get_grid_index(p);
        const auto &cand_elems = grid_to_elems[gidx];
        if (cand_elems.empty()) {
            fp.deleted = true;
            continue;
        }

        bool assigned = false;
        for (int eid : cand_elems) {
            if (vem_elements[eid].point_in_elem(p)) {//todo:vem_elements[eid].point_in_elem(p)
                fp.contained_in_element_id = eid;
                assigned = true;
                break;
            }
        }

        if (!assigned) fp.deleted = true;
    }

    for (int pid=0;pid<flip_particles.size();++pid) {
        const auto&flip_particle = flip_particles[pid];
        if (!flip_particle.deleted)
            vem_elements[flip_particle.contained_in_element_id].contained_flip_particles.push_back(pid);
    }

}

std::vector<int> VEM_Domain_3d::assign_pts_to_elements_3d(std::vector<_3d_vec> &pts) {
    // 清空旧分配
    FUNC_TIMER();
    auto get_grid_index = [&](const _3d_vec &p) noexcept -> int {
        int gx = std::min(grid_res - 1, std::max(0, int(p.vx * grid_res)));
        int gy = std::min(grid_res - 1, std::max(0, int(p.vy * grid_res)));
        int gz = std::min(grid_res - 1, std::max(0, int(p.vz * grid_res)));
        return (gz * grid_res + gy) * grid_res + gx;
    };

    // ===============================
    // 1️⃣ 并行处理 FLIP 粒子
    // ===============================
    std::vector<int>ans(pts.size(),-1);

#pragma omp parallel for schedule(static)
    for (int pid = 0; pid < static_cast<int>(pts.size()); ++pid) {

        const _3d_vec &p = pts[pid];
        if (out_of_domain(p)) {
            continue;
        }

        const int gidx = get_grid_index(p);
        const auto &cand_elems = grid_to_elems[gidx];
        if (cand_elems.empty()) {
            continue;
        }

        for (int eid : cand_elems) {
            if (vem_elements[eid].point_in_elem(p)) {//todo:vem_elements[eid].point_in_elem(p)
                ans[pid] = eid;
                break;
            }
        }
    }
    return ans;
}


std::vector<int> VEM_Domain_3d::assign_pts_to_elements_3d_with_BH(std::vector<_3d_vec> &pts,
    const std::vector<_3d_vec> &old_pts) {
    auto first_assigned = assign_pts_to_elements_3d(pts);
    std::vector<int> to_handle;
    for (int i=0;i<first_assigned.size();++i)
        if (first_assigned[i]==-1)
            to_handle.push_back(i);
    // 处理出界的flip粒子
    std::for_each(std::execution::par, to_handle.begin(), to_handle.end(),
        [&](int i) {
        Boundary_Handle(old_pts[i],pts[i],aabb_tree,0.1);
    }
    );
    std::vector<int> ans = assign_pts_to_elements_3d(pts);
    return ans;
}

std::vector<int> VEM_Domain_3d::assign_tracers_to_elements_3d() {
    // 清空旧分配
    FUNC_TIMER();
    auto get_grid_index = [&](const _3d_vec &p) noexcept -> int {
        int gx = std::min(grid_res - 1, std::max(0, int(p.vx * grid_res)));
        int gy = std::min(grid_res - 1, std::max(0, int(p.vy * grid_res)));
        int gz = std::min(grid_res - 1, std::max(0, int(p.vz * grid_res)));
        return (gz * grid_res + gy) * grid_res + gx;
    };

    // ===============================
    // 1️⃣ 并行处理 FLIP 粒子
    // ===============================
    std::vector<int>ans(tracers.size(),-1);

#pragma omp parallel for schedule(dynamic)
    for (int pid = 0; pid < static_cast<int>(tracers.size()); ++pid) {
        const _3d_vec p (tracers[pid].x,tracers[pid].y,tracers[pid].z);
        if (out_of_domain(p)) {
            continue;
        }

        const int gidx = get_grid_index(p);
        const auto &cand_elems = grid_to_elems[gidx];
        if (cand_elems.empty()) {
            continue;
        }

        for (int eid : cand_elems) {
            if (vem_elements[eid].point_in_elem(p,1e-10)) {
                ans[pid] = eid;
                break;
            }
        }
    }
    return ans;
}

std::vector<int> VEM_Domain_3d::assign_lucy_tracers1_to_elements_3d() {
    // 清空旧分配
    FUNC_TIMER();
    auto get_grid_index = [&](const _3d_vec &p) noexcept -> int {
        int gx = std::min(grid_res - 1, std::max(0, int(p.vx * grid_res)));
        int gy = std::min(grid_res - 1, std::max(0, int(p.vy * grid_res)));
        int gz = std::min(grid_res - 1, std::max(0, int(p.vz * grid_res)));
        return (gz * grid_res + gy) * grid_res + gx;
    };


        auto &tracers = lucy_tracers_1;
        std::vector<int>ans(tracers.size(),-1);

#pragma omp parallel for schedule(dynamic)
        for (int pid = 0; pid < static_cast<int>(tracers.size()); ++pid) {
            const _3d_vec p (tracers[pid].x,tracers[pid].y,tracers[pid].z);
            if (out_of_domain(p)) {
                continue;
            }

            const int gidx = get_grid_index(p);
            const auto &cand_elems = grid_to_elems[gidx];
            if (cand_elems.empty()) {
                continue;
            }

            for (int eid : cand_elems) {
                if (vem_elements[eid].point_in_elem(p,1e-10)) {
                    ans[pid] = eid;
                    break;
                }
            }
        }

    return ans;
}


std::vector<int> VEM_Domain_3d::assign_lucy_tracers2_to_elements_3d() {
    // 清空旧分配
    FUNC_TIMER();
    auto get_grid_index = [&](const _3d_vec &p) noexcept -> int {
        int gx = std::min(grid_res - 1, std::max(0, int(p.vx * grid_res)));
        int gy = std::min(grid_res - 1, std::max(0, int(p.vy * grid_res)));
        int gz = std::min(grid_res - 1, std::max(0, int(p.vz * grid_res)));
        return (gz * grid_res + gy) * grid_res + gx;
    };


    auto &tracers = lucy_tracers_2;
    std::vector<int>ans(tracers.size(),-1);

#pragma omp parallel for schedule(dynamic)
    for (int pid = 0; pid < static_cast<int>(tracers.size()); ++pid) {
        const _3d_vec p (tracers[pid].x,tracers[pid].y,tracers[pid].z);
        if (out_of_domain(p)) {
            continue;
        }

        const int gidx = get_grid_index(p);
        const auto &cand_elems = grid_to_elems[gidx];
        if (cand_elems.empty()) {
            continue;
        }

        for (int eid : cand_elems) {
            if (vem_elements[eid].point_in_elem(p,1e-10)) {
                ans[pid] = eid;
                break;
            }
        }
    }

    return ans;
}

inline Real rbf_gaussian(Real r, Real epsilon) {
    return std::exp(-(r * r) / (epsilon * epsilon));
}

void VEM_Domain_3d::PtoG(std::vector<FLIP_Particles_3d> &flip_particles,
    std::vector<FLIP_Particles_3d>& ghost_particles,Real cubk) {
    FUNC_TIMER();
    const Real eps = 0.001;
    // ----------------------------
    // 0) 统计 deleted（可选，但你原来有）
    // ----------------------------
    int deleted = 0;
    for (size_t pid = 0; pid < flip_particles.size(); ++pid)
        if (flip_particles[pid].deleted) ++deleted;

    RZ_INFO("deleted/all flip: ",
            (Real)deleted / std::max<size_t>(1, flip_particles.size()));

    k1_dof_u.setZero();
    k1_dof_v.setZero();
    k1_dof_w.setZero();
    k1_dof_density.setZero();
    k1_dof_temp.setZero();


    int n = std::ceil(1.0/cubk);
    RZ_INFO("ptogn:",n);
    UNGRID ptog_ugrid(n);
    ptog_ugrid.assign_particles_to_grid(flip_particles);
    // ptog_ugrid.add_ghost_particles_to_grid(ghost_particles);
    const Real h_2 = cubk * cubk;
    // ----------------------------
    // 3) DOF 点逐点做 P2G（粒子 -> DOF）重建
    // ----------------------------
// #pragma omp parallel for
    for (int i = 0; i < (int)k1_dof_u.size(); ++i) {
        const auto &cp = points[i];
        int grid_i, grid_j, grid_k;
        ptog_ugrid.get_grid_index(cp, grid_i, grid_j, grid_k);
        // 只取 1-ring 邻域 cell（3×3×3）
        auto linked_flips = ptog_ugrid.get_neighbors({grid_i, grid_j, grid_k}, 1);
        Real u_acc = 0.0, v_acc = 0.0, w_acc = 0.0;
        Real rho_acc = 0.0, temp_acc = 0.0;
        Real w_accum = 1e-16; // 防止除零（比 1e-4 更不偏置）
        std::vector<int> see_flips;

        CGAL_Point_3 source(cp.vx, cp.vy, cp.vz);
        // std::set<int> see_vem3d_ids;
        // std::set<int> tested_vem3d_ids;
        // for (auto&particle:linked_flips) {
        //     if (particle.second) {
        //         continue;
        //     }
        //     const auto &fp = flip_particles[particle.first];
        //     if (fp.contained_in_element_id == -1 || fp.deleted ||
        //         tested_vem3d_ids.count(fp.contained_in_element_id))continue;
        //     tested_vem3d_ids.insert(fp.contained_in_element_id);
        //     const _3d_vec& cell_c = vem_elements[fp.contained_in_element_id].centroid;
        //     CGAL_Point_3 end(cell_c.vx, cell_c.vy, cell_c.vz);
        //     if ((cp-cell_c).norm()<1e-25) {
        //         see_vem3d_ids.insert(fp.contained_in_element_id);
        //         continue;
        //     }
        //     K::Segment_3 seg(source,end);
        //     if (!aabb_tree.do_intersect(seg)) {
        //         see_vem3d_ids.insert(fp.contained_in_element_id);
        //     }
        // }
        //---------------------------------------
        for (auto&particle:linked_flips) {
            const auto &fp = flip_particles[particle.first];
            if (fp.deleted)continue;
            // if (!see_vem3d_ids.count(fp.contained_in_element_id)) continue;
            see_flips.push_back(particle.first);
        }


        for (auto &flipid : see_flips)
        {
            const auto &fp = flip_particles[flipid];
            if (fp.deleted) continue;
            const Real r_2 = (fp.pos-cp).squared_norm();
            const Real w = cubic_kernel(r_2,h_2);
            u_acc    += w * fp.vec.vx;
            v_acc    += w * fp.vec.vy;
            w_acc    += w * fp.vec.vz;
            rho_acc  += w * fp.rho;
            temp_acc += w * fp.temp;
            w_accum  += w;
        }

        // for (auto &ghostid : see_ghosts)
        // {
        //     const auto &fp = ghost_particles[ghostid];
        //     const Real r_2 = (fp.pos-cp).squared_norm();
        //     const Real w = cubic_kernel(r_2,h_2);
        //     w_accum  += w;
        // }

        k1_dof_u[i]       = u_acc / w_accum;
        k1_dof_v[i]       = v_acc / w_accum;
        k1_dof_w[i]       = w_acc / w_accum;
        k1_dof_density[i] = rho_acc / w_accum;
        k1_dof_temp[i]    = temp_acc / w_accum;

        // if (w_accum < eps) {
        //     k1_dof_density[i] = 0.0;
        //     k1_dof_temp[i]   = 0.0;
        //     k1_dof_u[i]      = 0.0;
        //     k1_dof_v[i]      = 0.0;
        //     k1_dof_w[i]      = 0.0;
        // }
    }


    for (int i = 0; i < (int)k1_dof_u.size(); ++i)
    {
        k1_dof_du[i] = k1_dof_u[i];
        k1_dof_dv[i] = k1_dof_v[i];
        k1_dof_dw[i] = k1_dof_w[i];
    }

    // convert to polynomial space as old v also temp new v
#pragma omp parallel for schedule(dynamic)
    for (auto &element : vem_elements) {
        const auto &projector_k1 =element.is_U?
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
        // if (element.has_boundary) {
        //     const auto &elem = element;
        //     for (int local_pid=0;local_pid<elem.polygons.size();++local_pid) {
        //         const int global_pid = elem.polygon_local_to_global_dof[local_pid];
        //         if (!is_polygon_boundary[global_pid])continue;
        //         std::vector<_3d_vec> constrained_dirs;
        //         const _3d_vec& normal = _3d_vec(0,0,0) - _3d_vec(elem.poly_planes[local_pid].orthogonal_vector().x(),
        //             elem.poly_planes[local_pid].orthogonal_vector().y(),
        //             elem.poly_planes[local_pid].orthogonal_vector().z());
        //         constrained_dirs.push_back(normal);
        //         for (auto& local_vid : elem.polygons[local_pid]) {
        //             {
        //                 // v
        //                 _3d_vec v(element_dof_values_u[local_vid], element_dof_values_v[local_vid],
        //                     element_dof_values_w[local_vid]);
        //                 for (auto& constrained_dir : constrained_dirs) {
        //                     Real dot_product = v.vx * constrained_dir.vx + v.vy * constrained_dir.vy + v.vz * constrained_dir.vz;
        //                     _3d_vec projection = constrained_dir * dot_product;
        //                     if (dot_product > 0.0)
        //                         v = v - projection;
        //                 }
        //                 element_dof_values_u[local_vid] = v.vx;
        //                 element_dof_values_v[local_vid] = v.vy;
        //                 element_dof_values_w[local_vid] = v.vz;
        //             }
        //         }
        //     }
        //
        // }
        RZ_ASSERT(projector_k1.cols() == ndof, "projector_k1.cols()==element_dof_values.size()");
        // === 计算该元素的多项式系数 ===
        element.poly_coeffs_oldu_k1 = projector_k1 * element_dof_values_u;
        element.poly_coeffs_oldv_k1 = projector_k1 * element_dof_values_v;
        element.poly_coeffs_oldw_k1 = projector_k1 * element_dof_values_w;
    }
    Real max_v = std::fmax(k1_dof_u.maxCoeff(), std::fmax(k1_dof_v.maxCoeff(),k1_dof_w.maxCoeff()));
    RZ_INFO("max v:",max_v);


}

void VEM_Domain_3d::apply_pressure_k1_poly(Real dt) {
    FUNC_TIMER();
    int element_id = 0;

    for (const auto &element: vem_elements) {
        Eigen::VectorXr element_dof_values_p;
        element_dof_values_p.resize(element.vertex_count());
        element_dof_values_p.setZero();
        const auto &proj = element.is_U==true ?
            U3d_project_matrix_star_k1:element.project_matrix_star_k1;
        const auto &verts = element.vertex_local_to_global_dof;
        int offset = 0;
        // === 1️⃣ 顶点自由度 ===
        const int nv = (int)verts.size();
        for (int i = 0; i < nv; ++i)
            element_dof_values_p[offset++] = global_p_k1[verts[i]];

        const Eigen::VectorXr poly_coeffs_p = proj * element_dof_values_p;

        Eigen::VectorXr pressure_du_poly_coeffs,pressure_dv_poly_coeffs,pressure_dw_poly_coeffs;
        pressure_du_poly_coeffs.resize(4);
        pressure_dv_poly_coeffs.resize(4);
        pressure_dw_poly_coeffs.resize(4);
        pressure_du_poly_coeffs.setZero();
        pressure_dv_poly_coeffs.setZero();
        pressure_dw_poly_coeffs.setZero();

        pressure_du_poly_coeffs[0] = -1.0/element.diameter*poly_coeffs_p[1]*dt;

        pressure_dv_poly_coeffs[0] = -1.0/element.diameter*poly_coeffs_p[2]*dt;

        pressure_dw_poly_coeffs[0] = -1.0/element.diameter*poly_coeffs_p[3]*dt;


        vem_elements[element_id].poly_coeffs_u_k1 += pressure_du_poly_coeffs;
        vem_elements[element_id].poly_coeffs_v_k1 += pressure_dv_poly_coeffs;
        vem_elements[element_id].poly_coeffs_w_k1 += pressure_dw_poly_coeffs;

        vem_elements[element_id].poly_coeffs_du_k1 = vem_elements[element_id].poly_coeffs_u_k1 -
            vem_elements[element_id].poly_coeffs_oldu_k1;
        vem_elements[element_id].poly_coeffs_dv_k1 = vem_elements[element_id].poly_coeffs_v_k1 -
            vem_elements[element_id].poly_coeffs_oldv_k1;
        vem_elements[element_id].poly_coeffs_dw_k1 = vem_elements[element_id].poly_coeffs_w_k1 -
            vem_elements[element_id].poly_coeffs_oldw_k1;
        element_id++;
    }
}

void VEM_Domain_3d::apply_pressure_k2_poly(Real dt) {
    FUNC_TIMER();
    int element_id = 0;
    const int base_v = 0;
    const int base_e = (int)points.size();
    const int base_f = base_e + (int)edges.size();
    const int base_c = base_f + (int)polygons.size();

    for (const auto &element: vem_elements) {
        Eigen::VectorXr element_dof_values_p;
        element_dof_values_p.resize(element.vertex_count()+
            element.edge_count()+
            element.polygons.size()
            +1);
        element_dof_values_p.setZero();
        const auto &proj = element.is_U==true ?
            U3d_project_matrix_star_k2:element.project_matrix_star_k2;
        const auto &verts = element.vertex_local_to_global_dof;
        int offset = 0;
        // === 1️⃣ 顶点自由度 ===
        const int nv = (int)verts.size();
        for (int i = 0; i < nv; ++i)
            element_dof_values_p[offset++] = global_p_k2[base_v + verts[i]];
        // === 2️⃣ 边自由度 ===
        for (int eid = 0; eid < (int)element.edge_count(); ++eid) {
            element_dof_values_p[offset++] =
                global_p_k2[base_e + element.edge_local_to_global_dof[eid]];
        }
        // === 3️⃣ 面自由度 ===
        const int npoly = (int)element.polygons.size();
        for (int i = 0; i < npoly; ++i)
            element_dof_values_p[offset++] =
                global_p_k2[base_f + element.polygon_local_to_global_dof[i]];
        // === 4️⃣ 体自由度 ===
        element_dof_values_p[offset++] = global_p_k2[base_c + element_id];
        // === 5️⃣ 投影 (无中间分配) ===
        const Eigen::VectorXr poly_coeffs_p = proj * element_dof_values_p;

        Eigen::VectorXr pressure_du_poly_coeffs,pressure_dv_poly_coeffs,pressure_dw_poly_coeffs;
        pressure_du_poly_coeffs.resize(4);
        pressure_dv_poly_coeffs.resize(4);
        pressure_dw_poly_coeffs.resize(4);
        pressure_du_poly_coeffs.setZero();
        pressure_dv_poly_coeffs.setZero();
        pressure_dw_poly_coeffs.setZero();

        pressure_du_poly_coeffs[0] = -1.0/element.diameter*poly_coeffs_p[1]*dt;
        pressure_du_poly_coeffs[1] = -2.0/element.diameter*poly_coeffs_p[4]*dt;
        pressure_du_poly_coeffs[2] = -1.0/element.diameter*poly_coeffs_p[5]*dt;
        pressure_du_poly_coeffs[3] = -1.0/element.diameter*poly_coeffs_p[6]*dt;

        pressure_dv_poly_coeffs[0] = -1.0/element.diameter*poly_coeffs_p[2]*dt;
        pressure_dv_poly_coeffs[1] = -1.0/element.diameter*poly_coeffs_p[5]*dt;
        pressure_dv_poly_coeffs[2] = -2.0/element.diameter*poly_coeffs_p[7]*dt;
        pressure_dv_poly_coeffs[3] = -1.0/element.diameter*poly_coeffs_p[8]*dt;

        pressure_dw_poly_coeffs[0] = -1.0/element.diameter*poly_coeffs_p[3]*dt;
        pressure_dw_poly_coeffs[1] = -1.0/element.diameter*poly_coeffs_p[6]*dt;
        pressure_dw_poly_coeffs[2] = -1.0/element.diameter*poly_coeffs_p[8]*dt;
        pressure_dw_poly_coeffs[3] = -2.0/element.diameter*poly_coeffs_p[9]*dt;

        vem_elements[element_id].poly_coeffs_u_k1 += pressure_du_poly_coeffs;
        vem_elements[element_id].poly_coeffs_v_k1 += pressure_dv_poly_coeffs;
        vem_elements[element_id].poly_coeffs_w_k1 += pressure_dw_poly_coeffs;

        vem_elements[element_id].poly_coeffs_du_k1 = vem_elements[element_id].poly_coeffs_u_k1 -
            vem_elements[element_id].poly_coeffs_oldu_k1;
        vem_elements[element_id].poly_coeffs_dv_k1 = vem_elements[element_id].poly_coeffs_v_k1 -
            vem_elements[element_id].poly_coeffs_oldv_k1;
        vem_elements[element_id].poly_coeffs_dw_k1 = vem_elements[element_id].poly_coeffs_w_k1 -
            vem_elements[element_id].poly_coeffs_oldw_k1;
        element_id++;
    }

}


void VEM_Domain_3d::GtoP_k2(std::vector<FLIP_Particles_3d>& flip_particles,Real flip_alpha,Real dt)
{
    FUNC_TIMER();
    // du
    for (int vid=0;vid<k1_dof_u.size();++vid) {
        k1_dof_du[vid] = k1_dof_u[vid] - k1_dof_du[vid];
        k1_dof_dv[vid] = k1_dof_v[vid] - k1_dof_dv[vid];
        k1_dof_dw[vid] = k1_dof_w[vid] - k1_dof_dw[vid];
    }
#pragma omp parallel for schedule(dynamic)
    for (const auto &element : vem_elements) {
        const auto &projector_k1 = element.is_U?
        U3d_project_matrix_star_k1 : element.project_matrix_star_k1;

        // === 从全局 DOF 提取当前元素的速度自由度 ===
        Eigen::VectorXr element_dof_values_u, element_dof_values_v, element_dof_values_w;
        Eigen::VectorXr element_dof_values_du, element_dof_values_dv, element_dof_values_dw;
        const auto &vert_ids = element.vertex_local_to_global_dof;
        const int ndof = (int)vert_ids.size();

        element_dof_values_u.resize(ndof);
        element_dof_values_v.resize(ndof);
        element_dof_values_w.resize(ndof);
        element_dof_values_u.setZero();
        element_dof_values_v.setZero();
        element_dof_values_w.setZero();
        element_dof_values_du.resize(ndof);
        element_dof_values_dv.resize(ndof);
        element_dof_values_dw.resize(ndof);
        element_dof_values_du.setZero();
        element_dof_values_dv.setZero();
        element_dof_values_dw.setZero();

        for (int i = 0; i < ndof; ++i) {
            int vid = vert_ids[i];
            element_dof_values_u[i] = k1_dof_u[vid];
            element_dof_values_v[i] = k1_dof_v[vid];
            element_dof_values_w[i] = k1_dof_w[vid];
            element_dof_values_du[i] = k1_dof_du[vid];
            element_dof_values_dv[i] = k1_dof_dv[vid];
            element_dof_values_dw[i] = k1_dof_dw[vid];
        }
        if (element.has_boundary) {
            const auto &elem = element;
            for (int local_pid=0;local_pid<elem.polygons.size();++local_pid) {
                const int global_pid = elem.polygon_local_to_global_dof[local_pid];
                if (!is_polygon_boundary[global_pid])continue;
                std::vector<_3d_vec> constrained_dirs;
                const _3d_vec& normal = _3d_vec(elem.poly_planes[local_pid].orthogonal_vector().x(),
                elem.poly_planes[local_pid].orthogonal_vector().y(),
                elem.poly_planes[local_pid].orthogonal_vector().z());
                constrained_dirs.push_back(normal);
                for (auto& local_vid : elem.polygons[local_pid]) {
                    {
                        // v
                        _3d_vec v(element_dof_values_u[local_vid], element_dof_values_v[local_vid],
                            element_dof_values_w[local_vid]);
                        for (auto& constrained_dir : constrained_dirs) {
                            Real dot_product = v.vx * constrained_dir.vx + v.vy * constrained_dir.vy + v.vz * constrained_dir.vz;
                            _3d_vec projection = constrained_dir * dot_product;
                            if (dot_product > 0.0)
                                v = v - projection;
                        }
                        element_dof_values_u[local_vid] = v.vx;
                        element_dof_values_v[local_vid] = v.vy;
                        element_dof_values_w[local_vid] = v.vz;
                    }
                    // dv
                    {
                        _3d_vec dv(element_dof_values_du[local_vid], element_dof_values_dv[local_vid],
                           element_dof_values_dw[local_vid]);
                        for (auto& constrained_dir : constrained_dirs) {
                            Real dot_product = dv.vx * constrained_dir.vx
                            + dv.vy * constrained_dir.vy + dv.vz * constrained_dir.vz;
                            _3d_vec projection = constrained_dir * dot_product;
                            if (dot_product > 0.0)
                                dv = dv - projection;
                        }
                        element_dof_values_du[local_vid] = dv.vx;
                        element_dof_values_dv[local_vid] = dv.vy;
                        element_dof_values_dw[local_vid] = dv.vz;
                    }
                }
            }

        }
        RZ_ASSERT(projector_k1.cols() == ndof, "projector_k1.cols()==element_dof_values.size()");
        // === 计算该元素的多项式系数 ===
        Eigen::VectorXr poly_coeffs_u = projector_k1 * element_dof_values_u;
        Eigen::VectorXr poly_coeffs_v = projector_k1 * element_dof_values_v;
        Eigen::VectorXr poly_coeffs_w = projector_k1 * element_dof_values_w;
        Eigen::VectorXr poly_coeffs_du = projector_k1 * element_dof_values_du;
        Eigen::VectorXr poly_coeffs_dv = projector_k1 * element_dof_values_dv;
        Eigen::VectorXr poly_coeffs_dw = projector_k1 * element_dof_values_dw;

        for (int p_idx = 0; p_idx < (int)element.contained_flip_particles.size(); ++p_idx) {
            int pid = element.contained_flip_particles[p_idx];
            auto &particle = flip_particles[pid];
            const auto &pos = particle.pos;

            Real vx = 0.0, vy = 0.0, vz = 0.0;
            Real dvx = 0.0, dvy = 0.0, dvz = 0.0;

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
                dvx += poly_coeffs_du(i) * val;
                dvy += poly_coeffs_dv(i) * val;
                dvz += poly_coeffs_dw(i) * val;
            }

            particle.pic_vec.vx = vx;
            particle.pic_vec.vy = vy;
            particle.pic_vec.vz = vz;

            const _3d_vec delta_vec(dvx,dvy,dvz);
            particle.vec = particle.pic_vec * flip_alpha + (particle.vec + delta_vec) * (1 - flip_alpha);
        }

    }

}

void VEM_Domain_3d::GtoP_k2_poly(std::vector<FLIP_Particles_3d> &flip_particles, Real flip_alpha) {
    FUNC_TIMER();
#pragma omp parallel for schedule(dynamic)
    for (const auto &element : vem_elements) {
        // === 计算该元素的多项式系数 ===
        const Eigen::VectorXr& poly_coeffs_u = element.poly_coeffs_u_k1;
        const Eigen::VectorXr& poly_coeffs_v = element.poly_coeffs_v_k1;
        const Eigen::VectorXr& poly_coeffs_w = element.poly_coeffs_w_k1;
        const Eigen::VectorXr& poly_coeffs_du = element.poly_coeffs_du_k1;
        const Eigen::VectorXr& poly_coeffs_dv = element.poly_coeffs_dv_k1;
        const Eigen::VectorXr& poly_coeffs_dw = element.poly_coeffs_dw_k1;

        for (int p_idx = 0; p_idx < (int)element.contained_flip_particles.size(); ++p_idx) {
            int pid = element.contained_flip_particles[p_idx];
            auto &particle = flip_particles[pid];
            const auto &pos = particle.pos;

            Real vx = 0.0, vy = 0.0, vz = 0.0;
            Real dvx = 0.0, dvy = 0.0, dvz = 0.0;

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
                dvx += poly_coeffs_du(i) * val;
                dvy += poly_coeffs_dv(i) * val;
                dvz += poly_coeffs_dw(i) * val;
            }

            particle.pic_vec.vx = vx;
            particle.pic_vec.vy = vy;
            particle.pic_vec.vz = vz;

            const _3d_vec delta_vec(dvx,dvy,dvz);
            particle.vec = particle.pic_vec * flip_alpha + (particle.vec + delta_vec) * (1 - flip_alpha);
        }
    }

}

void VEM_Domain_3d::CFL_GtoPicV(std::vector<FLIP_Particles_3d>& flip_particles, Real flip_alpha) {
    FUNC_TIMER();
#pragma omp parallel for schedule(dynamic)
    for (const auto &element : vem_elements) {
        // === 计算该元素的多项式系数 ===
        const Eigen::VectorXr& poly_coeffs_u = element.poly_coeffs_u_k1;
        const Eigen::VectorXr& poly_coeffs_v = element.poly_coeffs_v_k1;
        const Eigen::VectorXr& poly_coeffs_w = element.poly_coeffs_w_k1;
        const Eigen::VectorXr& poly_coeffs_du = element.poly_coeffs_du_k1;
        const Eigen::VectorXr& poly_coeffs_dv = element.poly_coeffs_dv_k1;
        const Eigen::VectorXr& poly_coeffs_dw = element.poly_coeffs_dw_k1;

        for (int p_idx = 0; p_idx < (int)element.contained_flip_particles.size(); ++p_idx) {
            int pid = element.contained_flip_particles[p_idx];
            auto &particle = flip_particles[pid];
            const auto &pos = particle.pos;

            Real vx = 0.0, vy = 0.0, vz = 0.0;
            Real dvx = 0.0, dvy = 0.0, dvz = 0.0;

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
                dvx += poly_coeffs_du(i) * val;
                dvy += poly_coeffs_dv(i) * val;
                dvz += poly_coeffs_dw(i) * val;
            }
            _3d_vec pic_vec(vx,vy,vz);

            particle.pic_vec = pic_vec ;
        }
    }
    Real max_v = 0.0;
    for (auto&p:flip_particles)max_v= std::max(max_v,p.pic_vec.norm());
    RZ_INFO("max particle velocity: ",max_v);
}

void VEM_Domain_3d::GtoP_k1(std::vector<FLIP_Particles_3d> &flip_particles, Real flip_alpha, Real dt) {
     FUNC_TIMER();
    // du
    for (int vid=0;vid<k1_dof_u.size();++vid) {
        k1_dof_du[vid] = k1_dof_u[vid] - k1_dof_du[vid];
        k1_dof_dv[vid] = k1_dof_v[vid] - k1_dof_dv[vid];
        k1_dof_dw[vid] = k1_dof_w[vid] - k1_dof_dw[vid];
    }
#pragma omp parallel for schedule(dynamic)
    for (const auto &element : vem_elements) {
        const auto &projector_k1 = element.is_U?
        U3d_project_matrix_star_k1 : element.project_matrix_star_k1;

        // === 从全局 DOF 提取当前元素的速度自由度 ===
        Eigen::VectorXr element_dof_values_u, element_dof_values_v, element_dof_values_w;
        Eigen::VectorXr element_dof_values_du, element_dof_values_dv, element_dof_values_dw;
        const auto &vert_ids = element.vertex_local_to_global_dof;
        const int ndof = (int)vert_ids.size();

        element_dof_values_u.resize(ndof);
        element_dof_values_v.resize(ndof);
        element_dof_values_w.resize(ndof);
        element_dof_values_u.setZero();
        element_dof_values_v.setZero();
        element_dof_values_w.setZero();
        element_dof_values_du.resize(ndof);
        element_dof_values_dv.resize(ndof);
        element_dof_values_dw.resize(ndof);
        element_dof_values_du.setZero();
        element_dof_values_dv.setZero();
        element_dof_values_dw.setZero();

        for (int i = 0; i < ndof; ++i) {
            int vid = vert_ids[i];
            element_dof_values_u[i] = k1_dof_u[vid];
            element_dof_values_v[i] = k1_dof_v[vid];
            element_dof_values_w[i] = k1_dof_w[vid];
            element_dof_values_du[i] = k1_dof_du[vid];
            element_dof_values_dv[i] = k1_dof_dv[vid];
            element_dof_values_dw[i] = k1_dof_dw[vid];
        }

        if (element.has_boundary) {
            const auto &elem = element;
            for (int local_pid=0;local_pid<elem.polygons.size();++local_pid) {
                const int global_pid = elem.polygon_local_to_global_dof[local_pid];
                if (!is_polygon_boundary[global_pid])continue;
                std::vector<_3d_vec> constrained_dirs;
                const _3d_vec& normal = _3d_vec(elem.poly_planes[local_pid].orthogonal_vector().x(),
                elem.poly_planes[local_pid].orthogonal_vector().y(),
                elem.poly_planes[local_pid].orthogonal_vector().z());
                constrained_dirs.push_back(normal);

                for (auto& local_vid : elem.polygons[local_pid]) {
                    {
                        // v
                        _3d_vec v(element_dof_values_u[local_vid], element_dof_values_v[local_vid],
                            element_dof_values_w[local_vid]);
                        for (auto& constrained_dir : constrained_dirs) {
                            Real dot_product = v.vx * constrained_dir.vx + v.vy * constrained_dir.vy + v.vz * constrained_dir.vz;
                            _3d_vec projection = constrained_dir * dot_product;
                            if (dot_product > 0.0)
                                v = v - projection;
                        }
                        element_dof_values_u[local_vid] = v.vx;
                        element_dof_values_v[local_vid] = v.vy;
                        element_dof_values_w[local_vid] = v.vz;
                    }
                    // dv
                    {
                        _3d_vec dv(element_dof_values_du[local_vid], element_dof_values_dv[local_vid],
                           element_dof_values_dw[local_vid]);
                        for (auto& constrained_dir : constrained_dirs) {
                            Real dot_product = dv.vx * constrained_dir.vx
                            + dv.vy * constrained_dir.vy + dv.vz * constrained_dir.vz;
                            _3d_vec projection = constrained_dir * dot_product;
                            if (dot_product > 0.0)
                                dv = dv - projection;
                        }
                        element_dof_values_du[local_vid] = dv.vx;
                        element_dof_values_dv[local_vid] = dv.vy;
                        element_dof_values_dw[local_vid] = dv.vz;
                    }
                }

            }
        }

        RZ_ASSERT(projector_k1.cols() == ndof, "projector_k1.cols()==element_dof_values.size()");

        // === 计算该元素的多项式系数 ===
        Eigen::VectorXr poly_coeffs_u = projector_k1 * element_dof_values_u;
        Eigen::VectorXr poly_coeffs_v = projector_k1 * element_dof_values_v;
        Eigen::VectorXr poly_coeffs_w = projector_k1 * element_dof_values_w;
        Eigen::VectorXr poly_coeffs_du = projector_k1 * element_dof_values_du;
        Eigen::VectorXr poly_coeffs_dv = projector_k1 * element_dof_values_dv;
        Eigen::VectorXr poly_coeffs_dw = projector_k1 * element_dof_values_dw;

        // === 遍历该元素内的所有粒子 ===
        // 假设：monomials_3d, poly_coeffs_u/v/w, element 都是只读的
        //       flip_particles 是可写的全局数组 (不同粒子 id 不重叠)

        for (int p_idx = 0; p_idx < (int)element.contained_flip_particles.size(); ++p_idx) {
            int pid = element.contained_flip_particles[p_idx];
            auto &particle = flip_particles[pid];
            const auto &pos = particle.pos;

            Real vx = 0.0, vy = 0.0, vz = 0.0;
            Real dvx = 0.0, dvy = 0.0, dvz = 0.0;

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
                dvx += poly_coeffs_du(i) * val;
                dvy += poly_coeffs_dv(i) * val;
                dvz += poly_coeffs_dw(i) * val;
            }

            particle.pic_vec.vx = vx;
            particle.pic_vec.vy = vy;
            particle.pic_vec.vz = vz;

            const _3d_vec delta_vec(dvx,dvy,dvz);
            particle.vec = particle.pic_vec * flip_alpha + (particle.vec + delta_vec) * (1 - flip_alpha);
        }

    }

}

void VEM_Domain_3d::resample_GtoP(std::vector<FLIP_Particles_3d> &flip_particles) {
#pragma omp parallel for schedule(dynamic)
    for (int eid=0;eid<vem_elements.size();++eid) {
        const auto&element = vem_elements[eid];
        // === 计算该元素的多项式系数 ===
        const Eigen::VectorXr& poly_coeffs_u = element.poly_coeffs_u_k1;
        const Eigen::VectorXr& poly_coeffs_v = element.poly_coeffs_v_k1;
        const Eigen::VectorXr& poly_coeffs_w = element.poly_coeffs_w_k1;
        // === 遍历该元素内的所有粒子 ===
        for (int p_idx = 0; p_idx < (int)element.contained_flip_particles.size(); ++p_idx) {
            int pid = element.contained_flip_particles[p_idx];
            auto &particle = flip_particles[pid];
            if (!particle.need_interpolate)continue;
            const auto &pos = particle.pos;

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
            particle.vec.vx = vx;
            particle.vec.vy = vy;
            particle.vec.vz = vz;
        }
    }
}

void VEM_Domain_3d::inter_rho_to_mg(MAC_Grid3D &mg) {
    FUNC_TIMER();
    mg._density.fill(0.0);
    std::vector<_3d_vec> pts;
    pts.resize(mg.nx*mg.ny*mg.nz);
// #pragma omp parallel for collapse(3) schedule(static)
    for (int k = 0; k < mg.nz; ++k)
        for (int j = 0; j < mg.ny; ++j)
            for (int i = 0; i < mg.nx; ++i) {
                const Real gx = (i + 0.5) * mg.dx;
                const Real gy = (j + 0.5) * mg.dx;
                const Real gz = (k + 0.5) * mg.dx;
                pts[i+j*mg.nx+k*mg.ny*mg.nz].vx = gx;
                pts[i+j*mg.nx+k*mg.ny*mg.nz].vy = gy;
                pts[i+j*mg.nx+k*mg.ny*mg.nz].vz = gz;
            }

#pragma omp parallel for schedule(dynamic)
    for (int eid=0;eid<vem_elements.size();++eid) {
        const auto& element = vem_elements[eid];
        const auto &projector_k1 = element.is_U?
        U3d_project_matrix_star_k1 : element.project_matrix_star_k1;

        // === 从全局 DOF 提取当前元素的速度自由度 ===
        Eigen::VectorXr element_dof_values_rho;
        const auto &vert_ids = element.vertex_local_to_global_dof;
        const int ndof = (int)vert_ids.size();

        element_dof_values_rho.resize(ndof);
        element_dof_values_rho.setZero();

        for (int i = 0; i < ndof; ++i) {
            int vid = vert_ids[i];
            element_dof_values_rho[i] = k1_dof_density[vid];
        }

        vem_elements[eid].poly_coeffs_rho_k1 = projector_k1 * element_dof_values_rho;
    }

    std::vector<int> elemids = assign_pts_to_elements_3d(pts);
#pragma omp parallel for schedule(dynamic)
    for (int k=0;k<pts.size();++k) {
        if (elemids[k] == -1) {
            continue;
        }
        const auto&element = vem_elements[elemids[k]];
        Eigen::VectorXr poly_coeffs_rho = vem_elements[elemids[k]].poly_coeffs_rho_k1 ;
        const auto &pos =pts[k];
        Real rho = 0.0;

        // === 插值多项式 ===
        const Real xc = element.centroid.vx;
        const Real yc = element.centroid.vy;
        const Real zc = element.centroid.vz;
        const Real h  = element.diameter;

        for (int i = 0; i < poly_coeffs_rho.size(); ++i) {
            const auto &mono = monomials_3d[i];
            Real val = mono.value(pos.vx, pos.vy, pos.vz, xc, yc, zc, h);
            rho += poly_coeffs_rho(i) * val;
        }
        int mgi = k%mg.nx;
        int mgj = (k/mg.nx)%mg.ny;
        int mgk = (k/mg.nx/mg.ny)%mg.nz;
        mg._density(mgi,mgj,mgk) = rho;
    }
    mg._density.clamp01();
}


void VEM_Domain_3d::flip_advection(std::vector<FLIP_Particles_3d> &flip_particles,const Real& dt) {
    FUNC_TIMER();
    for (auto &p: flip_particles) {
        if (p.deleted) {
            continue;
        }
        p.old_pos = p.pos;
        p.pos.vx += dt * p.vec.vx;
        p.pos.vy += dt * p.vec.vy;
        p.pos.vz += dt * p.vec.vz;
    }

    auto get_grid_index = [&](const _3d_vec &p) noexcept -> int {
        int gx = std::min(grid_res - 1, std::max(0, int(p.vx * grid_res)));
        int gy = std::min(grid_res - 1, std::max(0, int(p.vy * grid_res)));
        int gz = std::min(grid_res - 1, std::max(0, int(p.vz * grid_res)));
        return (gz * grid_res + gy) * grid_res + gx;
    };
    // ===============================
    // 1️⃣ 并行处理 FLIP 粒子
    // ===============================

#pragma omp parallel for schedule(dynamic)
    for (int pid = 0; pid < static_cast<int>(flip_particles.size()); ++pid) {
        auto &fp = flip_particles[pid];
        fp.contained_in_element_id = -1;
        if (fp.deleted) continue;

        const _3d_vec &p = fp.pos;
        if (out_of_domain(p)) {
            fp.deleted = true;
            continue;
        }

        const int gidx = get_grid_index(p);
        const auto &cand_elems = grid_to_elems[gidx];
        if (cand_elems.empty()) {
            fp.deleted = true;
            continue;
        }

        bool assigned = false;
        for (int eid : cand_elems) {
            if (vem_elements[eid].point_in_elem(p)) {
                fp.contained_in_element_id = eid;
                assigned = true;
                break;
            }
        }

        if (!assigned) fp.deleted = true;

    }
    // 处理出界的flip粒子
    std::vector<int> deleted_flip_particles;
    for (int particle_id=0;particle_id<flip_particles.size();++particle_id) {
        if (flip_particles[particle_id].deleted) {
            deleted_flip_particles.push_back(particle_id);
        }
    }
    // 处理出界的flip粒子
    std::for_each(std::execution::par, deleted_flip_particles.begin(), deleted_flip_particles.end(),
        [&](int i) {
        Boundary_Handle(flip_particles[i],aabb_tree,dt);
    }
    );
}

void VEM_Domain_3d::RK_3_advection_k2(std::vector<FLIP_Particles_3d> &flip_particles, const Real &dt) {
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
    boundary_aware_pos_fix(old_pos,temp_new_poses,dt);
    for (int i=0;i<flip_particles.size();++i) {
        flip_particles[i].pos = temp_new_poses[i];
    }

    assign_flip_smoke_particles_to_elements_3d(flip_particles);
    std::vector<FLIP_Particles_3d>temp;
    for (auto&p:flip_particles)if (!p.deleted)temp.push_back(p);
    flip_particles.swap(temp);
}

void VEM_Domain_3d::tracers_RK_3(const Real &dt) {
FUNC_TIMER();
    std::vector<_3d_vec>rk_pos;
    for (auto&p:tracers)rk_pos.push_back(_3d_vec(p.x,p.y,p.z));
    std::vector<_3d_vec>old_pos = rk_pos;

    std::vector<Real>u1(rk_pos.size(),0.0),v1(rk_pos.size(),0.0),w1(rk_pos.size(),0.0),
        u2(rk_pos.size(),0.0),v2(rk_pos.size(),0.0),w2(rk_pos.size(),0.0),
        u3(rk_pos.size(),0.0),v3(rk_pos.size(),0.0),w3(rk_pos.size(),0.0);
    //分配到 elem并插值速度 更新位置
    std::vector<int> elemids = assign_pts_to_elements_3d(rk_pos);

// #pragma omp parallel for schedule(static)
    Real max_smoke_v = 0.0;
    _3d_vec v_somke(0,0,0);
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
        if (std::sqrt(vx*vx+vy*vy+vz*vz)>max_smoke_v) {
            v_somke = _3d_vec(vx,vy,vz);
        }
        max_smoke_v = std::max(max_smoke_v,std::sqrt(vx*vx+vy*vy+vz*vz));
        rk_pos[k].vx = tracers[k].x + dt*0.5*vx;
        rk_pos[k].vy = tracers[k].y + dt*0.5*vy;
        rk_pos[k].vz = tracers[k].z + dt*0.5*vz;
    }
    RZ_INFO("max_smoke_v :",max_smoke_v,"Vy: ",v_somke.vy);
    boundary_aware_pos_fix(old_pos,rk_pos,dt * 0.5);
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

        rk_pos[k].vx = tracers[k].x + dt*0.75*vx;
        rk_pos[k].vy = tracers[k].y + dt*0.75*vy;
        rk_pos[k].vz = tracers[k].z + dt*0.75*vz;
    }
    //分配到 elem并插值速度
    boundary_aware_pos_fix(old_pos,rk_pos,dt * 0.75);
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
    for (int i=0;i<tracers.size();++i) {
        tracers[i].x += c1*u1[i]+c2*u2[i]+c3*u3[i];
        tracers[i].y += c1*v1[i]+c2*v2[i]+c3*v3[i];
        tracers[i].z += c1*w1[i]+c2*w2[i]+c3*w3[i];
        tracers[i].rest_age = tracers[i].rest_age/(1.0+0.1*dt);
    }
    std::vector<_3d_vec> temp_new_poses;
    for (auto&p:tracers)temp_new_poses.push_back({p.x,p.y,p.z});
    boundary_aware_pos_fix(old_pos,temp_new_poses,dt);
    for (int i=0;i<tracers.size();++i) {
        tracers[i].x = temp_new_poses[i].vx;
        tracers[i].y = temp_new_poses[i].vy;
        tracers[i].z = temp_new_poses[i].vz;
    }
}

void VEM_Domain_3d::flip_adv_skull(std::vector<FLIP_Particles_3d> &flip_particles, const Real &dt) {
    FUNC_TIMER();
    std::vector<_3d_vec> rk_pos;
    for (auto&p:flip_particles)rk_pos.push_back(p.pos);
    std::vector<Real>u1(rk_pos.size(),0.0),v1(rk_pos.size(),0.0),w1(rk_pos.size(),0.0);
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
    }
    Real c1 = 1.0 *  dt;

// #pragma omp parallel for schedule(dynamic)
    for (int i=0;i<flip_particles.size();++i) {
        flip_particles[i].old_pos = flip_particles[i].pos;
        flip_particles[i].pos.vx += c1*u1[i];
        flip_particles[i].pos.vy += c1*v1[i];
        flip_particles[i].pos.vz += c1*w1[i];
    }
    assign_flip_smoke_particles_to_elements_3d(flip_particles);
    std::vector<int> deleted_flip_particles;
    for (int particle_id=0;particle_id<flip_particles.size();++particle_id) {
        if (flip_particles[particle_id].deleted) {
            deleted_flip_particles.push_back(particle_id);
        }
    }
    RZ_INFO("out flips: ",deleted_flip_particles.size());
    // 处理出界的flip粒子
    if (deleted_flip_particles.size()) {
        std::for_each(std::execution::par, deleted_flip_particles.begin(), deleted_flip_particles.end(),
            [&](int i) {
            Boundary_Handle(flip_particles[i],aabb_tree,dt);
        }
        );
    }
    assign_flip_smoke_particles_to_elements_3d(flip_particles);
}

void VEM_Domain_3d::tracer_adv_skull(const Real &dt) {
    FUNC_TIMER();
    std::vector<_3d_vec>rk_pos;
    for (auto&p:tracers)rk_pos.push_back(_3d_vec(p.x,p.y,p.z));
    std::vector<Real>u1(rk_pos.size(),0.0),v1(rk_pos.size(),0.0),w1(rk_pos.size(),0.0);
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

    }

    //最后更新每个flip粒子的位置
    Real c1 = 1.0 *  dt;
// #pragma omp parallel for schedule(dynamic)
    for (int i=0;i<tracers.size();++i) {
        tracers[i].x += c1*u1[i];
        tracers[i].y += c1*v1[i];
        tracers[i].z += c1*w1[i];
        tracers[i].rest_age = tracers[i].rest_age/(1.0+0.1*dt);
    }
}

void VEM_Domain_3d::tracers_RK_3_lucy(const Real &dt) {
    FUNC_TIMER();
    {
        auto &tracers = lucy_tracers_1;
        std::vector<_3d_vec>rk_pos;
        for (auto&p:tracers)rk_pos.push_back(_3d_vec(p.x,p.y,p.z));
        std::vector<_3d_vec>old_pos = rk_pos;

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

            rk_pos[k].vx = tracers[k].x + dt*0.5*vx;
            rk_pos[k].vy = tracers[k].y + dt*0.5*vy;
            rk_pos[k].vz = tracers[k].z + dt*0.5*vz;
        }
        //分配到 elem并插值速度 更新位置
        boundary_aware_pos_fix(old_pos,rk_pos,dt * 0.5);
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

            rk_pos[k].vx = tracers[k].x + dt*0.75*vx;
            rk_pos[k].vy = tracers[k].y + dt*0.75*vy;
            rk_pos[k].vz = tracers[k].z + dt*0.75*vz;
        }
        //分配到 elem并插值速度
        boundary_aware_pos_fix(old_pos,rk_pos,dt * 0.75);
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
        for (int i=0;i<tracers.size();++i) {
            tracers[i].x += c1*u1[i]+c2*u2[i]+c3*u3[i];
            tracers[i].y += c1*v1[i]+c2*v2[i]+c3*v3[i];
            tracers[i].z += c1*w1[i]+c2*w2[i]+c3*w3[i];
            tracers[i].rest_age = tracers[i].rest_age/(1.0+0.1*dt);
        }
        std::vector<_3d_vec> temp_new_poses;
        for (auto&p:tracers)temp_new_poses.push_back({p.x,p.y,p.z});
        boundary_aware_pos_fix(old_pos,temp_new_poses,dt);
        for (int i=0;i<tracers.size();++i) {
            tracers[i].x = temp_new_poses[i].vx;
            tracers[i].y = temp_new_poses[i].vy;
            tracers[i].z = temp_new_poses[i].vz;
        }
    }
    {
        auto &tracers = lucy_tracers_2;
        std::vector<_3d_vec>rk_pos;
        for (auto&p:tracers)rk_pos.push_back(_3d_vec(p.x,p.y,p.z));
        std::vector<_3d_vec>old_pos = rk_pos;

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

            rk_pos[k].vx = tracers[k].x + dt*0.5*vx;
            rk_pos[k].vy = tracers[k].y + dt*0.5*vy;
            rk_pos[k].vz = tracers[k].z + dt*0.5*vz;
        }
        //分配到 elem并插值速度 更新位置
        boundary_aware_pos_fix(old_pos,rk_pos,dt * 0.5);
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

            rk_pos[k].vx = tracers[k].x + dt*0.75*vx;
            rk_pos[k].vy = tracers[k].y + dt*0.75*vy;
            rk_pos[k].vz = tracers[k].z + dt*0.75*vz;
        }
        //分配到 elem并插值速度
        boundary_aware_pos_fix(old_pos,rk_pos,dt * 0.75);
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
        for (int i=0;i<tracers.size();++i) {
            tracers[i].x += c1*u1[i]+c2*u2[i]+c3*u3[i];
            tracers[i].y += c1*v1[i]+c2*v2[i]+c3*v3[i];
            tracers[i].z += c1*w1[i]+c2*w2[i]+c3*w3[i];
            tracers[i].rest_age = tracers[i].rest_age/(1.0+0.1*dt);
        }
        std::vector<_3d_vec> temp_new_poses;
        for (auto&p:tracers)temp_new_poses.push_back({p.x,p.y,p.z});
        boundary_aware_pos_fix(old_pos,temp_new_poses,dt);
        for (int i=0;i<tracers.size();++i) {
            tracers[i].x = temp_new_poses[i].vx;
            tracers[i].y = temp_new_poses[i].vy;
            tracers[i].z = temp_new_poses[i].vz;
        }
    }
}

void VEM_Domain_3d::RK_3_advection_k2(
    ConcentrationField3D &concentration_field,
    const Real &dt)
{
    FUNC_TIMER();

    const int nx = concentration_field.nx;
    const int ny = concentration_field.ny;
    const int nz = concentration_field.nz;
    const Real dx = concentration_field.dx;

    // 旧场（semi-Lagrangian 必须）
    Array3D<Real> old_density = concentration_field._density;

    // ============================================================
    // 1. 取所有 cell center 作为 RK 点
    // ============================================================
    std::vector<_3d_vec> rk_pos = concentration_field.init_rk_pos;

    const int N = rk_pos.size();
    std::vector<Real>
        u1(N,0.0), v1(N,0.0), w1(N,0.0),
        u2(N,0.0), v2(N,0.0), w2(N,0.0),
        u3(N,0.0), v3(N,0.0), w3(N,0.0);
    std::vector<int> elemids = concentration_field.init_elemids;
    // ============================================================
    // stage 1
    // ============================================================

#pragma omp parallel for schedule(static)
    for (int p = 0; p < N; ++p) {
        int eid = elemids[p];
        if (eid == -1) continue;

        const auto &elem = vem_elements[eid];
        const auto &pos  = rk_pos[p];

        Real vx = 0, vy = 0, vz = 0;
        const Real xc = elem.centroid.vx;
        const Real yc = elem.centroid.vy;
        const Real zc = elem.centroid.vz;
        const Real h  = elem.diameter;

        const auto &pu = elem.poly_coeffs_u_k1;
        const auto &pv = elem.poly_coeffs_v_k1;
        const auto &pw = elem.poly_coeffs_w_k1;

        for (int m = 0; m < pu.size(); ++m) {
            Real val = monomials_3d[m].value(
                pos.vx, pos.vy, pos.vz, xc, yc, zc, h
            );
            vx += pu(m) * val;
            vy += pv(m) * val;
            vz += pw(m) * val;
        }

        u1[p] = vx; v1[p] = vy; w1[p] = vz;

        rk_pos[p].vx = concentration_field.init_rk_pos[p].vx - 0.5 * dt * vx;
        rk_pos[p].vy = concentration_field.init_rk_pos[p].vy - 0.5 * dt * vy;
        rk_pos[p].vz = concentration_field.init_rk_pos[p].vz - 0.5 * dt * vz;
    }

    // ============================================================
    // stage 2
    // ============================================================
    elemids = assign_pts_to_elements_3d_with_BH(rk_pos,concentration_field.init_rk_pos);
#pragma omp parallel for schedule(static)
    for (int p = 0; p < N; ++p) {
        int eid = elemids[p];
        if (eid == -1) {
            u2[p] = u1[p]; v2[p] = v1[p]; w2[p] = w1[p];
            continue;
        }

        const auto &elem = vem_elements[eid];
        const auto &pos  = rk_pos[p];

        Real vx = 0, vy = 0, vz = 0;
        const Real xc = elem.centroid.vx;
        const Real yc = elem.centroid.vy;
        const Real zc = elem.centroid.vz;
        const Real h  = elem.diameter;

        for (int m = 0; m < elem.poly_coeffs_u_k1.size(); ++m) {
            Real val = monomials_3d[m].value(
                pos.vx, pos.vy, pos.vz, xc, yc, zc, h
            );
            vx += elem.poly_coeffs_u_k1(m) * val;
            vy += elem.poly_coeffs_v_k1(m) * val;
            vz += elem.poly_coeffs_w_k1(m) * val;
        }

        u2[p] = vx; v2[p] = vy; w2[p] = vz;

        rk_pos[p].vx = concentration_field.init_rk_pos[p].vx - 0.75 * dt * vx;
        rk_pos[p].vy = concentration_field.init_rk_pos[p].vy - 0.75 * dt * vy;
        rk_pos[p].vz = concentration_field.init_rk_pos[p].vz - 0.75 * dt * vz;
    }

    // ============================================================
    // stage 3
    // ============================================================
    elemids = assign_pts_to_elements_3d_with_BH(rk_pos,concentration_field.init_rk_pos);

#pragma omp parallel for schedule(static)
    for (int p = 0; p < N; ++p) {
        int eid = elemids[p];
        if (eid == -1) {
            u3[p] = u2[p]; v3[p] = v2[p]; w3[p] = w2[p];
            continue;
        }

        const auto &elem = vem_elements[eid];
        const auto &pos  = rk_pos[p];

        Real vx = 0, vy = 0, vz = 0;
        const Real xc = elem.centroid.vx;
        const Real yc = elem.centroid.vy;
        const Real zc = elem.centroid.vz;
        const Real h  = elem.diameter;

        for (int m = 0; m < elem.poly_coeffs_u_k1.size(); ++m) {
            Real val = monomials_3d[m].value(
                pos.vx, pos.vy, pos.vz, xc, yc, zc, h
            );
            vx += elem.poly_coeffs_u_k1(m) * val;
            vy += elem.poly_coeffs_v_k1(m) * val;
            vz += elem.poly_coeffs_w_k1(m) * val;
        }

        u3[p] = vx; v3[p] = vy; w3[p] = vz;
    }

    // ============================================================
    // combine + sample
    // ============================================================
    const Real c1 = 2.0/9.0 * dt;
    const Real c2 = 3.0/9.0 * dt;
    const Real c3 = 4.0/9.0 * dt;

#pragma omp parallel for schedule(dynamic)
    for (int p = 0; p < N; ++p) {
        int i =  p % nx;
        int j = (p / nx) % ny;
        int k =  p / (nx * ny);
        if (concentration_field.init_elemids[p] == -1) {
            concentration_field._density(i,j,k)=0.0;
            continue;
        }
        _3d_vec x0(
            (i + 0.5) * dx,
            (j + 0.5) * dx,
            (k + 0.5) * dx
        );

        _3d_vec x_back = x0
            -  _3d_vec(u1[p], v1[p], w1[p]) * c1
            -  _3d_vec(u2[p], v2[p], w2[p]) * c2
            -  _3d_vec(u3[p], v3[p], w3[p]) * c3;

        concentration_field._density(i,j,k) =
            concentration_field.interpolate_density(x_back,old_density);
    }
}


void VEM_Domain_3d::smoke_advection(std::vector<Smoke_Particles_3d> &smoke_particles, const Real &dt) {
    // === 遍历每个元素 ===
    FUNC_TIMER();
#pragma omp parallel for schedule(dynamic)
    for (auto &element : vem_elements) {
        const auto &projector_k1 = element.is_U?
        U3d_project_matrix_star_k1 : element.project_matrix_star_k1;
        const auto &vert_ids = element.vertex_local_to_global_dof;
        const int ndof = (int)vert_ids.size();
        if (ndof == 0 || element.contained_smoke_particles.empty()) continue;

        // === 1️⃣ 从全局 DOF 取出该元素的速度自由度 ===
        Eigen::VectorXr element_dof_values_u(ndof), element_dof_values_v(ndof), element_dof_values_w(ndof);
        for (int i = 0; i < ndof; ++i) {
            int vid = vert_ids[i];
            element_dof_values_u[i] = k1_dof_u[vid];
            element_dof_values_v[i] = k1_dof_v[vid];
            element_dof_values_w[i] = k1_dof_w[vid];
        }

        if (element.has_boundary) {
            const auto &elem = element;
            // std::vector<_3d_vec> constrained_dirs;
            // for (int local_pid=0;local_pid<elem.polygons.size();++local_pid) {
            //     const int global_pid = elem.polygon_local_to_global_dof[local_pid];
            //     const _3d_vec& normal = elem.elements_2d[local_pid].plane.normal;
            //     if (!is_polygon_boundary[global_pid])continue;
            //     constrained_dirs.push_back(normal);
            // }
            for (int local_pid=0;local_pid<elem.polygons.size();++local_pid) {
                const int global_pid = elem.polygon_local_to_global_dof[local_pid];
                if (!is_polygon_boundary[global_pid])continue;
                std::vector<_3d_vec> constrained_dirs;
                const _3d_vec& normal = _3d_vec(elem.poly_planes[local_pid].orthogonal_vector().x(),
                elem.poly_planes[local_pid].orthogonal_vector().y(),
                elem.poly_planes[local_pid].orthogonal_vector().z());
                constrained_dirs.push_back(normal);
                for (auto& local_vid : elem.polygons[local_pid]) {
                    _3d_vec v(element_dof_values_u[local_vid], element_dof_values_v[local_vid],
                        element_dof_values_w[local_vid]);
                    for (auto& constrained_dir : constrained_dirs) {
                        Real dot_product = v.vx * constrained_dir.vx + v.vy * constrained_dir.vy + v.vz * constrained_dir.vz;
                        _3d_vec projection = constrained_dir * dot_product;
                        if (dot_product > 0.0)
                            v = v - projection;
                    }
                    element_dof_values_u[local_vid] = v.vx;
                    element_dof_values_v[local_vid] = v.vy;
                    element_dof_values_w[local_vid] = v.vz;
                }
            }
        }

        // === 2️⃣ 得到该元素的多项式系数 ===
        element.poly_coeffs_u_k1 = projector_k1 * element_dof_values_u;
        element.poly_coeffs_v_k1 = projector_k1 * element_dof_values_v;
        element.poly_coeffs_w_k1 = projector_k1 * element_dof_values_w;
        const auto& poly_coeffs_u = element.poly_coeffs_u_k1;
        const auto& poly_coeffs_v = element.poly_coeffs_v_k1;
        const auto& poly_coeffs_w = element.poly_coeffs_w_k1;

        // === 3️⃣ 遍历该元素内的所有烟雾粒子 ===
        for (int pid : element.contained_smoke_particles) {
            auto &sp = smoke_particles[pid];
            if (sp.deleted) continue;

            const auto &pos = sp.pos;
            Real vx = 0.0, vy = 0.0, vz = 0.0;

            // 基函数插值
            for (int i = 0; i < poly_coeffs_u.size(); ++i) {
                const auto &mono = monomials_3d[i];
                Real val = mono.value(
                    pos.vx, pos.vy, pos.vz,
                    element.centroid.vx,
                    element.centroid.vy,
                    element.centroid.vz,
                    element.diameter);

                vx += poly_coeffs_u(i) * val;
                vy += poly_coeffs_v(i) * val;
                vz += poly_coeffs_w(i) * val;
            }

            // === 4️⃣ 用欧拉步更新粒子位置 ===
            sp.old_pos = sp.pos;
            sp.v = _3d_vec(vx,vy,vz);
            sp.pos.vx  += dt*vx;
            sp.pos.vy  += dt*vy;
            sp.pos.vz  += dt*vz;

        }
    }
//     // 边界处理
    auto get_grid_index = [&](const _3d_vec &p) noexcept -> int {
        int gx = std::min(grid_res - 1, std::max(0, int(p.vx * grid_res)));
        int gy = std::min(grid_res - 1, std::max(0, int(p.vy * grid_res)));
        int gz = std::min(grid_res - 1, std::max(0, int(p.vz * grid_res)));
        return (gz * grid_res + gy) * grid_res + gx;
    };
    // ===============================
    // 2️⃣ 并行处理 SMOKE 粒子
    // ===============================
#pragma omp parallel for schedule(dynamic)
    for (int pid = 0; pid < static_cast<int>(smoke_particles.size()); ++pid) {
        auto &fp = smoke_particles[pid];
        fp.contained_in_element_id = -1;
        if (fp.deleted) continue;

        const _3d_vec &p = fp.pos;
        if (out_of_domain(p)) {
            fp.deleted = true;
            continue;
        }

        const int gidx = get_grid_index(p);
        const auto &cand_elems = grid_to_elems[gidx];
        if (cand_elems.empty()) {
            fp.deleted = true;
            continue;
        }

        bool assigned = false;
        for (int eid : cand_elems) {
            if (vem_elements[eid].point_in_elem(p)) {
                fp.contained_in_element_id = eid;
                assigned = true;
                break;
            }
        }

        if (!assigned) fp.deleted = true;
    }

    // 处理出界的烟雾粒子
    std::vector<int> deleted_smoke_particles;
    for (int particle_id=0;particle_id<smoke_particles.size();++particle_id) {
        if (smoke_particles[particle_id].deleted) {
            deleted_smoke_particles.push_back(particle_id);
        }
    }

    std::for_each(std::execution::par, deleted_smoke_particles.begin(), deleted_smoke_particles.end(),
        [&](int i) {
        Boundary_Handle(smoke_particles[i],aabb_tree,dt);
    }
    );

}

void VEM_Domain_3d::smoke_advection_RK2(std::vector<Smoke_Particles_3d> &smoke_particles, const Real &dt)
{
    auto get_grid_index = [&](const _3d_vec &p) noexcept -> int {
        int gx = std::min(grid_res - 1, std::max(0, int(p.vx * grid_res)));
        int gy = std::min(grid_res - 1, std::max(0, int(p.vy * grid_res)));
        int gz = std::min(grid_res - 1, std::max(0, int(p.vz * grid_res)));
        return (gz * grid_res + gy) * grid_res + gx;
    };
    // === 并行遍历每个元素 ===
#pragma omp parallel for schedule(dynamic)
    for (int eid = 0; eid < (int)vem_elements.size(); ++eid) {
        const auto &element = vem_elements[eid];
        const auto &vert_ids = element.vertex_local_to_global_dof;
        const int ndof = (int)vert_ids.size();
        if (ndof == 0 || element.contained_smoke_particles.empty()) continue;

        const auto &projector_k1 = element.is_U?
        U3d_project_matrix_star_k1 : element.project_matrix_star_k1;

        // --- 1️⃣ 从全局自由度取出速度 DOF ---
        Eigen::VectorXr element_dof_values_u(ndof), element_dof_values_v(ndof), element_dof_values_w(ndof);
        for (int i = 0; i < ndof; ++i) {
            int vid = vert_ids[i];
            element_dof_values_u[i] = k1_dof_u[vid];
            element_dof_values_v[i] = k1_dof_v[vid];
            element_dof_values_w[i] = k1_dof_w[vid];
        }

        // --- 2️⃣ 得到多项式系数 ---
        Eigen::VectorXr poly_coeffs_u = projector_k1 * element_dof_values_u;
        Eigen::VectorXr poly_coeffs_v = projector_k1 * element_dof_values_v;
        Eigen::VectorXr poly_coeffs_w = projector_k1 * element_dof_values_w;

        // --- 3️⃣ 遍历该元素内的粒子 ---
        for (int pid : element.contained_smoke_particles) {
            auto &sp = smoke_particles[pid];
            if (sp.deleted) continue;

            // ---------- Step 1: v1 = v(x_t) ----------
            const auto &pos = sp.pos;
            Real vx1 = 0.0, vy1 = 0.0, vz1 = 0.0;
            for (int i = 0; i < poly_coeffs_u.size(); ++i) {
                const auto &mono = monomials_3d[i];
                Real val = mono.value(pos.vx, pos.vy, pos.vz,
                                        element.centroid.vx, element.centroid.vy, element.centroid.vz,
                                        element.diameter);
                vx1 += poly_coeffs_u[i] * val;
                vy1 += poly_coeffs_v[i] * val;
                vz1 += poly_coeffs_w[i] * val;
            }

            // ---------- Step 2: 中点位置 ----------
            _3d_vec mid_pos(
                pos.vx + 0.5 * dt * vx1,
                pos.vy + 0.5 * dt * vy1,
                pos.vz + 0.5 * dt * vz1);

            // ---------- Step 3: 重新确定中点所在元素 ----------
            int mid_elem_id = sp.contained_in_element_id; // 从上次位置开始搜索
            bool found = false;

            // 如果仍在原元素内，直接使用
            if (vem_elements[mid_elem_id].point_in_elem(mid_pos)) {
                found = true;
            } else {
                // 否则在相邻候选元素中查找
                const auto &cands = grid_to_elems[get_grid_index(mid_pos)];
                for (int ceid : cands) {
                    if (vem_elements[ceid].point_in_elem(mid_pos)) {
                        mid_elem_id = ceid;
                        found = true;
                        break;
                    }
                }
            }

            if (!found) {
                sp.deleted = true;
                continue;
            }

            const auto &mid_elem = vem_elements[mid_elem_id];
            const auto &mid_proj = mid_elem.is_U?
        U3d_project_matrix_star_k1 : mid_elem.project_matrix_star_k1;
            const auto &mid_vert = mid_elem.vertex_local_to_global_dof;
            const int mid_ndof = (int)mid_vert.size();

            // --- 获取中点元素的多项式系数 ---
            Eigen::VectorXr mid_dof_u(mid_ndof), mid_dof_v(mid_ndof), mid_dof_w(mid_ndof);
            for (int i = 0; i < mid_ndof; ++i) {
                int vid = mid_vert[i];
                mid_dof_u[i] = k1_dof_u[vid];
                mid_dof_v[i] = k1_dof_v[vid];
                mid_dof_w[i] = k1_dof_w[vid];
            }

            Eigen::VectorXr mid_coeffs_u = mid_proj * mid_dof_u;
            Eigen::VectorXr mid_coeffs_v = mid_proj * mid_dof_v;
            Eigen::VectorXr mid_coeffs_w = mid_proj * mid_dof_w;

            // ---------- Step 4: v2 = v(mid_pos) ----------
            Real vx2 = 0.0, vy2 = 0.0, vz2 = 0.0;
            for (int i = 0; i < mid_coeffs_u.size(); ++i) {
                const auto &mono = monomials_3d[i];
                Real val = mono.value(mid_pos.vx, mid_pos.vy, mid_pos.vz,
                                        mid_elem.centroid.vx, mid_elem.centroid.vy, mid_elem.centroid.vz,
                                        mid_elem.diameter);
                vx2 += mid_coeffs_u[i] * val;
                vy2 += mid_coeffs_v[i] * val;
                vz2 += mid_coeffs_w[i] * val;
            }

            // ---------- Step 5: 更新粒子位置 ----------
            sp.pos.vx += dt * vx2;
            sp.pos.vy += dt * vy2;
            sp.pos.vz += dt * vz2;
            sp.contained_in_element_id = mid_elem_id; // 更新粒子当前所在元素

            // ---------- Step 6: 越界检测 ----------
            if (sp.pos.vx < 0.0 || sp.pos.vx > 1.0 ||
                sp.pos.vy < 0.0 || sp.pos.vy > 1.0 ||
                sp.pos.vz < 0.0 || sp.pos.vz > 1.0) {
                sp.deleted = true;
            }
        }
    }
}

void VEM_Domain_3d::add_external_forces(std::vector<Smoke_Particles_3d> &smoke_particles,
                                        const Real &dt, const Real &kernel_radius)
{
    FUNC_TIMER();
    const Real h = kernel_radius;
    const Real coeff = 0.0000003;
    const Real g = 9.8 * 2;

    const int n_dof = static_cast<int>(k1_dof_density.size());
    k1_dof_density.setZero();

    // ===== 每线程局部累加器 =====
    const int num_threads = omp_get_max_threads();
    std::vector<Eigen::VectorXr> local_density(num_threads, Eigen::VectorXr::Zero(n_dof));

    // ===== 并行遍历所有元素 =====
#pragma omp parallel for schedule(dynamic)
    for (int eid = 0; eid < static_cast<int>(vem_elements.size()); ++eid) {
        const auto &elem = vem_elements[eid];
        const int n_verts = static_cast<int>(elem.vertex_count());
        if (n_verts == 0 || elem.contained_smoke_particles.empty()) continue;

        int tid = omp_get_thread_num();
        auto &thread_density = local_density[tid];

        for (int lv = 0; lv < n_verts; ++lv) {
            const _3d_vec &cp = elem.vertex_position(lv);
            const int global_vid = elem.vertex_local_to_global_dof[lv];
            Real sum_density = 0.0;

            for (int pid : elem.contained_smoke_particles) {
                const auto &p = smoke_particles[pid];
                if (p.deleted) continue;

                const Real dx = cp.vx - p.pos.vx;
                const Real dy = cp.vy - p.pos.vy;
                const Real dz = cp.vz - p.pos.vz;
                const Real r2 = dx * dx + dy * dy + dz * dz;
                if (r2 >= h * h) continue;

                const Real r = std::sqrt(r2);
                sum_density += cubic_kernel(r, h) * coeff;
            }

            thread_density[global_vid] += sum_density;
        }
    }

    // ===== 合并局部累加结果 =====
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_dof; ++i) {
        Real total = 0.0;
        for (int t = 0; t < num_threads; ++t)
            total += local_density[t][i];
        k1_dof_density[i] = total;
    }

    // ===== 裁剪并加速度 =====
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n_dof; ++i) {
        Real &rho = k1_dof_density[i];
        // if (rho < 0.0) rho = 0.0;
        // else if (rho > 1.0) rho = 1.0;

        if (rho > 1e-20) {
            k1_dof_v[i] += rho * g * dt;
        }
    }


}

void VEM_Domain_3d::add_external_forces(const Real &dt)
{
    FUNC_TIMER();

    const int ndof = (int)k1_dof_v.size();

#pragma omp parallel for
    for (int i = 0; i < ndof; ++i)
    {
        // 读取 DOF 上的标量
        const Real rho  = k1_dof_density[i];
        const Real temp = k1_dof_temp[i];

        // Buoyancy（向上为正）
        const Real buoyancy = _beta * temp - _alpha * rho;
        // 施加到“向上速度”（v 分量）
        k1_dof_v[i] += dt * buoyancy;
    }

    // // set movable boundary after get poly_coeffs_u_k1_old
    // for (int vid=0;vid<k1_dof_u.size();++vid) {
    //     if (is_vertex_boundary[vid]) {
    //         _3d_vec old_v( k1_dof_u[vid], k1_dof_v[vid], k1_dof_w[vid]);
    //         _3d_vec old_normal_v = vertex_normals[vid]*(old_v.dot(vertex_normals[vid]));
    //         _3d_vec new_v = old_v - old_normal_v + vertex_normals[vid]*vertex_normals[vid].dot(vertex_neumann_vels[vid]);
    //         k1_dof_u[vid]       = new_v.vx;
    //         k1_dof_v[vid]       = new_v.vy;
    //         k1_dof_w[vid]       = new_v.vz;
    //     }
    // }

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
        // if (element.has_boundary) {
        //     const auto &elem = element;
        //     for (int local_pid=0;local_pid<elem.polygons.size();++local_pid) {
        //         const int global_pid = elem.polygon_local_to_global_dof[local_pid];
        //         if (!is_polygon_boundary[global_pid])continue;
        //         std::vector<_3d_vec> constrained_dirs;
        //         const _3d_vec& normal = _3d_vec(0,0,0) - _3d_vec(elem.poly_planes[local_pid].orthogonal_vector().x(),
        //         elem.poly_planes[local_pid].orthogonal_vector().y(),
        //         elem.poly_planes[local_pid].orthogonal_vector().z());
        //         constrained_dirs.push_back(normal);
        //         for (auto& local_vid : elem.polygons[local_pid]) {
        //             {
        //                 // v
        //                 _3d_vec v(element_dof_values_u[local_vid], element_dof_values_v[local_vid],
        //                     element_dof_values_w[local_vid]);
        //                 for (auto& constrained_dir : constrained_dirs) {
        //                     Real dot_product = v.vx * constrained_dir.vx + v.vy * constrained_dir.vy + v.vz * constrained_dir.vz;
        //                     _3d_vec projection = constrained_dir * dot_product;
        //                     if (dot_product > 0.0)
        //                         v = v - projection;
        //                 }
        //                 element_dof_values_u[local_vid] = v.vx;
        //                 element_dof_values_v[local_vid] = v.vy;
        //                 element_dof_values_w[local_vid] = v.vz;
        //             }
        //         }
        //     }
        //
        // }
        RZ_ASSERT(projector_k1.cols() == ndof, "projector_k1.cols()==element_dof_values.size()");
        // === 计算该元素的多项式系数 ===
        element.poly_coeffs_u_k1 = projector_k1 * element_dof_values_u;
        element.poly_coeffs_v_k1 = projector_k1 * element_dof_values_v;
        element.poly_coeffs_w_k1 = projector_k1 * element_dof_values_w;
    }

}


void VEM_Domain_3d::heat_decay(std::vector<FLIP_Particles_3d>& flip_particles, const Real &dt) {
    FUNC_TIMER();

    const int num = (int)flip_particles.size();

    #pragma omp parallel for
    for (int pid = 0; pid < num; ++pid)
    {
        if (flip_particles[pid].deleted) continue;

        flip_particles[pid].temp =
            flip_particles[pid].temp / (1.0 + _decay * dt);
    }
}

void VEM_Domain_3d::heat_decay(const Real &dt) {
    FUNC_TIMER();

    for (int pid = 0; pid < k1_dof_temp.size(); ++pid)
    {
        k1_dof_temp[pid] = k1_dof_temp[pid] / (1+_decay * dt);
    }
}




void VEM_Domain_3d::assemble_weakform_rhs_k1(Real dt) {
    FUNC_TIMER();
//-----------------------------------------------------------------------------------------------------------------------------------
    global_rhs_k1.resize(points.size());
    global_rhs_k1.setZero();
    for (int el_id = 0; el_id < (int)vem_elements.size(); ++el_id) {
        const auto& elem = vem_elements[el_id];
        const int nv = (int)elem.vertex_count();
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

void VEM_Domain_3d::assemble_weakform_rhs_k2(Real dt) {
    global_rhs_k2.setZero();
    const int n_vertices = points.size();
    const int n_edges = edges.size();
    const int n_polygons = polygons.size();
    for (int el_id = 0; el_id < (int)vem_elements.size(); ++el_id) {
        const auto& elem = vem_elements[el_id];

        const int nv = (int)elem.vertex_count();
        std::vector<int>element_to_global_dof;
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


        const int ndofs    = (int)element_to_global_dof.size(); // k2 pressure dofs
        const int ndofs_k1 = nv; // 顶点 dof（与你原注释一致）

        for (int i = 0; i < ndofs; ++i) {
            const int gi = element_to_global_dof[i];

            Real acc = 0.0;

            for (int j = 0; j < ndofs_k1; ++j) {
                const int gj = element_to_global_dof[j];

                const Real vx = elem.is_U
                    ? U3d_local_LX_k2(i, j)
                    : elem.local_LX_k2(i, j);

                const Real vy = elem.is_U
                    ? U3d_local_LY_k2(i, j)
                    : elem.local_LY_k2(i, j);

                const Real vz = elem.is_U
                    ? U3d_local_LZ_k2(i, j)
                    : elem.local_LZ_k2(i, j);

                // acc += vx * k1_dof_u[gj]
                //      + vy * k1_dof_v[gj]
                //      + vz * k1_dof_w[gj];
                RZ_ASSERT(!std::isnan(vy)&&!std::isnan(vx)&&!std::isnan(vz),"nan");
                Real vx0 = std::isnan(vx) ? 0.0 : vx;
                Real vy0 = std::isnan(vy) ? 0.0 : vy;
                Real vz0 = std::isnan(vz) ? 0.0 : vz;

                acc += vx0 * k1_dof_u[gj]
                     + vy0 * k1_dof_v[gj]
                     + vz0 * k1_dof_w[gj];
            }
            global_rhs_k2[gi] += acc;
            // global_rhs_k2[gi] -= acc;
        }
    }

    global_rhs_k2 /= dt;

    RZ_INFO("rhs norm:",global_rhs_k2.norm());

    for (int i = 0; i < global_rhs_k2.size(); ++i) {
        if (!std::isfinite(global_rhs_k2[i])) {
            RZ_INFO("nan in rhs");
            global_rhs_k2[i] = 0.0;
        }
    }
}


void VEM_Domain_3d::assemble_strongform_rhs_k1(Real dt) {
    FUNC_TIMER();
    Eigen::VectorXr k1_dof_divergence;
    k1_dof_divergence.resize(points.size());
    k1_dof_divergence.setZero();
    global_rhs_k1.setZero();
    for (const auto& element:vem_elements) {
        for (int vid=0;vid<element.vertex_count();++vid) {
            const int global_vid = element.vertex_local_to_global_dof[vid];
            //通过这个投影 来求这个点处的散度
            const _3d_vec pos = element.vertex_position(vid);
            Real divergence = 0.0;
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
                element_dof_values_u[i] = k1_dof_u[vert_ids[i]];
                element_dof_values_v[i] = k1_dof_v[vert_ids[i]];
                element_dof_values_w[i] = k1_dof_w[vert_ids[i]];
            }
            const auto &projector_k1 = element.is_U?
        U3d_project_matrix_star_k1 : element.project_matrix_star_k1;
            Eigen::VectorXr poly_coeffs_u = projector_k1 * element_dof_values_u;
            Eigen::VectorXr poly_coeffs_v = projector_k1 * element_dof_values_v;
            Eigen::VectorXr poly_coeffs_w = projector_k1 * element_dof_values_w;
            // === 插值多项式 ===
            const Real xc = element.centroid.vx;
            const Real yc = element.centroid.vy;
            const Real zc = element.centroid.vz;
            const Real h  = element.diameter;
            for (int i = 0; i < poly_coeffs_u.size(); ++i) {
                const auto &mono = monomials_3d[i];
                Real gradx = mono.derivative_x().value(pos.vx, pos.vy, pos.vz, xc, yc, zc, h);
                Real grady = mono.derivative_y().value(pos.vx, pos.vy, pos.vz, xc, yc, zc, h);
                Real gradz = mono.derivative_z().value(pos.vx, pos.vy, pos.vz, xc, yc, zc, h);
                divergence += poly_coeffs_u(i) * gradx;
                divergence += poly_coeffs_v(i) * grady;
                divergence += poly_coeffs_w(i) * gradz;
            }
            k1_dof_divergence[global_vid] = divergence;
        }
    }
    for (int i = 0; i < points.size(); ++i) {
        if (std::fabs(k1_dof_divergence[i])<1e-14)k1_dof_divergence[i] = 0.0;
        k1_dof_divergence[i] /= -dt;
    }
    // RZ_INFO(k1_dof_divergence);
    // exit(0);
    // for (int k=0;k<points.size();++k) {
    //     if (std::fabs(global_rhs_k1[k])>1e-10 || std::fabs(k1_dof_divergence[k])>1e-10) {
    //         RZ_INFO(global_rhs_k1[k]);
    //         RZ_INFO(k1_dof_divergence[k]);
    //         RZ_INFO(k1_dof_divergence[k]/global_rhs_k1[k]);
    //         RZ_INFO("------------------------");
    //     }
    // }
    // exit(0);
    global_rhs_k1 = k1_dof_divergence;


}

static bool inited = true;
static bool inited_k1 = false;

static VEX_PCG pcg_solver;
static AMGCLContext solver_context_k1;
static Eigen::SparseMatrixCr RZA;
static Eigen::VectorXr k2_k1_res;
static VEM_PMGSolver pmg_solver;



void VEM_Domain_3d::solve_k2(Real dt) {
    FUNC_TIMER();
    // using Clock = std::chrono::high_resolution_clock;
    // auto warm = global_p_k2;
    // {
    //     using SpMat = Eigen::SparseMatrix<Real, Eigen::RowMajor, int>;
    //     using Solver = Eigen::ConjugateGradient<
    //         SpMat,
    //         Eigen::Lower | Eigen::Upper,
    //         Eigen::DiagonalPreconditioner<Real>
    //     >;
    //
    //
    //     Solver solver;
    //     solver.setTolerance(1e-3);
    //     solver.setMaxIterations(10000);
    //
    //     solver.compute(global_stiffness_matrix_k2);
    //     if (solver.info() != Eigen::Success) {
    //         RZ_INFO("not sucess");
    //     }
    //
    // auto t0 = Clock::now();
    //
    // global_p_k2 = solver.solveWithGuess(global_rhs_k2,warm);
    //
    //
    // auto t1 = Clock::now();
    // Real ms = std::chrono::duration<Real, std::milli>(t1 - t0).count();
    //
    //     if (true) {
    //         std::cerr << "[DPCG] iters=" << solver.iterations()
    //                   << " rel_res=" << solver.error()
    //                   << " info=" << int(solver.info()) <<" ms="<<
    //                       ms<<"\n";
    //     }
    //
    // }

    // {
    //     using SpMat = Eigen::SparseMatrix<Real, Eigen::RowMajor, int>;
    //     using Solver = Eigen::ConjugateGradient<
    //         SpMat,
    //         Eigen::Lower | Eigen::Upper,
    //         Eigen::IncompleteCholesky<Real>
    //     >;
    //
    //
    //     Solver solver;
    //     solver.setTolerance(1e-3);
    //     solver.setMaxIterations(10000);
    //
    //     solver.compute(global_stiffness_matrix_k2);
    //     if (solver.info() != Eigen::Success) {
    //         RZ_INFO("not sucess");
    //     }
    //
    //     auto t0 = Clock::now();
    //
    //     global_p_k2 = solver.solveWithGuess(global_rhs_k2,warm);
    //
    //
    //     auto t1 = Clock::now();
    //     Real ms = std::chrono::duration<Real, std::milli>(t1 - t0).count();
    //
    //
    //     if (true) {
    //         std::cerr << "[ICPCG] iters=" << solver.iterations()
    //                   << " rel_res=" << solver.error()
    //         << " info=" << int(solver.info())<<" ms="<<
    //             ms<<"\n";
    //     }
    //
    // }

    {
        for (int i = 0; i < global_rhs_k2.size(); ++i) {
            if (!std::isfinite(global_rhs_k2[i])) {
                RZ_INFO("nan in rhs");
                global_rhs_k2[i] = 0.0;
            }
        }
        global_p_k2 = amgclSolve(solver_context, global_rhs_k2);
    }
    RZ_INFO("max pressure: ",global_p_k2.maxCoeff());
    // exit(0);
}

void VEM_Domain_3d::solve_k1(Real dt) {
    FUNC_TIMER();
    Eigen::VectorXr solved_p = amgclSolve(solver_context,global_rhs_k1);
    if (!liquid_internal_dofs.empty()
        && solved_p.size() == static_cast<int>(liquid_internal_dofs.size())) {
        global_p_k1 = Eigen::VectorXr::Zero(points.size());
        for (int i = 0; i < static_cast<int>(liquid_internal_dofs.size()); ++i) {
            global_p_k1[liquid_internal_dofs[i]] = solved_p[i];
        }
        global_rhs_k1.resize(points.size());
        global_rhs_k1.setZero();
    } else {
        global_p_k1.swap(solved_p);
    }
}

Eigen::SparseMatrixCr VEM_Domain_3d::build_Ak2() {
    return this->global_stiffness_matrix_k2;
    int n_vertices = (int) points.size();
    int n_edges = (int) edges.size();
    int n_polygons = (int) polygons.size();
    int n_elements = (int) vem_elements.size();
    int total_dofs = n_vertices + n_edges + n_polygons + n_elements;
    int num_idof = total_dofs;
    std::vector<int> internal_dofs;

    // -----------------------------
    // 1. 找 boundary dofs
    // -----------------------------
    std::vector<int> boundary_dofs;
    std::vector<_2d_vec> boundary_query_points;

    // 非边界顶点
    for (int v = 0; v < n_vertices; ++v) {
        internal_dofs.push_back(v);
    }

    // 非纯边界边
    for (int e_id = 0; e_id < n_edges; ++e_id) {
        internal_dofs.push_back(n_vertices + e_id);
    }
    // 非边界面
    for (int pid = 0; pid < n_polygons; ++pid) {
        internal_dofs.push_back(n_vertices + n_edges + pid);
    }
    // 全部单元 dofs
    for (int el = 0; el < n_elements; ++el) {
        internal_dofs.push_back(n_vertices + n_edges + n_polygons + el);
    }

    // -----------------------------
    // 4. 提取内部子系统 K_II
    // -----------------------------
    Eigen::VectorXi idofs = Eigen::VectorXi::Map(internal_dofs.data(), num_idof);
    const int nI = idofs.size();
    std::unordered_map<int, int> g2l;
    g2l.reserve(nI * 2);
    for (int i = 0; i < nI; ++i) g2l[idofs[i]] = i;
    Eigen::SparseMatrixR Krow = this->global_stiffness_matrix_k2;
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
    return K_II;
}

Eigen::SparseMatrixR VEM_Domain_3d::build_Ak1() {
    int n_vertices = (int)points.size();

    std::vector<int> internal_dofs;
    if (!liquid_internal_dofs.empty()) {
        internal_dofs = liquid_internal_dofs;
    } else {
        RZ_FATAL("");
    }

    int num_idof = (int)internal_dofs.size();

    Eigen::VectorXi idofs =
        Eigen::VectorXi::Map(internal_dofs.data(), num_idof);

    const int nI = idofs.size();

    std::unordered_map<int, int> g2l;
    g2l.reserve(nI * 2);

    for (int i = 0; i < nI; ++i)
        g2l[idofs[i]] = i;

    Eigen::SparseMatrixR Krow =
        this->global_stiffness_matrix_k1;

    Eigen::SparseMatrixR K_II(nI, nI);

    std::vector<Eigen::Triplet<GlobalReal>> trips;
    trips.reserve(static_cast<size_t>(Krow.nonZeros() * 0.3));

    for (int ii = 0; ii < nI; ++ii) {
        int gi = idofs[ii];

        for (Eigen::SparseMatrixR::InnerIterator it(Krow, gi); it; ++it) {
            int gj = it.col();

            auto found = g2l.find(gj);

            if (found != g2l.end()) {
                int jj = found->second;
                trips.emplace_back(ii, jj, it.value());
            }
        }
    }

    K_II.setFromTriplets(trips.begin(), trips.end());
    K_II.makeCompressed();

    return K_II;
}


#include"Visualization.h"
static int save_id = 0;

void VEM_Domain_3d::visualize_solved_pressure_k2(int nx,int ny,int nz) {
    // 先x 后 y 再z
    Eigen::VectorXr pressure;
    pressure.resize(nx*ny*nz);
    pressure.setZero();
    int element_id = 0;
    for (const auto &element : vem_elements) {
        const auto &P = element.is_U==true ?
                    U3d_project_matrix_star_k2:element.project_matrix_star_k2;
        std::vector<Real> tmp_vals;
        tmp_vals.reserve(P.cols());

        for (int vid : element.vertex_local_to_global_dof)
            tmp_vals.push_back(global_p_k2[vid]);

        for (int eid = 0; eid < (int)element.edge_count(); ++eid)
            tmp_vals.push_back(global_p_k2[points.size() + element.edge_local_to_global_dof[eid]]);

        for (int pid : element.polygon_local_to_global_dof)
            tmp_vals.push_back(global_p_k2[points.size() + edges.size() + pid]);

        tmp_vals.push_back(global_p_k2[points.size() + edges.size() + polygons.size() + element_id]);

        Eigen::Map<Eigen::VectorXr> elem_p(tmp_vals.data(), tmp_vals.size());
        Eigen::VectorXr coeff = P * elem_p;

        Real dx = 1.0 / nx, hdx = dx / 2.0;
        for (int ix = 0; ix < nx; ++ix)
            for (int iy = 0; iy < ny; ++iy)
                for (int iz = 0; iz < nz; ++iz) {
                    _3d_vec pos(ix * dx + hdx, iy * dx + hdx, iz * dx + hdx);
                    if (!point_in_convex_polyhedron(pos, element)) continue;

                    Real p_val = 0.0;
                    for (int mi = 0; mi < monomials_3d.size(); ++mi)
                        p_val += monomials_3d[mi].value(pos.vx, pos.vy, pos.vz,
                                                        element.centroid.vx, element.centroid.vy,
                                                        element.centroid.vz, element.diameter)
                                 * coeff(mi);

                    pressure(ix + iy * nx + iz * nx * ny) = p_val;
                }

        ++element_id;
    }

    save_pressure_vti(nx,ny,nz,1.0/(Real)nx,pressure,"vem_pressure"+std::to_string(save_id++)+".vti");

}

void VEM_Domain_3d::visualize_solved_pressure_k1(int nx, int ny, int nz) {
        // 先x 后 y 再z
    Eigen::VectorXr pressure;
    pressure.resize(nx*ny*nz);
    pressure.setZero();
    int element_id = 0;
    for (const auto &element : vem_elements) {
        const auto &P =  element.is_U?
        U3d_project_matrix_star_k1 : element.project_matrix_star_k1;

        std::vector<Real> tmp_vals;
        tmp_vals.reserve(P.cols());

        for (int vid : element.vertex_local_to_global_dof)
            tmp_vals.push_back(global_p_k1[vid]);

        Eigen::Map<Eigen::VectorXr> elem_p(tmp_vals.data(), tmp_vals.size());
        Eigen::VectorXr coeff = P * elem_p;

        Real dx = 1.0 / nx, hdx = dx / 2.0;
        for (int ix = 0; ix < nx; ++ix)
            for (int iy = 0; iy < ny; ++iy)
                for (int iz = 0; iz < nz; ++iz) {
                    _3d_vec pos(ix * dx + hdx, iy * dx + hdx, iz * dx + hdx);
                    if (!point_in_convex_polyhedron(pos, element)) continue;

                    Real p_val = 0.0;
                    for (int mi = 0; mi < 4; ++mi)
                        p_val += monomials_3d[mi].value(pos.vx, pos.vy, pos.vz,
                                                        element.centroid.vx, element.centroid.vy,
                                                        element.centroid.vz, element.diameter)
                                 * coeff(mi);

                    pressure(ix + iy * nx + iz * nx * ny) = p_val;
                }

        ++element_id;
    }

    save_pressure_vti(nx,ny,nz,1.0/(Real)nx,pressure,"vem_pressure_k1_"+std::to_string(save_id++)+".vti");
}

void VEM_Domain_3d::visualize_rhs_k1(int nx, int ny, int nz) {
    // 先x 后 y 再z
    Eigen::VectorXr rhs;
    rhs.resize(nx*ny*nz);
    rhs.setZero();
    int element_id = 0;
    for (const auto &element : vem_elements) {
        const auto &P = element.is_U?
        U3d_project_matrix_star_k1 : element.project_matrix_star_k1;

        std::vector<Real> tmp_vals;
        tmp_vals.reserve(P.cols());

        for (int vid : element.vertex_local_to_global_dof)
            tmp_vals.push_back(global_rhs_k1[vid]);

        Eigen::Map<Eigen::VectorXr> elem_p(tmp_vals.data(), tmp_vals.size());
        Eigen::VectorXr coeff = P * elem_p;

        Real dx = 1.0 / nx, hdx = dx / 2.0;
        for (int ix = 0; ix < nx; ++ix)
            for (int iy = 0; iy < ny; ++iy)
                for (int iz = 0; iz < nz; ++iz) {
                    _3d_vec pos(ix * dx + hdx, iy * dx + hdx, iz * dx + hdx);
                    if (!point_in_convex_polyhedron(pos, element)) continue;

                    Real p_val = 0.0;
                    for (int mi = 0; mi < 4; ++mi)
                        p_val += monomials_3d[mi].value(pos.vx, pos.vy, pos.vz,
                                                        element.centroid.vx, element.centroid.vy,
                                                        element.centroid.vz, element.diameter)
                                 * coeff(mi);

                    rhs(ix + iy * nx + iz * nx * ny) = p_val;
                }

        ++element_id;
    }

    save_pressure_vti(nx,ny,nz,1.0/(Real)nx,rhs,"vem_rhs_k1_"+std::to_string(save_id++)+".vti");
}

void VEM_Domain_3d::visualize_rhs_k2(int nx, int ny, int nz) {
        // 先x 后 y 再z
    Eigen::VectorXr rhs;
    rhs.resize(nx*ny*nz);
    rhs.setZero();
    int element_id = 0;
    for (const auto &element : vem_elements) {
        const auto &P = element.is_U==true ?
             U3d_project_matrix_star_k2:element.project_matrix_star_k2;

        std::vector<Real> tmp_vals;
        tmp_vals.reserve(P.cols());

        for (int vid : element.vertex_local_to_global_dof)
            tmp_vals.push_back(global_rhs_k2[vid]);

        for (int eid = 0; eid < (int)element.edge_count(); ++eid)
            tmp_vals.push_back(global_rhs_k2[points.size() + element.edge_local_to_global_dof[eid]]);

        for (int pid : element.polygon_local_to_global_dof)
            tmp_vals.push_back(global_rhs_k2[points.size() + edges.size() + pid]);

        tmp_vals.push_back(global_rhs_k2[points.size() + edges.size() + polygons.size() + element_id]);

        Eigen::Map<Eigen::VectorXr> elem_p(tmp_vals.data(), tmp_vals.size());
        Eigen::VectorXr coeff = P * elem_p;

        Real dx = 1.0 / nx, hdx = dx / 2.0;
        for (int ix = 0; ix < nx; ++ix)
            for (int iy = 0; iy < ny; ++iy)
                for (int iz = 0; iz < nz; ++iz) {
                    _3d_vec pos(ix * dx + hdx, iy * dx + hdx, iz * dx + hdx);
                    if (!point_in_convex_polyhedron(pos, element)) continue;

                    Real p_val = 0.0;
                    for (int mi = 0; mi < monomials_3d.size(); ++mi)
                        p_val += monomials_3d[mi].value(pos.vx, pos.vy, pos.vz,
                                                        element.centroid.vx, element.centroid.vy,
                                                        element.centroid.vz, element.diameter)
                                 * coeff(mi);

                    rhs(ix + iy * nx + iz * nx * ny) = p_val;
                }

        ++element_id;
    }

    save_pressure_vti(nx,ny,nz,1.0/(Real)nx,rhs,"vem_rhs_k2_"+std::to_string(save_id++)+".vti");

}

Real polygon_center_value(
    const std::vector<_3d_vec>& points,
    const std::vector<int>& polygon,
    const Eigen::VectorXr& dof_p_k1)
{
    assert(polygon.size() >= 3);
    auto V = [&](int idx)->Eigen::Vector3d {
        return Eigen::Vector3d(points[idx].vx, points[idx].vy, points[idx].vz);
    };

    Eigen::Vector3d c_sum = Eigen::Vector3d::Zero();
    Real f_sum = 0.0;
    Real area_sum = 0.0;

    const int v0 = polygon[0];
    Eigen::Vector3d p0 = V(v0);
    Real f0 = dof_p_k1[v0];

    for (size_t i = 1; i + 1 < polygon.size(); ++i) {
        int vi = polygon[i], vj = polygon[i + 1];
        Eigen::Vector3d pi = V(vi);
        Eigen::Vector3d pj = V(vj);

        // 三角形面积（向量叉积模长的一半）
        Eigen::Vector3d n = (pi - p0).cross(pj - p0);
        Real area = 0.5 * n.norm();

        // 三角形质心
        Eigen::Vector3d tri_c = (p0 + pi + pj) / 3.0;
        // 三角形平均函数值
        Real tri_f = (f0 + dof_p_k1[vi] + dof_p_k1[vj]) / 3.0;

        area_sum += area;
        c_sum += area * tri_c;
        f_sum += area * tri_f;
    }
    return f_sum;
}

Eigen::VectorXr VEM_Domain_3d::Warm_from_k1(const Eigen::VectorXr &dof_p_k1) {// todo : 这里是not scale as 1的吧
    const int ndof_k2 = global_rhs_k2.rows();
    const int ndof_k1 = global_rhs_k1.rows();
    Eigen::VectorXr ans(ndof_k2);
    ans.setZero();
    ans.head(dof_p_k1.rows()) = dof_p_k1;
    for (int eid = 0;eid<edges.size();++eid) {
        ans(ndof_k1 + eid) = (dof_p_k1[edges[eid].v1]+dof_p_k1[edges[eid].v2])/2;
    }
    for (int pid =0;pid<polygons.size();++pid) {
        ans(ndof_k1+edges.size()+pid) = polygon_center_value(points,polygons[pid],dof_p_k1) ;
    }
    for (int Vid=0;Vid<vem_elements.size();++Vid) {
        const auto&center = vem_elements[Vid].centroid;
        const auto& element = vem_elements[Vid];
        const auto &proj = element.is_U?
        U3d_project_matrix_star_k1 : element.project_matrix_star_k1;
        const auto &verts = element.vertex_local_to_global_dof;
        Eigen::VectorXr element_dof_values_p;
        element_dof_values_p.resize(element.vertex_local_to_global_dof.size());
        const int nv = (int)verts.size();
        for (int i = 0; i < nv; ++i)
            element_dof_values_p[i] = dof_p_k1[verts[i]];
        const Eigen::VectorXr poly_coeffs_p = proj * element_dof_values_p;
        const auto &pos = center;

        Real value = 0.0;
        const Real xc = element.centroid.vx;
        const Real yc = element.centroid.vy;
        const Real zc = element.centroid.vz;
        const Real h  = element.diameter;
        const int n_basis = 4;// k=1
        // === 梯度插值 ===
        for (int i = 0; i < n_basis; ++i) {
            const auto &mono = monomials_3d[i];
            const auto m = mono.value(pos.vx, pos.vy, pos.vz, xc, yc, zc, h);
            const Real coeff = poly_coeffs_p(i);
            value += m*coeff;
        }
        ans(ndof_k1 + edges.size() + polygons.size() + Vid) = value * element.volume;
    }
    return ans;
}

void VEM_Domain_3d::Renew_rhs_k1(const Eigen::VectorXr& p_k2, Real dt) {
    // 插值每个粒子处压强的梯度 并用之更新速度
    int element_id = 0;
    int max_per_element_dof_num_k2 = 0;
    for (const auto &element : vem_elements) {
        int ndof = (int)element.vertex_local_to_global_dof.size()
                  + (int)element.edge_count()
                  + (int)element.polygons.size()
                  + 1;
        max_per_element_dof_num_k2 = std::max(max_per_element_dof_num_k2, ndof);
    }
    k1_dof_u_pmg.setZero();
    k1_dof_v_pmg.setZero();
    k1_dof_w_pmg.setZero();
    // === 缓存常数偏移 ===
    const int base_v = 0;
    const int base_e = (int)points.size();
    const int base_f = base_e + (int)edges.size();
    const int base_c = base_f + (int)polygons.size();

    Eigen::VectorXr element_dof_values_p;
    element_dof_values_p.resize(max_per_element_dof_num_k2); // 可在函数外声明缓存
    Eigen::VectorXr weights;
    weights.resize(points.size());
    weights.setZero();
    for (const auto &element: vem_elements) {
        const auto &proj = element.is_U==true ?
            U3d_project_matrix_star_k2:element.project_matrix_star_k2;
        const auto &verts = element.vertex_local_to_global_dof;
        int offset = 0;
        // === 1️⃣ 顶点自由度 ===
        const int nv = (int)verts.size();
        for (int i = 0; i < nv; ++i)
            element_dof_values_p[offset++] = p_k2[ verts[i]];

        // === 2️⃣ 边自由度 ===
        for (int eid = 0; eid < (int)element.edge_count(); ++eid) {
            element_dof_values_p[offset++] =
                p_k2[base_e + element.edge_local_to_global_dof[eid]];
        }

        const int npoly = (int)element.polygons.size();
        for (int i = 0; i < npoly; ++i)
            element_dof_values_p[offset++] =
                p_k2[base_f + element.polygon_local_to_global_dof[i]];

        element_dof_values_p[offset++] = p_k2[base_c + element_id];

        const Eigen::VectorXr poly_coeffs_p = proj * element_dof_values_p.head(offset);

        for (int local_vid =0 ;local_vid<element.vertex_count();++local_vid) {
            int global_vid = element.vertex_local_to_global_dof[local_vid];
            const auto& pos = element.vertex_position(local_vid);
            Real dvx = 0.0, dvy = 0.0, dvz = 0.0;

            const Real xc = element.centroid.vx;
            const Real yc = element.centroid.vy;
            const Real zc = element.centroid.vz;
            const Real h  = element.diameter;
            const int n_basis = (int)monomials_3d.size();

            // === 梯度插值 ===
            for (int i = 0; i < n_basis; ++i) {
                const auto &mono = monomials_3d[i];
                const auto grad = mono.gradient(pos.vx, pos.vy, pos.vz, xc, yc, zc, h);
                const Real coeff = poly_coeffs_p(i);
                dvx += grad.vx * coeff;
                dvy += grad.vy * coeff;
                dvz += grad.vz * coeff;
            }

            // === 更新粒子速度 ===
            dvx *= dt;
            dvy *= dt;
            dvz *= dt;

            k1_dof_u_pmg[global_vid] -= dvx * element.volume;
            k1_dof_v_pmg[global_vid] -= dvy * element.volume;
            k1_dof_w_pmg[global_vid] -= dvz * element.volume;
            weights[global_vid] += element.volume;
        }
        element_id++;
    }
    for (int vid=0;vid<points.size();++vid) {
        k1_dof_u_pmg[vid] /= weights[vid];
        k1_dof_v_pmg[vid] /= weights[vid];
        k1_dof_w_pmg[vid] /= weights[vid];
        k1_dof_u_pmg[vid] += k1_dof_u[vid];
        k1_dof_v_pmg[vid] += k1_dof_v[vid];
        k1_dof_w_pmg[vid] += k1_dof_w[vid];
    }
    global_rhs_k1 =
        (global_LX_matrix_k1 * k1_dof_u_pmg.template cast<GlobalReal>()).template cast<Real>()
        + (global_LY_matrix_k1 * k1_dof_v_pmg.template cast<GlobalReal>()).template cast<Real>()
        + (global_LZ_matrix_k1 * k1_dof_w_pmg.template cast<GlobalReal>()).template cast<Real>();
    global_rhs_k1 /= dt;
}


void VEM_Domain_3d::save_boundary_to_obj() {
    RZ_ASSERT(is_polygon_boundary.size() == polygons.size(),"");
    std::string filename = "boundary.obj";
    std::ofstream file(filename.c_str());
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return;
    }

    for (int pid = 0; pid < polygons.size(); ++pid) {
        if (!is_obj_boundary[pid])continue;
        const auto&poly = polygons[pid];
        for (int vid : poly) {
            file << points[vid].to_obj_format() << std::endl;
        }
    }
    int vertex_cnt =0;
    for (int pid = 0; pid < polygons.size(); ++pid) {
        if (!is_obj_boundary[pid])continue;
        const auto&poly = polygons[pid];
        file << "f";
        for (int cc : poly) {
            file << " " << 1 + vertex_cnt++;
        }
        file << std::endl;
    }
    file.close();
}

void VEM_Domain_3d::save_obj_surface_to_obj() {
    RZ_ASSERT(is_obj_boundary.size() == polygons.size(),"");
    std::string filename = "obj_surface.obj";
    std::ofstream file(filename.c_str());
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return;
    }

    for (int pid = 0; pid < polygons.size(); ++pid) {
        if (!is_obj_boundary[pid])continue;
        const auto&poly = polygons[pid];
        for (int vid : poly) {
            file << points[vid].to_obj_format() << std::endl;
        }
    }
    int vertex_cnt =0;
    for (int pid = 0; pid < polygons.size(); ++pid) {
        if (!is_obj_boundary[pid])continue;
        const auto&poly = polygons[pid];
        file << "f";
        for (int cc : poly) {
            file << " " << 1 + vertex_cnt++;
        }
        file << std::endl;
    }
    file.close();
}

void VEM_Domain_3d::save_boundary_cells() {
    std::string filename = "boundary_cells.obj";
    int idx =0 ;

    std::ofstream file(filename.c_str());
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return;
    }
    for (auto&point:points) {
        file << point.to_obj_format() << std::endl;
    }
    for (const auto&element:vem_elements) {
        if (!element.has_boundary)continue;
        for (const auto& polygon : element.polygons) {
            file << "f";
            for (int idx : polygon) {
                // .obj 文件中索引从 1 开始，所以我们加 1
                file << " " << element.vertex_local_to_global_dof[idx] + 1;
            }
            file << std::endl;
        }
    }

    file.close();
    std::cout << "文件保存成功: " << filename << std::endl;
}
void save_tris_to_obj(const std::list<Triangle>& tris, const std::string& filename) {
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Failed to open file for writing!" << std::endl;
        return;
    }
    // 打印顶点
    int index = 1;
    for (const auto& tri : tris) {
        // 输出每个三角形的三个顶点
        outFile << "v " << tri[0].x() << " " << tri[0].y() << " " << tri[0].z() << std::endl;
        outFile << "v " << tri[1].x() << " " << tri[1].y() << " " << tri[1].z() << std::endl;
        outFile << "v " << tri[2].x() << " " << tri[2].y() << " " << tri[2].z() << std::endl;
    }
    // 打印三角形面
    for (const auto& tri : tris) {
        // 面的顶点索引是从1开始的，因此在这里我们使用固定的顺序
        int idx1 = index++;
        int idx2 = index++;
        int idx3 = index++;
        outFile << "f " << idx1 << " " << idx2 << " " << idx3 << std::endl;
    }

    outFile.close();
    std::cout << "Saved " << tris.size() << " triangles to " << filename << std::endl;
}

void VEM_Domain_3d::build_AABB() {
    FUNC_TIMER();
    tris.clear();
    for (int pid=0;pid<polygons.size();++pid) {
        if (!is_polygon_boundary[pid]) continue;
        const auto&poly = polygons[pid];
        int v0=poly[0];
        for (int i=0;i<poly.size()-2;++i) {
            int v1 = poly[i+1];
            int v2 = poly[i+2];
            // 获取三个顶点的坐标
            _3d_vec p0 = points[v0];
            _3d_vec p1 = points[v1];
            _3d_vec p2 = points[v2];
            CGAL_Point_3 cp0(p0.vx, p0.vy, p0.vz);
            CGAL_Point_3 cp1(p1.vx, p1.vy, p1.vz);
            CGAL_Point_3 cp2(p2.vx, p2.vy, p2.vz);
            auto tri = Triangle(cp0, cp1, cp2);
            // RZ_ASSERT(!tri.is_degenerate(),"");
            if (!tri.is_degenerate())
                tris.push_back(tri);
        }
    }
    RZ_INFO("AABB tris: ",tris.size());
    save_tris_to_obj(tris,"AABB_tris.obj");
    aabb_tree = AABBTree(tris.begin(), tris.end());
    RZ_INFO("AABB tree: ",aabb_tree.size());
}

void VEM_Domain_3d::direct_AABB(std::string aabbobj)
{
    FUNC_TIMER();

    CGAL_Mesh mesh;
    bool ok = CGAL::IO::read_polygon_mesh(aabbobj, mesh);
    RZ_ASSERT(ok, "[direct_AABB] failed to read obj");

    RZ_ASSERT(CGAL::is_triangle_mesh(mesh),
              "[direct_AABB] OBJ is not a pure triangle mesh");

    tris.clear();

    // 遍历所有三角面
    for (auto f : mesh.faces())
    {
        auto h = mesh.halfedge(f);

        auto v0 = mesh.target(h);
        auto v1 = mesh.target(mesh.next(h));
        auto v2 = mesh.target(mesh.next(mesh.next(h)));

        const CGAL_Point_3& p0 = mesh.point(v0);
        const CGAL_Point_3& p1 = mesh.point(v1);
        const CGAL_Point_3& p2 = mesh.point(v2);

        Triangle tri(p0, p1, p2);

        // 跳过退化三角形（极少见，但保险）
        if (!tri.is_degenerate())
            tris.push_back(tri);
    }

    RZ_ASSERT(!tris.empty(),
              "[direct_AABB] no valid triangles found");

    RZ_INFO("direct_AABB tris: ", tris.size());

    // 构建 AABB
    aabb_tree = AABBTree(tris.begin(), tris.end());


    RZ_INFO("direct_AABB tree size: ", aabb_tree.size());
}


void VEM_Domain_3d::boundary_aware_pos_fix(const std::vector<_3d_vec> &old_poses, std::vector<_3d_vec> &new_poses,const Real dt) {

    std::vector<int> to_handle;
    for (int k=0;k<old_poses.size();++k) {
        if (boundary_handle_grid(old_poses[k])) {
            to_handle.push_back(k);
        }
    }
    RZ_INFO("tohandle: ",to_handle.size()," all: ",old_poses.size());
    std::for_each(std::execution::par, to_handle.begin(), to_handle.end(),
       [&](int i) {
       Boundary_Handle(old_poses[i],new_poses[i],aabb_tree,dt);
   }
   );
}

void VEM_Domain_3d::clear_data_for_build() {
    clear_generate_elements_cache();
}

void VEM_Domain_3d::clear_generate_elements_cache(bool trim_allocator) {
    FUNC_MEMORY_TIMER();
    for (auto& elem_3d : vem_elements) {
        elem_3d.clear_generate_elements_cache();
    }

    Eigen::LocalMatrixXr().swap(U2d_stiffness_matrix_k1);
    Eigen::LocalMatrixXr().swap(U2d_stiffness_matrix);
    Eigen::LocalMatrixXr().swap(U3d_stiffness_matrix_k1);
    Eigen::LocalMatrixXr().swap(U3d_stiffness_matrix_k2);

#if defined(__GLIBC__)
    if (trim_allocator) {
        malloc_trim(0);
    }
#else
    (void)trim_allocator;
#endif
}

void VEM_Domain_3d::clear_hierarchy_build_cache() {
    FUNC_MEMORY_TIMER();
    for (auto& elem_3d : vem_elements) {
        elem_3d.clear_hierarchy_cache();
    }
    Eigen::MatrixXr().swap(U2d_project_matrix_star);
    Eigen::MatrixXr().swap(U2d_project_matrix_star_k1);
    Eigen::LocalMatrixXr().swap(U2d_stiffness_matrix_k1);
    Eigen::LocalMatrixXr().swap(U2d_stiffness_matrix);
    Eigen::MatrixXr().swap(U3d_project_matrix_star_k2);
    Eigen::MatrixXr().swap(U3d_project_matrix_star_k1);
    Eigen::LocalMatrixXr().swap(U3d_stiffness_matrix_k1);
    Eigen::LocalMatrixXr().swap(U3d_stiffness_matrix_k2);
    Eigen::MatrixXr().swap(U3d_local_LX_k1);
    Eigen::MatrixXr().swap(U3d_local_LY_k1);
    Eigen::MatrixXr().swap(U3d_local_LZ_k1);
    Eigen::MatrixXr().swap(U3d_local_LX_k2);
    Eigen::MatrixXr().swap(U3d_local_LY_k2);
    Eigen::MatrixXr().swap(U3d_local_LZ_k2);
}
