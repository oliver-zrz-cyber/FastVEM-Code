#include "sim3d.h"
#include "Render3d.h"
#include "AbSolver.h"
#include "mgsolver_vemg.h"
#include "mini_lcc.h"
#include "bsp_mesh.h"
#include "vem3d.h"
#include "vex_gmres.h"

static std::vector<FLIP_Particles_3d> flip_particles;
static std::vector<FLIP_Particles_3d> ghost_flip_particles;
static std::vector<RenderParticle> render_particles;
static std::vector<Smoke_Particles_3d> smoke_particles;
static Real dt = 0.01;
static bool inited_mac = false;
static AMGCLContext solver_context_mac;
static VEX_GMRES MG_gmres;
static Eigen::SparseMatrixR static_A;
static Eigen::VectorXr warm_pressure;




#pragma pack(push, 1)
struct TrancerBin {
    float x, y, z, rest_age;
    uint32_t id;
};
#pragma pack(pop)

static_assert(sizeof(TrancerBin) == 20);


void saveTrancersToBin(
    const std::vector<Trancer>& trancers,
    const std::string& filename
) {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }

    uint32_t count = static_cast<uint32_t>(trancers.size());
    ofs.write(reinterpret_cast<const char*>(&count), sizeof(uint32_t));

    if (count == 0) return;

    std::vector<TrancerBin> buf;
    buf.reserve(count);

    for (const auto& t : trancers) {
        buf.push_back({
            t.x, t.y, t.z, t.rest_age,
            static_cast<uint32_t>(t.id)
        });
    }

    ofs.write(reinterpret_cast<const char*>(buf.data()),
              sizeof(TrancerBin) * count);
}

void saveLqiudToBin(
    const std::vector<FLIP_Particles_3d>& trancers,
    const std::string& filename
) {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }

    uint32_t count = static_cast<uint32_t>(trancers.size());
    ofs.write(reinterpret_cast<const char*>(&count), sizeof(uint32_t));

    if (count == 0) return;

    std::vector<TrancerBin> buf;
    buf.reserve(count);
    int id =0;
    for (const auto& t : trancers) {
        buf.push_back({
            (float)t.pos.vx, (float)t.pos.vy, (float)t.pos.vz, 1.0,
            static_cast<uint32_t>(id++)
        });
    }

    ofs.write(reinterpret_cast<const char*>(buf.data()),
              sizeof(TrancerBin) * count);
}

static inline float frand(float a, float b)
{
    return a + (b - a) * (float(rand()) / float(RAND_MAX));
}
static uint32_t tracer_id=0;
//
void emit_tracers(
    VEM_Domain_3d& vem_domain,
    const sphere_source& src,
    int n
) {
    int emitted = 0;
    const float r = static_cast<float>(src.radius);
    auto& tracers = vem_domain.tracers;
    auto is_in = vem_domain.assign_tracers_to_elements_3d();
    int removed_cnt = 0;
    std::vector<Trancer> tracers_temp;
    for (int i=0;i<tracers.size();++i) {
        if (is_in[i] == -1) {
            ++removed_cnt;
            continue;
        }
        tracers_temp.push_back(tracers[i]);
    }
    // RZ_INFO("particles removed: ",removed_cnt);
    tracers.swap(tracers_temp);
    auto samples = src.sample_points(n);
    for (auto&pos:samples) {
        Trancer t;
        t.x = pos.vx;
        t.y = pos.vy;
        t.z = pos.vz;
        t.rest_age = 1.0f;
        t.id = tracer_id++;
        tracers.push_back(t);
    }

    is_in = vem_domain.assign_tracers_to_elements_3d();
    tracers_temp.clear();
    for (int i=0;i<tracers.size();++i) {
        if (is_in[i] == -1) {
            ++removed_cnt;
            continue;
        }
        tracers_temp.push_back(tracers[i]);
    }
    tracers.swap(tracers_temp);
}

void emit_tracers(
    VEM_Domain_3d& vem_domain,
    const cylinder_source& src,
    int n
) {
    int emitted = 0;
    const float r = static_cast<float>(src.radius);
    auto& tracers = vem_domain.tracers;
    auto is_in = vem_domain.assign_tracers_to_elements_3d();
    int removed_cnt = 0;
    std::vector<Trancer> tracers_temp;
    for (int i=0;i<tracers.size();++i) {
        if (is_in[i] == -1) {
            ++removed_cnt;
            continue;
        }
        tracers_temp.push_back(tracers[i]);
    }
    // RZ_INFO("particles removed: ",removed_cnt);
    tracers.swap(tracers_temp);
    auto samples = src.sample_points(n);
    for (auto&pos:samples) {
        Trancer t;
        t.x = pos.vx;
        t.y = pos.vy;
        t.z = pos.vz;
        t.rest_age = 1.0f;
        t.id = tracer_id++;
        tracers.push_back(t);
    }

    is_in = vem_domain.assign_tracers_to_elements_3d();
    tracers_temp.clear();
    for (int i=0;i<tracers.size();++i) {
        if (is_in[i] == -1) {
            ++removed_cnt;
            continue;
        }
        tracers_temp.push_back(tracers[i]);
    }
    tracers.swap(tracers_temp);
}


void emit_tracers(
    VEM_Domain_3d& vem_domain,
    const std::vector<cylinder_source>& srcs,
    int n
) {

     {
        auto src = srcs[0];

        auto& tracers = vem_domain.lucy_tracers_1;
        auto is_in = vem_domain.assign_lucy_tracers1_to_elements_3d();
        int removed_cnt = 0;
        std::vector<Trancer> tracers_temp;
        for (int i=0;i<tracers.size();++i) {
            if (is_in[i] == -1) {
                ++removed_cnt;
                continue;
            }
            tracers_temp.push_back(tracers[i]);
        }
        // RZ_INFO("particles removed: ",removed_cnt);
        tracers.swap(tracers_temp);
        auto samples = src.sample_points(n);
        for (auto&pos:samples) {
            Trancer t;
            t.x = pos.vx;
            t.y = pos.vy;
            t.z = pos.vz;
            t.rest_age = 1.0f;
            t.id = tracer_id++;
            tracers.push_back(t);
        }

        is_in = vem_domain.assign_lucy_tracers1_to_elements_3d();
        tracers_temp.clear();
        for (int i=0;i<tracers.size();++i) {
            if (is_in[i] == -1) {
                ++removed_cnt;
                continue;
            }
            tracers_temp.push_back(tracers[i]);
        }
        tracers.swap(tracers_temp);
    }
    {
         auto src = srcs[1];

         auto& tracers = vem_domain.lucy_tracers_2;
         auto is_in = vem_domain.assign_lucy_tracers2_to_elements_3d();
         int removed_cnt = 0;
         std::vector<Trancer> tracers_temp;
         for (int i=0;i<tracers.size();++i) {
             if (is_in[i] == -1) {
                 ++removed_cnt;
                 continue;
             }
             tracers_temp.push_back(tracers[i]);
         }
         // RZ_INFO("particles removed: ",removed_cnt);
         tracers.swap(tracers_temp);
         auto samples = src.sample_points(n);
         for (auto&pos:samples) {
             Trancer t;
             t.x = pos.vx;
             t.y = pos.vy;
             t.z = pos.vz;
             t.rest_age = 1.0f;
             t.id = tracer_id++;
             tracers.push_back(t);
         }

         is_in = vem_domain.assign_lucy_tracers2_to_elements_3d();
         tracers_temp.clear();
         for (int i=0;i<tracers.size();++i) {
             if (is_in[i] == -1) {
                 ++removed_cnt;
                 continue;
             }
             tracers_temp.push_back(tracers[i]);
         }
         tracers.swap(tracers_temp);
    }
}


void add_smoke_particles_in_range(
    const Source_range& range,
    Real dx = 0.02
)
{
    FUNC_TIMER();
    // const int ppc = 3;
    // const Real sub_dx = dx / ppc;
    //
    // for (int i = range.xmin; i <= range.xmax; ++i)
    //     for (int j = range.ymin; j <= range.ymax; ++j)
    //         for (int k = range.zmin; k <= range.zmax; ++k)
    //         {
    //             const Real base_x = i * dx;
    //             const Real base_y = j * dx;
    //             const Real base_z = k * dx;
    //
    //             for (int a = 0; a < ppc; ++a)
    //                 for (int b = 0; b < ppc; ++b)
    //                     for (int c = 0; c < ppc; ++c)
    //                     {
    //                         // 子 cell center
    //                         _3d_vec pos(
    //                             base_x + (a + 0.5) * sub_dx,
    //                             base_y + (b + 0.5) * sub_dx,
    //                             base_z + (c + 0.5) * sub_dx
    //                         );
    //
    //                         smoke_particles.emplace_back(pos);
    //                     }
    //         }



}

// void add_smoke_particles_in_range(
//     const sphere_source& range,
//     Real _dens,
//     Real _temp
// )
// {
//     FUNC_TIMER();
//
//     const _3d_vec& c = range.center;
//     const Real   r = range.radius;
//     const Real  r2 = r * r;
//
//     // ------------------------------------------------------------
//     // 1. 构建一个最简单的 uniform grid：cell -> particle indices
//     // ------------------------------------------------------------
//     const int nx = 100,ny=100,nz=100;
//     Real h = 1.0/100.0;
//     const int num_cells = nx * ny * nz;
//     std::vector<std::vector<int>> cell_particles(num_cells);
//
//     auto cell_id = [&](int i, int j, int k) {
//         return (k * ny + j) * nx + i;
//     };
//
//     // 把所有已有 FLIP 粒子分配到 cell
//     for (int pid = 0; pid < (int)flip_particles.size(); ++pid)
//     {
//         const _3d_vec& p = flip_particles[pid].pos;
//
//         int i = (int)std::floor(p.vx / h);
//         int j = (int)std::floor(p.vy / h);
//         int k = (int)std::floor(p.vz / h);
//
//         if (i < 0 || i >= nx ||
//             j < 0 || j >= ny ||
//             k < 0 || k >= nz)
//             continue;
//
//         cell_particles[cell_id(i, j, k)].push_back(pid);
//     }
//
//     // ------------------------------------------------------------
//     // 2. 计算 sphere 覆盖到的 cell AABB
//     // ------------------------------------------------------------
//
//     const int ix0 = std::max(0, (int)std::floor((c.vx - r) / h));
//     const int ix1 = std::min(nx - 1, (int)std::floor((c.vx + r) / h));
//     const int iy0 = std::max(0, (int)std::floor((c.vy - r) / h));
//     const int iy1 = std::min(ny - 1, (int)std::floor((c.vy + r) / h));
//     const int iz0 = std::max(0, (int)std::floor((c.vz - r) / h));
//     const int iz1 = std::min(nz - 1, (int)std::floor((c.vz + r) / h));
//
//     // ------------------------------------------------------------
//     // 3. 只遍历 AABB 内的 cell，只检查这些 cell 里的粒子
//     // ------------------------------------------------------------
//
//     for (int k = iz0; k <= iz1; ++k)
//     for (int j = iy0; j <= iy1; ++j)
//     for (int i = ix0; i <= ix1; ++i)
//     {
//         const int cid = cell_id(i, j, k);
//         auto& plist = cell_particles[cid];
//
//         for (int pid : plist)
//         {
//             FLIP_Particles_3d& p = flip_particles[pid];
//             _3d_vec d = p.pos - c;
//             if (d.dot(d) <= r2)
//             {
//                 p.rho  = _dens;
//                 p.temp = _temp;
//             }
//         }
//     }
// }

void add_smoke_particles_in_range(
    const sphere_source& range,
    Real _dens,
    Real _temp
)
{
    FUNC_TIMER();

    const _3d_vec& c = range.center;
    const Real   r = range.radius;
    const Real  r2 = r * r;

    // ------------------------------------------------------------
    // 1. uniform grid：cell -> particle indices
    // ------------------------------------------------------------
    const int nx = 100, ny = 100, nz = 100;
    const Real h = 1.0 / 100.0;
    const int num_cells = nx * ny * nz;
    std::vector<std::vector<int>> cell_particles(num_cells);

    auto cell_id = [&](int i, int j, int k) {
        return (k * ny + j) * nx + i;
    };

    for (int pid = 0; pid < (int)flip_particles.size(); ++pid)
    {
        const _3d_vec& p = flip_particles[pid].pos;

        int i = (int)std::floor(p.vx / h);
        int j = (int)std::floor(p.vy / h);
        int k = (int)std::floor(p.vz / h);

        if (i < 0 || i >= nx ||
            j < 0 || j >= ny ||
            k < 0 || k >= nz)
            continue;

        cell_particles[cell_id(i, j, k)].push_back(pid);
    }

    // ------------------------------------------------------------
    // 2. sphere AABB
    // ------------------------------------------------------------
    const int ix0 = std::max(0,      (int)std::floor((c.vx - r) / h));
    const int ix1 = std::min(nx - 1, (int)std::floor((c.vx + r) / h));
    const int iy0 = std::max(0,      (int)std::floor((c.vy - r) / h));
    const int iy1 = std::min(ny - 1, (int)std::floor((c.vy + r) / h));
    const int iz0 = std::max(0,      (int)std::floor((c.vz - r) / h));
    const int iz1 = std::min(nz - 1, (int)std::floor((c.vz + r) / h));

    // ------------------------------------------------------------
    // 3. 遍历 & 统计命中粒子
    // ------------------------------------------------------------
    int hit_count = 0;

    for (int k = iz0; k <= iz1; ++k)
    for (int j = iy0; j <= iy1; ++j)
    for (int i = ix0; i <= ix1; ++i)
    {
        const int cid = cell_id(i, j, k);
        auto& plist = cell_particles[cid];

        for (int pid : plist)
        {
            FLIP_Particles_3d& p = flip_particles[pid];
            if (range.contains(p.pos)) {
                p.rho  = _dens;
                p.temp = _temp;
                ++hit_count;
            }
        }
    }

    // // ------------------------------------------------------------
    // // 4. 若粒子数不足 16，则随机采样补足
    // // ------------------------------------------------------------
    // constexpr int kMinParticles = 1000;
    // if (hit_count >= kMinParticles)
    //     return;
    // auto sampled = range.sample_points(kMinParticles);
    // for (auto&pos: sampled) {
    //     FLIP_Particles_3d pnew;
    //     pnew.pos  =pos;
    //     pnew.old_pos  = pnew.pos ;
    //     pnew.rho  = _dens;
    //     pnew.temp = _temp;
    //     flip_particles.push_back(pnew);
    // }

}
//
void add_smoke_particles_in_range(
    const cylinder_source& range,
    Real _dens,
    Real _temp
)
{
    FUNC_TIMER();

    // ------------------------------------------------------------
    // 1. uniform grid：cell -> particle indices
    // ------------------------------------------------------------
    const int nx = 100, ny = 100, nz = 100;
    const Real h = 1.0 / 100.0;
    const int num_cells = nx * ny * nz;
    std::vector<std::vector<int>> cell_particles(num_cells);

    auto cell_id = [&](int i, int j, int k) {
        return (k * ny + j) * nx + i;
    };

    for (int pid = 0; pid < (int)flip_particles.size(); ++pid)
    {
        const _3d_vec& p = flip_particles[pid].pos;

        int i = (int)std::floor(p.vx / h);
        int j = (int)std::floor(p.vy / h);
        int k = (int)std::floor(p.vz / h);

        if (i < 0 || i >= nx ||
            j < 0 || j >= ny ||
            k < 0 || k >= nz)
            continue;

        cell_particles[cell_id(i, j, k)].push_back(pid);
    }

    // ------------------------------------------------------------
    // 2. cylinder AABB (tight, orientation-aware)
    // ------------------------------------------------------------

    // ⚠️ 注意：避免和 grid spacing 的 h 冲突
    const _3d_vec& c0 = range.center;
    const _3d_vec& a  = range.axis;
    const Real   hh = range.half_height;
    const Real   rr = range.radius;

    // 每个坐标方向的 half-extent
    const Real ex = std::abs(a.vx) * hh
                    + rr * std::sqrt(std::max(0.0, 1.0 - a.vx * a.vx));
    const Real ey = std::abs(a.vy) * hh
                    + rr * std::sqrt(std::max(0.0, 1.0 - a.vy * a.vy));
    const Real ez = std::abs(a.vz) * hh
                    + rr * std::sqrt(std::max(0.0, 1.0 - a.vz * a.vz));

    // 映射到 grid index（这里的 h 是 grid spacing）
    const int ix0 = std::max(0,      (int)std::floor((c0.vx - ex) / h));
    const int ix1 = std::min(nx - 1, (int)std::floor((c0.vx + ex) / h));
    const int iy0 = std::max(0,      (int)std::floor((c0.vy - ey) / h));
    const int iy1 = std::min(ny - 1, (int)std::floor((c0.vy + ey) / h));
    const int iz0 = std::max(0,      (int)std::floor((c0.vz - ez) / h));
    const int iz1 = std::min(nz - 1, (int)std::floor((c0.vz + ez) / h));


    // ------------------------------------------------------------
    // 3. 遍历 & 统计命中粒子
    // ------------------------------------------------------------
    int hit_count = 0;

    for (int k = iz0; k <= iz1; ++k)
    for (int j = iy0; j <= iy1; ++j)
    for (int i = ix0; i <= ix1; ++i)
    {
        const int cid = cell_id(i, j, k);
        auto& plist = cell_particles[cid];

        for (int pid : plist)
        {
            FLIP_Particles_3d& p = flip_particles[pid];
            if (p.rho>1)continue;
            if (range.contains(p.pos)) {
                p.rho  = _dens;
                p.temp = _temp;
                p.vec = range.init_vec;
                ++hit_count;
            }
        }
    }


}


void add_smoke_particles_in_range_skull(
    const cylinder_source& range,
    Real _dens,
    Real _temp
)
{
    FUNC_TIMER();

    // ------------------------------------------------------------
    // 1. uniform grid：cell -> particle indices
    // ------------------------------------------------------------
    const int nx = 100, ny = 100, nz = 100;
    const Real h = 1.0 / 100.0;
    const int num_cells = nx * ny * nz;
    std::vector<std::vector<int>> cell_particles(num_cells);

    auto cell_id = [&](int i, int j, int k) {
        return (k * ny + j) * nx + i;
    };

    for (int pid = 0; pid < (int)flip_particles.size(); ++pid)
    {
        const _3d_vec& p = flip_particles[pid].pos;

        int i = (int)std::floor(p.vx / h);
        int j = (int)std::floor(p.vy / h);
        int k = (int)std::floor(p.vz / h);

        if (i < 0 || i >= nx ||
            j < 0 || j >= ny ||
            k < 0 || k >= nz)
            continue;

        cell_particles[cell_id(i, j, k)].push_back(pid);
    }

    // ------------------------------------------------------------
    // 2. cylinder AABB (tight, orientation-aware)
    // ------------------------------------------------------------

    // ⚠️ 注意：避免和 grid spacing 的 h 冲突
    const _3d_vec& c0 = range.center;
    const _3d_vec& a  = range.axis;
    const Real   hh = range.half_height;
    const Real   rr = range.radius;

    // 每个坐标方向的 half-extent
    const Real ex = std::abs(a.vx) * hh
                    + rr * std::sqrt(std::max(0.0, 1.0 - a.vx * a.vx));
    const Real ey = std::abs(a.vy) * hh
                    + rr * std::sqrt(std::max(0.0, 1.0 - a.vy * a.vy));
    const Real ez = std::abs(a.vz) * hh
                    + rr * std::sqrt(std::max(0.0, 1.0 - a.vz * a.vz));

    // 映射到 grid index（这里的 h 是 grid spacing）
    const int ix0 = std::max(0,      (int)std::floor((c0.vx - ex) / h));
    const int ix1 = std::min(nx - 1, (int)std::floor((c0.vx + ex) / h));
    const int iy0 = std::max(0,      (int)std::floor((c0.vy - ey) / h));
    const int iy1 = std::min(ny - 1, (int)std::floor((c0.vy + ey) / h));
    const int iz0 = std::max(0,      (int)std::floor((c0.vz - ez) / h));
    const int iz1 = std::min(nz - 1, (int)std::floor((c0.vz + ez) / h));


    // ------------------------------------------------------------
    // 3. 遍历 & 统计命中粒子
    // ------------------------------------------------------------
    int hit_count = 0;

    for (int k = iz0; k <= iz1; ++k)
    for (int j = iy0; j <= iy1; ++j)
    for (int i = ix0; i <= ix1; ++i)
    {
        const int cid = cell_id(i, j, k);
        auto& plist = cell_particles[cid];

        for (int pid : plist)
        {
            FLIP_Particles_3d& p = flip_particles[pid];
            if (p.rho > 0)continue;
            if (range.contains(p.pos)) {
                p.rho  = _dens;
                p.temp = _temp;
                p.vec =  range.axis * 10;
            }
        }
    }

    // // ------------------------------------------------------------
    // // 4. 若粒子数不足 16，则随机采样补足
    // // ------------------------------------------------------------
    constexpr int kMinParticles = 10000;
    auto sampled = range.sample_points(kMinParticles);
    for (auto& pos : sampled)
    {
        FLIP_Particles_3d pnew;
        pnew.pos     = pos;
        pnew.old_pos = pos;
        pnew.rho     = _dens;
        pnew.temp    = _temp;

        // ------------------------------------------------------------
        // 根据轴向位置决定速度方向
        // ------------------------------------------------------------
        _3d_vec d = pos - range.center;
        // pnew.vec =  range.axis * 10;


        pnew.vec =  range.axis * 10;

        flip_particles.push_back(pnew);
        // RZ_INFO("here");
    }


}

void add_smoke_particles_in_range(
    const std::vector<cylinder_source>& ranges,
    Real _dens,
    Real _temp
)
{
    FUNC_TIMER();

    // ------------------------------------------------------------
    // 1. uniform grid：cell -> particle indices
    // ------------------------------------------------------------
    const int nx = 100, ny = 100, nz = 100;
    const Real h = 1.0 / 100.0;
    const int num_cells = nx * ny * nz;
    std::vector<std::vector<int>> cell_particles(num_cells);

    auto cell_id = [&](int i, int j, int k) {
        return (k * ny + j) * nx + i;
    };

    for (int pid = 0; pid < (int)flip_particles.size(); ++pid)
    {
        const _3d_vec& p = flip_particles[pid].pos;

        int i = (int)std::floor(p.vx / h);
        int j = (int)std::floor(p.vy / h);
        int k = (int)std::floor(p.vz / h);

        if (i < 0 || i >= nx ||
            j < 0 || j >= ny ||
            k < 0 || k >= nz)
            continue;

        cell_particles[cell_id(i, j, k)].push_back(pid);
    }

    // ------------------------------------------------------------
    // 2. cylinder AABB (tight, orientation-aware)
    // ------------------------------------------------------------
    for (auto&range: ranges) {
        // ⚠️ 注意：避免和 grid spacing 的 h 冲突
        const _3d_vec& c0 = range.center;
        const _3d_vec& a  = range.axis;
        const Real   hh = range.half_height;
        const Real   rr = range.radius;

        // 每个坐标方向的 half-extent
        const Real ex = std::abs(a.vx) * hh
                        + rr * std::sqrt(std::max(0.0, 1.0 - a.vx * a.vx));
        const Real ey = std::abs(a.vy) * hh
                        + rr * std::sqrt(std::max(0.0, 1.0 - a.vy * a.vy));
        const Real ez = std::abs(a.vz) * hh
                        + rr * std::sqrt(std::max(0.0, 1.0 - a.vz * a.vz));

        // 映射到 grid index（这里的 h 是 grid spacing）
        const int ix0 = std::max(0,      (int)std::floor((c0.vx - ex) / h));
        const int ix1 = std::min(nx - 1, (int)std::floor((c0.vx + ex) / h));
        const int iy0 = std::max(0,      (int)std::floor((c0.vy - ey) / h));
        const int iy1 = std::min(ny - 1, (int)std::floor((c0.vy + ey) / h));
        const int iz0 = std::max(0,      (int)std::floor((c0.vz - ez) / h));
        const int iz1 = std::min(nz - 1, (int)std::floor((c0.vz + ez) / h));


        // ------------------------------------------------------------
        // 3. 遍历 & 统计命中粒子
        // ------------------------------------------------------------
        int hit_count = 0;

        for (int k = iz0; k <= iz1; ++k)
            for (int j = iy0; j <= iy1; ++j)
                for (int i = ix0; i <= ix1; ++i)
                {
                    const int cid = cell_id(i, j, k);
                    auto& plist = cell_particles[cid];

                    for (int pid : plist)
                    {

                        FLIP_Particles_3d& p = flip_particles[pid];
                        if (p.rho>1.0)continue;
                        if (range.contains(p.pos)) {
                            p.rho  = _dens;
                            p.temp = _temp;
                            p.vec = range.init_vec;
                            ++hit_count;
                        }
                    }
                }
    }
}



void record_concentration_cells_in_range(
    const sphere_source& range,
    ConcentrationField3D& cfield
)
{
    FUNC_TIMER();

    cfield.inside_source.clear();

    const _3d_vec& c = range.center;
    const Real   r = range.radius;
    const Real  r2 = r * r;

    const int nx = cfield.nx;
    const int ny = cfield.ny;
    const int nz = cfield.nz;
    const Real dx = cfield.dx;

    auto cell_id = [&](int i, int j, int k) {
        return (k * ny + j) * nx + i;
    };

    // ------------------------------------------------------------
    // 1. sphere 对应的 cell AABB（和你粒子版完全一致）
    // ------------------------------------------------------------
    const int ix0 = std::max(0, (int)std::floor((c.vx - r) / dx));
    const int ix1 = std::min(nx - 1, (int)std::floor((c.vx + r) / dx));
    const int iy0 = std::max(0, (int)std::floor((c.vy - r) / dx));
    const int iy1 = std::min(ny - 1, (int)std::floor((c.vy + r) / dx));
    const int iz0 = std::max(0, (int)std::floor((c.vz - r) / dx));
    const int iz1 = std::min(nz - 1, (int)std::floor((c.vz + r) / dx));

    // ------------------------------------------------------------
    // 2. 只遍历 AABB 内的 cell center
    // ------------------------------------------------------------
    for (int k = iz0; k <= iz1; ++k)
        for (int j = iy0; j <= iy1; ++j)
            for (int i = ix0; i <= ix1; ++i)
            {
                _3d_vec x(
                    (i + 0.5) * dx,
                    (j + 0.5) * dx,
                    (k + 0.5) * dx
                );

                _3d_vec d = x - c;
                if (d.dot(d) <= r2)
                {
                    cfield.inside_source.push_back(cell_id(i, j, k));
                }
            }
}




void init_flip(MAC_Grid3D &mac)
{
    flip_particles.clear();

    // 每个cell内均匀采样 3x3x3
    const int samples_per_axis = 2;
    const Real spacing = mac.dx / samples_per_axis;

    for (int i = 0; i < mac.nx; ++i) {
        for (int j = 0; j < mac.ny; ++j) {
            for (int k = 0; k < mac.nz; ++k) {
                // cell 的物理坐标原点（左下前角）
                Real cell_origin_x = i * mac.dx;
                Real cell_origin_y = j * mac.dx;
                Real cell_origin_z = k * mac.dx;

                for (int si = 0; si < samples_per_axis; ++si) {
                    for (int sj = 0; sj < samples_per_axis; ++sj) {
                        for (int sk = 0; sk < samples_per_axis; ++sk) {
                            FLIP_Particles_3d p;

                            // 小立方体内采样点的中心
                            Real px = cell_origin_x + (si + 0.5f) * spacing;
                            Real py = cell_origin_y + (sj + 0.5f) * spacing;
                            Real pz = cell_origin_z + (sk + 0.5f) * spacing;

                            p.pos = _3d_vec(px, py, pz);   // 世界坐标
                            p.vec = _3d_vec(0.0, 0.0, 0.0); // 初速度设为0
                            // {
                            //     p.vec.vy = 30.f;
                            //     p.vec.vx = -2.5f;
                            //     p.vec.vz = -1.75f;
                            // }
                            // // // // 示例：下半区域加入向下流动速度
                            // if (i > 0 && i < 50 && j > 0 && j < 50 && k > 0 && k < 50) {
                            //     p.vec = _3d_vec(0.0, 3.0, 0.0);
                            // }

                            flip_particles.push_back(p);
                        }
                    }
                }
            }
        }
    }
}
void remove_out_of_bounds_particles_3d(std::vector<FLIP_Particles_3d>& particles, Real dx=0.02)
{
    const Real margin = 0.01f * dx * 0.99f;
    const Real xmin = margin, ymin = margin, zmin = margin;
    const Real xmax = 1.0 - margin, ymax = 1.0 - margin, zmax = 1.0 - margin;
    std::for_each(particles.begin(), particles.end(), [&](FLIP_Particles_3d& p) {
        if (p.pos.vx < xmin || p.pos.vx > xmax ||
            p.pos.vy < ymin || p.pos.vy > ymax ||
            p.pos.vz < zmin || p.pos.vz > zmax)
        {
            p.deleted = true;
        }
    });


    // particles.erase(
    //     std::remove_if(particles.begin(), particles.end(),
    //         [&](const FLIP_Particles_3d& p) {
    //             return (p.pos.vx < xmin || p.pos.vx > xmax ||
    //                     p.pos.vy < ymin || p.pos.vy > ymax ||
    //                     p.pos.vz < zmin || p.pos.vz > zmax);
    //         }),
    //     particles.end());
}
void remove_out_of_bounds_density_particles_3d(std::vector<Smoke_Particles_3d>& particles, Real dx = 0.02) {
    const Real margin = 0.01f * dx * 0.99f;
    const Real xmin = margin, ymin = margin, zmin = margin;
    const Real xmax = 1.0 - margin, ymax = 1.0 - margin, zmax = 1.0 - margin;
    particles.erase(
        std::remove_if(particles.begin(), particles.end(),
            [&](const Smoke_Particles_3d& p) {
                return (p.pos.vx < xmin || p.pos.vx > xmax ||
                        p.pos.vy < ymin || p.pos.vy > ymax ||
                        p.pos.vz < zmin || p.pos.vz > zmax);
            }),
        particles.end());
}
void convert_to_render_particles(const std::vector<FLIP_Particles_3d>& particles)
{
    FUNC_TIMER();
    render_particles.clear();
    for (const auto& p : particles) {
        if (p.deleted) continue;  // 跳过被标记删除的粒子
        RenderParticle rp;
        rp.pos = glm::vec3(p.pos.vx, p.pos.vy, p.pos.vz);
        rp.vel = glm::vec3(p.vec.vx, p.vec.vy, p.vec.vz);
        rp.speed = std::sqrt(p.vec.vx * p.vec.vx +
                             p.vec.vy * p.vec.vy +
                             p.vec.vz * p.vec.vz);
        render_particles.push_back(rp);
    }
}

// 每个 grid cell 存放包含该格子内粒子的索引列表
static std::vector<std::vector<size_t>> grid_particle_indices_3d;

void build_grid_particle_map_3d(const MAC_Grid3D &mac)
{
    FUNC_TIMER();
    size_t nx = mac._density.nx();
    size_t ny = mac._density.ny();
    size_t nz = mac._density.nz();

    // 每个格子对应一个 vector<size_t>
    grid_particle_indices_3d.clear();
    grid_particle_indices_3d.resize(nx * ny * nz);

    for (size_t p_idx = 0; p_idx < flip_particles.size(); ++p_idx) {
        const auto &p = flip_particles[p_idx];
        if (p.deleted) continue;

        // 计算粒子所在 cell 索引
        int i = static_cast<int>(p.pos.vx / mac.dx);
        int j = static_cast<int>(p.pos.vy / mac.dx);
        int k = static_cast<int>(p.pos.vz / mac.dx);

        // 边界检查
        if (i >= 0 && i < (int)nx &&
            j >= 0 && j < (int)ny &&
            k >= 0 && k < (int)nz)
        {
            size_t cell_index = i + nx * (j + ny * k);
            grid_particle_indices_3d[cell_index].push_back(p_idx);
        }
    }
}
Real Weight_function(const _3d_vec& xp,const _3d_vec& xc, Real dx)
{
    Real ans = 1.0;
    ans *= std::fmax(1.0 - std::abs((xp.vx - xc.vx) / dx), 0.0);
    ans *= std::fmax(1.0 - std::abs((xp.vy - xc.vy) / dx), 0.0);
    ans *= std::fmax(1.0 - std::abs((xp.vz - xc.vz) / dx), 0.0);
    return ans;
}

//parallel version
void ptogrid(MAC_Grid3D &mac)
{
    FUNC_TIMER();

    const size_t nx = mac._density.nx();
    const size_t ny = mac._density.ny();
    const size_t nz = mac._density.nz();
    const Real  dx = mac.dx;

    build_grid_particle_map_3d(mac);

    // -------------------------------
    // U component (x-face centers)
    // -------------------------------
    auto & _u = mac._u;
    _u.fill(0.0);

    #pragma omp parallel for collapse(1) schedule(dynamic)
    for (size_t i = 1; i < nx; ++i) {
        for (size_t j = 0; j < ny; ++j)
            for (size_t k = 0; k < nz; ++k)
            {
                const Real px = i * dx;
                const Real py = j * dx + 0.5f * dx;
                const Real pz = k * dx + 0.5f * dx;
                const _3d_vec world_coord(px, py, pz);

                Real sum_w = 0.0;
                Real sum_u = 0.0;

                for (int di = -1; di <= 0; ++di)
                    for (int dj = -1; dj <= 1; ++dj)
                        for (int dk = -1; dk <= 1; ++dk)
                        {
                            int ci = static_cast<int>(i) + di;
                            int cj = static_cast<int>(j) + dj;
                            int ck = static_cast<int>(k) + dk;

                            if (ci < 0 || cj < 0 || ck < 0 ||
                                ci >= (int)nx || cj >= (int)ny || ck >= (int)nz)
                                continue;

                            size_t cell_index = ci + nx * (cj + ny * ck);
                            const auto &particles = grid_particle_indices_3d[cell_index];

                            for (auto pid : particles)
                            {
                                const auto &p = flip_particles[pid];
                                const Real w = Weight_function(p.pos, world_coord, dx);
                                sum_w += w;
                                sum_u += w * p.vec.vx;
                            }
                        }
                _u(i, j, k) = (sum_w > 1e-5f) ? sum_u / sum_w : 0.0;
            }
    }

    mac._delta_u = mac._u;

    // -------------------------------
    // V component (y-face centers)
    // -------------------------------
    auto & _v = mac._v;
    _v.fill(0.0);

    #pragma omp parallel for collapse(1) schedule(dynamic)
    for (size_t i = 0; i < nx; ++i) {
        for (size_t j = 1; j < ny; ++j)
            for (size_t k = 0; k < nz; ++k)
            {
                const Real px = i * dx + 0.5f * dx;
                const Real py = j * dx;
                const Real pz = k * dx + 0.5f * dx;
                const _3d_vec world_coord(px, py, pz);

                Real sum_w = 0.0;
                Real sum_v = 0.0;

                for (int di = -1; di <= 1; ++di)
                    for (int dj = -1; dj <= 0; ++dj)
                        for (int dk = -1; dk <= 1; ++dk)
                        {
                            int ci = static_cast<int>(i) + di;
                            int cj = static_cast<int>(j) + dj;
                            int ck = static_cast<int>(k) + dk;

                            if (ci < 0 || cj < 0 || ck < 0 ||
                                ci >= (int)nx || cj >= (int)ny || ck >= (int)nz)
                                continue;

                            size_t cell_index = ci + nx * (cj + ny * ck);
                            const auto &particles = grid_particle_indices_3d[cell_index];

                            for (auto pid : particles)
                            {
                                const auto &p = flip_particles[pid];
                                const Real w = Weight_function(p.pos, world_coord, dx);
                                sum_w += w;
                                sum_v += w * p.vec.vy;
                            }
                        }
                _v(i, j, k) = (sum_w > 1e-5f) ? sum_v / sum_w : 0.0;
            }
    }

    mac._delta_v = mac._v;

    // -------------------------------
    // W component (z-face centers)
    // -------------------------------
    auto & _w = mac._w;
    _w.fill(0.0);

    #pragma omp parallel for collapse(1) schedule(dynamic)
    for (size_t i = 0; i < nx; ++i) {
        for (size_t j = 0; j < ny; ++j)
            for (size_t k = 1; k < nz; ++k)
            {
                const Real px = i * dx + 0.5f * dx;
                const Real py = j * dx + 0.5f * dx;
                const Real pz = k * dx;
                const _3d_vec world_coord(px, py, pz);

                Real sum_w = 0.0;
                Real sum_wv = 0.0;

                for (int di = -1; di <= 1; ++di)
                    for (int dj = -1; dj <= 1; ++dj)
                        for (int dk = -1; dk <= 0; ++dk)
                        {
                            int ci = static_cast<int>(i) + di;
                            int cj = static_cast<int>(j) + dj;
                            int ck = static_cast<int>(k) + dk;

                            if (ci < 0 || cj < 0 || ck < 0 ||
                                ci >= (int)nx || cj >= (int)ny || ck >= (int)nz)
                                continue;

                            size_t cell_index = ci + nx * (cj + ny * ck);
                            const auto &particles = grid_particle_indices_3d[cell_index];

                            for (auto pid : particles)
                            {
                                const auto &p = flip_particles[pid];
                                const Real w = Weight_function(p.pos, world_coord, dx);
                                sum_w += w;
                                sum_wv += w * p.vec.vz;
                            }
                        }
                _w(i, j, k) = (sum_w > 1e-5f) ? sum_wv / sum_w : 0.0;
            }
    }

    mac._delta_w = mac._w;
}
#include "Visualization.h"

static int save_id = 0;
void grid_pressure_projection(MAC_Grid3D &mac)
{
    FUNC_TIMER();
    const int nx = mac.nx;
    const int ny = mac.ny;
    const int nz = mac.nz;

    const int matrix_dim = nx * ny * nz;

    Eigen::SparseMatrixR A(matrix_dim, matrix_dim);
    Eigen::VectorXr b(matrix_dim);
    std::vector<Eigen::Triplet<GlobalReal>> A_triplets;

    Real scale_dx = 1.0 / mac.dx;
    Real scale_A  = dt / (mac.dx * mac.dx); // suppose density = 1

    const auto& _u = mac._u;
    const auto& _v = mac._v;
    const auto& _w = mac._w;



    // ==============================
    // Step 1: Build Laplacian Matrix A
    // ==============================
    if (!inited_mac) {
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
                for (int k = 0; k < nz; ++k)
                {
                    int row = i + nx * (j + ny * k);
                    Real xl, xr, yl, yr, zl, zr, c;

                    xl = (i > 0)      ? -scale_A : 0.0;
                    xr = (i < nx - 1) ? -scale_A : 0.0;
                    yl = (j > 0)      ? -scale_A : 0.0;
                    yr = (j < ny - 1) ? -scale_A : 0.0;
                    zl = (k > 0)      ? -scale_A : 0.0;
                    zr = (k < nz - 1) ? -scale_A : 0.0;

                    c = -(xl + xr + yl + yr + zl + zr);

                    // Center
                    A_triplets.emplace_back(row, row, c);

                    // X-neighbors
                    if (i > 0)
                        A_triplets.emplace_back(row, row - 1, xl);
                    if (i < nx - 1)
                        A_triplets.emplace_back(row, row + 1, xr);

                    // Y-neighbors
                    if (j > 0)
                        A_triplets.emplace_back(row, row - nx, yl);
                    if (j < ny - 1)
                        A_triplets.emplace_back(row, row + nx, yr);

                    // Z-neighbors
                    if (k > 0)
                        A_triplets.emplace_back(row, row - nx * ny, zl);
                    if (k < nz - 1)
                        A_triplets.emplace_back(row, row + nx * ny, zr);
                }

        A.setFromTriplets(A_triplets.begin(), A_triplets.end());
        A.makeCompressed();
        inited_mac = true;
        amgclInit(solver_context_mac,A,1e-3,10000);
        // amgclInitCPU(solver_context_mac_cpu,A,1e-3,10000);
        // static_A = A;
        // mg_solver.build_hierarchy(A,mac.nx,mac.ny,mac.nz,4);
        // mg_solver_cuda.build_hierarchy(A,mac.nx,mac.ny,mac.nz,5);
    }

    // ==============================
    // Step 2: Build right-hand side b = -div(u)
    // ==============================
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
            {
                Real div =
                    (_u(i + 1, j, k) - _u(i, j, k)) +
                    (_v(i, j + 1, k) - _v(i, j, k)) +
                    (_w(i, j, k + 1) - _w(i, j, k));

                b(i + nx * (j + ny * k)) = -scale_dx * div;
            }

    // ==============================
    // Step 3: Solid boundary correction
    // ==============================
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
            {
                int id = i + nx * (j + ny * k);

                if (i == 0)       b(id) -= scale_dx * _u(i, j, k);
                if (i == nx - 1)  b(id) -= scale_dx * _u(i + 1, j, k);
                if (j == 0)       b(id) -= scale_dx * _v(i, j, k);
                if (j == ny - 1)  b(id) -= scale_dx * _v(i, j + 1, k);
                if (k == 0)       b(id) -= scale_dx * _w(i, j, k);
                if (k == nz - 1)  b(id) -= scale_dx * _w(i, j, k + 1);
            }
    // ==============================
    // Step 4: Solve Ap = b
    // ==============================
    // Eigen::VectorXr pressure;
    // mg_solver_cuda.set_rhs(b);
    // mg_solver_cuda.VC(pressure);
    // mg_solver.set_rhs(b);
    // mg_solver.FMG(pressure);
    // mg_solver.VC(pressure);
    // const auto& pressure = warm_pressure;

    Eigen::VectorXr pressure = amgclSolve(solver_context_mac, b);
    // Eigen::VectorXr pressure = amgclSolveCPU(A, b, 1e-3, -1, 10000);
    // Eigen::VectorXr pressure = amgclSolveCPU(solver_context_mac_cpu, b);

    // gaussSeidel(static_A,b,warm_pressure);
    // const auto& pressure = warm_pressure;
    // save_pressure_vti(mac.nx,mac.ny,mac.nz,mac.dx,pressure,"flip_pressure"+std::to_string(save_id++)+".vti");
    // exit(0);
    // ==============================
    // Step 5: Apply pressure gradient
    // ==============================

    // --- U component ---
    for (int i = 0; i <= nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
            {
                if (i == 0 || i == nx)
                    mac._u(i, j, k) = 0.0;
                else {
                    int id_r = i + nx * (j + ny * k);
                    int id_l = (i - 1) + nx * (j + ny * k);
                    mac._u(i, j, k) -= Real((pressure(id_r) - pressure(id_l)) * dt / mac.dx);
                }
            }

    // --- V component ---
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j <= ny; ++j)
            for (int k = 0; k < nz; ++k)
            {
                if (j == 0 || j == ny)
                    mac._v(i, j, k) = 0.0;
                else {
                    int id_u = i + nx * (j + ny * k);
                    int id_d = i + nx * ((j - 1) + ny * k);
                    mac._v(i, j, k) -= Real((pressure(id_u) - pressure(id_d)) * dt / mac.dx);
                }
            }

    // --- W component ---
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k <= nz; ++k)
            {
                if (k == 0 || k == nz)
                    mac._w(i, j, k) = 0.0;
                else {
                    int id_f = i + nx * (j + ny * k);
                    int id_b = i + nx * (j + ny * (k - 1));
                    mac._w(i, j, k) -= Real((pressure(id_f) - pressure(id_b)) * dt / mac.dx);
                }
            }
}

void gridtop(MAC_Grid3D &mac, Real flip_alpha)
{
    FUNC_TIMER();

    mac._delta_u = mac._u - mac._delta_u;
    mac._delta_v = mac._v - mac._delta_v;
    mac._delta_w = mac._w - mac._delta_w;

    const size_t n = flip_particles.size();

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i)
    {
        auto &p = flip_particles[i];
        if (p.deleted) continue;

        _3d_vec pic_vel   = mac.interpolate_velocity(p.pos);
        _3d_vec flip_delta = mac.interpolate_delta_velocity(p.pos);// filed上速度的增量

        // 混合更新
        p.vec = pic_vel * flip_alpha +
                (p.vec + flip_delta) * (1.0 - flip_alpha);
    }
}

void flip_advection(MAC_Grid3D &mac)
{
    FUNC_TIMER();
    for (auto &p : flip_particles) {
        if (p.deleted)continue;;
        p.pos.vx += dt * p.vec.vx;
        p.pos.vy += dt * p.vec.vy;
        p.pos.vz += dt * p.vec.vz;
    }
}

void simple_CFL(MAC_Grid3D &mac)
{
    dt = 0.01f;
    // === 粒子速度 CFL 限制 ===
    for (auto &p : flip_particles)
    {
        dt = std::min<Real>(3.0 * mac.dx / (std::fabs(p.vec.vx) + 1e-5f), dt);
        dt = std::min<Real>(3.0 * mac.dx / (std::fabs(p.vec.vy) + 1e-5f), dt);
        dt = std::min<Real>(3.0 * mac.dx / (std::fabs(p.vec.vz) + 1e-5f), dt);
    }
}
void simple_CFL(VEM_UGRID &mac)
{
    dt = 0.01f;
    // === 粒子速度 CFL 限制 ===
    for (auto &p : flip_particles)
    {
        dt = std::min<Real>(3.0 * mac.dx / (std::fabs(p.vec.vx) + 1e-5f), dt);
        dt = std::min<Real>(3.0 * mac.dx / (std::fabs(p.vec.vy) + 1e-5f), dt);
        dt = std::min<Real>(3.0 * mac.dx / (std::fabs(p.vec.vz) + 1e-5f), dt);
    }
}
void simple_CFL(Real h) {
    dt = 0.01f;
    // === 粒子速度 CFL 限制 ===
    for (auto &p : flip_particles)
    {
        dt = std::min(h / (std::fabs(p.pic_vec.vx) + 1e-5f), dt);
        dt = std::min(h / (std::fabs(p.pic_vec.vy) + 1e-5f), dt);
        dt = std::min(h / (std::fabs(p.pic_vec.vz) + 1e-5f), dt);
    }
    dt = std::max<Real>(dt,2e-4);
}



// ====================================================
// 主函数：以粒子为主循环计算烟雾密度
// ====================================================
void compute_smoke_density_from_particles(MAC_Grid3D &mac, Real kernel_radius)// todo: use grids for density
{
    FUNC_TIMER();

    const Real h  = kernel_radius;
    const Real dx = mac.dx;
    const Real inv_dx = 1.0 / dx;
    const int nx = mac.nx;
    const int ny = mac.ny;
    const int nz = mac.nz;
    const int range = static_cast<int>(std::ceil(h * inv_dx));

    int deleted = 0;
    for (size_t pid = 0; pid < smoke_particles.size(); ++pid) {
        const auto &p = smoke_particles[pid];
        if (p.deleted) ++deleted;
    }
    RZ_INFO("deleted/all: ", (Real)deleted/(Real)smoke_particles.size());


    mac._density.fill(0.0);

    const Real h2 = h * h;
    const Real coeff = 0.0000003f; // 原缩放常数

    // 用于减少原子操作次数的线程局部缓存
    const int num_threads = omp_get_max_threads();
    std::vector<Array3D<Real>> local_density(num_threads,
        Array3D<Real>(nx, ny, nz, 0.0));

#pragma omp parallel for schedule(dynamic)
    for (size_t pid = 0; pid < smoke_particles.size(); ++pid)
    {
        const auto &p = smoke_particles[pid];
        if (p.deleted) continue;

        const int tid = omp_get_thread_num();
        auto &local = local_density[tid];

        // 粒子所在格点索引
        const int i = static_cast<int>(p.pos.vx * inv_dx);
        const int j = static_cast<int>(p.pos.vy * inv_dx);
        const int k = static_cast<int>(p.pos.vz * inv_dx);

        // 预计算：粒子位置
        const Real px = p.pos.vx;
        const Real py = p.pos.vy;
        const Real pz = p.pos.vz;

        for (int di = -range; di <= range; ++di)
        {
            const int gi = i + di;
            if (gi < 0 || gi >= nx) continue;
            const Real gx = (gi + 0.5f) * dx - px;

            for (int dj = -range; dj <= range; ++dj)
            {
                const int gj = j + dj;
                if (gj < 0 || gj >= ny) continue;
                const Real gy = (gj + 0.5f) * dx - py;

                for (int dk = -range; dk <= range; ++dk)
                {
                    const int gk = k + dk;
                    if (gk < 0 || gk >= nz) continue;
                    const Real gz = (gk + 0.5f) * dx - pz;

                    const Real r2 = gx * gx + gy * gy + gz * gz;
                    if (r2 < h2) {
                        const Real r = std::sqrt(r2);
                        const Real w = cubic_kernel(r, h);
                        local(gi, gj, gk) += w * coeff;
                    }
                }
            }
        }
    }

    // ===============================
    // 2️⃣ 合并线程局部密度结果
    // ===============================
#pragma omp parallel for collapse(3)
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
            {
                Real sum = 0.0;
                for (int t = 0; t < num_threads; ++t)
                    sum += local_density[t](i, j, k);
                mac._density(i, j, k) += sum;
            }

    mac._density.clamp01();

    // for (size_t pid = 0; pid < smoke_particles.size(); ++pid)
    // {
    //     const auto &p = smoke_particles[pid];
    //     if (p.deleted) continue;
    //
    //     // 粒子所在格点索引
    //     int i = static_cast<int>(p.pos.vx / dx);
    //     int j = static_cast<int>(p.pos.vy / dx);
    //     int k = static_cast<int>(p.pos.vz / dx);
    //
    //     // 在邻域内更新格点密度
    //     for (int di = -range; di <= range; ++di)
    //         for (int dj = -range; dj <= range; ++dj)
    //             for (int dk = -range; dk <= range; ++dk)
    //             {
    //                 int gi = i + di;
    //                 int gj = j + dj;
    //                 int gk = k + dk;
    //                 if (gi < 0 || gj < 0 || gk < 0 ||
    //                     gi >= nx || gj >= ny || gk >= nz)
    //                     continue;
    //
    //                 _3d_vec grid_center((gi + 0.5f) * dx,
    //                                     (gj + 0.5f) * dx,
    //                                     (gk + 0.5f) * dx);
    //
    //                 Real r = (grid_center - p.pos).norm();
    //                 if (r < h)
    //                     mac._density(gi, gj, gk) += cubic_kernel(r, h)*0.0000003;
    //             }
    // }
    // mac._density.clamp01();
}

void compute_smoke_density_from_particles(MAC_Grid3D &mac)
{
    FUNC_TIMER();

    // kernel_radius 作为 IVOCK kernel 的尺度
    const Real h = mac.dx;
    IVOCK_Kernel K(h);

    const Real dx     = mac.dx;
    const Real inv_dx = 1.0 / dx;

    const int nx = mac.nx;
    const int ny = mac.ny;
    const int nz = mac.nz;

    const int num_cells = nx * ny * nz;

    auto cell_id = [&](int i, int j, int k) {
        return (k * ny + j) * nx + i;
    };

    // --------------------------------------------------
    // 统计 deleted（保持你原逻辑）
    // --------------------------------------------------
    int deleted = 0;
    for (size_t pid = 0; pid < flip_particles.size(); ++pid)
        if (flip_particles[pid].deleted) ++deleted;

    RZ_INFO("deleted/all: ",
            (Real)deleted / (Real)flip_particles.size());

    // --------------------------------------------------
    // 1) particle -> cell hash
    // --------------------------------------------------
    std::vector<std::vector<int>> cell_particles(num_cells);

    for (int pid = 0; pid < (int)flip_particles.size(); ++pid)
    {
        const auto &p = flip_particles[pid];
        if (p.deleted) continue;

        int i = (int)std::floor(p.pos.vx * inv_dx);
        int j = (int)std::floor(p.pos.vy * inv_dx);
        int k = (int)std::floor(p.pos.vz * inv_dx);

        if (i < 0 || i >= nx ||
            j < 0 || j >= ny ||
            k < 0 || k >= nz)
            continue;

        cell_particles[cell_id(i, j, k)].push_back(pid);
    }

    // --------------------------------------------------
    // 2) P2G: cell-centered density reconstruction
    // --------------------------------------------------
    mac._density.fill(0.0);

#pragma omp parallel for collapse(3) schedule(static)
    for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i)
    {
        const Real gx = (i + 0.5) * dx;
        const Real gy = (j + 0.5) * dx;
        const Real gz = (k + 0.5) * dx;

        Real total_w   = 1e-12; // 防止除零
        Real total_val = 0.0;

        for (int kk = k - 1; kk <= k + 1; ++kk)
        for (int jj = j - 1; jj <= j + 1; ++jj)
        for (int ii = i - 1; ii <= i + 1; ++ii)
        {
            if (ii < 0 || ii >= nx ||
                jj < 0 || jj >= ny ||
                kk < 0 || kk >= nz)
                continue;

            const auto &plist = cell_particles[cell_id(ii, jj, kk)];

            for (int pid : plist)
            {
                const auto &p = flip_particles[pid];

                const Real px = p.pos.vx;
                const Real py = p.pos.vy;
                const Real pz = p.pos.vz;

                if (!K.in_support(gx, gy, gz, px, py, pz))
                    continue;

                const Real w = K.weight(gx, gy, gz, px, py, pz);
                if (w <= 0.0) continue;

                // 如果你希望用粒子的 rho
                total_val += w * p.rho;
                total_w += w;
            }
        }

        mac._density(i, j, k) = total_val/total_w;
    }

    mac._density.clamp01();
}


void add_external_force(MAC_Grid3D &mac, Real dt)
{
    size_t nx = mac._density.nx();
    size_t ny = mac._density.ny();
    size_t nz = mac._density.nz();

    // 在3D中，重力通常作用在 v 分量上（即 Y 方向）
    for (size_t i = 0; i < nx; ++i)
        for (size_t j = 1; j < ny; ++j)
            for (size_t k = 0; k < nz; ++k)
            {
                // 平均两层密度作为插值后的密度
                Real avg_density = (mac._density(i, j - 1, k) + mac._density(i, j, k)) * 0.5f;
                mac._v(i, j, k) += avg_density * 9.8f * dt;
            }
}
void smoke_advection(MAC_Grid3D &mac)
{
    FUNC_TIMER();

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < smoke_particles.size(); ++i)
    {
        auto &p = smoke_particles[i];
        if (p.deleted) continue;

        // Step 1: 初始点速度
        _3d_vec v1 = mac.interpolate_velocity(p.pos);

        // Step 2: 预测中点位置
        _3d_vec mid_pos = p.pos + v1 * (0.5f * dt);

        // Step 3: 在中点插值得到速度
        _3d_vec v2 = mac.interpolate_velocity(mid_pos);

        // Step 4: RK2 积分更新位置
        p.pos =p.pos + v2 * dt;
    }
}


void render_loop_3d(RGB_Screen &screen) {
    MAC_Grid3D mg(100,100,100);
    MAC_Grid3D density_mg(100,100,100);
    init_flip(mg);
    // Render render(screen.width,screen.height);
    Source_range source(20,30,0,10,20,30);
    VolumeRenderer render;
    render.initWindow();
    int step = 0;
    while(!render.should_close()){
        simple_CFL(mg);
        RZ_INFO("running step: ",step++);
        remove_out_of_bounds_density_particles_3d(smoke_particles,mg.dx);
        add_smoke_particles_in_range(source);
        // smoke density
        compute_smoke_density_from_particles(mg,mg.dx*1.5);
        compute_smoke_density_from_particles(density_mg,density_mg.dx*1.5);
        // save smoke

        // simulation
        remove_out_of_bounds_particles_3d(flip_particles, mg.dx);

        ptogrid(mg);

        add_external_force(mg,dt);

        grid_pressure_projection(mg); // boundary condition!

        gridtop(mg,0.95);//bigger for pic

        flip_advection(mg);

        smoke_advection(mg);
        // advect smoke particles

        convert_to_render_particles(flip_particles);

        // render smoke

        render.draw_frame(density_mg._density);
        render.drawParticles(smoke_particles);
        render.poll_events();
    }
}

void render_loop_3d_VEM(RGB_Screen &screen, std::string domain,std::string save_name) {//
    VEM_Domain_3d vem_simulator;
    vem_simulator._alpha = 0.5;
    vem_simulator._beta = 0.35;
    vem_simulator._decay = 4.0;
    vem_simulator.resample_grid.resize(150,150,150);

    MAC_Grid3D mg(100,100,100);// 辅助用于密度
    vem_simulator.read_from_file(domain);
    vem_simulator.save_boundary_to_obj();
    vem_simulator.build_AABB();
    vem_simulator.init();
    vem_simulator.preprocess_polygons();
    vem_simulator.generate_elements();

    int dofs = 0;
    for (const auto&element:vem_simulator.vem_elements) {
        int element_dof = 1 + element.vertex_count() + element.edge_count() + element.polygons.size();
        dofs+=element_dof*element_dof;
    }
    RZ_INFO("n^2 sum dof: ",dofs);
    vem_simulator.save_boundary_cells();
    vem_simulator.assemble_global_stiffness_matrix_k2();
    if (true) {//init
        visualize_A_sparsity(vem_simulator.global_stiffness_matrix_k2,"./"+save_name+"_k2.png");
        int N = vem_simulator.global_stiffness_matrix_k2.rows();
        RZ_INFO("no zeros:",vem_simulator.global_stiffness_matrix_k2.nonZeros());
        std::vector<GlobalReal> nullspaces(N);  // 1 column
        for (int i = 0; i < N; ++i) nullspaces[i] = 1.0;
        eigen_to_csr64(vem_simulator.csr_k2,vem_simulator.global_stiffness_matrix_k2);
        amgclInit(vem_simulator.solver_context, vem_simulator.csr_k2, 1e-3, 10000,0.0,nullspaces);
    }
    // vem_simulator.assemble_global_stiffness_matrix_k1();
//-------------------------------
    vem_simulator.grid_res = 100; // 可按场景调整
    vem_simulator.grid_to_elems.clear();
    vem_simulator.build_element_bboxes();
    vem_simulator.link_ele_with_uniform_grid_3d();
    Real kernel_radius = 1 / 200.0;
    vem_simulator.init_flip_particles(flip_particles,ghost_flip_particles,kernel_radius);
    // vem_simulator.element_wise_init_flip_particles(flip_particles);
//-------------------------------
    // for (auto&vem:vem_simulator.vem_elements) cubk+=vem.diameter;
    // cubk /= vem_simulator.vem_elements.size();
    // cubk /= 3.0;
//------------------------
    // vem_simulator.init_advection_concentration_filed(mg);


    sphere_source sp_source_ball(0.5,0.1,0.5,0.08);// ball

    // sphere_source sp_source_brush(0.6,0.1,0.45,0.08);
    // sp_source_brush.init_vec = _3d_vec(0,0.0,0.0);// brush

    // record_concentration_cells_in_range(sp_source,mg);
    VolumeRenderer render;
    render.initWindow();
    RZ_INFO("here");
    int step = 0;
    int frame_num = 0;
    const Real frame_time = 0.001;
    const Real flip_alpha = 0.05;
    while(!render.should_close()){
        if (frame_num>420)break;
        RZ_INFO("running frame: ",frame_num++);
        vem_simulator.CFL_GtoPicV(flip_particles,flip_alpha);
        simple_CFL(1/100.0);
        Real ft = frame_time;
        while (ft > 1e-6) {
            if (ft<dt)dt = ft;
            ft -= dt;
            RZ_INFO("running step: ",step++);
            RZ_INFO("dt: ",dt);
            vem_simulator.GtoP_k2_poly(flip_particles,flip_alpha);
            {// tracers step
                emit_tracers(vem_simulator,sp_source_ball,7777);
                vem_simulator.tracers_RK_3(dt/2.0);
                emit_tracers(vem_simulator,sp_source_ball,7777);
                vem_simulator.tracers_RK_3(dt/2.0);
                // for test
                // emit_tracers(vem_simulator,sp_source_brush,6666);
                // vem_simulator.tracers_RK_3(dt);
            }
            vem_simulator.RK_3_advection_k2(flip_particles,dt);
            vem_simulator.PtoG(flip_particles,ghost_flip_particles,kernel_radius);

            compute_smoke_density_from_particles(mg);

            vem_simulator.heat_decay(flip_particles,dt*1.0);
            add_smoke_particles_in_range(sp_source_ball,10.0,500.0);

            // vem_simulator.resample_grid.gather(flip_particles);
            // mg.add_smoke_concentration(dt,1);

            vem_simulator.add_external_forces(dt);


            vem_simulator.assemble_weakform_rhs_k1(dt);
            {
                vem_simulator.solve_k2(dt);
                vem_simulator.apply_pressure_k2_poly(dt);
            }
            // {
            //     vem_simulator.solve_k1(dt);
            //     vem_simulator.apply_pressure_k1(dt);
            // }
        }

        // render smoke
        saveTrancersToBin(vem_simulator.tracers,"/media/lhw/PS1008/SIG_Results/SmokeBins/"+save_name+"_"+std::to_string(frame_num)+".bin");
        render.draw_frame(mg._density);
        render.poll_events();
    }
}


void render_loop_3d_VEM(std::string mesh_file, std::string save_path, const std::string save_name,
    const sphere_source &source, int FULL_CFL, Real kernel_radius, int n, std::string aabb_path) {
    RZ_INFO("running cfl:",FULL_CFL);
    VEM_Domain_3d vem_simulator;
    vem_simulator._alpha = 0.5;
    vem_simulator._beta = 0.35;
    vem_simulator._decay = 4.0;
    vem_simulator.resample_grid.resize(150,150,150);
    vem_simulator.grid_res = (int)(100);
//-------------------------------
    vem_simulator.boundary_mesh = std::make_shared<BSP_Mesh>();
    auto& bsp_mesh = (*vem_simulator.boundary_mesh);
    bsp_mesh.read_mesh(mesh_file);
    bsp_mesh.sample_points_on_mesh();
    bsp_mesh.vx = 10;
    bsp_mesh.construct_planars_from_input_mesh();
    // lcc.build(bsp_mesh.planars,n);
    // lcc.split();
    // lcc.label_lcc(bsp_mesh.planars,0.65,80);
    // vem_simulator.load_from_lcc(lcc);
    MiniLcc mlcc;

    mlcc.build(bsp_mesh.planars,n);
    mlcc.split();
    mlcc.label_cvxs(bsp_mesh.planars,0.65,80);
    // mlcc.save_lcc("vis.vtu");
    // std::vector<int> level;
    // for (int i=0;i<mlcc.cvxs.size();++i)level.push_back(i);
    // // mlcc.save_vis_lcc("dragon_vis.vtu");
    // mlcc.save_lcc_obj("dragon_out_lcc.obj","dragon_in_lcc.obj");
    // mlcc.extract_surface("./check.obj");
    // exit(0);
    vem_simulator.load_from_Minilcc(mlcc);

    vem_simulator.save_boundary_to_obj();
    vem_simulator.build_AABB();
    vem_simulator.init();
    vem_simulator.preprocess_polygons();
    vem_simulator.save_obj_surface_to_obj();
    // exit(0);
    vem_simulator.generate_elements();// todo: speed up, maybe use neural network
    // exit(0);
    vem_simulator.compute_face_neumann_velocity();
    int dofs = 0;
    for (const auto&element:vem_simulator.vem_elements) {
        int element_dof = 1 + element.vertex_count() + element.edge_count() + element.polygons.size();
        dofs+=element_dof*element_dof;
    }
    RZ_INFO("n^2 sum dof: ",dofs);
    vem_simulator.assemble_global_stiffness_matrix_k2();
    if (true) {//init
        eigen_to_csr64(vem_simulator.csr_k2,vem_simulator.global_stiffness_matrix_k2);
        amgclInit(vem_simulator.solver_context, vem_simulator.csr_k2, 1e-3, 10000,0.0);
    }
    vem_simulator.boundary_handle_grid.resize(vem_simulator.grid_res);
    vem_simulator.build_boundary_handle_grid();
    vem_simulator.grid_to_elems.clear();
    vem_simulator.build_element_bboxes();
    vem_simulator.link_ele_with_uniform_grid_3d();
    vem_simulator.init_flip_particles(flip_particles,ghost_flip_particles,kernel_radius);
//-------------------------------
    Real cubk = kernel_radius;
    int step = 0;
    int frame_num = 0;
    const Real frame_time = 0.001;
    const Real flip_alpha = 0.05;
    RZ_ASSERT(FULL_CFL == 1,"cfl");
    while(true){
        if (frame_num>360)break;
        RZ_INFO("running frame: ",frame_num++);
        RZ_INFO("running step: ",step++);
        RZ_INFO("frame time: ",frame_time);

        vem_simulator.CFL_GtoPicV(flip_particles,flip_alpha);
        simple_CFL(1/120.0);
        vem_simulator.GtoP_k2_poly(flip_particles,flip_alpha);
        Real ft = frame_time;
        while (ft > 1e-5) {
            if (ft<dt)dt = ft;
            RZ_INFO("RK dt: ",dt);
            ft -= dt;
            {// tracers step
                emit_tracers(vem_simulator,source,6666);
                vem_simulator.tracers_RK_3(dt);
            }
            vem_simulator.RK_3_advection_k2(flip_particles,dt);
        }
        
        vem_simulator.assign_flip_smoke_particles_to_elements_3d(flip_particles);

        vem_simulator.PtoG(flip_particles,ghost_flip_particles,cubk);

        vem_simulator.heat_decay(flip_particles,frame_time);

        add_smoke_particles_in_range(source,10.0,500.0);

        vem_simulator.add_external_forces(frame_time);

        vem_simulator.assemble_weakform_rhs_k2(frame_time);

        // vem_simulator.assemble_weakform_rhs_boundary_k1(frame_time);
        // vem_simulator.assemble_weakform_rhs_boundary_k2(frame_time);

        // {
        //     vem_simulator.solve_k2(frame_time);
        //     vem_simulator.apply_pressure_k2_poly(frame_time);
        // }
        {
            vem_simulator.solve_k2(frame_time);
            vem_simulator.apply_pressure_k2_poly(frame_time);
        }
        //
        // exit(0);

        saveTrancersToBin(vem_simulator.tracers,
            save_path+save_name+"_"+std::to_string(frame_num)+".bin");
    }
}

void render_loop_3d_VEM(std::string mesh_file, std::string save_path, const std::string save_name,
    const sphere_source &source, int FULL_CFL, int init_n, int level, std::string aabb_path) {
    RZ_INFO("running cfl:",FULL_CFL);
    const int finest_res = init_n << (level - 1);
    const Real kernel_radius = 1.0 / static_cast<Real>(finest_res);
    RZ_INFO("computed kernel radius:",kernel_radius," finest res:",finest_res);
    int vem_k = 2;
    VEM_GMGSolver vem_gmg_simulator(vem_k);
    BSP_Mesh bsp_mesh;
    bsp_mesh.read_mesh(mesh_file);
    bsp_mesh.sample_points_on_mesh();
    bsp_mesh.vx = 10;
    bsp_mesh.construct_planars_from_input_mesh();

    MiniLcc mlcc;
    mlcc.MultiLevelbuild(bsp_mesh.planars,init_n,level);
    mlcc.split();
    mlcc.GMGlabel_cvxs(bsp_mesh.planars,0.65,80);
    mlcc.calculate_cut_values_v2();
    auto lods = mlcc.generate_lods(level,init_n);

// //------------------------------------------------------
    vem_gmg_simulator.load_mGrids(mlcc,lods);
    // Fixed coarse work per V-cycle; configure before init_for_pcg().
    vem_gmg_simulator.direct_coarse = false;
    vem_gmg_simulator.coarse_smoothing_iterations = 8;
    vem_gmg_simulator.pre_smoothing_iterations = 1;
    vem_gmg_simulator.post_smoothing_iterations = 1;
    vem_gmg_simulator.diagnostic_checks = false;
    vem_gmg_simulator.build_hierarchy();
    vem_gmg_simulator.init_for_pcg();
    // Use AMGCL's GMRES implementation with the fixed GMG V-cycle as its
    // right preconditioner. These match the AMGCL branch's tolerance and
    // default restart length (30) for a directly comparable outer solver.
    MG_gmres.bprm_ = vem_gmg_simulator.bprm_;
    {
        FunctionTimer timer("GMG_GMRES_build");
        MG_gmres.build(vem_gmg_simulator.finest_A(), vem_gmg_simulator.As_gpu.back(),
            10000, 1e-3, 30, false);
    }
 
    vem_gmg_simulator.mGrids.back()._alpha = 0.5;
    vem_gmg_simulator.mGrids.back()._beta = 0.35;
    vem_gmg_simulator.mGrids.back()._decay = 4.0;
    vem_gmg_simulator.mGrids.back().resample_grid.resize(150,150,150);
    vem_gmg_simulator.mGrids.back().grid_res = (int)(100);
//-------------------------------
    vem_gmg_simulator.mGrids.back().boundary_mesh = std::make_shared<BSP_Mesh>(bsp_mesh);

    //--------------------------------
    vem_gmg_simulator.mGrids.back().save_boundary_to_obj();
    vem_gmg_simulator.mGrids.back().build_AABB();

    int dofs = 0;
    for (const auto&element:vem_gmg_simulator.mGrids.back().vem_elements) {
        int element_dof = 1 + element.vertex_count() + element.edge_count() + element.polygons.size();
        dofs+=element_dof*element_dof;
    }

    vem_gmg_simulator.mGrids.back().boundary_handle_grid.resize(vem_gmg_simulator.mGrids.back().grid_res);
    vem_gmg_simulator.mGrids.back().build_boundary_handle_grid();
    vem_gmg_simulator.mGrids.back().grid_to_elems.clear();
    vem_gmg_simulator.mGrids.back().build_element_bboxes();
    vem_gmg_simulator.mGrids.back().link_ele_with_uniform_grid_3d();
    vem_gmg_simulator.mGrids.back().init_flip_particles(flip_particles,ghost_flip_particles,kernel_radius);
//-------------------------------
    Real cubk = kernel_radius;
    int step = 0;
    int frame_num = 0;
    const Real frame_time = 0.001;
    const Real flip_alpha = 0.05;
    RZ_ASSERT(FULL_CFL == 1,"cfl");
    while(true){
        if (frame_num>360)break;
        RZ_INFO("running frame: ",frame_num++);
        RZ_INFO("running step: ",step++);
        RZ_INFO("frame time: ",frame_time);

        vem_gmg_simulator.mGrids.back().CFL_GtoPicV(flip_particles,flip_alpha);
        simple_CFL(1/120.0);
        vem_gmg_simulator.mGrids.back().GtoP_k2_poly(flip_particles,flip_alpha);
        Real ft = frame_time;
        while (ft > 1e-5) {
            if (ft<dt)dt = ft;
            RZ_INFO("RK dt: ",dt);
            ft -= dt;
            {// tracers step
                emit_tracers(vem_gmg_simulator.mGrids.back(),source,6666);
                vem_gmg_simulator.mGrids.back().tracers_RK_3(dt);
            }
            vem_gmg_simulator.mGrids.back().RK_3_advection_k2(flip_particles,dt);
        }

        vem_gmg_simulator.mGrids.back().assign_flip_smoke_particles_to_elements_3d(flip_particles);

        vem_gmg_simulator.mGrids.back().PtoG(flip_particles,ghost_flip_particles,cubk);

        vem_gmg_simulator.mGrids.back().heat_decay(flip_particles,frame_time);

        add_smoke_particles_in_range(source,10.0,500.0);

        vem_gmg_simulator.mGrids.back().add_external_forces(frame_time);

        vem_gmg_simulator.mGrids.back().assemble_weakform_rhs_k2(frame_time);

        if (vem_k == 2) {
            // If CPU code changes pressure, call reset_device_initial_guess()
            // before solving to upload that new initial guess.
            {
                // Includes GMRES, all GMG V-cycles, and device-to-host pressure copy.
                FunctionTimer timer("GMG_GMRES_solve");
                MG_gmres.solve(vem_gmg_simulator,
                    vem_gmg_simulator.mGrids.back().global_rhs_k2,
                    vem_gmg_simulator.mGrids.back().global_p_k2,
                    VEX_GMRES::InitialGuess::Device);
            }
            vem_gmg_simulator.mGrids.back().apply_pressure_k2_poly(frame_time);
        }
        else {
            RZ_FATAL("");
        }

        saveTrancersToBin(vem_gmg_simulator.mGrids.back().tracers,
            save_path+save_name+"_"+std::to_string(frame_num)+".bin");
    }
}
void render_loop_3d_VEM(RGB_Screen &screen,std::vector<std::string> domains) {// todo : ptog or gtop disspsion
    int vem_k = 2;
    VEM_GMGSolver vem_gmg_simulator(vem_k);
    vem_gmg_simulator.load_mGrids(domains);
    vem_gmg_simulator.direct_coarse = false;
    vem_gmg_simulator.build_hierarchy();
    vem_gmg_simulator.init_for_pcg();
    MG_gmres.bprm_ = vem_gmg_simulator.bprm_;
    {
        FunctionTimer timer("GMG_GMRES_build");
        MG_gmres.build(vem_gmg_simulator.finest_A(), vem_gmg_simulator.As_gpu.back(),
            10000, 1e-3, 30, false);
    }
    exit(0);

    MAC_Grid3D mg(50,50,50);// 辅助用于密度
    VEM_Domain_3d& vem_simulator = vem_gmg_simulator.mGrids.back();
    vem_simulator._alpha = 0.15;
    vem_simulator._beta = 0.25;
    vem_simulator._decay = 10.0;
    Real cubk = 1 / 150.0;
    vem_simulator.init_flip_particles(flip_particles,ghost_flip_particles,cubk);
    // Source_range source(20,30,0,5,20,30);
    sphere_source sp_source(0.5,0.1,0.5,0.05);
    VolumeRenderer render;
    render.initWindow();
    int step = 0;
    while(!render.should_close()){
        simple_CFL(mg);//
        if (step>600)break;
        RZ_INFO("running step: ",step++);
        RZ_INFO("dt: ",dt);
        vem_gmg_simulator.dt = dt;

        vem_simulator.GtoP_k2_poly(flip_particles,0.03);
        vem_simulator.RK_3_advection_k2(flip_particles,dt);
        vem_simulator.PtoG(flip_particles,ghost_flip_particles,cubk);
        compute_smoke_density_from_particles(mg);

        vem_simulator.heat_decay(flip_particles,dt*0.5);
        add_smoke_particles_in_range(sp_source,10.0,500.0);

        vem_simulator.add_external_forces(dt);

        vem_simulator.heat_decay(flip_particles,dt*0.5);
        add_smoke_particles_in_range(sp_source,10.0,500.0);
        vem_simulator.assemble_weakform_rhs_k1(dt);

        if (vem_k == 2) {
            {
                FunctionTimer timer("GMG_GMRES_solve");
                MG_gmres.solve(vem_gmg_simulator, vem_simulator.global_rhs_k2,
                    vem_simulator.global_p_k2, VEX_GMRES::InitialGuess::Device);
            }
            vem_simulator.apply_pressure_k2_poly(dt);
        }

        mg.saveDensityToVDB("/home/lhw/桌面/VortexCoupling/PIC/RzPIC/outputs/smoke/ball/short_"+std::to_string(step)+".vdb");

        // render smoke
        render.draw_frame(mg._density);
        render.poll_events();
    }
}
