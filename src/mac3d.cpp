#include "mac3d.h"
#include <openvdb/openvdb.h>
// ======== 三维三线性插值辅助 ========
static inline Real trilinear_interp(
    Real c000, Real c100, Real c010, Real c110,
    Real c001, Real c101, Real c011, Real c111,
    Real tx, Real ty, Real tz)
{
    return
        c000 * (1 - tx) * (1 - ty) * (1 - tz) +
        c100 * tx       * (1 - ty) * (1 - tz) +
        c010 * (1 - tx) * ty       * (1 - tz) +
        c110 * tx       * ty       * (1 - tz) +
        c001 * (1 - tx) * (1 - ty) * tz +
        c101 * tx       * (1 - ty) * tz +
        c011 * (1 - tx) * ty       * tz +
        c111 * tx       * ty       * tz;
}

// ======== 各方向速度插值 ========
Real MAC_Grid3D::interpolate_u_velocity(_3d_vec u_coord, const Array3D<Real>& inter_u)
{
    int lx = u_coord.x_to_lower(), ly = u_coord.y_to_lower(), lz = u_coord.z_to_lower();
    int ux = u_coord.x_to_upper(), uy = u_coord.y_to_upper(), uz = u_coord.z_to_upper();

    // 边界安全
    lx = std::max(0, std::min(lx, nx));
    ux = std::max(0, std::min(ux, nx));
    ly = std::max(0, std::min(ly, ny - 1));
    uy = std::max(0, std::min(uy, ny - 1));
    lz = std::max(0, std::min(lz, nz - 1));
    uz = std::max(0, std::min(uz, nz - 1));

    Real c000 = inter_u(lx, ly, lz);
    Real c100 = inter_u(ux, ly, lz);
    Real c010 = inter_u(lx, uy, lz);
    Real c110 = inter_u(ux, uy, lz);
    Real c001 = inter_u(lx, ly, uz);
    Real c101 = inter_u(ux, ly, uz);
    Real c011 = inter_u(lx, uy, uz);
    Real c111 = inter_u(ux, uy, uz);

    Real tx = u_coord.vx - lx;
    Real ty = u_coord.vy - ly;
    Real tz = u_coord.vz - lz;

    return trilinear_interp(c000, c100, c010, c110, c001, c101, c011, c111, tx, ty, tz);
}

Real MAC_Grid3D::interpolate_v_velocity(_3d_vec v_coord, const Array3D<Real>& inter_v)
{
    int lx = v_coord.x_to_lower(), ly = v_coord.y_to_lower(), lz = v_coord.z_to_lower();
    int ux = v_coord.x_to_upper(), uy = v_coord.y_to_upper(), uz = v_coord.z_to_upper();

    lx = std::max(0, std::min(lx, nx - 1));
    ux = std::max(0, std::min(ux, nx - 1));
    ly = std::max(0, std::min(ly, ny));
    uy = std::max(0, std::min(uy, ny));
    lz = std::max(0, std::min(lz, nz - 1));
    uz = std::max(0, std::min(uz, nz - 1));

    Real c000 = inter_v(lx, ly, lz);
    Real c100 = inter_v(ux, ly, lz);
    Real c010 = inter_v(lx, uy, lz);
    Real c110 = inter_v(ux, uy, lz);
    Real c001 = inter_v(lx, ly, uz);
    Real c101 = inter_v(ux, ly, uz);
    Real c011 = inter_v(lx, uy, uz);
    Real c111 = inter_v(ux, uy, uz);

    Real tx = v_coord.vx - lx;
    Real ty = v_coord.vy - ly;
    Real tz = v_coord.vz - lz;

    return trilinear_interp(c000, c100, c010, c110, c001, c101, c011, c111, tx, ty, tz);
}

Real MAC_Grid3D::interpolate_w_velocity(_3d_vec w_coord, const Array3D<Real>& inter_w)
{
    int lx = w_coord.x_to_lower(), ly = w_coord.y_to_lower(), lz = w_coord.z_to_lower();
    int ux = w_coord.x_to_upper(), uy = w_coord.y_to_upper(), uz = w_coord.z_to_upper();

    lx = std::max(0, std::min(lx, nx - 1));
    ux = std::max(0, std::min(ux, nx - 1));
    ly = std::max(0, std::min(ly, ny - 1));
    uy = std::max(0, std::min(uy, ny - 1));
    lz = std::max(0, std::min(lz, nz));
    uz = std::max(0, std::min(uz, nz));

    Real c000 = inter_w(lx, ly, lz);
    Real c100 = inter_w(ux, ly, lz);
    Real c010 = inter_w(lx, uy, lz);
    Real c110 = inter_w(ux, uy, lz);
    Real c001 = inter_w(lx, ly, uz);
    Real c101 = inter_w(ux, ly, uz);
    Real c011 = inter_w(lx, uy, uz);
    Real c111 = inter_w(ux, uy, uz);

    Real tx = w_coord.vx - lx;
    Real ty = w_coord.vy - ly;
    Real tz = w_coord.vz - lz;

    return trilinear_interp(c000, c100, c010, c110, c001, c101, c011, c111, tx, ty, tz);
}

// ======== 密度插值 ========
Real MAC_Grid3D::interpolate_density(_3d_vec density_world_coord)
{
    _3d_vec grid_coord = coord_trans_rtom(density_world_coord);
    if (grid_coord.vx < 0 || grid_coord.vy < 0 || grid_coord.vz < 0 ||
        grid_coord.vx > nx || grid_coord.vy > ny || grid_coord.vz > nz)
        return 0.0f;

    _3d_vec density_coord = grid_coord - _3d_vec(0.5f, 0.5f, 0.5f);

    int lx = density_coord.x_to_lower();
    int ly = density_coord.y_to_lower();
    int lz = density_coord.z_to_lower();
    int ux = density_coord.x_to_upper();
    int uy = density_coord.y_to_upper();
    int uz = density_coord.z_to_upper();

    lx = std::max(0, std::min(lx, nx - 1));
    ux = std::max(0, std::min(ux, nx - 1));
    ly = std::max(0, std::min(ly, ny - 1));
    uy = std::max(0, std::min(uy, ny - 1));
    lz = std::max(0, std::min(lz, nz - 1));
    uz = std::max(0, std::min(uz, nz - 1));

    Real c000 = _density(lx, ly, lz);
    Real c100 = _density(ux, ly, lz);
    Real c010 = _density(lx, uy, lz);
    Real c110 = _density(ux, uy, lz);
    Real c001 = _density(lx, ly, uz);
    Real c101 = _density(ux, ly, uz);
    Real c011 = _density(lx, uy, uz);
    Real c111 = _density(ux, uy, uz);

    Real tx = density_coord.vx - lx;
    Real ty = density_coord.vy - ly;
    Real tz = density_coord.vz - lz;

    return trilinear_interp(c000, c100, c010, c110, c001, c101, c011, c111, tx, ty, tz);
}

// void MAC_Grid3D::saveDensityToVDB(const std::string &filename) {
//     {
//         // Initialize OpenVDB library
//         openvdb::initialize();
//
//         // 创建一个新的 VDB 文件
//         openvdb::GridPtrVec grids;
//
//         // 创建一个浮动点格子来存储密度数据
//         openvdb::FloatGrid::Ptr densityGrid = openvdb::FloatGrid::create();
//
//         // 获取密度数据的值
//         openvdb::FloatGrid::Accessor accessor = densityGrid->getAccessor();
//
//         // 通过循环访问 _density 数组中的每个元素，并将其转换为 VDB 格式
//         for (int k = 0; k < nz; ++k) {
//             for (int j = 0; j < ny; ++j) {
//                 for (int i = 0; i < nx; ++i) {
//                     // 计算网格的世界坐标
//                     openvdb::Coord ijk(i, j, k);
//                     float densityValue = _density(i, j, k);
//
//                     // 设置 VDB 格子中的值
//                     accessor.setValue(ijk, densityValue);
//                 }
//             }
//         }
//
//         // 将格子添加到 VDB 文件中
//         grids.push_back(densityGrid);
//
//         // 将格子写入 VDB 文件
//         openvdb::io::File file(filename);
//         file.write(grids);
//         std::cout << "Density data saved to " << filename << std::endl;
//     }
// }
void MAC_Grid3D::saveDensityToVDB(const std::string &filename) {
    // Initialize OpenVDB library (建议全局一次，这里保留你原逻辑)
    openvdb::initialize();

    openvdb::GridPtrVec grids;

    // 1️⃣ 创建 FloatGrid
    openvdb::FloatGrid::Ptr densityGrid = openvdb::FloatGrid::create(0.0f);

    // ——【关键 1】设置 grid 名字，Blender / Cycles 直接可用
    densityGrid->setName("density");

    // 2️⃣ 计算 density 的 min / max（用于 scale 到 [0,1]）
    float dmin = std::numeric_limits<float>::max();
    float dmax = std::numeric_limits<float>::lowest();

    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                float d = _density(i, j, k);
                dmin = std::min(dmin, d);
                dmax = std::max(dmax, d);
            }

    float invRange = (dmax > dmin) ? 1.0f / (dmax - dmin) : 0.0f;

    // 3️⃣ 写入 VDB（含 scale + X 轴旋转 90°）
    openvdb::FloatGrid::Accessor accessor = densityGrid->getAccessor();

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {

                // ——【关键 2】scale / clamp 到 [0,1]
                float d = _density(i, j, k);
                float d01 = (d - dmin) * invRange;
                d01 = std::clamp(d01, 0.0f, 1.0f);

                // ——【关键 3】沿 X 轴旋转 +90°
                //
                // 原坐标: (i, j, k)
                // 绕 X 轴 +90°:
                //   y' = -z
                //   z' =  y
                //
                // 为了保持 index 非负，这里做一个平移
                int ii = i;
                int jj = nz - 1 - k;   // -z → [0, nz-1]
                int kk = j;

                openvdb::Coord ijk(ii, jj, kk);
                accessor.setValue(ijk, d01);
            }
        }
    }

    // 4️⃣ （强烈推荐）设置 voxel 尺寸（否则 Blender 默认 1m/voxel）
    // 假设你的 MAC grid spacing 是 dx
    densityGrid->setTransform(
        openvdb::math::Transform::createLinearTransform(dx)
    );

    // 5️⃣ 写文件
    grids.push_back(densityGrid);

    openvdb::io::File file(filename);
    file.write(grids);
    file.close();
    
}

void VEM_UGRID::saveDensityToVDB(const std::string &filename) {
    // Initialize OpenVDB library (建议全局一次，这里保留你原逻辑)
    openvdb::initialize();

    openvdb::GridPtrVec grids;

    // 1️⃣ 创建 FloatGrid
    openvdb::FloatGrid::Ptr densityGrid = openvdb::FloatGrid::create(0.0f);

    // ——【关键 1】设置 grid 名字，Blender / Cycles 直接可用
    densityGrid->setName("density");

    // 2️⃣ 计算 density 的 min / max（用于 scale 到 [0,1]）
    float dmin = std::numeric_limits<float>::max();
    float dmax = std::numeric_limits<float>::lowest();

    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                float d = _density(i, j, k);
                dmin = std::min(dmin, d);
                dmax = std::max(dmax, d);
            }

    float invRange = (dmax > dmin) ? 1.0f / (dmax - dmin) : 0.0f;

    // 3️⃣ 写入 VDB（含 scale + X 轴旋转 90°）
    openvdb::FloatGrid::Accessor accessor = densityGrid->getAccessor();

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {

                // ——【关键 2】scale / clamp 到 [0,1]
                float d = _density(i, j, k);
                float d01 = (d - dmin) * invRange;
                d01 = std::clamp(d01, 0.0f, 1.0f);

                // ——【关键 3】沿 X 轴旋转 +90°
                //
                // 原坐标: (i, j, k)
                // 绕 X 轴 +90°:
                //   y' = -z
                //   z' =  y
                //
                // 为了保持 index 非负，这里做一个平移
                int ii = i;
                int jj = nz - 1 - k;   // -z → [0, nz-1]
                int kk = j;

                openvdb::Coord ijk(ii, jj, kk);
                accessor.setValue(ijk, d01);
            }
        }
    }

    // 4️⃣ （强烈推荐）设置 voxel 尺寸（否则 Blender 默认 1m/voxel）
    // 假设你的 MAC grid spacing 是 dx
    densityGrid->setTransform(
        openvdb::math::Transform::createLinearTransform(dx)
    );

    // 5️⃣ 写文件
    grids.push_back(densityGrid);

    openvdb::io::File file(filename);
    file.write(grids);
    file.close();
}

void VEM_UGRID::gather(const std::vector<FLIP_Particles_3d> &flip_particles)
{
    FUNC_TIMER();
    ivock_kernel._h = dx;

    // =========================
    // 0. 清空网格
    // =========================
    _density.fill(0.0);
    _temp.fill(0.0);

    Array3D<Real> coef(nx, ny, nz, 0.0);

    const Real eps = 1e-4;

    // =========================
    // 1. 构建粒子 hash（cell -> particle indices）
    //    注意：hash 仍然是 cell-based，用于加速邻域查询
    // =========================
    std::vector<std::vector<int>> particle_hash(nx * ny * nz);

    auto clampi = [&](int v, int lo, int hi) {
        return std::max(lo, std::min(hi, v));
    };

    for (int p = 0; p < (int)flip_particles.size(); ++p) {
        if (flip_particles[p].deleted) continue;

        const auto& pos = flip_particles[p].pos;
        int ci = clampi(int(pos.vx / dx), 0, nx - 1);
        int cj = clampi(int(pos.vy / dx), 0, ny - 1);
        int ck = clampi(int(pos.vz / dx), 0, nz - 1);

        int hash_idx = ck * nx * ny + cj * nx + ci;
        particle_hash[hash_idx].push_back(p);
    }

    // =========================
    // 2. 粒子 -> 顶点 gather
    // =========================
#pragma omp parallel for schedule(static)
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {

                // ===== 顶点坐标 =====
                Real gx = i * dx;
                Real gy = j * dx;
                Real gz = k * dx;

                Real total_w = eps;
                Real rho = 0.0, temp = 0.0;

                // 顶点影响的粒子，来自相邻 cell
                for (int kk = -1; kk <= 1; ++kk)
                    for (int jj = -1; jj <= 1; ++jj)
                        for (int ii = -1; ii <= 1; ++ii) {

                            int ni = i + ii;
                            int nj = j + jj;
                            int nk = k + kk;
                            if (ni < 0 || ni >= nx ||
                                nj < 0 || nj >= ny ||
                                nk < 0 || nk >= nz)
                                continue;

                            int hash_idx = nk * nx * ny + nj * nx + ni;
                            const auto& plist = particle_hash[hash_idx];

                            for (int pid : plist) {
                                const auto& p = flip_particles[pid];
                                Real r = (_3d_vec(gx, gy, gz) - p.pos).norm();

                                Real wgt = ivock_kernel.weight(
                                    gx, gy, gz,
                                    p.pos.vx, p.pos.vy, p.pos.vz
                                );

                                if (wgt <= 0.0) continue;

                                total_w += wgt;

                                rho  += wgt * p.rho;
                                temp += wgt * p.temp;
                            }
                        }

                coef(i, j, k) = total_w;
                _density(i, j, k) = rho  / total_w;
                _temp(i, j, k)    = temp / total_w;
            }

    // =========================
    // 3. 低权重点的简单修补（保持与你原逻辑一致）
    // =========================
#pragma omp parallel for schedule(static)
    for (int k = 1; k < nz - 1; ++k)
        for (int j = 1; j < ny - 1; ++j)
            for (int i = 1; i < nx - 1; ++i) {

                if (coef(i, j, k) < 1e-3) {
                    _density(i, j, k) =
                        (_density(i - 1, j, k) + _density(i + 1, j, k) +
                         _density(i, j - 1, k) + _density(i, j + 1, k) +
                         _density(i, j, k - 1) + _density(i, j, k + 1)) / 6.0;

                    _temp(i, j, k) =
                        (_temp(i - 1, j, k) + _temp(i + 1, j, k) +
                         _temp(i, j - 1, k) + _temp(i, j + 1, k) +
                         _temp(i, j, k - 1) + _temp(i, j, k + 1)) / 6.0;
                }
            }
}


static inline Real clampd(Real v, Real lo, Real hi) {
    return std::max(lo, std::min(hi, v));
}
static inline int clampi(int v, int lo, int hi) {
    return std::max(lo, std::min(hi, v));
}
static inline Real lerp(Real a, Real b, Real t) {
    return (1.0 - t) * a + t * b;
}
inline Real sample_linear(
    const Array3D<Real>& g,
    Real world_x, Real world_y, Real world_z
) {
    // We store values on vertex nodes: i=0..nx-1, etc.
    // Clamp in grid space first to keep (i,i+1) valid and ensure cx,cy,cz in [0,1).
    const Real eps = 1e-12;
    const Real dx = 1.0/g.nx_;
    const int nx = g.nx_;
    const int ny = g.ny_;
    const int nz = g.nz_;

    Real gx = world_x / dx;
    Real gy = world_y / dx;
    Real gz = world_z / dx;

    gx = clampd(gx, 0.0, (Real)(nx - 1) - eps);
    gy = clampd(gy, 0.0, (Real)(ny - 1) - eps);
    gz = clampd(gz, 0.0, (Real)(nz - 1) - eps);

    int i = (int)std::floor(gx);
    int j = (int)std::floor(gy);
    int k = (int)std::floor(gz);

    // Ensure i+1 exists
    i = clampi(i, 0, nx - 2);
    j = clampi(j, 0, ny - 2);
    k = clampi(k, 0, nz - 2);

    const Real cx = gx - (Real)i;
    const Real cy = gy - (Real)j;
    const Real cz = gz - (Real)k;

    const Real v000 = g(i,   j,   k);
    const Real v100 = g(i+1, j,   k);
    const Real v010 = g(i,   j+1, k);
    const Real v110 = g(i+1, j+1, k);

    const Real v001 = g(i,   j,   k+1);
    const Real v101 = g(i+1, j,   k+1);
    const Real v011 = g(i,   j+1, k+1);
    const Real v111 = g(i+1, j+1, k+1);

    const Real v00 = lerp(v000, v100, cx);
    const Real v10 = lerp(v010, v110, cx);
    const Real v01 = lerp(v001, v101, cx);
    const Real v11 = lerp(v011, v111, cx);

    const Real v0  = lerp(v00, v10, cy);
    const Real v1  = lerp(v01, v11, cy);

    return lerp(v0, v1, cz);
}


inline void VEM_UGRID::assign(FLIP_Particles_3d& p) const
{
    if (p.deleted) return;

    const Real x = p.pos.vx;
    const Real y = p.pos.vy;
    const Real z = p.pos.vz;
    // ===== density / temperature =====
    p.rho  = sample_linear(_density, x, y, z);
    p.temp = sample_linear(_temp,    x, y, z);
}

void VEM_UGRID::resample(std::vector<FLIP_Particles_3d> &particles) {
    FUNC_TIMER();

#pragma omp parallel for schedule(dynamic)
    for (int i=0;i<particles.size();++i) {
        if (particles[i].need_interpolate)
            assign(particles[i]);
    }

}

Real ConcentrationField3D::interpolate_density(
    const _3d_vec& real_world_pos ,const Array3D<Real>& density
) const
{
    // ------------------------------------------------------------
    // 1. world -> cell index space (cell-centered)
    // ------------------------------------------------------------
    const Real gx = real_world_pos.vx / dx - 0.5;
    const Real gy = real_world_pos.vy / dx - 0.5;
    const Real gz = real_world_pos.vz / dx - 0.5;

    int i0 = static_cast<int>(std::floor(gx));
    int j0 = static_cast<int>(std::floor(gy));
    int k0 = static_cast<int>(std::floor(gz));

    Real fx = gx - i0;
    Real fy = gy - j0;
    Real fz = gz - k0;

    // ------------------------------------------------------------
    // 2. clamp indices (IMPORTANT)
    // ------------------------------------------------------------
    auto clampi = [&](int v, int lo, int hi) {
        return std::max(lo, std::min(hi, v));
    };

    int i1 = clampi(i0 + 1, 0, nx - 1);
    int j1 = clampi(j0 + 1, 0, ny - 1);
    int k1 = clampi(k0 + 1, 0, nz - 1);

    i0 = clampi(i0, 0, nx - 1);
    j0 = clampi(j0, 0, ny - 1);
    k0 = clampi(k0, 0, nz - 1);

    // ------------------------------------------------------------
    // 3. fetch 8 corner samples
    // ------------------------------------------------------------
    const Real c000 = density(i0, j0, k0);
    const Real c100 = density(i1, j0, k0);
    const Real c010 = density(i0, j1, k0);
    const Real c110 = density(i1, j1, k0);

    const Real c001 = density(i0, j0, k1);
    const Real c101 = density(i1, j0, k1);
    const Real c011 = density(i0, j1, k1);
    const Real c111 = density(i1, j1, k1);

    // ------------------------------------------------------------
    // 4. trilinear interpolation
    // ------------------------------------------------------------
    const Real c00 = c000 * (1.0 - fx) + c100 * fx;
    const Real c10 = c010 * (1.0 - fx) + c110 * fx;
    const Real c01 = c001 * (1.0 - fx) + c101 * fx;
    const Real c11 = c011 * (1.0 - fx) + c111 * fx;

    const Real c0 = c00 * (1.0 - fy) + c10 * fy;
    const Real c1 = c01 * (1.0 - fy) + c11 * fy;

    return c0 * (1.0 - fz) + c1 * fz;
}

void ConcentrationField3D::add_smoke_concentration(const Real&dt,const Real&_dens) {
    FUNC_TIMER();
#pragma omp parallel for schedule(static)
    for (int idx = 0; idx < (int)inside_source.size(); ++idx)
    {
        int cid = inside_source[idx];

        int i =  cid % nx;
        int j = (cid / nx) % ny;
        int k =  cid / (nx * ny);

        _density(i, j, k) = _dens;
    }
    // _density.clamp01();
}




void ConcentrationField3D::saveDensityToVDB(const std::string &filename) const {
    // Initialize OpenVDB library (建议全局一次，这里保留你原逻辑)
    openvdb::initialize();

    openvdb::GridPtrVec grids;

    // 1️⃣ 创建 FloatGrid
    openvdb::FloatGrid::Ptr densityGrid = openvdb::FloatGrid::create(0.0f);

    // ——【关键 1】设置 grid 名字，Blender / Cycles 直接可用
    densityGrid->setName("density");

    // 2️⃣ 计算 density 的 min / max（用于 scale 到 [0,1]）
    float dmin = std::numeric_limits<float>::max();
    float dmax = std::numeric_limits<float>::lowest();

    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                float d = _density(i, j, k);
                dmin = std::min(dmin, d);
                dmax = std::max(dmax, d);
            }

    float invRange = (dmax > dmin) ? 1.0f / (dmax - dmin) : 0.0f;

    // 3️⃣ 写入 VDB（含 scale + X 轴旋转 90°）
    openvdb::FloatGrid::Accessor accessor = densityGrid->getAccessor();

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {

                // ——【关键 2】scale / clamp 到 [0,1]
                float d = _density(i, j, k);
                float d01 = (d - dmin) * invRange;
                d01 = std::clamp(d01, 0.0f, 1.0f);

                // ——【关键 3】沿 X 轴旋转 +90°
                //
                // 原坐标: (i, j, k)
                // 绕 X 轴 +90°:
                //   y' = -z
                //   z' =  y
                //
                // 为了保持 index 非负，这里做一个平移
                int ii = i;
                int jj = nz - 1 - k;   // -z → [0, nz-1]
                int kk = j;

                openvdb::Coord ijk(ii, jj, kk);
                accessor.setValue(ijk, d01);
            }
        }
    }

    // 4️⃣ （强烈推荐）设置 voxel 尺寸（否则 Blender 默认 1m/voxel）
    // 假设你的 MAC grid spacing 是 dx
    densityGrid->setTransform(
        openvdb::math::Transform::createLinearTransform(dx)
    );

    // 5️⃣ 写文件
    grids.push_back(densityGrid);

    openvdb::io::File file(filename);
    file.write(grids);
    file.close();
}


void MAC_Grid3D::resize(int n) {
    nx = n, ny = n, nz = n;
    _p.resize(nx, ny, nz, 0.0f),
    _u.resize(nx + 1, ny, nz, 0.0f),
    _v.resize(nx, ny + 1, nz, 0.0f),
    _w.resize(nx, ny, nz + 1, 0.0f),
    _delta_u.resize(nx + 1, ny, nz, 0.0f),
    _delta_v.resize(nx, ny + 1, nz, 0.0f),
    _delta_w.resize(nx, ny, nz + 1, 0.0f),
    _density.resize(nx, ny, nz, 0.0f),
    dx = 1.0/(Real)n;
}

// ======== 速度场插值 ========
_3d_vec MAC_Grid3D::interpolate_velocity(const _3d_vec& real_world_pos)
{
    _3d_vec grid_coord = coord_trans_rtom(real_world_pos);
    if (grid_coord.vx < 0 || grid_coord.vy < 0 || grid_coord.vz < 0 ||
        grid_coord.vx > nx || grid_coord.vy > ny || grid_coord.vz > nz)
        return _3d_vec(0, 0, 0);

    _3d_vec u_coord(grid_coord.vx, grid_coord.vy - 0.5f, grid_coord.vz - 0.5f);
    _3d_vec v_coord(grid_coord.vx - 0.5f, grid_coord.vy, grid_coord.vz - 0.5f);
    _3d_vec w_coord(grid_coord.vx - 0.5f, grid_coord.vy - 0.5f, grid_coord.vz);

    _3d_vec ans;
    ans.vx = interpolate_u_velocity(u_coord, _u);
    ans.vy = interpolate_v_velocity(v_coord, _v);
    ans.vz = interpolate_w_velocity(w_coord, _w);
    return ans;
}

_3d_vec MAC_Grid3D::interpolate_delta_velocity(const _3d_vec& real_world_pos)
{
    _3d_vec grid_coord = coord_trans_rtom(real_world_pos);
    if (grid_coord.vx < 0 || grid_coord.vy < 0 || grid_coord.vz < 0 ||
        grid_coord.vx > nx || grid_coord.vy > ny || grid_coord.vz > nz)
        return _3d_vec(0, 0, 0);

    _3d_vec u_coord(grid_coord.vx, grid_coord.vy - 0.5f, grid_coord.vz - 0.5f);
    _3d_vec v_coord(grid_coord.vx - 0.5f, grid_coord.vy, grid_coord.vz - 0.5f);
    _3d_vec w_coord(grid_coord.vx - 0.5f, grid_coord.vy - 0.5f, grid_coord.vz);

    _3d_vec ans;
    ans.vx = interpolate_u_velocity(u_coord, _delta_u);
    ans.vy = interpolate_v_velocity(v_coord, _delta_v);
    ans.vz = interpolate_w_velocity(w_coord, _delta_w);
    return ans;
}

// ======== 世界坐标 → 网格坐标 ========
_3d_vec MAC_Grid3D::coord_trans_rtom(const _3d_vec& real_world_pos)
{
    return _3d_vec(real_world_pos.vx / dx,
                   real_world_pos.vy / dx,
                   real_world_pos.vz / dx);
}
