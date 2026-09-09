#pragma once
#include "dfn.h"

class MAC_Grid3D
{
public:
    MAC_Grid3D(size_t nx = 0, size_t ny = 0, size_t nz = 0)
        : nx(nx), ny(ny), nz(nz),
          _p(nx, ny, nz, 0.0f),
          _u(nx + 1, ny, nz, 0.0f),
          _v(nx, ny + 1, nz, 0.0f),
          _w(nx, ny, nz + 1, 0.0f),
          _delta_u(nx + 1, ny, nz, 0.0f),
          _delta_v(nx, ny + 1, nz, 0.0f),
          _delta_w(nx, ny, nz + 1, 0.0f),
          _density(nx, ny, nz, 0.0f),
          dx(1.0f / nx)
    {}

    // ====== 主数据 ======
    Array3D<Real> _p, _u, _v, _w;
    Array3D<Real> _delta_u, _delta_v, _delta_w;
    Array3D<Real> _density;
    Real dx;
    int nx, ny, nz;

    // ====== 函数 ======
    void resize(int n);

    _3d_vec interpolate_velocity(const _3d_vec& real_world_pos);
    _3d_vec interpolate_delta_velocity(const _3d_vec& real_world_pos);
    _3d_vec coord_trans_rtom(const _3d_vec& real_world_pos);

    Real interpolate_u_velocity(_3d_vec u_coord, const Array3D<Real>& inter_u);
    Real interpolate_v_velocity(_3d_vec v_coord, const Array3D<Real>& inter_v);
    Real interpolate_w_velocity(_3d_vec w_coord, const Array3D<Real>& inter_w);
    Real interpolate_density(_3d_vec density_world_coord);

    // 将 _density 存储为 VDB 文件
    void saveDensityToVDB(const std::string& filename);
};

// class VEM_UGRID
// {
// public:
//     VEM_UGRID(size_t nx = 0, size_t ny = 0, size_t nz = 0)
//         : nx(nx), ny(ny), nz(nz),
//           _density(nx, ny, nz, 0.0f),
//           _temp(nx, ny, nz, 0.0f),
//           dx(1.0f / nx),
//             ivock_kernel(1.0f/nx)
//     {}
//
//     // ====== 主数据 ======
//     Array3D<Real>  _density,_temp;
//     IVOCK_Kernel ivock_kernel;
//     Real dx;
//     int nx, ny, nz;
//
//     void saveDensityToVDB(const std::string& filename);
//     void gather(const std::vector<FLIP_Particles_3d>& flip_particles) ;
//     inline void assign(FLIP_Particles_3d& p) const;
//     void resample(std::vector<FLIP_Particles_3d>& particles);
// };
class VEM_UGRID
{
public:
    // ====== 默认构造：200^3 ======
    VEM_UGRID():ivock_kernel(1.0f/200.0)
    {
        resize(200, 200, 200);
        ivock_kernel = IVOCK_Kernel(1/200.0);
    }

    // ====== 显式构造 ======
    VEM_UGRID(size_t nx_, size_t ny_, size_t nz_):ivock_kernel(1.0f/200.0)
    {
        resize(nx_, ny_, nz_);
    }

    // ====== 调整分辨率（核心接口） ======
    void resize(size_t nx_, size_t ny_, size_t nz_)
    {
        nx = static_cast<int>(nx_);
        ny = static_cast<int>(ny_);
        nz = static_cast<int>(nz_);

        // 网格步长（假设单位盒 [0,1]^3）
        dx = 1.0 / Real(nx);

        // 重建场
        _density.resize(nx, ny, nz);
        _temp.resize(nx, ny, nz);
        _density.fill(0.0);
        _temp.fill(0.0);

        // 重建核（重要：kernel 半径和 dx 强相关）
        ivock_kernel = IVOCK_Kernel(dx);
    }

    // ====== 主数据 ======
    Array3D<Real> _density, _temp;
    IVOCK_Kernel ivock_kernel;
    Real dx = 0.0;
    int nx = 0, ny = 0, nz = 0;

    // ====== 功能接口 ======
    void saveDensityToVDB(const std::string& filename);
    void gather(const std::vector<FLIP_Particles_3d>& flip_particles);
    inline void assign(FLIP_Particles_3d& p) const;
    void resample(std::vector<FLIP_Particles_3d>& particles);
};


class ConcentrationField3D
{
public:
    ConcentrationField3D(size_t nx = 0, size_t ny = 0, size_t nz = 0)
        : nx(nx), ny(ny), nz(nz),
          _density(nx, ny, nz, 0.0),
          dx(1.0 / Real(nx))
    {}

    // ====== 主数据 ======
    Array3D<Real> _density;   // cell-centered scalar field
    Real dx;
    int nx, ny, nz;

    // ====== 核心接口 ======

    Real interpolate_density(
        const _3d_vec& real_world_pos,const Array3D<Real>& _density
    ) const;

    void add_smoke_concentration(const Real&dt,const Real&_dens);

    // VDB 导出
    void saveDensityToVDB(const std::string& filename) const;

    std::vector<_3d_vec> init_rk_pos;
    std::vector<int> init_elemids;
    std::vector<int> inside_source;
};
