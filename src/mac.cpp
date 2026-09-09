#include "mac.h"
// 这一块插值基本思路：首先grid是左下角0，0 超出grid的就是0 然后就是不同的coord:p,u,v
// 可以画一下 做一个坐标转换，要进一步处理 坐标在grid内 但是坐标转换后用到的插值点可能超出范围的情况
Real MAC_Grid::interpolate_u_velocity(_2d_vec u_coord,const Array2D<Real>& inter_u) {
    int low_x = u_coord.x_to_lower(),low_y = u_coord.y_to_lower(),
    upper_x = u_coord.x_to_upper(),upper_y = u_coord.y_to_upper();

    Real Qll = low_y < 0?inter_u(low_x,0):inter_u(low_x,low_y),
    Qul = low_y < 0?inter_u(upper_x,0):inter_u(upper_x,low_y),
    Qlu = upper_y == ny?inter_u(low_x, ny-1):inter_u(low_x, upper_y),
    Quu = upper_y == ny?inter_u(upper_x, ny-1):inter_u(upper_x, upper_y);
    // Real Qll = low_y < 0?0.0:_u(low_x,low_y),
    // Qul = low_y < 0?0.0:_u(upper_x,low_y),
    // Qlu = upper_y == ny?0.0:_u(low_x, upper_y),
    // Quu = upper_y == ny?0.0:_u(upper_x, upper_y);

    Real alpha_x = u_coord.vx - low_x, alpha_y = u_coord.vy - low_y;

    return (1 - alpha_x) * (1 - alpha_y) * Qll +
           alpha_x       * (1 - alpha_y) * Qul +
           (1 - alpha_x) * alpha_y       * Qlu +
           alpha_x       * alpha_y       * Quu;
}

Real MAC_Grid::interpolate_v_velocity(_2d_vec v_coord,const Array2D<Real>& inter_v) {
    int low_x = v_coord.x_to_lower(), low_y = v_coord.y_to_lower(),
        upper_x = v_coord.x_to_upper(), upper_y = v_coord.y_to_upper();

    Real Qll = low_x < 0 ? inter_v(0, low_y) : inter_v(low_x, low_y);
    Real Qul = upper_x == nx ? inter_v(nx - 1, low_y) : inter_v(upper_x, low_y);
    Real Qlu = low_x < 0 ? inter_v(0, upper_y) : inter_v(low_x, upper_y);
    Real Quu = upper_x == nx ? inter_v(nx - 1, upper_y) : inter_v(upper_x, upper_y);
    // Real Qll = low_x < 0 ? 0.0 : _v(low_x, low_y);
    // Real Qul = upper_x == nx ? 0.0 : _v(upper_x, low_y);
    // Real Qlu = low_x < 0 ? 0.0 : _v(low_x, upper_y);
    // Real Quu = upper_x == nx ? 0.0 : _v(upper_x, upper_y);
    Real alpha_x = v_coord.vx - low_x;
    Real alpha_y = v_coord.vy - low_y;

    return (1 - alpha_x) * (1 - alpha_y) * Qll +
           alpha_x       * (1 - alpha_y) * Qul +
           (1 - alpha_x) * alpha_y       * Qlu +
           alpha_x       * alpha_y       * Quu;
}

// Real MAC_Grid::interpolate_density(_2d_vec density_world_coord)
// {
//     _2d_vec grid_coord = coord_trans_rtom(density_world_coord);
//
//     // 如果完全落到域外，返回 0
//     if (grid_coord.vx < 0 || grid_coord.vy < 0 ||
//         grid_coord.vx > nx  || grid_coord.vy > ny) {
//         return 0.0;
//         }
//
//     _2d_vec density_coord = grid_coord - _2d_vec(0.5, 0.5);
//
//     // ==== Clamp 到合法范围 ====
//     int low_x   = std::max(0, std::min((int)nx - 1, density_coord.x_to_lower()));
//     int low_y   = std::max(0, std::min((int)ny - 1, density_coord.y_to_lower()));
//     int upper_x = std::max(0, std::min((int)nx - 1, density_coord.x_to_upper()));
//     int upper_y = std::max(0, std::min((int)ny - 1, density_coord.y_to_upper()));
//
//     Real Qll = _density(low_x,   low_y);
//     Real Qul = _density(upper_x, low_y);
//     Real Qlu = _density(low_x,   upper_y);
//     Real Quu = _density(upper_x, upper_y);
//
//     Real alpha_x = density_coord.vx - low_x;
//     Real alpha_y = density_coord.vy - low_y;
//
//     return (1 - alpha_x) * (1 - alpha_y) * Qll +
//            alpha_x       * (1 - alpha_y) * Qul +
//            (1 - alpha_x) * alpha_y       * Qlu +
//            alpha_x       * alpha_y       * Quu;
// }

Real MAC_Grid::interpolate_density(_2d_vec density_world_coord) {
    _2d_vec grid_coord = coord_trans_rtom(density_world_coord);
    if (grid_coord.vx<0||grid_coord.vy<0||grid_coord.vx>nx||grid_coord.vy>ny)
        { return 0.0; }
    _2d_vec density_coord = grid_coord - _2d_vec(0.5,0.5);
    int low_x = density_coord.x_to_lower(), low_y = density_coord.y_to_lower(),
    upper_x = density_coord.x_to_upper(), upper_y = density_coord.y_to_upper();
    Real Qll = (low_x < 0||low_y<0) ? 0 : _density(low_x, low_y);
    Real Qul = (upper_x >= nx||low_y<0)? 0 : _density(upper_x, low_y);
    Real Qlu = (low_x < 0||upper_y>=ny) ? 0 : _density(low_x, upper_y);
    Real Quu = (upper_x >= nx||upper_y>=ny) ? 0 : _density(upper_x, upper_y);
    Real alpha_x = density_coord.vx - low_x;
    Real alpha_y = density_coord.vy - low_y;
    return (1 - alpha_x) * (1 - alpha_y) * Qll
    + alpha_x * (1 - alpha_y) * Qul
    + (1 - alpha_x) * alpha_y * Qlu
    + alpha_x * alpha_y * Quu;
}

_2d_vec MAC_Grid::interpolate_delta_velocity(const _2d_vec &real_world_pos)
{
    _2d_vec ans(0,0);
    _2d_vec grid_coord = coord_trans_rtom(real_world_pos);
    if (grid_coord.vx<0||grid_coord.vy<0||grid_coord.vx>nx||grid_coord.vy>ny) {
        // RZ_FATAL("failed to interpolate velocity");
        RZ_INFO("real_world_pos:vx,vy:",real_world_pos.vx,real_world_pos.vy);
        RZ_INFO("boundary handled:vx,vy:",grid_coord.vx,grid_coord.vy);
        return ans;
    }
    _2d_vec u_coord(grid_coord.vx,grid_coord.vy-0.5f);
    _2d_vec v_coord(grid_coord.vx-0.5f,grid_coord.vy);
    ans.vx = interpolate_u_velocity(u_coord, _delta_u);
    ans.vy = interpolate_v_velocity(v_coord, _delta_v);


    return ans;
}

_2d_vec MAC_Grid::interpolate_velocity(const _2d_vec& real_world_pos) {
    _2d_vec ans(0,0);
    _2d_vec grid_coord = coord_trans_rtom(real_world_pos);
    if (grid_coord.vx<0||grid_coord.vy<0||grid_coord.vx>nx||grid_coord.vy>ny) {
        // RZ_FATAL("failed to interpolate velocity");
        RZ_INFO("real_world_pos:vx,vy:",real_world_pos.vx,real_world_pos.vy);
        RZ_INFO("boundary handled:vx,vy:",grid_coord.vx,grid_coord.vy);
        return ans;
    }
    _2d_vec u_coord(grid_coord.vx,grid_coord.vy-0.5f);
    _2d_vec v_coord(grid_coord.vx-0.5f,grid_coord.vy);
    ans.vx = interpolate_u_velocity(u_coord,_u);
    ans.vy = interpolate_v_velocity(v_coord,_v);
    return ans;
}

_2d_vec MAC_Grid::coord_trans_rtom(const _2d_vec& real_world_pos) {
    return _2d_vec(real_world_pos.vx/dx,real_world_pos.vy/dx);
}