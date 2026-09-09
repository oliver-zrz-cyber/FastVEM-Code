#pragma once
#include "dfn.h"
class MAC_Grid
{
public:
    MAC_Grid(size_t nx = 0, size_t ny = 0)
    : nx(nx), ny(ny), _p(nx,ny,0.0),_u(nx+1,ny,0.0),_v(nx,ny+1,0.0),_delta_u(nx+1,ny,0.0),_delta_v(nx,ny+1,0.0),_density(nx,ny,0.0) ,dx(1.0/nx){}
    Array2D<Real> _p,_u,_v,_density,_delta_u,_delta_v;
    Real dx;
    _2d_vec interpolate_velocity(const _2d_vec& real_world_pos);
    _2d_vec coord_trans_rtom(const _2d_vec& real_world_pos);
    Real interpolate_u_velocity(_2d_vec u_coord,const Array2D<Real>& inter_v) ;
    Real interpolate_v_velocity(_2d_vec v_coord,const Array2D<Real>& inter_v) ;
    Real interpolate_density(_2d_vec density_world_coord) ;
    _2d_vec interpolate_delta_velocity(const _2d_vec& real_world_pos);
    int nx,ny;
private:


};


