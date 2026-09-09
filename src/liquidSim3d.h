#pragma once

#include "mac3d.h"
class simple_liquid {
public:
    void run();
private:
    MAC_Grid3D grid;
    std::vector<FLIP_Particles_3d> particles;// both marker and flip
    void init(int n);
    void init_cube(int n,Real lx,Real ly,Real lz);
    void advection(Real dt);
    void add_external_force(Real dt,_3d_vec f);
    void assemble_and_solve(Real dt);
    void PtoG();
    void GtoP(Real alpha);
    void apply_pressure(Real dt);
    void save_particles(std::string filename);
    Real CFL(Real dt);
    void label_liquid();
};


