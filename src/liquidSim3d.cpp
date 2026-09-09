#include "liquidSim3d.h"
#include <Eigen/Sparse>
#include <fstream>

void simple_liquid::run() {
    init_cube(100,0.25,0.3,0.25);
    int tot_step = 800;
    Real dt = 0.01;
    _3d_vec g(0,-9.8,0);
    for (int k=0;k<tot_step;++k) {
        RZ_INFO("running : ",k);
        GtoP(0.95);
        Real real_dt = CFL(dt);
        advection(real_dt);
        label_liquid();
        PtoG();
        add_external_force(real_dt,g);
        assemble_and_solve(real_dt);
        apply_pressure(real_dt);
        save_particles("/media/lhw/PS1008/SIG_Results/SmokeBins/liquid/liquid_"+std::to_string(k)+".bin");
    }
}

void simple_liquid::init(int n) {
    grid.resize(n);
    _3d_vec center(0.5,0.5,0.5);
    Real r = 0.25;
    Real r2 = r * r;
    Real dx = grid.dx;
    particles.clear();
    for (int i = 0; i < grid.nx; ++i) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int k = 0; k < grid.nz; ++k) {

                // cell center in world space
                _3d_vec cell_center(
                    (i + 0.5) * dx,
                    (j + 0.5) * dx,
                    (k + 0.5) * dx
                );

                _3d_vec diff = cell_center - center;
                Real dist2 = diff.vx * diff.vx + diff.vy * diff.vy + diff.vz * diff.vz;

                if (dist2 <= r2) {
                    // mark this cell as full of water
                    grid._density(i, j, k) = 1.0;

                    // sample 8 particles inside this cell: 2x2x2 layout
                    for (int a = 0; a < 2; ++a) {
                        for (int b = 0; b < 2; ++b) {
                            for (int c = 0; c < 2; ++c) {
                                _3d_vec p(
                                    (i + 0.25 + 0.5 * a) * dx,
                                    (j + 0.25 + 0.5 * b) * dx,
                                    (k + 0.25 + 0.5 * c) * dx
                                );

                                particles.emplace_back(p, _3d_vec(0.0, 0.0, 0.0));
                            }
                        }
                    }
                }
            }
        }
    }

}

void simple_liquid::init_cube(int n, Real sx, Real sy, Real sz) {
    grid.resize(n);
    Real dx = grid.dx;

    particles.clear();

    // 清空 density（防止残留）
    for (int i = 0; i < grid.nx; ++i)
        for (int j = 0; j < grid.ny; ++j)
            for (int k = 0; k < grid.nz; ++k)
                grid._density(i, j, k) = 0.0;

    for (int i = 0; i < grid.nx; ++i) {
        for (int j = 0; j < grid.ny; ++j) {
            for (int k = 0; k < grid.nz; ++k) {

                // cell center
                _3d_vec cell_center(
                    (i + 0.5) * dx,
                    (j + 0.5) * dx,
                    (k + 0.5) * dx
                );

                // 判断是否在左下角 cube 内
                if (cell_center.vx <= sx &&
                    cell_center.vy <= sy &&
                    cell_center.vz <= sz) {

                    // 标记为流体
                    grid._density(i, j, k) = 1.0;

                    // 每个 cell 放 2x2x2 粒子
                    for (int a = 0; a < 2; ++a) {
                        for (int b = 0; b < 2; ++b) {
                            for (int c = 0; c < 2; ++c) {
                                _3d_vec p(
                                    (i + 0.25 + 0.5 * a) * dx,
                                    (j + 0.25 + 0.5 * b) * dx,
                                    (k + 0.25 + 0.5 * c) * dx
                                );

                                particles.emplace_back(p, _3d_vec(0.0, 0.0, 0.0));
                            }
                        }
                    }
                    }
            }
        }
    }
}

void simple_liquid::advection(Real dt) {
    for (auto&particle:particles) {
        particle.pos = particle.pos + particle.vec*dt;
        if (particle.pos.vx<0)particle.pos.vx = 0+1e-9,particle.vec.vx = 0;
        if (particle.pos.vy<0)particle.pos.vy = 0+1e-9,particle.vec.vy = 0;
        if (particle.pos.vz<0)particle.pos.vz = 0+1e-9,particle.vec.vz = 0;
        if (particle.pos.vx>1)particle.pos.vx = 1-1e-9,particle.vec.vx = 0;
        if (particle.pos.vy>1)particle.pos.vy = 1-1e-9,particle.vec.vy = 0;
        if (particle.pos.vz>1)particle.pos.vz = 1-1e-9,particle.vec.vz = 0;
    }
}

void simple_liquid::add_external_force(Real dt, _3d_vec f) {
    const int nx = grid.nx;
    const int ny = grid.ny;
    const int nz = grid.nz;

    auto is_liquid = [&](int i, int j, int k) -> bool {
        if (i < 0 || i >= nx ||
            j < 0 || j >= ny ||
            k < 0 || k >= nz) {
            return false;
        }
        return grid._density(i, j, k) > 0.5;
    };

    // ============================================================
    // u-face: adjacent cells are (i-1,j,k) and (i,j,k)
    // ============================================================
    for (int i = 0; i < nx + 1; ++i) {
        for (int j = 0; j < ny; ++j) {
            for (int k = 0; k < nz; ++k) {
                bool left_liquid  = is_liquid(i - 1, j, k);
                bool right_liquid = is_liquid(i,     j, k);

                if (left_liquid || right_liquid) {
                    grid._u(i, j, k) += dt * f.vx;
                }
            }
        }
    }

    // ============================================================
    // v-face: adjacent cells are (i,j-1,k) and (i,j,k)
    // ============================================================
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny + 1; ++j) {
            for (int k = 0; k < nz; ++k) {
                bool down_liquid = is_liquid(i, j - 1, k);
                bool up_liquid   = is_liquid(i, j,     k);

                if (down_liquid || up_liquid) {
                    grid._v(i, j, k) += dt * f.vy;
                }
            }
        }
    }

    // ============================================================
    // w-face: adjacent cells are (i,j,k-1) and (i,j,k)
    // ============================================================
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            for (int k = 0; k < nz + 1; ++k) {
                bool back_liquid  = is_liquid(i, j, k - 1);
                bool front_liquid = is_liquid(i, j, k);

                if (back_liquid || front_liquid) {
                    grid._w(i, j, k) += dt * f.vz;
                }
            }
        }
    }

    // 可选：重新施加边界法向速度为 0
    for (int j = 0; j < ny; ++j) {
        for (int k = 0; k < nz; ++k) {
            grid._u(0,  j, k) = 0.0;
            grid._u(nx, j, k) = 0.0;
        }
    }

    for (int i = 0; i < nx; ++i) {
        for (int k = 0; k < nz; ++k) {
            grid._v(i, 0,  k) = 0.0;
            grid._v(i, ny, k) = 0.0;
        }
    }

    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            grid._w(i, j, 0)  = 0.0;
            grid._w(i, j, nz) = 0.0;
        }
    }
}

void simple_liquid::assemble_and_solve(Real dt) {
    const int nx = grid.nx;
    const int ny = grid.ny;
    const int nz = grid.nz;
    const Real dx = grid.dx;
    const Real inv_dx = 1.0 / dx;
    const Real scale_A = 1.0 / (dx * dx);
    const Real scale_b = -1.0 / dt; // 为了保证 SPD

    // ============================================================
    // 0. 先施加边界条件：域边界法向速度为 0
    // ============================================================
    for (int j = 0; j < ny; ++j) {
        for (int k = 0; k < nz; ++k) {
            grid._u(0,  j, k) = 0.0;
            grid._u(nx, j, k) = 0.0;
        }
    }

    for (int i = 0; i < nx; ++i) {
        for (int k = 0; k < nz; ++k) {
            grid._v(i, 0,  k) = 0.0;
            grid._v(i, ny, k) = 0.0;
        }
    }

    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            grid._w(i, j, 0)  = 0.0;
            grid._w(i, j, nz) = 0.0;
        }
    }

    // ============================================================
    // 1. helper：用 density > 0.5 判断 liquid
    // ============================================================
    auto is_inside = [&](int i, int j, int k) -> bool {
        return (i >= 0 && i < nx &&
                j >= 0 && j < ny &&
                k >= 0 && k < nz);
    };

    auto is_fluid = [&](int i, int j, int k) -> bool {
        return is_inside(i, j, k) && (grid._density(i, j, k) > 0.5);
    };

    // ============================================================
    // 2. 给 liquid cell 编号
    // ============================================================
    Array3D<int> cell_id(nx, ny, nz, -1);

    int fluid_cell_count = 0;
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            for (int k = 0; k < nz; ++k) {
                if (is_fluid(i, j, k)) {
                    cell_id(i, j, k) = fluid_cell_count++;
                }
            }
        }
    }

    if (fluid_cell_count == 0) {
        // 顺手把压力清零，避免残留旧值
        for (int i = 0; i < nx; ++i)
            for (int j = 0; j < ny; ++j)
                for (int k = 0; k < nz; ++k)
                    grid._p(i, j, k) = 0.0;
        return;
    }

    // ============================================================
    // 3. 分配 A 和 b
    // ============================================================
    Eigen::SparseMatrixCr A(fluid_cell_count, fluid_cell_count);
    Eigen::VectorXr b(fluid_cell_count);
    b.setZero();

    std::vector<Eigen::Triplet<GlobalReal>> triplets;
    triplets.reserve(fluid_cell_count * 7);

    // ============================================================
    // 4. 组装 A 和 b
    // ============================================================
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            for (int k = 0; k < nz; ++k) {

                if (!is_fluid(i, j, k)) continue;

                const int row = cell_id(i, j, k);

                // ------------------------------------------------
                // RHS: divergence of u*
                // ------------------------------------------------
                Real div = (
                    grid._u(i + 1, j, k) - grid._u(i, j, k) +
                    grid._v(i, j + 1, k) - grid._v(i, j, k) +
                    grid._w(i, j, k + 1) - grid._w(i, j, k)
                ) * inv_dx;

                b(row) = scale_b * div;

                // ------------------------------------------------
                // Matrix A
                // ------------------------------------------------
                Real diag = 0.0;

                const int di[6] = {+1, -1,  0,  0,  0,  0};
                const int dj[6] = { 0,  0, +1, -1,  0,  0};
                const int dk[6] = { 0,  0,  0,  0, +1, -1};

                for (int dir = 0; dir < 6; ++dir) {
                    const int ni = i + di[dir];
                    const int nj = j + dj[dir];
                    const int nk = k + dk[dir];

                    if (is_inside(ni, nj, nk)) {
                        if (is_fluid(ni, nj, nk)) {
                            // fluid-fluid coupling
                            const int col = cell_id(ni, nj, nk);
                            triplets.emplace_back(row, col, -scale_A);
                            diag += scale_A;
                        } else {
                            // liquid-air interface: p_air = 0
                            diag += scale_A;
                        }
                    } else {
                        // 域外是固体边界
                        // 这里不引入额外未知量
                        // 边界法向速度已经在前面强制为 0
                    }
                }

                triplets.emplace_back(row, row, diag);
            }
        }
    }

    // ============================================================
    // 5. 构建稀疏矩阵
    // ============================================================
    A.setFromTriplets(triplets.begin(), triplets.end());

    // ============================================================
    // 6. 求解
    // ============================================================
    Eigen::ConjugateGradient<
        Eigen::SparseMatrixCr,
        Eigen::Lower | Eigen::Upper
    > solver;

    solver.setTolerance(1e-6);
    solver.setMaxIterations(1000);

    solver.compute(A);
    if (solver.info() != Eigen::Success) {
        std::cerr << "[ERROR] CG decomposition failed.\n";
        return;
    }

    Eigen::GlobalVectorXr b_global = b.template cast<GlobalReal>();
    Eigen::VectorXr p = solver.solve(b_global).template cast<Real>();
    if (solver.info() != Eigen::Success) {
        std::cerr << "[ERROR] CG solve failed.\n";
        return;
    }

    if (!p.allFinite()) {
        std::cerr << "[ERROR] Pressure contains NaN or Inf.\n";
        return;
    }

    RZ_INFO("max pressure: ", p.maxCoeff());
    RZ_INFO("cg iterations: ", solver.iterations());
    RZ_INFO("cg error: ", solver.error());

    // ============================================================
    // 7. 写回 pressure 到网格
    // ============================================================
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            for (int k = 0; k < nz; ++k) {
                if (cell_id(i, j, k) >= 0) {
                    grid._p(i, j, k) = p(cell_id(i, j, k));
                } else {
                    grid._p(i, j, k) = 0.0;
                }
            }
        }
    }
}

void simple_liquid::PtoG() {
        // 先清空网格速度
    const int nx = grid.nx;
    const int ny = grid.ny;
    const int nz = grid.nz;

    const Real dx = grid.dx;
    const Real inv_dx = 1.0 / dx;

    // ============================================================
    // 0. 清空网格速度
    // ============================================================
    for (int i = 0; i < nx + 1; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
                grid._u(i, j, k) = 0.0;

    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny + 1; ++j)
            for (int k = 0; k < nz; ++k)
                grid._v(i, j, k) = 0.0;

    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz + 1; ++k)
                grid._w(i, j, k) = 0.0;

    // ============================================================
    // 1. 权重累积
    // ============================================================
    Array3D<Real> u_weight(nx + 1, ny, nz, 0.0);
    Array3D<Real> v_weight(nx, ny + 1, nz, 0.0);
    Array3D<Real> w_weight(nx, ny, nz + 1, 0.0);

    // ============================================================
    // 2. helper
    // ============================================================
    auto lerp_weight = [](Real x) -> Real {
        x = std::abs(x);
        return (x < 1.0) ? (1.0 - x) : 0.0;
    };

    auto is_liquid = [&](int i, int j, int k) -> bool {
        if (i < 0 || i >= nx ||
            j < 0 || j >= ny ||
            k < 0 || k >= nz) {
            return false;
        }
        return grid._density(i, j, k) > 0.5;
    };

    // ============================================================
    // 3. splat to u
    //    u-face center = (i*dx, (j+0.5)dx, (k+0.5)dx)
    //    adjacent cells: (i-1,j,k), (i,j,k)
    // ============================================================
    auto splat_to_u = [&](const _3d_vec& pos, Real value) {
        Real gx = pos.vx * inv_dx;
        Real gy = pos.vy * inv_dx - 0.5;
        Real gz = pos.vz * inv_dx - 0.5;

        int i0 = static_cast<int>(std::floor(gx));
        int j0 = static_cast<int>(std::floor(gy));
        int k0 = static_cast<int>(std::floor(gz));

        for (int di = 0; di <= 1; ++di) {
            for (int dj = 0; dj <= 1; ++dj) {
                for (int dk = 0; dk <= 1; ++dk) {
                    int i = i0 + di;
                    int j = j0 + dj;
                    int k = k0 + dk;

                    if (i < 0 || i >= nx + 1 ||
                        j < 0 || j >= ny ||
                        k < 0 || k >= nz)
                        continue;

                    bool left_liquid  = is_liquid(i - 1, j, k);
                    bool right_liquid = is_liquid(i,     j, k);
                    if (!(left_liquid || right_liquid))
                        continue;

                    Real wx = lerp_weight(gx - i);
                    Real wy = lerp_weight(gy - j);
                    Real wz = lerp_weight(gz - k);
                    Real w = wx * wy * wz;

                    if (w <= 0.0) continue;

                    grid._u(i, j, k) += w * value;
                    u_weight(i, j, k) += w;
                }
            }
        }
    };

    // ============================================================
    // 4. splat to v
    //    v-face center = ((i+0.5)dx, j*dx, (k+0.5)dx)
    //    adjacent cells: (i,j-1,k), (i,j,k)
    // ============================================================
    auto splat_to_v = [&](const _3d_vec& pos, Real value) {
        Real gx = pos.vx * inv_dx - 0.5;
        Real gy = pos.vy * inv_dx;
        Real gz = pos.vz * inv_dx - 0.5;

        int i0 = static_cast<int>(std::floor(gx));
        int j0 = static_cast<int>(std::floor(gy));
        int k0 = static_cast<int>(std::floor(gz));

        for (int di = 0; di <= 1; ++di) {
            for (int dj = 0; dj <= 1; ++dj) {
                for (int dk = 0; dk <= 1; ++dk) {
                    int i = i0 + di;
                    int j = j0 + dj;
                    int k = k0 + dk;

                    if (i < 0 || i >= nx ||
                        j < 0 || j >= ny + 1 ||
                        k < 0 || k >= nz)
                        continue;

                    bool down_liquid = is_liquid(i, j - 1, k);
                    bool up_liquid   = is_liquid(i, j,     k);
                    if (!(down_liquid || up_liquid))
                        continue;

                    Real wx = lerp_weight(gx - i);
                    Real wy = lerp_weight(gy - j);
                    Real wz = lerp_weight(gz - k);
                    Real w = wx * wy * wz;

                    if (w <= 0.0) continue;

                    grid._v(i, j, k) += w * value;
                    v_weight(i, j, k) += w;
                }
            }
        }
    };

    // ============================================================
    // 5. splat to w
    //    w-face center = ((i+0.5)dx, (j+0.5)dx, k*dx)
    //    adjacent cells: (i,j,k-1), (i,j,k)
    // ============================================================
    auto splat_to_w = [&](const _3d_vec& pos, Real value) {
        Real gx = pos.vx * inv_dx - 0.5;
        Real gy = pos.vy * inv_dx - 0.5;
        Real gz = pos.vz * inv_dx;

        int i0 = static_cast<int>(std::floor(gx));
        int j0 = static_cast<int>(std::floor(gy));
        int k0 = static_cast<int>(std::floor(gz));

        for (int di = 0; di <= 1; ++di) {
            for (int dj = 0; dj <= 1; ++dj) {
                for (int dk = 0; dk <= 1; ++dk) {
                    int i = i0 + di;
                    int j = j0 + dj;
                    int k = k0 + dk;

                    if (i < 0 || i >= nx ||
                        j < 0 || j >= ny ||
                        k < 0 || k >= nz + 1)
                        continue;

                    bool back_liquid  = is_liquid(i, j, k - 1);
                    bool front_liquid = is_liquid(i, j, k);
                    if (!(back_liquid || front_liquid))
                        continue;

                    Real wx = lerp_weight(gx - i);
                    Real wy = lerp_weight(gy - j);
                    Real wz = lerp_weight(gz - k);
                    Real w = wx * wy * wz;

                    if (w <= 0.0) continue;

                    grid._w(i, j, k) += w * value;
                    w_weight(i, j, k) += w;
                }
            }
        }
    };

    // ============================================================
    // 6. particle -> grid
    // ============================================================
    for (const auto& p : particles) {
        if (p.deleted) continue;

        if (p.pos.vx < 0.0 || p.pos.vx > 1.0 ||
            p.pos.vy < 0.0 || p.pos.vy > 1.0 ||
            p.pos.vz < 0.0 || p.pos.vz > 1.0)
            continue;

        splat_to_u(p.pos, p.vec.vx);
        splat_to_v(p.pos, p.vec.vy);
        splat_to_w(p.pos, p.vec.vz);
    }

    // ============================================================
    // 7. normalize
    // ============================================================
    for (int i = 0; i < nx + 1; ++i) {
        for (int j = 0; j < ny; ++j) {
            for (int k = 0; k < nz; ++k) {
                if (u_weight(i, j, k) > 1e-12) {
                    grid._u(i, j, k) /= u_weight(i, j, k);
                } else {
                    grid._u(i, j, k) = 0.0;
                }
            }
        }
    }

    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny + 1; ++j) {
            for (int k = 0; k < nz; ++k) {
                if (v_weight(i, j, k) > 1e-12) {
                    grid._v(i, j, k) /= v_weight(i, j, k);
                } else {
                    grid._v(i, j, k) = 0.0;
                }
            }
        }
    }

    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            for (int k = 0; k < nz + 1; ++k) {
                if (w_weight(i, j, k) > 1e-12) {
                    grid._w(i, j, k) /= w_weight(i, j, k);
                } else {
                    grid._w(i, j, k) = 0.0;
                }
            }
        }
    }

    // ============================================================
    // 8. 保存 pressure projection 前的网格速度
    // ============================================================
    grid._delta_u = grid._u;
    grid._delta_v = grid._v;
    grid._delta_w = grid._w;

    // ============================================================
    // 9. 边界法向速度置 0
    // ============================================================
    for (int j = 0; j < ny; ++j) {
        for (int k = 0; k < nz; ++k) {
            grid._u(0,  j, k) = 0.0;
            grid._u(nx, j, k) = 0.0;
        }
    }

    for (int i = 0; i < nx; ++i) {
        for (int k = 0; k < nz; ++k) {
            grid._v(i, 0,  k) = 0.0;
            grid._v(i, ny, k) = 0.0;
        }
    }

    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            grid._w(i, j, 0)  = 0.0;
            grid._w(i, j, nz) = 0.0;
        }
    }
}

void simple_liquid::GtoP(Real alpha) {
    // alpha: FLIP blending weight
    // alpha = 1 -> pure FLIP
    // alpha = 0 -> pure PIC

    alpha = std::max<Real>(0.0, std::min<Real>(1.0, alpha));

    for (auto& p : particles) {
        if (p.deleted) continue;

        // PIC velocity: directly interpolate current grid velocity
        _3d_vec pic_vel = grid.interpolate_velocity(p.pos);

        // FLIP velocity: old particle velocity + interpolated grid delta velocity
        _3d_vec flip_vel = p.vec + grid.interpolate_delta_velocity(p.pos);

        // store PIC velocity if needed
        p.pic_vec = pic_vel;

        // blended PIC/FLIP update
        p.vec =  pic_vel * (1.0 - alpha) + flip_vel * alpha;
    }
}

void simple_liquid::save_particles(std::string filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    // 统计有效粒子数
    uint32_t count = 0;
    for (const auto& p : particles) {
        if (!p.deleted) {
            ++count;
        }
    }

    // 写入 count
    out.write(reinterpret_cast<const char*>(&count), sizeof(uint32_t));
    if (!out) {
        std::cerr << "Failed to write particle count to file: " << filename << std::endl;
        return;
    }

    // 写入 xyz，类型必须是 float，和 bin2abc 完全匹配
    for (const auto& p : particles) {
        if (p.deleted) continue;

        float x = static_cast<float>(p.pos.vx);
        float y = static_cast<float>(p.pos.vy);
        float z = static_cast<float>(p.pos.vz);

        out.write(reinterpret_cast<const char*>(&x), sizeof(float));
        out.write(reinterpret_cast<const char*>(&y), sizeof(float));
        out.write(reinterpret_cast<const char*>(&z), sizeof(float));

        if (!out) {
            std::cerr << "Failed while writing particle data to file: " << filename << std::endl;
            return;
        }
    }

    out.close();
}

Real simple_liquid::CFL(Real dt) {
    const Real dx = grid.dx;
    const Real eps = 1e-12;

    Real max_speed = 0.0;

    for (const auto& p : particles) {
        if (p.deleted) continue;

        Real vx = p.vec.vx;
        Real vy = p.vec.vy;
        Real vz = p.vec.vz;

        Real speed = std::sqrt(vx * vx + vy * vy + vz * vz);
        if (speed > max_speed) {
            max_speed = speed;
        }
    }

    // 如果所有粒子都静止，则直接返回原 dt
    if (max_speed < eps) {
        return dt;
    }

    // 要求一个时间步最多跨越 1 个格子
    Real dt_cfl = dx / max_speed;

    // 最大 dt 不超过输入的 dt
    return std::min(dt, dt_cfl);
}

void simple_liquid::label_liquid() {
    const int nx = grid.nx;
    const int ny = grid.ny;
    const int nz = grid.nz;
    const Real dx = grid.dx;
    const Real inv_dx = 1.0 / dx;

    // ------------------------------------------------------------
    // 1. 全部清零
    // ------------------------------------------------------------
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            for (int k = 0; k < nz; ++k) {
                grid._density(i, j, k) = 0.0;
            }
        }
    }

    // clamp helper
    auto clamp_int = [](int x, int lo, int hi) -> int {
        return (x < lo) ? lo : ((x > hi) ? hi : x);
    };

    // ------------------------------------------------------------
    // 2. 根据粒子标记 liquid cell
    // ------------------------------------------------------------
    for (const auto& p : particles) {
        if (p.deleted) continue;

        int i = static_cast<int>(std::floor(p.pos.vx * inv_dx));
        int j = static_cast<int>(std::floor(p.pos.vy * inv_dx));
        int k = static_cast<int>(std::floor(p.pos.vz * inv_dx));

        i = clamp_int(i, 0, nx - 1);
        j = clamp_int(j, 0, ny - 1);
        k = clamp_int(k, 0, nz - 1);

        grid._density(i, j, k) = 1.0;
    }
}


void simple_liquid::apply_pressure(Real dt) {
    const int nx = grid.nx;
    const int ny = grid.ny;
    const int nz = grid.nz;
    const Real dx = grid.dx;
    const Real scale = dt / dx;   // rho = 1

    // ------------------------------------------------------------
    // 1. helper：统一用 density > 0.5 判断 liquid
    // ------------------------------------------------------------
    auto is_inside = [&](int i, int j, int k) -> bool {
        return (i >= 0 && i < nx &&
                j >= 0 && j < ny &&
                k >= 0 && k < nz);
    };

    auto is_fluid = [&](int i, int j, int k) -> bool {
        return is_inside(i, j, k) && (grid._density(i, j, k) > 0.5);
    };

    // ------------------------------------------------------------
    // 2. 用 pressure 更新 u faces
    //    u(i,j,k) 位于 cell (i-1,j,k) 和 (i,j,k) 之间
    // ------------------------------------------------------------
    for (int i = 1; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            for (int k = 0; k < nz; ++k) {
                const bool left_fluid  = is_fluid(i - 1, j, k);
                const bool right_fluid = is_fluid(i,     j, k);

                if (!left_fluid && !right_fluid) continue;

                const Real pL = left_fluid  ? grid._p(i - 1, j, k) : 0.0;
                const Real pR = right_fluid ? grid._p(i,     j, k) : 0.0;

                const Real du = -scale * (pR - pL);
                grid._u(i, j, k) += du;
            }
        }
    }

    // ------------------------------------------------------------
    // 3. 用 pressure 更新 v faces
    //    v(i,j,k) 位于 cell (i,j-1,k) 和 (i,j,k) 之间
    // ------------------------------------------------------------
    for (int i = 0; i < nx; ++i) {
        for (int j = 1; j < ny; ++j) {
            for (int k = 0; k < nz; ++k) {
                const bool down_fluid = is_fluid(i, j - 1, k);
                const bool up_fluid   = is_fluid(i, j,     k);

                if (!down_fluid && !up_fluid) continue;

                const Real pD = down_fluid ? grid._p(i, j - 1, k) : 0.0;
                const Real pU = up_fluid   ? grid._p(i, j,     k) : 0.0;

                const Real dv = -scale * (pU - pD);
                grid._v(i, j, k) += dv;
            }
        }
    }

    // ------------------------------------------------------------
    // 4. 用 pressure 更新 w faces
    //    w(i,j,k) 位于 cell (i,j,k-1) 和 (i,j,k) 之间
    // ------------------------------------------------------------
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            for (int k = 1; k < nz; ++k) {
                const bool back_fluid  = is_fluid(i, j, k - 1);
                const bool front_fluid = is_fluid(i, j, k);

                if (!back_fluid && !front_fluid) continue;

                const Real pB = back_fluid  ? grid._p(i, j, k - 1) : 0.0;
                const Real pF = front_fluid ? grid._p(i, j, k)     : 0.0;

                const Real dw = -scale * (pF - pB);
                grid._w(i, j, k) += dw;
            }
        }
    }

    // ------------------------------------------------------------
    // 5. 记录 FLIP 用的网格速度增量
    //    这里默认：在 PtoG() 结束后
    //    grid._delta_u/v/w 保存的是 projection 前速度
    // ------------------------------------------------------------
    grid._delta_u = grid._u - grid._delta_u;
    grid._delta_v = grid._v - grid._delta_v;
    grid._delta_w = grid._w - grid._delta_w;

    // ------------------------------------------------------------
    // 6. 最后重新施加边界条件：域边界法向速度为 0
    // ------------------------------------------------------------
    for (int j = 0; j < ny; ++j) {
        for (int k = 0; k < nz; ++k) {
            grid._u(0,  j, k) = 0.0;
            grid._u(nx, j, k) = 0.0;

            grid._delta_u(0,  j, k) = 0.0;
            grid._delta_u(nx, j, k) = 0.0;
        }
    }

    for (int i = 0; i < nx; ++i) {
        for (int k = 0; k < nz; ++k) {
            grid._v(i, 0,  k) = 0.0;
            grid._v(i, ny, k) = 0.0;

            grid._delta_v(i, 0,  k) = 0.0;
            grid._delta_v(i, ny, k) = 0.0;
        }
    }

    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            grid._w(i, j, 0)  = 0.0;
            grid._w(i, j, nz) = 0.0;

            grid._delta_w(i, j, 0)  = 0.0;
            grid._delta_w(i, j, nz) = 0.0;
        }
    }
}
