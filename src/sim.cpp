#include "sim.h"
#include <GLFW/glfw3.h>
#include <iostream>
#include <cmath>
#include <boost/mpl/size.hpp>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include "AbSolver.h"
#include "vem2d.h"
#include "stb_image_write.h"
#include "Render3d.h"
static std::vector<FLIP_Particles> flip_particles;
static Real dt = 0.01;
// 全局指针，供回调访问
RGB_Screen* rgb_grid = nullptr;
bool g_should_close = false;
bool g_mouse_down = false;
Real g_last_update_time = 0.0;
Real g_mouse_x = 0.0, g_mouse_y = 0.0;
void remove_out_of_bounds_particles(std::vector<FLIP_Particles>& particles, Real dx)
{
    const Real margin = 0.01f * dx * 0.99f;
    const Real xmin = margin, ymin = margin;
    const Real xmax = 1.0f - margin, ymax = 1.0f - margin;

    particles.erase(
        std::remove_if(particles.begin(), particles.end(),
            [&](const FLIP_Particles& p) {
                return (p.pos.vx < xmin || p.pos.vx > xmax ||
                        p.pos.vy < ymin || p.pos.vy > ymax);
            }),
        particles.end());
}

Real Weight_function(_2d_vec& xp,_2d_vec& xc,Real dx)
{
    Real ans = 1.0;
    ans *= std::fmax(1.0f-std::abs((xp.vx-xc.vx)/dx),0.0f);
    ans *= std::fmax(1.0f-std::abs((xp.vy-xc.vy)/dx),0.0f);
    return ans;
}

bool isSymmetricEfficient(const Eigen::SparseMatrixCr& A, Real tol = 1e-6f) {
    for (int k=0; k<A.outerSize(); ++k)
        for (Eigen::SparseMatrixCr::InnerIterator it(A,k); it; ++it) {
            Real a_ij = it.value();
            Real a_ji = A.coeff(it.col(), it.row());
            if (std::abs(a_ij - a_ji) > tol)
                return false;
        }
    return true;
}
bool isSPD(const Eigen::SparseMatrixCr& A) {
    // // 先快速检查是否对称
    if (!isSymmetricEfficient(A)) {
        RZ_FATAL("A is not symmetric!");
        return false;
    }
    // // 尝试做Cholesky分解
    Eigen::SimplicialLLT<Eigen::SparseMatrixCr> llt;
    llt.compute(A);
    return (llt.info() == Eigen::Success);
}
// 鼠标点击回调
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            g_mouse_down = true;
            double mouse_x = g_mouse_x;
            double mouse_y = g_mouse_y;
            glfwGetCursorPos(window, &mouse_x, &mouse_y);
            g_mouse_x = static_cast<Real>(mouse_x);
            g_mouse_y = static_cast<Real>(mouse_y);
            g_last_update_time = glfwGetTime(); // 记录时间
        } else if (action == GLFW_RELEASE) {
            g_mouse_down = false;
        }
    }
}

// 鼠标移动回调
void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    g_mouse_x = static_cast<Real>(xpos);
    g_mouse_y = static_cast<Real>(ypos);
}


// 键盘回调，捕获 ESC
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        g_should_close = true;
    }
}

static void hsv2rgb(Real h, Real s, Real v, uint8_t &r, uint8_t &g, uint8_t &b) {
    Real c = v * s;
    Real x = c * (1 - fabsf(fmodf(h / 60.0f, 2) - 1));
    Real m = v - c;
    Real r_, g_, b_;
    if (h < 60)      { r_ = c; g_ = x; b_ = 0; }
    else if (h < 120){ r_ = x; g_ = c; b_ = 0; }
    else if (h < 180){ r_ = 0; g_ = c; b_ = x; }
    else if (h < 240){ r_ = 0; g_ = x; b_ = c; }
    else if (h < 300){ r_ = x; g_ = 0; b_ = c; }
    else             { r_ = c; g_ = 0; b_ = x; }

    r = static_cast<uint8_t>((r_ + m) * 255);
    g = static_cast<uint8_t>((g_ + m) * 255);
    b = static_cast<uint8_t>((b_ + m) * 255);
}

Real arrow_scale = 30.0f;
Real max_speed   = 1.0f;

auto clamp01 = [](Real x) { return std::fmax(0.0f, std::fmin(1.0f, x)); };

void render_loop(RGB_Screen &screen) {

    MAC_Grid mg(50,50);
    init_flip(mg);
    rgb_grid = &screen; // 设置全局指针
    if (!glfwInit()) {
        std::cout << "Failed to initialize GLFW\n";
        return;
    }
    GLFWwindow* window = glfwCreateWindow(screen.width, screen.height, "MAC Grid Renderer", NULL, NULL);
    if (!window) {
        glfwTerminate();
        std::cerr << "Failed to create window\n";
        return;
    }

    glfwMakeContextCurrent(window);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetKeyCallback(window, key_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);

    int frame = 0;
    while (!glfwWindowShouldClose(window) && !g_should_close) {
        RZ_INFO("runnnnnn frame: ",frame);
        glClear(GL_COLOR_BUFFER_BIT);
        // simlation
        // === 鼠标持续加密度 ===
        if (g_mouse_down) {
            Real now = glfwGetTime();
            Real dt = now - g_last_update_time;
            g_last_update_time = now;

            int pixel_x = static_cast<int>(g_mouse_x);
            int pixel_y = rgb_grid->height - static_cast<int>(g_mouse_y);
            int i_center = static_cast<int>((Real)pixel_x / screen.width * mg._density.nx());
            int j_center = static_cast<int>((Real)pixel_y / screen.height * mg._density.ny());

            // 9点邻域：中心 + 上下左右 + 四个对角
            Real sigma = 1.0f;  // 控制分布宽度，越大扩散越平滑
            Real two_sigma2 = 2.0f * sigma * sigma;

            for (int di = -1; di <= 1; ++di) {
                for (int dj = -1; dj <= 1; ++dj) {
                    int i = i_center + di;
                    int j = j_center + dj;
                    if (i >= 0 && i < mg._density.nx() &&
                        j >= 0 && j < mg._density.ny()) {

                        // 计算距离
                        Real dist2 = static_cast<Real>(di * di + dj * dj);
                        // 高斯权重
                        Real weight = std::exp(-dist2 / two_sigma2);

                        Real& cell = mg._density(i, j);
                        cell += static_cast<Real>(dt * 5 * weight);  // 加权加密度
                        if (cell > 1.0f) cell = 1.0f;
                        }
                }
            }
        }
        remove_out_of_bounds_particles(flip_particles,mg.dx);
        // compute dt ...
        simple_CFL(mg);
        // PtoG
        ptogrid(mg); // ptog's boundary!!! and old u 应该用 ptog之后的 也就是 delta是因为中间解算导致的增量
        // add external force
        add_external_force(mg);
        // diffuse velocity

        // pressure projection
        grid_pressure_projection(mg);

        // advect velocity
        // velocity_advection(mg);
        gridtop(mg,0.05);// bigger for pic
        flip_advection(mg);
        // advect density
        density_advection(mg);
        // damp density

        // diffuse density



        // === 根据 mg._density 渲染灰度图 (像素反查) ===
        int nx = mg._density.nx();
        int ny = mg._density.ny();

        for (int sy = 0; sy < screen.height; ++sy) {
            // 计算该像素落在哪个 cell
            Real j = ((Real)sy / (Real)screen.height * (Real)ny);
            if (j >= (Real)ny) j = (Real)ny - 1;
            for (int sx = 0; sx < screen.width; ++sx) {
                Real i = ((Real)sx / (Real)screen.width * (Real)nx);
                if (i >= (Real)nx) i = (Real)nx - 1;
                _2d_vec world_coord(i*mg.dx,j*mg.dx);
                Real d = mg.interpolate_density(world_coord);
                // std::cout<<d<<"\n";
                Real clamped = std::fmax(0.0f, std::fmin(1.0f, d));
                unsigned char gray = static_cast<unsigned char>(255 * (1.0f - clamped));

                int idx = (sy * screen.width + sx) * 3;
                screen.pixels[idx + 0] = gray;
                screen.pixels[idx + 1] = gray;
                screen.pixels[idx + 2] = gray;
            }
        }
        //  渲染速度
        for (auto &p : flip_particles) {
            Real speed = std::sqrt(p.vec.vx * p.vec.vx + p.vec.vy * p.vec.vy);
            Real t = clamp01(speed / max_speed);

            Real hue = (1.0f - t) * 240.0f;
            uint8_t r, g, b;
            hsv2rgb(hue, 1.0f, 1.0f, r, g, b);

            int px = static_cast<int>(p.pos.vx * screen.width);
            int py = static_cast<int>(p.pos.vy * screen.height);
            if (px < 0 || px >= screen.width || py < 0 || py >= screen.height) continue;

            int idx = (py * screen.width + px) * 3;
            screen.pixels[idx + 0] = r;
            screen.pixels[idx + 1] = g;
            screen.pixels[idx + 2] = b;

            int tx = px + static_cast<int>(p.vec.vx * arrow_scale);
            int ty = py + static_cast<int>(p.vec.vy * arrow_scale);

            int dx = abs(tx - px), sx = px < tx ? 1 : -1;
            int dy = -abs(ty - py), sy = py < ty ? 1 : -1;
            int err = dx + dy, e2;
            int x = px, y = py;

            while (true) {
                if (x >= 0 && x < screen.width && y >= 0 && y < screen.height) {
                    int idx2 = (y * screen.width + x) * 3;
                    screen.pixels[idx2 + 0] = r;
                    screen.pixels[idx2 + 1] = g;
                    screen.pixels[idx2 + 2] = b;
                }
                if (x == tx && y == ty) break;
                e2 = 2 * err;
                if (e2 >= dy) { err += dy; x += sx; }
                if (e2 <= dx) { err += dx; y += sy; }
            }
        }
        // 渲染像素数组
        glRasterPos2f(-1.0f, -1.0f);
        glDrawPixels(screen.width, screen.height, GL_RGB, GL_UNSIGNED_BYTE, screen.pixels.data());

        glfwSwapBuffers(window);
        glfwPollEvents();

        frame++;
    }

    glfwDestroyWindow(window);
    glfwTerminate();
}


void add_external_force(MAC_Grid &mac)
{
    size_t nx = mac._density.nx(),ny = mac._density.ny();
    for (size_t i=0; i<nx; ++i)
        for (size_t j=1; j<ny; ++j)
            mac._v(i,j) -= (mac._density(i,j-1) + mac._density(i,j)) / 2.0f * 9.8f * dt ;
            // mac._v(i,j) -= 9.8f * dt ;
}

void velocity_advection(MAC_Grid &mac)// 考虑边界 & second order RK
{
    size_t nx = mac._density.nx(),ny = mac._density.ny();
    // u adv
    Array2D<Real> advected_u(nx+1,ny,0.0);
    for (size_t i=0; i< nx + 1; ++i)
        for (size_t j=0; j< ny; ++j) {
            // if (i==0||i==nx) {
            //     advected_u(i,j) = 0.0;
            //     continue;
            // }
            Real px = i * mac.dx;
            Real py = j * mac.dx + 0.5 * mac.dx;
            _2d_vec world_coord(px,py);
            _2d_vec vl = mac.interpolate_velocity(world_coord);
            _2d_vec mid_p_world_coord = world_coord - vl*dt*0.5;
            _2d_vec mid_p_velocity = mac.interpolate_velocity(mid_p_world_coord);
            _2d_vec start_p_world_coord = world_coord - mid_p_velocity*dt;
            _2d_vec start_velocity = mac.interpolate_velocity(start_p_world_coord);
            advected_u(i,j) = start_velocity.vx;
        }
    mac._u.swap(advected_u);
    // v adv
    Array2D<Real> advected_v(nx,ny+1,0.0);
    for (size_t i=0; i< nx; ++i)
        for (size_t j=0; j< ny + 1; ++j) {
            // if (j==0||j==ny) {
            //     advected_v(i,j) = 0.0;
            //     continue;
            // }
            Real px = i * mac.dx + 0.5 * mac.dx;
            Real py = j * mac.dx;
            _2d_vec world_coord(px,py);
            _2d_vec vl = mac.interpolate_velocity(world_coord);
            _2d_vec mid_p_world_coord = world_coord - vl*dt*0.5;
            _2d_vec mid_p_velocity = mac.interpolate_velocity(mid_p_world_coord);
            _2d_vec start_p_world_coord = world_coord - mid_p_velocity*dt;
            _2d_vec start_velocity = mac.interpolate_velocity(start_p_world_coord);
            advected_v(i,j) = start_velocity.vy;
        }
    mac._v.swap(advected_v);
}

void save_pressure_png(
    const MAC_Grid &mac,
    const Eigen::VectorXr &pressure,
    const std::string &filename,
    int width = 400,
    int height = 400)
{
    const int nx = mac.nx;
    const int ny = mac.ny;

    if (pressure.size() != nx * ny) {
        std::cerr << "[save_pressure_png] pressure size mismatch!\n";
        return;
    }

    // 计算压力范围
    Real p_min = pressure.minCoeff();
    Real p_max = pressure.maxCoeff();
    Real range = (p_max - p_min);
    if (range < 1e-12) range = 1.0;

    // 在右侧留出 color bar 的宽度（20像素）
    const int colorbar_width = 20;
    const int total_width = width + colorbar_width;

    std::vector<unsigned char> img(total_width * height * 3, 255);

    // 主图绘制
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            Real gx = (Real)i / (width - 1) * nx;
            Real gy = (Real)j / (height - 1) * ny;

            int ix = std::clamp((int)std::floor(gx), 0, nx - 1);
            int iy = std::clamp((int)std::floor(gy), 0, ny - 1);

            Real val = (pressure(ix * nx + iy) - p_min) / range;
            val = std::clamp<Real>(val, 0.0, 1.0);

            unsigned char r = (unsigned char)(val * 255);
            unsigned char g = 0;
            unsigned char b = (unsigned char)((1.0 - val) * 255);

            int idx = ((height - 1 - j) * total_width + i) * 3;
            img[idx + 0] = r;
            img[idx + 1] = g;
            img[idx + 2] = b;
        }
    }

    // 绘制 color bar （右侧 20 像素）
    for (int j = 0; j < height; ++j) {
        Real val = (Real)j / (height - 1); // 从底到顶
        unsigned char r = (unsigned char)(val * 255);
        unsigned char g = 0;
        unsigned char b = (unsigned char)((1.0 - val) * 255);

        for (int i = width; i < total_width; ++i) {
            int idx = ((height - 1 - j) * total_width + i) * 3;
            img[idx + 0] = r;
            img[idx + 1] = g;
            img[idx + 2] = b;
        }
    }

    // 保存 PNG
    if (!stbi_write_png(filename.c_str(), total_width, height, 3, img.data(), total_width * 3))
        std::cerr << "[save_pressure_png] Failed to write PNG: " << filename << "\n";
    else {
        std::cout << "[save_pressure_png] Saved pressure visualization to " << filename << "\n";
        std::cout << "Color bar scale:\n";
        std::cout << "  Top (red):   " << p_max << "\n";
        std::cout << "  Middle:      " << (p_max + p_min) * 0.5 << "\n";
        std::cout << "  Bottom (blue): " << p_min << "\n";
    }
}
void grid_pressure_projection(MAC_Grid &mac)
{
    // use Real
    const int matrix_dim = mac.nx*mac.ny;
    Eigen::SparseMatrixR A(matrix_dim,matrix_dim);
    Eigen::VectorXr b(matrix_dim);
    std::vector<Eigen::Triplet<GlobalReal>> A_triplets;
    Real scale_dx = 1.0/mac.dx;
    Real scale_A = dt/mac.dx/mac.dx; // suppose density is 1
    const auto& _u = mac._u;
    const auto& _v = mac._v;
    // set A
    for (int i = 0;i<mac.nx;i++)
        for (int j=0;j<mac.ny;j++) {
            int row  = i*mac.nx+j;
            Real l,r,u,bl,c;
            if (i!=0) l = -scale_A;
            else l = 0.0;
            if (i!=mac.nx-1) r = -scale_A;
            else r = 0.0;
            if (j!=0)bl = -scale_A;
            else bl = 0.0;
            if (j!=mac.ny-1) u = -scale_A;
            else u = 0.0;
            c = -(l+r+u+bl);

            int col_c = i*mac.nx+j;
            A_triplets.emplace_back(row,col_c,c);

            int col_l = (i-1)*mac.nx+j;
            if ((i-1)>=0)A_triplets.emplace_back(row,col_l,l);

            int col_r = (i+1)*mac.nx+j;
            if (i+1<mac.nx)A_triplets.emplace_back(row,col_r,r);

            int col_b = i*mac.nx+j-1;
            if ((j-1)>=0)A_triplets.emplace_back(row,col_b,bl);

            int col_u = i*mac.nx+j+1;
            if ((j+1)<mac.ny)A_triplets.emplace_back(row,col_u,u);
        }
    A.setFromTriplets(A_triplets.begin(),A_triplets.end());

    // RZ_ASSERT(isSymmetricEfficient(A),"A not sym!");
    // set b
    for (int i = 0;i<mac.nx;i++)
        for (int j=0;j<mac.ny;j++) {
            b(i*mac.nx+j) = -scale_dx *
                (_u(i+1,j) - _u(i,j)+_v(i,j+1)-_v(i,j));
        }
    // count solid boundary for b
    for (int i = 0;i<mac.nx;i++)
        for (int j=0;j<mac.ny;j++) {
            if (i==0) b(i*mac.nx+j) -= scale_dx * _u(i,j);
            if (i==mac.nx-1)b(i*mac.nx+j) -= scale_dx * _u(i+1,j);
            if (j==0) b(i*mac.nx+j) -= scale_dx * _v(i,j);
            if (j==mac.ny-1)b(i*mac.nx+j) -= scale_dx * _v(i,j+1);
        }
    // //solve

    // Eigen::ConjugateGradient<Eigen::SparseMatrixCr, Eigen::Lower|Eigen::Upper, Eigen::IncompleteCholesky<Real>> pcg;
    // pcg.setMaxIterations(10000);
    // pcg.setTolerance(1e-5);
    // pcg.compute(A);
    // Eigen::VectorXr pressure = pcg.solve(b);
    // if (pcg.info() == Eigen::Success) {
    //     std::cout << "PCG 收敛成功！迭代次数：" << pcg.iterations()
    //               << "，最终误差：" << pcg.error() << std::endl;
    // } else {
    //     std::cerr << "PCG 未收敛，状态码：" << pcg.info()
    //               << "，迭代次数：" << pcg.iterations()
    //               << "，最终误差：" << pcg.error() << std::endl;
    //     RZ_ASSERT(false,"PCG 不收敛");
    // }

    Eigen::VectorXr pressure = amgclSolve(A,b,1e-3, -1, 10000);
    save_pressure_png(mac, pressure, "/home/lhw/桌面/VortexCoupling/PIC/RzPIC/outputs/solutionpic.png", 400, 400);
    exit(0);
    // apply pressure
    for (int i=0;i<mac.nx+1;++i)
        for (int j=0;j<mac.ny;++j) {
            if (i==0||i==mac.nx)mac._u(i,j) = 0;
            else {
                mac._u(i,j) -= Real(pressure(i*mac.nx+j) - pressure((i-1) * mac.nx + j)) * dt / mac.dx;
            }
        }
    for (int i=0;i<mac.nx;++i)
        for (int j=0;j<mac.ny+1;++j) {
            if (j==0||j==mac.ny)mac._v(i,j) = 0;
            else {
                mac._v(i,j) -= Real(pressure(i*mac.nx+j) - pressure(i * mac.nx + j-1)) * dt / mac.dx;
            }
        }

}
void density_advection(MAC_Grid &mac)
{
    size_t nx = mac._density.nx(),ny = mac._density.ny();
    Array2D<Real> advected_density(nx,ny,0.0);
    for (size_t i=0; i< nx; ++i)
        for (size_t j=0; j< ny; ++j) {
            Real px = i * mac.dx + 0.5 * mac.dx;
            Real py = j * mac.dx + 0.5 * mac.dx;
            _2d_vec world_coord(px,py);
            _2d_vec vl = mac.interpolate_velocity(world_coord);
            _2d_vec mid_p_world_coord = world_coord - vl*dt*0.5;
            _2d_vec mid_p_velocity = mac.interpolate_velocity(mid_p_world_coord);
            _2d_vec start_p_world_coord = world_coord - mid_p_velocity*dt;
            advected_density(i,j) = mac.interpolate_density(start_p_world_coord);
        }
    mac._density.swap(advected_density);
}

void simple_CFL(MAC_Grid &mac) {
    dt = 0.01;
    size_t nx = mac._density.nx(),ny = mac._density.ny();
    for (size_t i=1; i< nx ; ++i)
        for (size_t j=1; j< ny; ++j) {
            dt = std::min(1.0f * mac.dx / (std::fabs(mac._v(i,j))+1e-5f),dt);
            dt = std::min(1.0f * mac.dx / (std::fabs(mac._u(i,j))+1e-5f),dt);
        }
    for (auto &p:flip_particles) {
        dt = std::min(1.0f * mac.dx / (std::fabs(p.vec.vx)+1e-5f),dt);
        dt = std::min(1.0f * mac.dx / (std::fabs(p.vec.vy)+1e-5f),dt);
    }
}


void init_flip(MAC_Grid &mac)
{
    flip_particles.clear();

    // 每个cell内采样3x3均匀分布的点
    const int samples_per_axis = 3;
    const Real spacing = mac.dx / samples_per_axis;

    for (int i = 0; i < mac.nx; ++i) {
        for (int j = 0; j < mac.ny; ++j) {
            // 左下角起点 (cell 的物理坐标原点)
            Real cell_origin_x = i * mac.dx;
            Real cell_origin_y = j * mac.dx;

            for (int si = 0; si < samples_per_axis; ++si) {
                for (int sj = 0; sj < samples_per_axis; ++sj) {
                    FLIP_Particles p;

                    // 均匀采样，取每个小格子的中心
                    Real px = cell_origin_x + (si + 0.5f) * spacing;
                    Real py = cell_origin_y + (sj + 0.5f) * spacing;

                    p.pos = _2d_vec(px, py); // 世界坐标
                    p.vec = _2d_vec(0.0f, 0.0f); // 初速度设为0
                    if (i>15&&i<35&&j>45&&j<50) {
                        p.vec = _2d_vec(0.0f, -3.0f);
                    }
                    flip_particles.push_back(p);
                }
            }
        }
    }
}
// 存储每个cell中包含的粒子索引
static std::vector<std::vector<size_t>> grid_particle_indices;

void build_grid_particle_map(const MAC_Grid &mac)
{
    size_t nx = mac._density.nx();
    size_t ny = mac._density.ny();

    // 每个格子一个vector
    grid_particle_indices.clear();
    grid_particle_indices.resize(nx * ny);

    for (size_t p_idx = 0; p_idx < flip_particles.size(); ++p_idx) {
        const auto &p = flip_particles[p_idx];

        // 计算粒子所在的cell索引
        int i = static_cast<int>(p.pos.vx / mac.dx);
        int j = static_cast<int>(p.pos.vy / mac.dx);

        // 边界检查，防止粒子越界
        if (i >= 0 && i < (int)nx && j >= 0 && j < (int)ny) {
            size_t cell_index = i + j * nx;
            grid_particle_indices[cell_index].push_back(p_idx);
        }
    }
}

// 辅助函数：查询某个cell内的粒子索引
const std::vector<size_t>& get_particles_in_cell(size_t i, size_t j, size_t nx)
{
    size_t cell_index = i + j * nx;
    return grid_particle_indices[cell_index];
}

void ptogrid(MAC_Grid &mac)
{
    size_t nx = mac._density.nx(),ny = mac._density.ny();
    build_grid_particle_map(mac);
    // u ptog

    auto& _u =mac._u;
    _u.fill(0.0);
    for (size_t i=1; i< nx; ++i)
        for (size_t j=0; j< ny; ++j) {
            Real px = i * mac.dx;
            Real py = j * mac.dx + 0.5 * mac.dx;
            _2d_vec world_coord(px,py);
            Real sum_w = 0.0;
            Real sum_u = 0.0;
            for (int di = -1; di <= 0; ++di) {
                for (int dj = -1; dj <= 1; ++dj) {
                    int ci = static_cast<int>(i) + di;
                    int cj = static_cast<int>(j) + dj;
                    if (ci >= 0 && ci < (int)nx && cj >= 0 && cj < (int)ny) {
                        size_t cell_index = ci + cj * nx;
                        auto &particles = grid_particle_indices[cell_index];
                        for (auto& pid:particles) {
                            auto& p = flip_particles[pid];
                            Real w = Weight_function(p.pos,world_coord,mac.dx);
                            sum_w += w;
                            sum_u += w*p.vec.vx;
                        }
                    }
                }
            }
            if (sum_w > 1e-5f)
                _u(i,j) = sum_u/sum_w;
            else
                _u(i,j) = 0.0;
        }
    mac._delta_u =  (mac._u);
    // v ptog

    auto& _v = mac._v;
    _v.fill(0.0);

    for (size_t i = 0; i < nx; ++i) {
        for (size_t j = 1; j < ny; ++j) {
            // v-grid 世界坐标（位于 cell 水平边中心）
            Real px = i * mac.dx + 0.5f * mac.dx;
            Real py = j * mac.dx;
            _2d_vec world_coord(px, py);

            Real sum_w = 0.0f;
            Real sum_v = 0.0f;
            for (int di = -1; di <= 1; ++di) {
                for (int dj = -1; dj <= 0; ++dj) {
                    int ci = static_cast<int>(i) + di;
                    int cj = static_cast<int>(j) + dj;
                    if (ci >= 0 && ci < (int)nx && cj >= 0 && cj < (int)ny) {
                        size_t cell_index = ci + cj * nx;
                        auto &particles = grid_particle_indices[cell_index];
                        for (auto& pid : particles) {
                            auto &p = flip_particles[pid];
                            Real w = Weight_function(p.pos, world_coord, mac.dx);
                            sum_w += w;
                            sum_v += w * p.vec.vy;
                        }
                    }
                }
            }
            if (sum_w > 1e-5f)
                _v(i, j) = sum_v / sum_w;
            else
                _v(i, j) = 0.0f;
        }
    }
    mac._delta_v = (mac._v);
}

void gridtop(MAC_Grid &mac,Real flip_alpha)
{
    mac._delta_u = mac._u - mac._delta_u;
    mac._delta_v = mac._v - mac._delta_v;
    // rk2
    for (auto&p:flip_particles) {
        // p.vec = mac.interpolate_velocity(p.pos);
        // auto temp = mac.interpolate_velocity(p.pos) * flip_alpha
        // + (p.vec + mac.interpolate_delta_velocity(p.pos))*(1 - flip_alpha);
        // auto midpos = p.pos - temp * (dt / 2.0);
        p.vec = mac.interpolate_velocity(p.pos) * flip_alpha
        + (p.vec + mac.interpolate_delta_velocity(p.pos))*(1 - flip_alpha);
    }
}

void flip_advection(MAC_Grid &mac)
{
    for (auto &p : flip_particles) {
        p.pos.vx += dt * p.vec.vx;
        p.pos.vy += dt * p.vec.vy;
    }
}


void render_loop_vem(RGB_Screen &screen,std::string domain)
{
    RZ_INFO("-------------------------------");
    VEM_Domain vem_simulator;
    vem_simulator.read_from_file(domain);
    vem_simulator.init_flip_particles();
    vem_simulator.init();
    vem_simulator.generate_elements();
    vem_simulator.assemble_global_stiffness_matrix();
    vem_simulator.assemble_global_stiffness_matrix_k1();
    RZ_INFO("stiffness matrix");
    rgb_grid = &screen; // 设置全局指针
    if (!glfwInit()) {
        std::cout << "Failed to initialize GLFW\n";
        return;
    }
    GLFWwindow* window = glfwCreateWindow(screen.width, screen.height, "MAC Grid Renderer", NULL, NULL);
    if (!window) {
        glfwTerminate();
        std::cerr << "Failed to create window\n";
        return;
    }

    glfwMakeContextCurrent(window);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetKeyCallback(window, key_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);

    int frame = 0;
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    while (!glfwWindowShouldClose(window) && !g_should_close) {
        glClear(GL_COLOR_BUFFER_BIT);
        std::fill(screen.pixels.begin(), screen.pixels.end(), 255);
        // PtoG
        vem_simulator.PtoG();
        // solve pressure
        RZ_INFO("runnnnnn frame:",frame);
        vem_simulator.assemble_weakform_rhs(dt);
        vem_simulator.solve_pic();
        // vem_simulator.save_solution_png("/home/lhw/桌面/VortexCoupling/PIC/RzPIC/outputs/solutionvem.png");
        // exit(0);
        // GtoP
        vem_simulator.GtoP(0.95,dt);// bigger for pic
        // advection
        vem_simulator.flip_advection(dt);

        //  渲染速度
        for (auto &p : vem_simulator.flip_particles) {
            if (p.deleted)continue;
            Real speed = std::sqrt(p.vec.vx * p.vec.vx + p.vec.vy * p.vec.vy);
            Real t = clamp01(speed / max_speed);

            Real hue = (1.0f - t) * 240.0f;
            uint8_t r, g, b;
            hsv2rgb(hue, 1.0f, 1.0f, r, g, b);

            int px = static_cast<int>(p.pos.vx * screen.width);
            int py = static_cast<int>(p.pos.vy * screen.height);
            if (px < 0 || px >= screen.width || py < 0 || py >= screen.height) continue;

            int idx = (py * screen.width + px) * 3;
            screen.pixels[idx + 0] = r;
            screen.pixels[idx + 1] = g;
            screen.pixels[idx + 2] = b;

            int tx = px + static_cast<int>(p.vec.vx * arrow_scale);
            int ty = py + static_cast<int>(p.vec.vy * arrow_scale);

            int dx = abs(tx - px), sx = px < tx ? 1 : -1;
            int dy = -abs(ty - py), sy = py < ty ? 1 : -1;
            int err = dx + dy, e2;
            int x = px, y = py;

            while (true) {
                if (x >= 0 && x < screen.width && y >= 0 && y < screen.height) {
                    int idx2 = (y * screen.width + x) * 3;
                    screen.pixels[idx2 + 0] = r;
                    screen.pixels[idx2 + 1] = g;
                    screen.pixels[idx2 + 2] = b;
                }
                if (x == tx && y == ty) break;
                e2 = 2 * err;
                if (e2 >= dy) { err += dy; x += sx; }
                if (e2 <= dx) { err += dx; y += sy; }
            }
        }
        // 渲染像素数组
        glRasterPos2f(-1.0f, -1.0f);
        glDrawPixels(screen.width, screen.height, GL_RGB, GL_UNSIGNED_BYTE, screen.pixels.data());

        glfwSwapBuffers(window);
        glfwPollEvents();

        frame++;
    }

    glfwDestroyWindow(window);
    glfwTerminate();
}
