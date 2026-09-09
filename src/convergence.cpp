#include "convergence.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <map>
#include <functional>
#include <limits>
#include <stdexcept>
#include <Eigen/Sparse>
#include <fstream>

namespace {
struct Geometry2D {
    std::string name;
    std::string title;
    Real cx;
    Real cy;
    std::function<bool(Real, Real)> is_inside;
    std::function<Real(Real, Real)> phi;
    std::function<Real(Real, Real)> dphidx;
    std::function<Real(Real, Real)> dphidy;
    std::function<Real(Real, Real)> lap_phi;
};

struct PoissonCase2D {
    std::string name;
    std::string title;
    std::function<Real(Real, Real)> u_exact;
    std::function<Real(Real, Real)> rhs_f;
};

std::vector<Geometry2D> build_geometries() {
    const Real cx = 0.5;
    const Real cy = 0.5;
    const Real r = 0.25;

    const Real a = 0.30;
    const Real b = 0.20;
    const Real theta = M_PI / 6.0;
    const Real c = std::cos(theta);
    const Real s = std::sin(theta);
    const Real ia2 = 1.0 / (a * a);
    const Real ib2 = 1.0 / (b * b);
    const Real A11 = c * c * ia2 + s * s * ib2;
    const Real A22 = s * s * ia2 + c * c * ib2;
    const Real A12 = c * s * (ia2 - ib2);

    const Real sx = 0.28;
    const Real sy = 0.20;
    const Real isx4 = 1.0 / std::pow(sx, 4);
    const Real isy4 = 1.0 / std::pow(sy, 4);

    return {
        {
            "circle",
            "Circle",
            cx, cy,
            [=](Real x, Real y) -> bool {
                const Real dx = x - cx;
                const Real dy = y - cy;
                return dx * dx + dy * dy < r * r;
            },
            [=](Real x, Real y) -> Real {
                const Real dx = x - cx;
                const Real dy = y - cy;
                return dx * dx + dy * dy - r * r;
            },
            [=](Real x, Real) -> Real { return 2.0 * (x - cx); },
            [=](Real, Real y) -> Real { return 2.0 * (y - cy); },
            [](Real, Real) -> Real { return 4.0; }
        },
        {
            "ellipse",
            "Ellipse",
            cx, cy,
            [=](Real x, Real y) -> bool {
                const Real dx = x - cx;
                const Real dy = y - cy;
                return dx * dx * ia2 + dy * dy * ib2 < 1.0;
            },
            [=](Real x, Real y) -> Real {
                const Real dx = x - cx;
                const Real dy = y - cy;
                return dx * dx * ia2 + dy * dy * ib2 - 1.0;
            },
            [=](Real x, Real) -> Real { return 2.0 * (x - cx) * ia2; },
            [=](Real, Real y) -> Real { return 2.0 * (y - cy) * ib2; },
            [=](Real, Real) -> Real { return 2.0 * ia2 + 2.0 * ib2; }
        },
        {
            "ellipse_rotated",
            "Rotated ellipse",
            cx, cy,
            [=](Real x, Real y) -> bool {
                const Real dx = x - cx;
                const Real dy = y - cy;
                return A11 * dx * dx + 2.0 * A12 * dx * dy + A22 * dy * dy < 1.0;
            },
            [=](Real x, Real y) -> Real {
                const Real dx = x - cx;
                const Real dy = y - cy;
                return A11 * dx * dx + 2.0 * A12 * dx * dy + A22 * dy * dy - 1.0;
            },
            [=](Real x, Real y) -> Real {
                const Real dx = x - cx;
                const Real dy = y - cy;
                return 2.0 * A11 * dx + 2.0 * A12 * dy;
            },
            [=](Real x, Real y) -> Real {
                const Real dx = x - cx;
                const Real dy = y - cy;
                return 2.0 * A12 * dx + 2.0 * A22 * dy;
            },
            [=](Real, Real) -> Real { return 2.0 * (A11 + A22); }
        },
        {
            "superellipse4",
            "Superellipse p=4",
            cx, cy,
            [=](Real x, Real y) -> bool {
                const Real dx = x - cx;
                const Real dy = y - cy;
                return std::pow(dx, 4) * isx4 + std::pow(dy, 4) * isy4 < 1.0;
            },
            [=](Real x, Real y) -> Real {
                const Real dx = x - cx;
                const Real dy = y - cy;
                return std::pow(dx, 4) * isx4 + std::pow(dy, 4) * isy4 - 1.0;
            },
            [=](Real x, Real) -> Real {
                const Real dx = x - cx;
                return 4.0 * std::pow(dx, 3) * isx4;
            },
            [=](Real, Real y) -> Real {
                const Real dy = y - cy;
                return 4.0 * std::pow(dy, 3) * isy4;
            },
            [=](Real x, Real y) -> Real {
                const Real dx = x - cx;
                const Real dy = y - cy;
                return 12.0 * dx * dx * isx4 + 12.0 * dy * dy * isy4;
            }
        }
    };
}

std::vector<Geometry2D> select_geometries(const std::string &geometry_name) {
    const auto geometries = build_geometries();
    if (geometry_name == "all") return geometries;

    for (const auto &geometry : geometries) {
        if (geometry.name == geometry_name) return {geometry};
    }

    std::string available;
    for (size_t i = 0; i < geometries.size(); ++i) {
        if (i) available += ", ";
        available += geometries[i].name;
    }
    throw std::runtime_error("Unknown geometry: " + geometry_name +
                             ". Available geometries: " + available + ", all");
}

std::vector<PoissonCase2D> build_poisson_cases(const Geometry2D &geometry) {
    const Real cx = geometry.cx;
    const Real cy = geometry.cy;
    const auto &phi = geometry.phi;
    const auto &dphidx = geometry.dphidx;
    const auto &dphidy = geometry.dphidy;
    const auto &lap_phi = geometry.lap_phi;

    return {
        {
            "quadratic",
            "Boundary function",
            [=](Real x, Real y) -> Real { return phi(x, y); },
            [=](Real x, Real y) -> Real { return -lap_phi(x, y); }
        },
        {
            "quartic_phi",
            "Squared boundary function",
            [=](Real x, Real y) -> Real {
                const Real p = phi(x, y);
                return p * p;
            },
            [=](Real x, Real y) -> Real {
                const Real p = phi(x, y);
                const Real px = dphidx(x, y);
                const Real py = dphidy(x, y);
                return -2.0 * (px * px + py * py + p * lap_phi(x, y));
            }
        },
        {
            "linear_x",
            "Linear-x weighted boundary function",
            [=](Real x, Real y) -> Real { return phi(x, y) * (x - cx); },
            [=](Real x, Real y) -> Real {
                return -((x - cx) * lap_phi(x, y) + 2.0 * dphidx(x, y));
            }
        },
        {
            "linear_y",
            "Linear-y weighted boundary function",
            [=](Real x, Real y) -> Real { return phi(x, y) * (y - cy); },
            [=](Real x, Real y) -> Real {
                return -((y - cy) * lap_phi(x, y) + 2.0 * dphidy(x, y));
            }
        },
        {
            "minus4",
            "u = -4 * phi",
            [=](Real x, Real y) -> Real { return -4.0 * phi(x, y); },
            [=](Real x, Real y) -> Real { return 4.0 * lap_phi(x, y); }
        }
    };
}

std::vector<PoissonCase2D> select_poisson_cases(const std::string &case_name,
                                                const Geometry2D &geometry) {
    const auto cases = build_poisson_cases(geometry);
    if (case_name == "all") return cases;

    for (const auto &poisson_case : cases) {
        if (poisson_case.name == case_name) return {poisson_case};
    }

    std::string available;
    for (size_t i = 0; i < cases.size(); ++i) {
        if (i) available += ", ";
        available += cases[i].name;
    }
    throw std::runtime_error("Unknown Poisson case: " + case_name +
                             ". Available cases: " + available + ", all");
}

void write_vem_error_field_csv(const std::string &filename,
                               const std::vector<_2d_vec> &points,
                               const Eigen::VectorXr &uh,
                               const std::unordered_set<int> &active_dofs,
                               Real h,
                               const std::function<Real(Real, Real)> &u_exact) {
    std::ofstream fout(filename);
    if (!fout.is_open()) {
        std::cerr << "Failed to open " << filename << "\n";
        return;
    }

    fout << "x,y,i,j,active,error\n";

    for (int gid = 0; gid < static_cast<int>(points.size()); ++gid) {
        const Real x = points[gid].vx;
        const Real y = points[gid].vy;
        const int i = static_cast<int>(std::round(x / h));
        const int j = static_cast<int>(std::round(y / h));
        const int active = active_dofs.count(gid) ? 1 : 0;
        const Real err = active ? std::abs(uh[gid] - u_exact(x, y)) : 0.0;

        fout << std::setprecision(16)
             << x << ","
             << y << ","
             << i << ","
             << j << ","
             << active << ","
             << err << "\n";
    }
}

void print_convergence_header() {
    std::cout << std::setw(8)  << "N"
              << std::setw(14) << "h"
              << std::setw(14) << "Unknowns"
              << std::setw(18) << "L2 Error"
              << std::setw(12) << "Rate"
              << std::setw(18) << "Linf Error"
              << std::setw(12) << "Rate"
              << std::setw(12) << "Iters"
              << std::setw(18) << "Residual"
              << std::endl;
}

void write_failed_convergence_row(std::ofstream &csv, int N, Real h, int unknowns,
                                  int iterations = -1, Real residual = std::numeric_limits<Real>::quiet_NaN()) {
    csv << N << "," << h << "," << unknowns
        << ",nan,nan,nan,nan," << iterations << "," << residual << "\n";
}

void write_convergence_row(std::ofstream &csv, int N, Real h, int unknowns,
                           Real err_l2, Real err_linf,
                           Real rate_l2, Real rate_li,
                           int iterations, Real residual) {
    csv << N << ","
        << std::setprecision(16) << h << ","
        << unknowns << ","
        << err_l2 << ","
        << err_linf << ","
        << rate_l2 << ","
        << rate_li << ","
        << iterations << ","
        << residual << "\n";
}

}

void MAC_DB::convergence_test_simple(const std::string& case_name,
                                     const std::string& geometry_name)
{
    using SpMat = Eigen::SparseMatrixCr;
    using Triplet = Eigen::Triplet<GlobalReal>;

    const std::vector<int> resolutions = {32, 64, 128, 256, 512};

    for (const auto &geometry : select_geometries(geometry_name)) {
        const auto &is_inside = geometry.is_inside;
        for (const auto &poisson_case : select_poisson_cases(case_name, geometry)) {
        const auto &u_exact = poisson_case.u_exact;
        const auto &rhs_f = poisson_case.rhs_f;

        std::ofstream csv("convergence_results_mac_simple_" + geometry.name + "_" + poisson_case.name + ".csv");
        if (!csv.is_open()) {
            std::cerr << "Failed to open convergence_results_mac_simple_" << geometry.name
                      << "_" << poisson_case.name << ".csv for writing.\n";
            continue;
        }

        csv << "N,h,unknowns,L2_error,Linf_error,L2_rate,Linf_rate,iterations,residual\n";
        std::cout << "\n[MAC simple] geometry = " << geometry.name
                  << ", case = " << poisson_case.name << "\n";
        print_convergence_header();

        Real prev_l2 = -1.0;
        Real prev_li = -1.0;

        for (int N : resolutions)
        {
            grid = MAC_Grid(N, N);

            const int nx = grid.nx;
            const int ny = grid.ny;
            const Real h  = grid.dx;
            const Real h2 = h * h;

            Array2D<int> active(nx, ny, 0);

            for (int i = 0; i < nx; ++i) {
                for (int j = 0; j < ny; ++j) {
                    const Real x = (i + 0.5) * h;
                    const Real y = (j + 0.5) * h;
                    if (is_inside(x, y)) {
                        active(i, j) = 1;
                    }
                }
            }

            Array2D<int> id(nx, ny, -1);
            int unknowns = 0;
            for (int i = 0; i < nx; ++i) {
                for (int j = 0; j < ny; ++j) {
                    if (active(i, j)) {
                        id(i, j) = unknowns++;
                    }
                }
            }

            std::vector<Triplet> trips;
            trips.reserve(5 * unknowns);

            Eigen::VectorXr b = Eigen::VectorXr::Zero(unknowns);

            auto is_active = [&](int i, int j) -> bool {
                return (i >= 0 && i < nx && j >= 0 && j < ny && active(i, j));
            };

            for (int i = 0; i < nx; ++i) {
                for (int j = 0; j < ny; ++j) {
                    if (!active(i, j)) continue;

                    const int row = id(i, j);
                    const Real x = (i + 0.5) * h;
                    const Real y = (j + 0.5) * h;

                    Real diag = 0.0;
                    Real rhs  = rhs_f(x, y);

                    diag += 1.0 / h2;
                    if (is_active(i - 1, j)) trips.emplace_back(row, id(i - 1, j), -1.0 / h2);

                    diag += 1.0 / h2;
                    if (is_active(i + 1, j)) trips.emplace_back(row, id(i + 1, j), -1.0 / h2);

                    diag += 1.0 / h2;
                    if (is_active(i, j - 1)) trips.emplace_back(row, id(i, j - 1), -1.0 / h2);

                    diag += 1.0 / h2;
                    if (is_active(i, j + 1)) trips.emplace_back(row, id(i, j + 1), -1.0 / h2);

                    trips.emplace_back(row, row, diag);
                    b(row) = rhs;
                }
            }

            SpMat A(unknowns, unknowns);
            A.setFromTriplets(trips.begin(), trips.end());
            A.makeCompressed();

            Eigen::ConjugateGradient<SpMat, Eigen::Lower | Eigen::Upper> solver;
            solver.setTolerance(1e-12);
            solver.setMaxIterations(20000);
            solver.compute(A);

            if (solver.info() != Eigen::Success) {
                std::cerr << "Solver factorization failed at N = " << N << std::endl;
                write_failed_convergence_row(csv, N, h, unknowns);
                continue;
            }

            Eigen::VectorXr x = solver.solve(b.template cast<GlobalReal>()).template cast<Real>();

            if (solver.info() != Eigen::Success) {
                std::cerr << "Solver failed at N = " << N << std::endl;
                write_failed_convergence_row(csv, N, h, unknowns, solver.iterations(), solver.error());
                continue;
            }

            Array2D<Real> uh(nx, ny, 0.0);
            for (int i = 0; i < nx; ++i) {
                for (int j = 0; j < ny; ++j) {
                    if (active(i, j)) {
                        uh(i, j) = x(id(i, j));
                    }
                }
            }

            Real err_l2_sum = 0.0;
            Real err_linf   = 0.0;

            for (int i = 0; i < nx; ++i) {
                for (int j = 0; j < ny; ++j) {
                    if (!active(i, j)) continue;

                    const Real xc = (i + 0.5) * h;
                    const Real yc = (j + 0.5) * h;
                    const Real e  = std::abs(uh(i, j) - u_exact(xc, yc));

                    err_l2_sum += e * e * h2;
                    err_linf = std::max(err_linf, e);
                }
            }

            const Real err_l2 = std::sqrt(err_l2_sum);

            Real rate_l2 = 0.0;
            Real rate_li = 0.0;

            if (prev_l2 > 0.0) rate_l2 = std::log(prev_l2 / err_l2) / std::log(2.0);
            if (prev_li > 0.0) rate_li = std::log(prev_li / err_linf) / std::log(2.0);

            std::cout << std::setw(8)  << N
                      << std::setw(14) << std::scientific << h
                      << std::setw(14) << std::fixed << unknowns
                      << std::setw(18) << std::scientific << err_l2
                      << std::setw(12) << std::fixed << std::setprecision(4) << rate_l2
                      << std::setw(18) << std::scientific << err_linf
                      << std::setw(12) << std::fixed << std::setprecision(4) << rate_li
                      << std::setw(12) << solver.iterations()
                      << std::setw(18) << std::scientific << solver.error()
                      << std::endl;

            write_convergence_row(csv, N, h, unknowns,
                                  err_l2, err_linf, rate_l2, rate_li,
                                  solver.iterations(), solver.error());

            prev_l2 = err_l2;
            prev_li = err_linf;

            std::string filename = "error_field_mac_simple_" + geometry.name + "_" +
                                   poisson_case.name + "_N" + std::to_string(N) + ".csv";
            std::ofstream fout(filename);
            if (!fout.is_open()) {
                std::cerr << "Failed to open " << filename << "\n";
            } else {
                fout << "x,y,i,j,active,error\n";
                for (int j = ny - 1; j >= 0; --j) {
                    for (int i = 0; i < nx; ++i) {
                        const Real xcc = (i + 0.5) * h;
                        const Real ycc = (j + 0.5) * h;
                        const Real err = active(i, j) ? std::abs(uh(i, j) - u_exact(xcc, ycc)) : 0.0;
                        fout << std::setprecision(16)
                             << xcc << "," << ycc << "," << i << "," << j << ","
                             << active(i, j) << "," << err << "\n";
                    }
                }
            }
        }

        csv.close();
        std::cout << "\nSaved CSV to convergence_results_mac_simple_" << geometry.name
                  << "_" << poisson_case.name << ".csv\n";
        }
    }
}

void MAC_DB::convergence_test_ghost_fluid(const std::string& case_name,
                                          const std::string& geometry_name)
{
    using SpMat = Eigen::SparseMatrixCr;
    using Triplet = Eigen::Triplet<GlobalReal>;

    const std::vector<int> resolutions = {32, 64, 128, 256, 512};

    // free-surface pressure
    auto p_gamma = [&](Real, Real) -> Real {
        return 0.0;
    };

    for (const auto &geometry : select_geometries(geometry_name)) {
        const auto &phi = geometry.phi;
        for (const auto &poisson_case : select_poisson_cases(case_name, geometry)) {
        const auto &u_exact = poisson_case.u_exact;
        const auto &rhs_f = poisson_case.rhs_f;

        std::ofstream csv("convergence_results_mac_ghost_fluid_" + geometry.name + "_" + poisson_case.name + ".csv");
        if (!csv.is_open()) {
            std::cerr << "Failed to open convergence_results_mac_ghost_fluid_" << geometry.name
                      << "_" << poisson_case.name << ".csv for writing.\n";
            continue;
        }

        csv << "N,h,unknowns,L2_error,Linf_error,L2_rate,Linf_rate,iterations,residual\n";
        std::cout << "\n[MAC ghost fluid] geometry = " << geometry.name
                  << ", case = " << poisson_case.name << "\n";
        print_convergence_header();

        Real prev_l2 = -1.0;
        Real prev_li = -1.0;

        for (int N : resolutions)
        {
        grid = MAC_Grid(N, N);

        const int nx = grid.nx;
        const int ny = grid.ny;
        const Real h  = grid.dx;
        const Real h2 = h * h;

        Array2D<int> active(nx, ny, 0);
        Array2D<Real> phi_cell(nx, ny, 0.0);

        // build level set and active mask
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                const Real x = (i + 0.5) * h;
                const Real y = (j + 0.5) * h;

                phi_cell(i, j) = phi(x, y);
                if (phi_cell(i, j) <= 0.0) {
                    active(i, j) = 1;
                }
            }
        }

        // numbering
        Array2D<int> id(nx, ny, -1);
        int unknowns = 0;
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                if (active(i, j)) {
                    id(i, j) = unknowns++;
                }
            }
        }

        std::vector<Triplet> trips;
        trips.reserve(5 * unknowns);

        Eigen::VectorXr b = Eigen::VectorXr::Zero(unknowns);

        auto in_grid = [&](int i, int j) -> bool {
            return (i >= 0 && i < nx && j >= 0 && j < ny);
        };

        auto is_active = [&](int i, int j) -> bool {
            return in_grid(i, j) && active(i, j);
        };

        auto clamp_theta = [&](Real theta) -> Real {
            // avoid too small theta causing very large coefficients
            const Real eps = 1e-6;
            return std::max(theta, eps);
        };

        auto add_neighbor_contribution = [&](int row,
                                             int i, int j,
                                             int ni, int nj,
                                             Real& diag,
                                             Real& rhs)
        {
            // out of domain grid boundary: treat as zero Dirichlet
            // for this test, outer box should not be hit by the circle,
            // but keeping this makes the code robust.
            if (!in_grid(ni, nj)) {
                diag += 1.0 / h2;
                return;
            }

            if (active(ni, nj)) {
                diag += 1.0 / h2;
                trips.emplace_back(row, id(ni, nj), -1.0 / h2);
                return;
            }

            // Ghost Fluid free-surface treatment
            // cell (i,j) is fluid, neighbor (ni,nj) is air
            // interface fraction theta = |phi_i| / (|phi_i| + |phi_j|)
            const Real phi_i = phi_cell(i, j);
            const Real phi_j = phi_cell(ni, nj);

            Real theta = std::abs(phi_i) / (std::abs(phi_i) + std::abs(phi_j));
            theta = clamp_theta(theta);

            diag += 1.0 / (theta * h2);

            // free-surface Dirichlet value p_gamma = 0
            // if nonzero BC is needed, keep this term
            rhs += p_gamma(0.0, 0.0) / (theta * h2);
        };

        // assemble
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                if (!active(i, j)) continue;

                const int row = id(i, j);
                const Real x = (i + 0.5) * h;
                const Real y = (j + 0.5) * h;

                Real diag = 0.0;
                Real rhs  = rhs_f(x, y);

                add_neighbor_contribution(row, i, j, i - 1, j, diag, rhs); // left
                add_neighbor_contribution(row, i, j, i + 1, j, diag, rhs); // right
                add_neighbor_contribution(row, i, j, i, j - 1, diag, rhs); // down
                add_neighbor_contribution(row, i, j, i, j + 1, diag, rhs); // up

                trips.emplace_back(row, row, diag);
                b(row) = rhs;
            }
        }

        SpMat A(unknowns, unknowns);
        A.setFromTriplets(trips.begin(), trips.end());
        A.makeCompressed();

        Eigen::ConjugateGradient<SpMat, Eigen::Lower | Eigen::Upper> solver;
        solver.setTolerance(1e-12);
        solver.setMaxIterations(20000);
        solver.compute(A);

        if (solver.info() != Eigen::Success) {
            std::cerr << "Solver factorization failed at N = " << N << std::endl;
            write_failed_convergence_row(csv, N, h, unknowns);
            continue;
        }

        Eigen::VectorXr x = solver.solve(b.template cast<GlobalReal>()).template cast<Real>();

        if (solver.info() != Eigen::Success) {
            std::cerr << "Solver failed at N = " << N << std::endl;
            write_failed_convergence_row(csv, N, h, unknowns, solver.iterations(), solver.error());
            continue;
        }

        // recover solution
        Array2D<Real> uh(nx, ny, 0.0);
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                if (active(i, j)) {
                    uh(i, j) = x(id(i, j));
                }
            }
        }

        // errors
        Real err_l2_sum = 0.0;
        Real err_linf   = 0.0;

        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                if (!active(i, j)) continue;

                const Real xc = (i + 0.5) * h;
                const Real yc = (j + 0.5) * h;
                const Real e  = std::abs(uh(i, j) - u_exact(xc, yc));

                err_l2_sum += e * e * h2;
                err_linf = std::max(err_linf, e);
            }
        }

        const Real err_l2 = std::sqrt(err_l2_sum);

        Real rate_l2 = 0.0;
        Real rate_li = 0.0;
        if (prev_l2 > 0.0) rate_l2 = std::log(prev_l2 / err_l2) / std::log(2.0);
        if (prev_li > 0.0) rate_li = std::log(prev_li / err_linf) / std::log(2.0);

        std::cout << std::setw(8)  << N
                  << std::setw(14) << std::scientific << h
                  << std::setw(14) << std::fixed << unknowns
                  << std::setw(18) << std::scientific << err_l2
                  << std::setw(12) << std::fixed << std::setprecision(4) << rate_l2
                  << std::setw(18) << std::scientific << err_linf
                  << std::setw(12) << std::fixed << std::setprecision(4) << rate_li
                  << std::setw(12) << solver.iterations()
                  << std::setw(18) << std::scientific << solver.error()
                  << std::endl;

        write_convergence_row(csv, N, h, unknowns,
                              err_l2, err_linf, rate_l2, rate_li,
                              solver.iterations(), solver.error());

        prev_l2 = err_l2;
        prev_li = err_linf;

        // error field
        {
            std::string filename = "error_field_mac_ghost_fluid_" + geometry.name + "_" +
                                   poisson_case.name + "_N" + std::to_string(N) + ".csv";
            std::ofstream fout(filename);

            if (!fout.is_open()) {
                std::cerr << "Failed to open " << filename << "\n";
            } else {
                fout << "x,y,i,j,active,error\n";

                for (int j = ny - 1; j >= 0; --j) {
                    for (int i = 0; i < nx; ++i) {
                        const Real xcc = (i + 0.5) * h;
                        const Real ycc = (j + 0.5) * h;

                        Real err = 0.0;
                        if (active(i, j)) {
                            err = std::abs(uh(i, j) - u_exact(xcc, ycc));
                        }

                        fout << std::setprecision(16)
                             << xcc << ","
                             << ycc << ","
                             << i << ","
                             << j << ","
                             << active(i, j) << ","
                             << err << "\n";
                    }
                }

                fout.close();
            }
        }
        }

        csv.close();
        std::cout << "\nSaved CSV to convergence_results_mac_ghost_fluid_" << geometry.name
                  << "_" << poisson_case.name << ".csv\n";
        }
    }
}

void VEM_DB::convergence_test_simple(const std::string& case_name,
                                     const std::string& geometry_name) {
    const std::vector<int> resolutions = {32, 64, 128, 256, 512};

    for (const auto &geometry : select_geometries(geometry_name)) {
        const auto &is_inside = geometry.is_inside;
        for (const auto &poisson_case : select_poisson_cases(case_name, geometry)) {
        const auto &u_exact = poisson_case.u_exact;
        const auto &rhs_f = poisson_case.rhs_f;

        std::ofstream csv("convergence_results_vem_simple_" + geometry.name + "_" + poisson_case.name + ".csv");
        if (!csv.is_open()) {
            std::cerr << "Failed to open convergence_results_vem_simple_" << geometry.name
                      << "_" << poisson_case.name << ".csv for writing.\n";
            continue;
        }

        csv << "N,h,unknowns,L2_error,Linf_error,L2_rate,Linf_rate,iterations,residual\n";
        std::cout << "\n[VEM simple] geometry = " << geometry.name
                  << ", case = " << poisson_case.name << "\n";
        print_convergence_header();

        Real prev_l2 = -1.0;
        Real prev_li = -1.0;

        for (int k = 0; k < (int)resolutions.size(); ++k) {
        const int N = resolutions[k];
        const Real h = 1.0 / N;
        const Real h2 = h * h;

        vem_grid.uniform_construct(N, N);
        vem_grid.init();
        vem_grid.generate_elements();

        const int n_points   = (int)vem_grid.points.size();
        const int n_elements = (int)vem_grid.vem_elements.size();

        // --------------------------------------------------
        // 0. 标记 active elements
        // --------------------------------------------------
        std::vector<char> active_elem(n_elements, 0);
        for (int el_id = 0; el_id < n_elements; ++el_id) {
            const auto &elem = vem_grid.vem_elements[el_id];
            Real ex = elem.centroid.vx;
            Real ey = elem.centroid.vy;
            if (is_inside(ex, ey)) {
                active_elem[el_id] = 1;
            }
        }

        // --------------------------------------------------
        // 1. 收集 used_dofs，并建立 点->相邻单元 列表
        // --------------------------------------------------
        std::unordered_set<int> used_dofs;
        std::vector<std::vector<int>> point_to_elements(n_points);

        for (int el_id = 0; el_id < n_elements; ++el_id) {
            const auto &elem = vem_grid.vem_elements[el_id];
            int n_sides = (int)elem.local_to_global_dof.size();
            for (int j = 0; j < n_sides; ++j) {
                int gid = elem.local_to_global_dof[j];
                point_to_elements[gid].push_back(el_id);
                if (active_elem[el_id]) {
                    used_dofs.insert(gid);
                }
            }
        }

        // --------------------------------------------------
        // 2. 识别 boundary_dofs:
        //    一个 used_dof 如果连接到任何 inactive element，
        //    就把它视为离散边界点
        // --------------------------------------------------
        std::unordered_set<int> boundary_dofs;
        std::unordered_set<int> interior_dofs;

        for (int gid : used_dofs) {
            bool is_boundary_dof = false;
            for (int el_id : point_to_elements[gid]) {
                if (!active_elem[el_id]) {
                    is_boundary_dof = true;
                    break;
                }
            }

            if (is_boundary_dof) boundary_dofs.insert(gid);
            else                 interior_dofs.insert(gid);
        }

        // --------------------------------------------------
        // 3. 只给 interior dofs 编号
        // --------------------------------------------------
        std::unordered_map<int, int> dof_remap;
        int new_id = 0;
        for (int gid = 0; gid < n_points; ++gid) {
            if (interior_dofs.count(gid)) {
                dof_remap[gid] = new_id++;
            }
        }

        using SpMat = Eigen::SparseMatrixCr;
        using Triplet = Eigen::Triplet<GlobalReal>;

        SpMat A(new_id, new_id);
        std::vector<Triplet> triplets;
        Eigen::VectorXr b = Eigen::VectorXr::Zero(new_id);

        // --------------------------------------------------
        // 4. 装配 reduced matrix A 和 rhs b
        //    只对 interior dofs 建立方程
        //    boundary dofs 的值固定为 0
        // --------------------------------------------------
        for (int el_id = 0; el_id < n_elements; ++el_id) {
            if (!active_elem[el_id]) continue;

            const auto &elem = vem_grid.vem_elements[el_id];
            const Eigen::MatrixXr &Ke = elem.stiffness_matrix_k1;

            const int n_sides = (int)elem.local_to_global_dof.size();
            const Real area = elem.area;
            const _2d_vec centroid = elem.centroid;

            // 局部 load vector：这里沿用你原来的 lumped 写法
            Eigen::VectorXr fe = Eigen::VectorXr::Zero(n_sides);
            for (int i = 0; i < n_sides; ++i) {
                fe[i] = rhs_f(centroid.vx, centroid.vy) * (area / n_sides);
            }

            for (int i = 0; i < n_sides; ++i) {
                int gi = elem.local_to_global_dof[i];

                // 只有 interior dof 才有对应方程
                if (!interior_dofs.count(gi)) continue;

                int row = dof_remap[gi];

                // RHS
                b[row] += fe[i];

                // stiffness
                for (int j = 0; j < n_sides; ++j) {
                    int gj = elem.local_to_global_dof[j];
                    Real val = Ke(i, j);

                    if (std::fabs(val) < 1e-14) continue;

                    if (interior_dofs.count(gj)) {
                        int col = dof_remap[gj];
                        triplets.emplace_back(row, col, val);
                    }
                    else if (boundary_dofs.count(gj)) {
                        // Dirichlet: u_B = 0
                        // b[row] -= Ke(i,j) * 0.0;
                        // 齐次边界这里不用做任何事
                    }
                }
            }
        }

        A.setFromTriplets(triplets.begin(), triplets.end());

        // --------------------------------------------------
        // 5. 求解
        // --------------------------------------------------
        Eigen::ConjugateGradient<SpMat, Eigen::Lower | Eigen::Upper> solver;
        solver.setTolerance(1e-12);
        solver.setMaxIterations(20000);
        solver.compute(A);

        if (solver.info() != Eigen::Success) {
            std::cerr << "Solver factorization failed at N = " << N << std::endl;
            write_failed_convergence_row(csv, N, h, new_id);
            continue;
        }

        Eigen::VectorXr x = solver.solve(b.template cast<GlobalReal>()).template cast<Real>();

        if (solver.info() != Eigen::Success) {
            std::cerr << "Solver failed at N = " << N << std::endl;
            write_failed_convergence_row(csv, N, h, new_id, solver.iterations(), solver.error());
            continue;
        }

        // --------------------------------------------------
        // 6. 恢复完整解向量 uh
        //    interior: 解出来的值
        //    boundary: 0
        //    outside : 0
        // --------------------------------------------------
        Eigen::VectorXr uh = Eigen::VectorXr::Zero(n_points);
        for (int gid = 0; gid < n_points; ++gid) {
            if (interior_dofs.count(gid)) {
                uh[gid] = x[dof_remap[gid]];
            } else if (boundary_dofs.count(gid)) {
                uh[gid] = 0.0;
            }
        }

        // --------------------------------------------------
        // 7. 误差
        //    这里只在 used_dofs 上评估
        // --------------------------------------------------
        Real err_l2_sum = 0.0;
        Real err_linf   = 0.0;

        for (int gid = 0; gid < n_points; ++gid) {
            if (!used_dofs.count(gid)) continue;

            const Real xc = vem_grid.points[gid].vx;
            const Real yc = vem_grid.points[gid].vy;

            const Real e = std::abs(uh[gid] - u_exact(xc, yc));
            err_l2_sum += e * e * h2;
            err_linf = std::max(err_linf, e);
        }

        const Real err_l2 = std::sqrt(err_l2_sum);

        Real rate_l2 = 0.0;
        Real rate_li = 0.0;

        if (prev_l2 > 0.0) rate_l2 = std::log(prev_l2 / err_l2) / std::log(2.0);
        if (prev_li > 0.0) rate_li = std::log(prev_li / err_linf) / std::log(2.0);

        std::cout << std::setw(8)  << N
                  << std::setw(14) << std::scientific << h
                  << std::setw(14) << std::fixed << new_id
                  << std::setw(18) << std::scientific << err_l2
                  << std::setw(12) << std::fixed << std::setprecision(4) << rate_l2
                  << std::setw(18) << std::scientific << err_linf
                  << std::setw(12) << std::fixed << std::setprecision(4) << rate_li
                  << std::setw(12) << solver.iterations()
                  << std::setw(18) << std::scientific << solver.error()
                  << std::endl;

        write_convergence_row(csv, N, h, new_id,
                              err_l2, err_linf, rate_l2, rate_li,
                              solver.iterations(), solver.error());

        write_vem_error_field_csv(
            "error_field_vem_simple_" + geometry.name + "_" + poisson_case.name +
            "_N" + std::to_string(N) + ".csv",
            vem_grid.points, uh, used_dofs, h, u_exact);

        prev_l2 = err_l2;
        prev_li = err_linf;
    }

        csv.close();
        std::cout << "\nSaved CSV to convergence_results_vem_simple_" << geometry.name
                  << "_" << poisson_case.name << ".csv\n";
    }
    }
}

void VEM_DB::convergence_test_simple_points_based_boundary(const std::string& case_name,
                                                           const std::string& geometry_name) {
    const std::vector<int> resolutions = {32, 64, 128, 256, 512};

    for (const auto &geometry : select_geometries(geometry_name)) {
        const auto &is_inside = geometry.is_inside;
        for (const auto &poisson_case : select_poisson_cases(case_name, geometry)) {
        const auto &u_exact = poisson_case.u_exact;
        const auto &rhs_f = poisson_case.rhs_f;

        std::ofstream csv("convergence_results_vem_simple_points_based_boundary_" + geometry.name + "_" + poisson_case.name + ".csv");
        if (!csv.is_open()) {
            std::cerr << "Failed to open convergence_results_vem_simple_points_based_boundary_" << geometry.name
                      << "_" << poisson_case.name << ".csv for writing.\n";
            continue;
        }

        csv << "N,h,unknowns,L2_error,Linf_error,L2_rate,Linf_rate,iterations,residual\n";
        std::cout << "\n[VEM simple points based boundary] geometry = " << geometry.name
                  << ", case = " << poisson_case.name << "\n";
        print_convergence_header();

        Real prev_l2 = -1.0;
        Real prev_li = -1.0;

        for (int k = 0; k < (int)resolutions.size(); ++k) {
        const int N = resolutions[k];
        const Real h = 1.0 / N;
        const Real h2 = h * h;

        vem_grid.uniform_construct(N, N);
        vem_grid.init();
        vem_grid.generate_elements();

        const int n_points   = (int)vem_grid.points.size();
        const int n_elements = (int)vem_grid.vem_elements.size();


        std::unordered_set<int> boundary_dofs;
        std::unordered_set<int> interior_dofs;

        for (int i=0;i<vem_grid.points.size();++i) {
            bool is_boundary_dof = false;
            if (!is_inside(vem_grid.points[i].vx,vem_grid.points[i].vy))
                is_boundary_dof = true;
            if (is_boundary_dof) boundary_dofs.insert(i);
            else                 interior_dofs.insert(i);
        }

        // --------------------------------------------------
        // 3. 只给 interior dofs 编号
        // --------------------------------------------------
        std::unordered_map<int, int> dof_remap;
        int new_id = 0;
        for (int gid = 0; gid < n_points; ++gid) {
            if (interior_dofs.count(gid)) {
                dof_remap[gid] = new_id++;
            }
        }

        using SpMat = Eigen::SparseMatrixCr;
        using Triplet = Eigen::Triplet<GlobalReal>;

        SpMat A(new_id, new_id);
        std::vector<Triplet> triplets;
        Eigen::VectorXr b = Eigen::VectorXr::Zero(new_id);

        for (int el_id = 0; el_id < n_elements; ++el_id) {
            const auto &elem = vem_grid.vem_elements[el_id];
            const Eigen::MatrixXr &Ke = elem.stiffness_matrix_k1;

            const int n_sides = (int)elem.local_to_global_dof.size();
            const Real area = elem.area;
            const _2d_vec centroid = elem.centroid;

            // 局部 load vector：这里沿用你原来的 lumped 写法
            Eigen::VectorXr fe = Eigen::VectorXr::Zero(n_sides);
            for (int i = 0; i < n_sides; ++i) {
                fe[i] = rhs_f(centroid.vx, centroid.vy) * (area / n_sides);
            }

            for (int i = 0; i < n_sides; ++i) {
                int gi = elem.local_to_global_dof[i];

                // 只有 interior dof 才有对应方程
                if (!interior_dofs.count(gi)) continue;

                int row = dof_remap[gi];

                // RHS
                b[row] += fe[i];

                // stiffness
                for (int j = 0; j < n_sides; ++j) {
                    int gj = elem.local_to_global_dof[j];
                    Real val = Ke(i, j);

                    if (std::fabs(val) < 1e-14) continue;

                    if (interior_dofs.count(gj)) {
                        int col = dof_remap[gj];
                        triplets.emplace_back(row, col, val);
                    }
                    else if (boundary_dofs.count(gj)) {
                        // Dirichlet: u_B = 0
                        // b[row] -= Ke(i,j) * 0.0;
                        // 齐次边界这里不用做任何事
                    }
                }
            }
        }

        A.setFromTriplets(triplets.begin(), triplets.end());

        // --------------------------------------------------
        // 5. 求解
        // --------------------------------------------------
        Eigen::ConjugateGradient<SpMat, Eigen::Lower | Eigen::Upper> solver;
        solver.setTolerance(1e-12);
        solver.setMaxIterations(20000);
        solver.compute(A);

        if (solver.info() != Eigen::Success) {
            std::cerr << "Solver factorization failed at N = " << N << std::endl;
            write_failed_convergence_row(csv, N, h, new_id);
            continue;
        }

        Eigen::VectorXr x = solver.solve(b.template cast<GlobalReal>()).template cast<Real>();

        if (solver.info() != Eigen::Success) {
            std::cerr << "Solver failed at N = " << N << std::endl;
            write_failed_convergence_row(csv, N, h, new_id, solver.iterations(), solver.error());
            continue;
        }

        // --------------------------------------------------
        // 6. 恢复完整解向量 uh
        //    interior: 解出来的值
        //    boundary: 0
        //    outside : 0
        // --------------------------------------------------
        Eigen::VectorXr uh = Eigen::VectorXr::Zero(n_points);
        for (int gid = 0; gid < n_points; ++gid) {
            if (interior_dofs.count(gid)) {
                uh[gid] = x[dof_remap[gid]];
            } else if (boundary_dofs.count(gid)) {
                uh[gid] = 0.0;
            }
        }

        // --------------------------------------------------
        // 7. 误差
        //    这里只在 used_dofs 上评估
        // --------------------------------------------------
        Real err_l2_sum = 0.0;
        Real err_linf   = 0.0;

        for (int gid = 0; gid < n_points; ++gid) {
            if (!interior_dofs.count(gid)) continue;

            const Real xc = vem_grid.points[gid].vx;
            const Real yc = vem_grid.points[gid].vy;

            const Real e = std::abs(uh[gid] - u_exact(xc, yc));
            err_l2_sum += e * e * h2;
            err_linf = std::max(err_linf, e);
        }

        const Real err_l2 = std::sqrt(err_l2_sum);

        Real rate_l2 = 0.0;
        Real rate_li = 0.0;

        if (prev_l2 > 0.0) rate_l2 = std::log(prev_l2 / err_l2) / std::log(2.0);
        if (prev_li > 0.0) rate_li = std::log(prev_li / err_linf) / std::log(2.0);

        std::cout << std::setw(8)  << N
                  << std::setw(14) << std::scientific << h
                  << std::setw(14) << std::fixed << new_id
                  << std::setw(18) << std::scientific << err_l2
                  << std::setw(12) << std::fixed << std::setprecision(4) << rate_l2
                  << std::setw(18) << std::scientific << err_linf
                  << std::setw(12) << std::fixed << std::setprecision(4) << rate_li
                  << std::setw(12) << solver.iterations()
                  << std::setw(18) << std::scientific << solver.error()
                  << std::endl;

        write_convergence_row(csv, N, h, new_id,
                              err_l2, err_linf, rate_l2, rate_li,
                              solver.iterations(), solver.error());

        write_vem_error_field_csv(
            "error_field_vem_simple_points_based_boundary_" + geometry.name + "_" +
            poisson_case.name + "_N" + std::to_string(N) + ".csv",
            vem_grid.points, uh, interior_dofs, h, u_exact);

        prev_l2 = err_l2;
        prev_li = err_linf;
    }

        csv.close();
        std::cout << "\nSaved CSV to convergence_results_vem_simple_points_based_boundary_" << geometry.name
                  << "_" << poisson_case.name << ".csv\n";
    }
    }
}


void VEM_DB::convergence_test_fraction_fluid(const std::string& case_name,
                                             const std::string& geometry_name) {
    const std::vector<int> resolutions = {32, 64, 128, 256, 512};

    for (const auto &geometry : select_geometries(geometry_name)) {
        const auto &is_inside = geometry.is_inside;
        auto cell_area_fraction = [&](Real x0, Real y0, Real h) -> Real {
            const int S = 8;
            int cnt = 0;

            for (int a = 0; a < S; ++a) {
                for (int b = 0; b < S; ++b) {
                    Real x = x0 + (a + 0.5) * h / S;
                    Real y = y0 + (b + 0.5) * h / S;
                    if (is_inside(x, y)) cnt++;
                }
            }

            return Real(cnt) / Real(S * S);
        };

        for (const auto &poisson_case : select_poisson_cases(case_name, geometry)) {
        const auto &u_exact = poisson_case.u_exact;
        const auto &rhs_f = poisson_case.rhs_f;

        std::ofstream csv("convergence_results_vem_fraction_fluid_" + geometry.name + "_" + poisson_case.name + ".csv");
        if (!csv.is_open()) {
            std::cerr << "Failed to open convergence_results_vem_fraction_fluid_" << geometry.name
                      << "_" << poisson_case.name << ".csv for writing.\n";
            continue;
        }

        csv << "N,h,unknowns,L2_error,Linf_error,L2_rate,Linf_rate,iterations,residual\n";
        std::cout << "\n[VEM fraction fluid] geometry = " << geometry.name
                  << ", case = " << poisson_case.name << "\n";
        print_convergence_header();

        Real prev_l2 = -1.0;
        Real prev_li = -1.0;

        for (int k = 0; k < (int)resolutions.size(); ++k) {
        const int N = resolutions[k];
        const Real h = 1.0 / N;
        const Real h2 = h * h;

        vem_grid.uniform_construct(N, N);
        vem_grid.init();
        vem_grid.generate_elements();

        const int n_points   = (int)vem_grid.points.size();
        const int n_elements = (int)vem_grid.vem_elements.size();

        std::unordered_set<int> boundary_dofs;
        std::unordered_set<int> interior_dofs;

        // 节点分类：圆内 unknown，圆外 Dirichlet 0
        for (int i = 0; i < n_points; ++i) {
            Real x = vem_grid.points[i].vx;
            Real y = vem_grid.points[i].vy;

            if (is_inside(x, y)) interior_dofs.insert(i);
            else                 boundary_dofs.insert(i);
        }

        std::unordered_map<int, int> dof_remap;
        int new_id = 0;

        for (int gid = 0; gid < n_points; ++gid) {
            if (interior_dofs.count(gid)) {
                dof_remap[gid] = new_id++;
            }
        }

        using SpMat = Eigen::SparseMatrixCr;
        using Triplet = Eigen::Triplet<GlobalReal>;

        SpMat A(new_id, new_id);
        std::vector<Triplet> triplets;
        Eigen::VectorXr b = Eigen::VectorXr::Zero(new_id);

        for (int el_id = 0; el_id < n_elements; ++el_id) {
            const auto &elem = vem_grid.vem_elements[el_id];
            const Eigen::MatrixXr &Ke0 = elem.stiffness_matrix_k1;

            const int n_sides = (int)elem.local_to_global_dof.size();
            const Real area = elem.area;
            const _2d_vec centroid = elem.centroid;

            // 假设规则网格单元，centroid 是 cell 中心
            Real x0 = centroid.vx - 0.5 * h;
            Real y0 = centroid.vy - 0.5 * h;

            Real theta = cell_area_fraction(x0, y0, h);

            // 完全在圆外：跳过
            if (theta < 1e-12) continue;

            // 用面积 fraction 缩放刚度矩阵和 load
            Eigen::MatrixXr Ke = theta * Ke0;

            Eigen::VectorXr fe = Eigen::VectorXr::Zero(n_sides);
            for (int i = 0; i < n_sides; ++i) {
                fe[i] = rhs_f(centroid.vx, centroid.vy)
                      * (theta * area / n_sides);
            }

            for (int i = 0; i < n_sides; ++i) {
                int gi = elem.local_to_global_dof[i];

                if (!interior_dofs.count(gi)) continue;

                int row = dof_remap[gi];

                b[row] += fe[i];

                for (int j = 0; j < n_sides; ++j) {
                    int gj = elem.local_to_global_dof[j];
                    Real val = Ke(i, j);

                    if (std::fabs(val) < 1e-14) continue;

                    if (interior_dofs.count(gj)) {
                        int col = dof_remap[gj];
                        triplets.emplace_back(row, col, val);
                    } else {
                        // 圆外节点作为齐次 Dirichlet：u = 0
                        // b[row] -= val * 0.0;
                    }
                }
            }
        }

        A.setFromTriplets(triplets.begin(), triplets.end());

        Eigen::ConjugateGradient<SpMat, Eigen::Lower | Eigen::Upper> solver;
        solver.setTolerance(1e-12);
        solver.setMaxIterations(20000);
        solver.compute(A);

        if (solver.info() != Eigen::Success) {
            std::cerr << "Solver factorization failed at N = " << N << std::endl;
            write_failed_convergence_row(csv, N, h, new_id);
            continue;
        }

        Eigen::VectorXr x = solver.solve(b.template cast<GlobalReal>()).template cast<Real>();

        if (solver.info() != Eigen::Success) {
            std::cerr << "Solver failed at N = " << N << std::endl;
            write_failed_convergence_row(csv, N, h, new_id, solver.iterations(), solver.error());
            continue;
        }

        Eigen::VectorXr uh = Eigen::VectorXr::Zero(n_points);

        for (int gid = 0; gid < n_points; ++gid) {
            if (interior_dofs.count(gid)) {
                uh[gid] = x[dof_remap[gid]];
            }
        }

        Real err_l2_sum = 0.0;
        Real err_linf   = 0.0;
        Real area_sum   = 0.0;

        for (int gid = 0; gid < n_points; ++gid) {
            if (!interior_dofs.count(gid)) continue;

            Real xg = vem_grid.points[gid].vx;
            Real yg = vem_grid.points[gid].vy;

            Real e = std::abs(uh[gid] - u_exact(xg, yg));

            err_l2_sum += e * e * h2;
            area_sum   += h2;
            err_linf = std::max(err_linf, e);
        }

        Real err_l2 = std::sqrt(err_l2_sum);

        Real rate_l2 = 0.0;
        Real rate_li = 0.0;

        if (prev_l2 > 0.0) rate_l2 = std::log(prev_l2 / err_l2) / std::log(2.0);
        if (prev_li > 0.0) rate_li = std::log(prev_li / err_linf) / std::log(2.0);

        std::cout << std::setw(8)  << N
                  << std::setw(14) << std::scientific << h
                  << std::setw(14) << std::fixed << new_id
                  << std::setw(18) << std::scientific << err_l2
                  << std::setw(12) << std::fixed << std::setprecision(4) << rate_l2
                  << std::setw(18) << std::scientific << err_linf
                  << std::setw(12) << std::fixed << std::setprecision(4) << rate_li
                  << std::setw(12) << solver.iterations()
                  << std::setw(18) << std::scientific << solver.error()
                  << std::endl;

        write_convergence_row(csv, N, h, new_id,
                              err_l2, err_linf, rate_l2, rate_li,
                              solver.iterations(), solver.error());

        write_vem_error_field_csv(
            "error_field_vem_fraction_fluid_" + geometry.name + "_" + poisson_case.name +
            "_N" + std::to_string(N) + ".csv",
            vem_grid.points, uh, interior_dofs, h, u_exact);

        prev_l2 = err_l2;
        prev_li = err_linf;
    }

        csv.close();
        std::cout << "\nSaved CSV to convergence_results_vem_fraction_fluid_" << geometry.name
                  << "_" << poisson_case.name << ".csv\n";
    }
    }
}

void VEM_DB::convergence_test_ghost_fluid(const std::string& case_name,
                                          const std::string& geometry_name) {
    const std::vector<int> resolutions = {32, 64, 128, 256, 512};

    auto clamp_theta = [&](Real theta) -> Real {
        // avoid too small theta causing very large coefficients
        const Real eps = 1e-6;
        return std::max(theta, eps);
    };

    for (const auto &geometry : select_geometries(geometry_name)) {
        const auto &is_inside = geometry.is_inside;
        const auto &phi = geometry.phi;
        for (const auto &poisson_case : select_poisson_cases(case_name, geometry)) {
        const auto &u_exact = poisson_case.u_exact;
        const auto &rhs_f = poisson_case.rhs_f;

        std::ofstream csv("convergence_results_vem_ghost_fluid_" + geometry.name + "_" + poisson_case.name + ".csv");
        if (!csv.is_open()) {
            std::cerr << "Failed to open convergence_results_vem_ghost_fluid_" << geometry.name
                      << "_" << poisson_case.name << ".csv for writing.\n";
            continue;
        }

        csv << "N,h,unknowns,L2_error,Linf_error,L2_rate,Linf_rate,iterations,residual\n";
        std::cout << "\n[VEM ghost fluid] geometry = " << geometry.name
                  << ", case = " << poisson_case.name << "\n";
        print_convergence_header();

        Real prev_l2 = -1.0;
        Real prev_li = -1.0;

        for (int k = 0; k < (int)resolutions.size(); ++k) {
        const int N = resolutions[k];
        const Real h = 1.0 / N;
        const Real h2 = h * h;

        vem_grid.uniform_construct(N, N);
        vem_grid.init();
        vem_grid.generate_elements();

        const int n_points   = (int)vem_grid.points.size();
        const int n_elements = (int)vem_grid.vem_elements.size();


        std::unordered_set<int> boundary_dofs;
        std::unordered_set<int> interior_dofs;

        for (int i=0;i<vem_grid.points.size();++i) {
            bool is_boundary_dof = false;
            if (!is_inside(vem_grid.points[i].vx,vem_grid.points[i].vy))
                is_boundary_dof = true;
            if (is_boundary_dof) boundary_dofs.insert(i);
            else                 interior_dofs.insert(i);
        }

        // --------------------------------------------------
        // 3. 只给 interior dofs 编号
        // --------------------------------------------------
        std::unordered_map<int, int> dof_remap;
        int new_id = 0;
        for (int gid = 0; gid < n_points; ++gid) {
            if (interior_dofs.count(gid)) {
                dof_remap[gid] = new_id++;
            }
        }

        using SpMat = Eigen::SparseMatrixCr;
        using Triplet = Eigen::Triplet<GlobalReal>;

        SpMat A(new_id, new_id);
        std::vector<Triplet> triplets;
        Eigen::VectorXr b = Eigen::VectorXr::Zero(new_id);

        for (int el_id = 0; el_id < n_elements; ++el_id) {
            const auto &elem = vem_grid.vem_elements[el_id];
            const Eigen::MatrixXr &Ke = elem.stiffness_matrix_k1;

            const int n_sides = (int)elem.local_to_global_dof.size();
            const Real area = elem.area;
            const _2d_vec centroid = elem.centroid;

            // 局部 load vector：这里沿用你原来的 lumped 写法
            Eigen::VectorXr fe = Eigen::VectorXr::Zero(n_sides);
            for (int i = 0; i < n_sides; ++i) {
                fe[i] = rhs_f(centroid.vx, centroid.vy) * (area / n_sides);
            }

            for (int i = 0; i < n_sides; ++i) {
                int gi = elem.local_to_global_dof[i];

                // 只有 interior dof 才有对应方程
                if (!interior_dofs.count(gi)) continue;

                int row = dof_remap[gi];

                // RHS
                b[row] += fe[i];

                // stiffness
                for (int j = 0; j < n_sides; ++j) {
                    int gj = elem.local_to_global_dof[j];
                    Real val = Ke(i, j);

                    if (std::fabs(val) < 1e-14) continue;

                    if (interior_dofs.count(gj)) {
                        int col = dof_remap[gj];
                        triplets.emplace_back(row, col, val);
                    }
                    else if (boundary_dofs.count(gj)) {
                        Real phi_i = phi(elem.points[i].vx,elem.points[i].vy);
                        Real phi_j = phi(elem.points[j].vx,elem.points[j].vy);
                        Real theta = phi_j/phi_i;
                        triplets.emplace_back(row, row, val*theta);
                    }
                }
            }
        }

        A.setFromTriplets(triplets.begin(), triplets.end());

        // --------------------------------------------------
        // 5. 求解
        // --------------------------------------------------
        Eigen::ConjugateGradient<SpMat, Eigen::Lower | Eigen::Upper> solver;
        solver.setTolerance(1e-12);
        solver.setMaxIterations(20000);
        solver.compute(A);

        if (solver.info() != Eigen::Success) {
            std::cerr << "Solver factorization failed at N = " << N << std::endl;
            write_failed_convergence_row(csv, N, h, new_id);
            continue;
        }

        Eigen::VectorXr x = solver.solve(b.template cast<GlobalReal>()).template cast<Real>();

        if (solver.info() != Eigen::Success) {
            std::cerr << "Solver failed at N = " << N << std::endl;
            write_failed_convergence_row(csv, N, h, new_id, solver.iterations(), solver.error());
            continue;
        }

        // --------------------------------------------------
        // 6. 恢复完整解向量 uh
        //    interior: 解出来的值
        //    boundary: 0
        //    outside : 0
        // --------------------------------------------------
        Eigen::VectorXr uh = Eigen::VectorXr::Zero(n_points);
        for (int gid = 0; gid < n_points; ++gid) {
            if (interior_dofs.count(gid)) {
                uh[gid] = x[dof_remap[gid]];
            } else if (boundary_dofs.count(gid)) {
                uh[gid] = 0.0;
            }
        }

        // --------------------------------------------------
        // 7. 误差
        //    这里只在 used_dofs 上评估
        // --------------------------------------------------
        Real err_l2_sum = 0.0;
        Real err_linf   = 0.0;

        for (int gid = 0; gid < n_points; ++gid) {
            if (!interior_dofs.count(gid)) continue;

            const Real xc = vem_grid.points[gid].vx;
            const Real yc = vem_grid.points[gid].vy;

            const Real e = std::abs(uh[gid] - u_exact(xc, yc));
            err_l2_sum += e * e * h2;
            err_linf = std::max(err_linf, e);
        }

        const Real err_l2 = std::sqrt(err_l2_sum);

        Real rate_l2 = 0.0;
        Real rate_li = 0.0;

        if (prev_l2 > 0.0) rate_l2 = std::log(prev_l2 / err_l2) / std::log(2.0);
        if (prev_li > 0.0) rate_li = std::log(prev_li / err_linf) / std::log(2.0);

        std::cout << std::setw(8)  << N
                  << std::setw(14) << std::scientific << h
                  << std::setw(14) << std::fixed << new_id
                  << std::setw(18) << std::scientific << err_l2
                  << std::setw(12) << std::fixed << std::setprecision(4) << rate_l2
                  << std::setw(18) << std::scientific << err_linf
                  << std::setw(12) << std::fixed << std::setprecision(4) << rate_li
                  << std::setw(12) << solver.iterations()
                  << std::setw(18) << std::scientific << solver.error()
                  << std::endl;

        write_convergence_row(csv, N, h, new_id,
                              err_l2, err_linf, rate_l2, rate_li,
                              solver.iterations(), solver.error());

        write_vem_error_field_csv(
            "error_field_vem_ghost_fluid_" + geometry.name + "_" + poisson_case.name +
            "_N" + std::to_string(N) + ".csv",
            vem_grid.points, uh, interior_dofs, h, u_exact);

        prev_l2 = err_l2;
        prev_li = err_linf;
    }

        csv.close();
        std::cout << "\nSaved CSV to convergence_results_vem_ghost_fluid_" << geometry.name
                  << "_" << poisson_case.name << ".csv\n";
        }
    }
}

void VEM_DB::convergence_test_Nitsche_fluid() {
    
}
