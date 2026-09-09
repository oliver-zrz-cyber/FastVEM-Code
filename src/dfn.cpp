#include "dfn.h"
#include <unordered_map>
#include <mutex>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <limits>
#include <unistd.h>

#if __has_include(<cuda_runtime.h>)
#include <cuda_runtime.h>
#define VEMFLUID_HAS_CUDA_RUNTIME 1
#else
#define VEMFLUID_HAS_CUDA_RUNTIME 0
#endif

// -------------------------
// CSR64
// -------------------------
void CSR64::clear() {
    nrows = ncols = 0;
    row_ptr.clear();
    col_idx.clear();
    values.clear();
    row_ptr.shrink_to_fit();
    col_idx.shrink_to_fit();
    values.shrink_to_fit();
}

// -------------------------
// FunctionTimer / RZTimer
// -------------------------
FunctionTimer::FunctionTimer(const std::string& name)
    : name_(name), start_(std::chrono::high_resolution_clock::now()) {}

FunctionTimer::~FunctionTimer() {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<Real, std::milli> elapsed = end - start_;
    std::cout << "[Timer] " << name_ << " took " << elapsed.count() << " ms\n";
}

namespace {
    size_t current_rss_kb() {
        std::ifstream status_file("/proc/self/status");
        std::string line;
        while (std::getline(status_file, line)) {
            if (line.substr(0, 6) == "VmRSS:") {
                std::istringstream iss(line);
                std::string key, unit;
                size_t value = 0;
                iss >> key >> value >> unit;
                return value;
            }
        }
        return 0;
    }

    FunctionMemoryTimer::GpuMemorySnapshot current_gpu_memory() {
        FunctionMemoryTimer::GpuMemorySnapshot snapshot;
#if VEMFLUID_HAS_CUDA_RUNTIME
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count <= 0) {
            cudaGetLastError();
            return snapshot;
        }

        size_t free_bytes = 0;
        size_t total_bytes = 0;
        err = cudaMemGetInfo(&free_bytes, &total_bytes);
        if (err != cudaSuccess) {
            cudaGetLastError();
            return snapshot;
        }

        snapshot.available = true;
        snapshot.free_bytes = free_bytes;
        snapshot.total_bytes = total_bytes;
        snapshot.used_bytes = total_bytes - free_bytes;
#endif
        return snapshot;
    }

    Real kb_to_mb(size_t kb) {
        return static_cast<Real>(kb) / 1024.0;
    }

    Real bytes_to_mb(size_t bytes) {
        return static_cast<Real>(bytes) / (1024.0 * 1024.0);
    }

    std::string csv_escape(const std::string& value) {
        std::string escaped = "\"";
        for (char ch : value) {
            if (ch == '"') {
                escaped += "\"\"";
            } else {
                escaped += ch;
            }
        }
        escaped += "\"";
        return escaped;
    }

    std::string profile_csv_path() {
        const char* path = std::getenv("VEMFLUID_PROFILE_CSV");
        if (path && path[0] != '\0') {
            return path;
        }
        return "function_memory_profile.csv";
    }

    std::string profile_run_id() {
        static const std::string run_id = [] {
            std::ostringstream oss;
            const std::time_t now = std::time(nullptr);
            std::tm tm{};
            localtime_r(&now, &tm);
            oss << std::put_time(&tm, "%Y%m%d_%H%M%S") << "_pid" << getpid();
            return oss.str();
        }();
        return run_id;
    }

    std::string profile_timestamp() {
        std::ostringstream oss;
        const std::time_t now = std::time(nullptr);
        std::tm tm{};
        localtime_r(&now, &tm);
        oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
        return oss.str();
    }

    std::mutex& profile_csv_mutex() {
        static std::mutex mtx;
        return mtx;
    }

    void append_memory_profile_csv(
        const std::string& function_name,
        Real elapsed_ms,
        Real cpu_rss_before_mb,
        Real cpu_rss_after_mb,
        Real cpu_rss_delta_mb,
        const FunctionMemoryTimer::GpuMemorySnapshot& gpu_before,
        const FunctionMemoryTimer::GpuMemorySnapshot& gpu_after)
    {
        std::lock_guard<std::mutex> lock(profile_csv_mutex());
        const std::string path = profile_csv_path();

        bool write_header = true;
        {
            std::ifstream existing(path);
            write_header = !existing.good() || existing.peek() == std::ifstream::traits_type::eof();
        }

        std::ofstream csv(path, std::ios::app);
        if (!csv) {
            std::cerr << "[MemoryTimer] warning: failed to open profile CSV: "
                      << path << std::endl;
            return;
        }

        if (write_header) {
            csv << "run_id,timestamp,function,elapsed_ms,"
                << "cpu_rss_before_mb,cpu_rss_after_mb,cpu_rss_delta_mb,"
                << "gpu_available,gpu_used_before_mb,gpu_used_after_mb,gpu_used_delta_mb\n";
        }

        const bool gpu_available = gpu_before.available && gpu_after.available;
        const Real nan = std::numeric_limits<Real>::quiet_NaN();
        const Real gpu_before_mb = gpu_available ? bytes_to_mb(gpu_before.used_bytes) : nan;
        const Real gpu_after_mb = gpu_available ? bytes_to_mb(gpu_after.used_bytes) : nan;
        const Real gpu_delta_mb = gpu_available ? gpu_after_mb - gpu_before_mb : nan;

        csv << csv_escape(profile_run_id()) << ','
            << csv_escape(profile_timestamp()) << ','
            << csv_escape(function_name) << ','
            << elapsed_ms << ','
            << cpu_rss_before_mb << ','
            << cpu_rss_after_mb << ','
            << cpu_rss_delta_mb << ','
            << (gpu_available ? 1 : 0) << ',';
        if (gpu_available) {
            csv << gpu_before_mb << ',' << gpu_after_mb << ',' << gpu_delta_mb;
        } else {
            csv << ",,";
        }
        csv << '\n';
    }
}

FunctionMemoryTimer::FunctionMemoryTimer(const std::string& name)
    : name_(name),
      start_(std::chrono::high_resolution_clock::now()),
      rss_before_kb_(current_rss_kb()),
      gpu_before_(current_gpu_memory()) {}

FunctionMemoryTimer::~FunctionMemoryTimer() {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<Real, std::milli> elapsed = end - start_;

    const size_t rss_after_kb = current_rss_kb();
    const Real rss_before_mb = kb_to_mb(rss_before_kb_);
    const Real rss_after_mb = kb_to_mb(rss_after_kb);
    const Real rss_delta_mb = rss_after_mb - rss_before_mb;

    std::cout << "[MemoryTimer] " << name_
              << " took " << elapsed.count() << " ms"
              << ", CPU RSS: " << rss_before_mb
              << " -> " << rss_after_mb
              << " MB, delta = " << rss_delta_mb << " MB";

    const GpuMemorySnapshot gpu_after = current_gpu_memory();
    append_memory_profile_csv(
        name_,
        elapsed.count(),
        rss_before_mb,
        rss_after_mb,
        rss_delta_mb,
        gpu_before_,
        gpu_after);

    if (gpu_before_.available && gpu_after.available) {
        const Real gpu_delta_mb = bytes_to_mb(gpu_after.used_bytes)
                                  - bytes_to_mb(gpu_before_.used_bytes);
        std::cout << ", CUDA GPU used: " << bytes_to_mb(gpu_before_.used_bytes)
                  << " -> " << bytes_to_mb(gpu_after.used_bytes)
                  << " MB, delta = " << gpu_delta_mb << " MB";
    } else {
        std::cout << ", CUDA GPU used: unavailable";
    }

    std::cout << std::endl;
}

namespace {
    std::unordered_map<int, std::chrono::high_resolution_clock::time_point>& rz_times() {
        static std::unordered_map<int, std::chrono::high_resolution_clock::time_point> times_;
        return times_;
    }

    std::mutex& rz_mutex() {
        static std::mutex mtx_;
        return mtx_;
    }
}

void RZTimer::mark(int idx) {
    std::lock_guard<std::mutex> lock(rz_mutex());
    auto now = std::chrono::high_resolution_clock::now();
    auto& times = rz_times();

    auto it = times.find(idx);
    if (it == times.end()) {
        times[idx] = now;
    } else {
        std::chrono::duration<Real, std::milli> elapsed = now - it->second;
        std::cout << "[RZTIME] idx " << idx << " took " << elapsed.count() << " ms\n";
        times.erase(it);
    }
}

// -------------------------
// _3d_vec
// -------------------------
_3d_vec _3d_vec::operator+(const _3d_vec& rhs) const { return _3d_vec(vx + rhs.vx, vy + rhs.vy, vz + rhs.vz); }
_3d_vec _3d_vec::operator-(const _3d_vec& rhs) const { return _3d_vec(vx - rhs.vx, vy - rhs.vy, vz - rhs.vz); }
_3d_vec _3d_vec::operator*(const Real& rhs) const { return _3d_vec(vx * rhs, vy * rhs, vz * rhs); }
_3d_vec _3d_vec::operator/(const Real& rhs) const { return _3d_vec(vx / rhs, vy / rhs, vz / rhs); }
Real _3d_vec::dot(const _3d_vec& rhs) const { return vx * rhs.vx + vy * rhs.vy + vz * rhs.vz; }

_3d_vec _3d_vec::cross(const _3d_vec& rhs) const {
    return _3d_vec(
        vy * rhs.vz - vz * rhs.vy,
        vz * rhs.vx - vx * rhs.vz,
        vx * rhs.vy - vy * rhs.vx
    );
}

_3d_vec _3d_vec::opposite() const { return _3d_vec(-vx, -vy, -vz); }
Real _3d_vec::norm() const { return std::sqrt(vx * vx + vy * vy + vz * vz); }
Real _3d_vec::squared_norm() const { return vx * vx + vy * vy + vz * vz; }

_3d_vec _3d_vec::normalized() const {
    Real n = norm();
    return (n > 1e-15) ? _3d_vec(vx / n, vy / n, vz / n) : _3d_vec(0.0, 0.0, 0.0);
}

std::string _3d_vec::to_obj_format() const {
    return "v " + std::to_string(vx) + " " + std::to_string(vy) + " " + std::to_string(vz);
}

int _3d_vec::x_to_lower() const { return static_cast<int>(std::floor(vx)); }
int _3d_vec::x_to_upper() const { return static_cast<int>(std::ceil(vx)); }
int _3d_vec::y_to_lower() const { return static_cast<int>(std::floor(vy)); }
int _3d_vec::y_to_upper() const { return static_cast<int>(std::ceil(vy)); }
int _3d_vec::z_to_lower() const { return static_cast<int>(std::floor(vz)); }
int _3d_vec::z_to_upper() const { return static_cast<int>(std::ceil(vz)); }

bool _3d_vec::sort(const _3d_vec& a, const _3d_vec& b) {// 注意这个sort!
    if (a.vx < b.vx && std::fabs(a.vx - b.vx) > 1e-15) return true;
    if (a.vx > b.vx && std::fabs(a.vx - b.vx) > 1e-15) return false;
    if (a.vy < b.vy && std::fabs(a.vy - b.vy) > 1e-15) return true;
    if (a.vy > b.vy && std::fabs(a.vy - b.vy) > 1e-15) return false;
    return a.vz < b.vz;
}

// -------------------------
// 粒子 / range
// -------------------------
FLIP_Particles::FLIP_Particles()
    : pos(0.0, 0.0), vec(0.0, 0.0), pic_vec(0.0, 0.0), deleted(false) {}

FLIP_Particles::FLIP_Particles(const _2d_vec& p, const _2d_vec& v)
    : pos(p), vec(v), pic_vec(0.0, 0.0), deleted(false) {}

FLIP_Particles_3d::FLIP_Particles_3d()
    : pos(0.0, 0.0, 0.0), old_pos(0.0, 0.0, 0.0), vec(0.0, 0.0, 0.0),
      pic_vec(0.0, 0.0, 0.0), contained_in_element_id(-1),
      deleted(false), need_interpolate(false) {}

FLIP_Particles_3d::FLIP_Particles_3d(const _3d_vec& p, const _3d_vec& v)
    : pos(p), old_pos(p), vec(v), pic_vec(0.0, 0.0, 0.0),
      contained_in_element_id(-1), deleted(false), need_interpolate(false) {}

Source_range::Source_range(int xmin_, int xmax_, int ymin_, int ymax_, int zmin_, int zmax_)
    : xmin(xmin_), xmax(xmax_), ymin(ymin_), ymax(ymax_), zmin(zmin_), zmax(zmax_) {}

// -------------------------
// source
// -------------------------
sphere_source::sphere_source(Real xc_, Real yc_, Real zc_, Real r_)
    : center(xc_, yc_, zc_), radius(r_), init_vec(0.0, 0.0, 0.0) {}

bool sphere_source::contains(const _3d_vec& p) const {
    _3d_vec d = p - center;
    return d.dot(d) <= radius * radius;
}

std::vector<_3d_vec> sphere_source::sample_points(int n) const {
    std::vector<_3d_vec> pts;
    pts.reserve(n);

    static thread_local std::mt19937 rng(123456);
    std::uniform_real_distribution<Real> uni01(0.0, 1.0);

    for (int i = 0; i < n; ++i) {
        const Real u1 = uni01(rng);
        const Real u2 = uni01(rng);
        const Real u3 = uni01(rng);

        const Real r  = radius * std::cbrt(u1);
        const Real phi   = std::acos(2.0 * u2 - 1.0);
        const Real theta = 2.0 * M_PI * u3;

        _3d_vec offset;
        offset.vx = r * std::sin(phi) * std::cos(theta);
        offset.vy = r * std::sin(phi) * std::sin(theta);
        offset.vz = r * std::cos(phi);

        pts.push_back(center + offset);
    }

    return pts;
}

cylinder_source::cylinder_source(const _3d_vec& c, const _3d_vec& axis_dir, Real r, Real h)
    : center(c), axis(axis_dir), radius(r), half_height(h), init_vec(0.0, 0.0, 0.0) {
    const Real len = std::sqrt(axis.dot(axis));
    if (len > 0.0) axis = axis / len;
}

bool cylinder_source::contains(const _3d_vec& p) const {
    const _3d_vec d = p - center;
    const Real t = d.dot(axis);
    if (std::abs(t) > half_height) return false;
    const _3d_vec radial = d - axis * t;
    return radial.dot(radial) <= radius * radius;
}

std::vector<_3d_vec> cylinder_source::sample_points(int n) const {
    std::vector<_3d_vec> pts;
    pts.reserve(n);

    _3d_vec u;
    if (std::abs(axis.vx) < 0.9) u = _3d_vec(1, 0, 0);
    else u = _3d_vec(0, 1, 0);

    u = (u - axis * u.dot(axis));
    const Real ulen = std::sqrt(u.dot(u));
    if (ulen > 0.0) u = u / ulen;

    _3d_vec v = axis.cross(u);

    static thread_local std::mt19937 rng(123456);
    std::uniform_real_distribution<Real> uni01(0.0, 1.0);
    std::uniform_real_distribution<Real> uni_theta(0.0, 2.0 * M_PI);

    for (int i = 0; i < n; ++i) {
        const Real t = (2.0 * uni01(rng) - 1.0) * half_height;
        const Real rr = radius * std::sqrt(uni01(rng));
        const Real theta = uni_theta(rng);

        const Real cx = rr * std::cos(theta);
        const Real cy = rr * std::sin(theta);

        _3d_vec p = center + axis * t + u * cx + v * cy;
        pts.push_back(p);
    }

    return pts;
}

// -------------------------
// kernels
// -------------------------
Real cubic_kernel(Real r, Real h) {
    if (r >= h) return 0.0;
    Real q = r / h;
    Real t = 1.0 - q;
    return t * t * t;
}

Real desbrun_gascuel_cubic(Real r, Real h) {
    if (r >= h) return 0.0;
    Real q = r / h;
    Real t = 1.0 - q;
    return t * t * t;
}

Real monaghan_cubic(Real r, Real h) {
    Real q = r / h;
    if (q >= 2.0) return 0.0;

    Real alpha = 1.0 / (M_PI * h * h * h);
    if (q < 1.0) {
        return alpha * (1.0 - 1.5*q*q + 0.75*q*q*q);
    } else {
        Real t = 2.0 - q;
        return alpha * 0.25 * t * t * t;
    }
}

Real cubic_kernel_new(Real r, Real h) {
    const Real q = (r * r) / (h * h);
    if (q >= 1.0) return 0.0;
    const Real t = 1.0 - q;
    return t * t * t;
}

Smoke_Particles_3d::Smoke_Particles_3d()
    : old_pos(0.0, 0.0, 0.0), pos(0.0, 0.0, 0.0), v(0.0, 0.0, 0.0),
      deleted(false), contained_in_element_id(-1) {}

Smoke_Particles_3d::Smoke_Particles_3d(const _3d_vec& p)
    : old_pos(p), pos(p), v(0.0, 0.0, 0.0),
      deleted(false), contained_in_element_id(-1) {}

// -------------------------
// VEM_Monomial_2d
// -------------------------
VEM_Monomial_2d::VEM_Monomial_2d(int p_degree, int q_degree, Real c_coeff)
    : p(p_degree), q(q_degree), c(c_coeff) {}

void VEM_Monomial_2d::pow_table_2d(Real x, int max_deg, Real* out) const noexcept {
    out[0] = 1.0;
    for (int i = 1; i <= max_deg; ++i) out[i] = out[i - 1] * x;
}

Real VEM_Monomial_2d::value(Real x, Real y, Real xc, Real yc, Real hd) const noexcept {
    const Real dx = x - xc;
    const Real dy = y - yc;
    Real px[4], py[4], hpow[8];
    pow_table_2d(dx, p, px);
    pow_table_2d(dy, q, py);
    pow_table_2d(hd, p + q, hpow);
    return c * px[p] * py[q] / hpow[p + q];
}

_2d_vec VEM_Monomial_2d::gradient(Real x, Real y, Real xc, Real yc, Real hd) const noexcept {
    const Real dx = x - xc;
    const Real dy = y - yc;
    Real px[4], py[4], hpow[8];
    pow_table_2d(dx, p, px);
    pow_table_2d(dy, q, py);
    pow_table_2d(hd, p + q, hpow);

    const Real base = c / hpow[p + q];
    Real gx = 0.0, gy = 0.0;
    if (p > 0) gx = base * p * px[p - 1] * py[q];
    if (q > 0) gy = base * q * px[p] * py[q - 1];
    return _2d_vec{gx, gy};
}

VEM_Monomial_2d VEM_Monomial_2d::derivative_x() const {
    if (p == 0) return VEM_Monomial_2d(0, 0, 0.0);
    return VEM_Monomial_2d(p - 1, q, c).scale(p);
}

VEM_Monomial_2d VEM_Monomial_2d::derivative_y() const {
    if (q == 0) return VEM_Monomial_2d(0, 0, 0.0);
    return VEM_Monomial_2d(p, q - 1, c).scale(q);
}

VEM_Monomial_2d VEM_Monomial_2d::scale(Real s) const {
    return VEM_Monomial_2d(p, q, c * s);
}

VEM_Monomial_2d VEM_Monomial_2d::operator*(const VEM_Monomial_2d& other) const {
    return VEM_Monomial_2d(p + other.p, q + other.q, c * other.c);
}

VEM_Monomial_2d& VEM_Monomial_2d::operator*=(const VEM_Monomial_2d& other) {
    p += other.p;
    q += other.q;
    c *= other.c;
    return *this;
}

// -------------------------
// VEM_Monomial_3d
// -------------------------
VEM_Monomial_3d::VEM_Monomial_3d(int p_degree, int q_degree, int r_degree, Real c_coeff)
    : p(p_degree), q(q_degree), r(r_degree), c(c_coeff) {}

void VEM_Monomial_3d::pow_table(Real x, int max_deg, Real* out) const noexcept {
    out[0] = 1.0;
    for (int i = 1; i <= max_deg; ++i) out[i] = out[i - 1] * x;
}

Real VEM_Monomial_3d::value(Real x, Real y, Real z,
                              Real xc, Real yc, Real zc,
                              Real hd) const noexcept {
    const Real dx = x - xc;
    const Real dy = y - yc;
    const Real dz = z - zc;

    Real px[4], py[4], pz[4], hpow[10];
    pow_table(dx, p, px);
    pow_table(dy, q, py);
    pow_table(dz, r, pz);
    pow_table(hd, p + q + r, hpow);

    return c * px[p] * py[q] * pz[r] / hpow[p + q + r];
}

_3d_vec VEM_Monomial_3d::gradient(Real x, Real y, Real z,
                                  Real xc, Real yc, Real zc,
                                  Real hd) const noexcept {
    const Real dx = x - xc;
    const Real dy = y - yc;
    const Real dz = z - zc;

    Real px[4], py[4], pz[4], hpow[10];
    pow_table(dx, p, px);
    pow_table(dy, q, py);
    pow_table(dz, r, pz);
    pow_table(hd, p + q + r, hpow);

    const Real base = c / hpow[p + q + r];
    Real gx = 0.0, gy = 0.0, gz = 0.0;

    if (p > 0) gx = base * p * px[p - 1] * py[q] * pz[r];
    if (q > 0) gy = base * q * px[p] * py[q - 1] * pz[r];
    if (r > 0) gz = base * r * px[p] * py[q] * pz[r - 1];

    return _3d_vec{gx, gy, gz};
}

VEM_Monomial_3d VEM_Monomial_3d::derivative_x() const {
    if (p == 0) return VEM_Monomial_3d(0, 0, 0, 0.0);
    return VEM_Monomial_3d(p - 1, q, r, c).scale(p);
}

VEM_Monomial_3d VEM_Monomial_3d::derivative_y() const {
    if (q == 0) return VEM_Monomial_3d(0, 0, 0, 0.0);
    return VEM_Monomial_3d(p, q - 1, r, c).scale(q);
}

VEM_Monomial_3d VEM_Monomial_3d::derivative_z() const {
    if (r == 0) return VEM_Monomial_3d(0, 0, 0, 0.0);
    return VEM_Monomial_3d(p, q, r - 1, c).scale(r);
}

VEM_Monomial_3d VEM_Monomial_3d::scale(Real s) const {
    return VEM_Monomial_3d(p, q, r, c * s);
}

VEM_Monomial_3d VEM_Monomial_3d::operator*(const VEM_Monomial_3d& other) const {
    return VEM_Monomial_3d(p + other.p, q + other.q, r + other.r, c * other.c);
}

VEM_Monomial_3d& VEM_Monomial_3d::operator*=(const VEM_Monomial_3d& other) {
    p += other.p;
    q += other.q;
    r += other.r;
    c *= other.c;
    return *this;
}

void VEM_Monomial_3d::print() const {
    std::cout << "(cx*(x-xc))^" << p
              << " * (cy*(y-yc))^" << q
              << " * (cz*(z-zc))^" << r << std::endl;
}

// -------------------------
// UNGRID
// -------------------------
UNGRID::UNGRID() : nx(0), ny(0), nz(0), grid_size(0.0) {}

UNGRID::UNGRID(int n)
    : nx(n), ny(n), nz(n), grid_size(1.0 / (Real)n) {
    grid.resize(
        nx,
        std::vector<std::vector<std::vector<std::pair<int,bool>>>>(
            ny,
            std::vector<std::vector<std::pair<int,bool>>>(nz)
        )
    );
}

void UNGRID::resize(int n) {
    nx = ny = nz = n;
    grid_size = 1.0 / (Real)n;

    grid.clear();
    grid.resize(
        nx,
        std::vector<std::vector<std::vector<std::pair<int,bool>>>>(
            ny,
            std::vector<std::vector<std::pair<int,bool>>>(nz)
        )
    );
}

void UNGRID::get_grid_index(const _3d_vec& pos, int& i, int& j, int& k) const {
    i = static_cast<int>(pos.vx / grid_size);
    j = static_cast<int>(pos.vy / grid_size);
    k = static_cast<int>(pos.vz / grid_size);

    i = std::min(std::max(i, 0), nx - 1);
    j = std::min(std::max(j, 0), ny - 1);
    k = std::min(std::max(k, 0), nz - 1);
}

std::pair<int,bool> UNGRID::operator()(const _3d_vec& pos) const {
    int i, j, k;
    get_grid_index(pos, i, j, k);
    if (grid[i][j][k].empty()) return std::make_pair(0, false);
    return grid[i][j][k][0];
}

std::vector<std::pair<int,bool>> UNGRID::get_neighbors(const UGID& grid_id, int n) const {
    std::vector<std::pair<int,bool>> neighbors;
    for (int dx = -n; dx <= n; ++dx)
        for (int dy = -n; dy <= n; ++dy)
            for (int dz = -n; dz <= n; ++dz) {
                int ni = grid_id.x + dx;
                int nj = grid_id.y + dy;
                int nk = grid_id.z + dz;
                if (ni >= 0 && ni < nx && nj >= 0 && nj < ny && nk >= 0 && nk < nz) {
                    for (auto particle_index : grid[ni][nj][nk]) {
                        neighbors.push_back(particle_index);
                    }
                }
            }
    return neighbors;
}

// -------------------------
// Boundary_UGRID
// -------------------------
Boundary_UGRID::Boundary_UGRID() : nx(0), ny(0), nz(0), grid_size(0.0) {}

Boundary_UGRID::Boundary_UGRID(int n)
    : nx(n), ny(n), nz(n), grid_size(1.0 / (Real)n) {
    grid.resize(
        nx,
        std::vector<std::vector<bool>>(ny, std::vector<bool>(nz, false))
    );
}

void Boundary_UGRID::resize(int n) {
    nx = ny = nz = n;
    grid_size = 1.0 / (Real)n;

    grid.clear();
    grid.resize(
        nx,
        std::vector<std::vector<bool>>(ny, std::vector<bool>(nz, false))
    );
}

void Boundary_UGRID::get_grid_index(const _3d_vec& pos, int& i, int& j, int& k) const {
    i = static_cast<int>(pos.vx / grid_size);
    j = static_cast<int>(pos.vy / grid_size);
    k = static_cast<int>(pos.vz / grid_size);

    i = std::min(std::max(i, 0), nx - 1);
    j = std::min(std::max(j, 0), ny - 1);
    k = std::min(std::max(k, 0), nz - 1);
}

bool Boundary_UGRID::operator()(const _3d_vec& pos) const {
    int i, j, k;
    get_grid_index(pos, i, j, k);
    return grid[i][j][k];
}

bool Boundary_UGRID::operator()(int i, int j, int k) const {
    if (i < 0 || i >= nx || j < 0 || j >= ny || k < 0 || k >= nz) return false;
    return grid[i][j][k];
}

std::vector<UGID> Boundary_UGRID::get_neighbors(const UGID& id, int n) const {
    std::vector<UGID> neighbors;
    neighbors.reserve((2*n+1)*(2*n+1)*(2*n+1));

    for (int dx = -n; dx <= n; ++dx)
        for (int dy = -n; dy <= n; ++dy)
            for (int dz = -n; dz <= n; ++dz) {
                int ni = id.x + dx;
                int nj = id.y + dy;
                int nk = id.z + dz;

                if (ni < 0 || ni >= nx || nj < 0 || nj >= ny || nk < 0 || nk >= nz)
                    continue;

                neighbors.emplace_back(ni, nj, nk);
            }

    return neighbors;
}
