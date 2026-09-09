// #pragma once
// #include <vector>
// #include <cstdio>
// #include <cstdlib>
// #include <cmath>
// #include <sstream>
// #include <iostream>
// #include <chrono>
// #include <string>
// #include <random>
//
//
// struct CSR64 {
//     using ValueType = Real;
//     using ColumnType = int;
//     using PointerType = ptrdiff_t;
//
//     PointerType nrows = 0;
//     PointerType ncols = 0;
//
//     std::vector<PointerType> row_ptr;
//     std::vector<ColumnType> col_idx;
//     std::vector<ValueType> values;
//     void clear() {
//         nrows = ncols = 0;
//         row_ptr.clear();
//         col_idx.clear();
//         values.clear();
//
//         // 立刻归还内存给系统（可选但在大矩阵时很有用）
//         row_ptr.shrink_to_fit();
//         col_idx.shrink_to_fit();
//         values.shrink_to_fit();
//     }
// };
//
// class FunctionTimer {
// public:
//     explicit FunctionTimer(const std::string& name)
//         : name_(name), start_(std::chrono::high_resolution_clock::now()) {}
//
//     ~FunctionTimer() {
//         auto end = std::chrono::high_resolution_clock::now();
//         std::chrono::duration<Real, std::milli> elapsed = end - start_;
//         std::cout << "[Timer] " << name_ << " took " << elapsed.count() << " ms\n";
//     }
//
// private:
//     std::string name_;
//     std::chrono::high_resolution_clock::time_point start_;
// };
//
// #define FUNC_TIMER() FunctionTimer __function_timer__(__func__)
// #include <unordered_map>
// #include <mutex>
// class RZTimer {
// public:
//     static void mark(int idx) {
//         std::lock_guard<std::mutex> lock(getMutex());
//         auto now = std::chrono::high_resolution_clock::now();
//         auto& times = getTimes();
//
//         auto it = times.find(idx);
//         if (it == times.end()) {
//             // 第一次调用，记录开始时间
//             times[idx] = now;
//         } else {
//             // 第二次调用，计算耗时并输出
//             std::chrono::duration<Real, std::milli> elapsed = now - it->second;
//             std::cout << "[RZTIME] idx " << idx << " took "
//                       << elapsed.count() << " ms\n";
//             times.erase(it);
//         }
//     }
//
// private:
//     static std::unordered_map<int, std::chrono::high_resolution_clock::time_point>& getTimes() {
//         static std::unordered_map<int, std::chrono::high_resolution_clock::time_point> times_;
//         return times_;
//     }
//
//     static std::mutex& getMutex() {
//         static std::mutex mtx_;
//         return mtx_;
//     }
// };
//
// #define RZTIME(idx) RZTimer::mark(idx)
//
// template<typename T>
// void rz_stream_args(std::ostringstream& oss, T&& arg) {
//     oss << std::forward<T>(arg);
// }
//
// template<typename T, typename... Args>
// void rz_stream_args(std::ostringstream& oss, T&& arg, Args&&... args) {
//     oss << std::forward<T>(arg);
//     rz_stream_args(oss, std::forward<Args>(args)...);
// }
//
// // =========================
// // RZ_INFO：直接拼接参数
// // =========================
// #define RZ_INFO(...)                                                     \
// do {                                                                     \
//     std::ostringstream _rz_oss;                                          \
//     rz_stream_args(_rz_oss, __VA_ARGS__);                                \
//     std::cout << "[RZ_INFO] " << _rz_oss.str() << std::endl;             \
// } while (0)
// // =========================
// // RZ_DEBUG_INFO：输出文件与行号
// // =========================
// #define RZ_DEBUG_INFO()                                                  \
// do {                                                                     \
// std::cout << "[RZ_DEBUG] "                                           \
// << __FILE__ << ":" << __LINE__ << std::endl;               \
// } while (0)
// // =========================
// // RZ_ASSERT：条件不成立时输出文件、行号和表达式
// // =========================
// #define RZ_ASSERT(cond, ...)                                                             \
// do {                                                                                     \
//     if (!(cond)) {                                                                       \
//         std::ostringstream _rz_oss;                                                      \
//         rz_stream_args(_rz_oss, __VA_ARGS__);                                            \
//         std::cerr << "[RZ_ASSERT FAILED] " << __FILE__ << ":" << __LINE__ << " (" << #cond << ")\n"; \
//         std::cerr << "[RZ_ASSERT] " << _rz_oss.str() << std::endl;                       \
//         std::abort();                                                                    \
//     }                                                                                    \
// } while (0)
//
// // =========================
// // RZ_CHECK：条件不成立时输出，但不中断程序
// // =========================
// #define RZ_CHECK(cond, ...)                                                              \
// do {                                                                                     \
//     if (!(cond)) {                                                                       \
//         std::ostringstream _rz_oss;                                                      \
//         rz_stream_args(_rz_oss, __VA_ARGS__);                                            \
//         std::cerr << "[RZ_CHECK FAILED] " << __FILE__ << ":" << __LINE__ << " (" << #cond << ")\n"; \
//         std::cerr << "[RZ_CHECK] " << _rz_oss.str() << std::endl;                        \
//     }                                                                                    \
// } while (0)
//
// #define RZ_FATAL(...)                                                                     \
// do {                                                                                      \
// std::ostringstream _rz_oss;                                                           \
// rz_stream_args(_rz_oss, __VA_ARGS__);                                                 \
// std::cerr << "[RZ_FATAL] " << __FILE__ << ":" << __LINE__ << "\n";                    \
// std::cerr << "[RZ_FATAL] " << _rz_oss.str() << std::endl;                             \
// std::abort();                                                                         \
// } while (0)
//
// struct RGB_Screen {
//     int width, height;
//     std::vector<unsigned char> pixels; // RGB，每像素3字节
// };
// template<typename T>
// class Array2D {
// public:
//     Array2D(size_t nx = 0, size_t ny = 0, T init = T{})
//         : nx_(nx), ny_(ny), data_(nx * ny, init) {}
//
//     inline T& operator()(size_t i, size_t j) {
//         RZ_ASSERT(i < nx_ && j < ny_, "out of range");
//         return data_[i + nx_ * j];
//     }
//
//     inline const T& operator()(size_t i, size_t j) const {
//         RZ_ASSERT(i < nx_ && j < ny_, "out of range");
//         return data_[i + nx_ * j];
//     }
//
//     inline void resize(size_t nx, size_t ny, T init = T{}) {
//         nx_ = nx; ny_ = ny;
//         data_.assign(nx_ * ny_, init);
//     }
//
//     inline void fill(T value) { std::fill(data_.begin(), data_.end(), value); }
//
//     inline size_t nx() const { return nx_; }
//     inline size_t ny() const { return ny_; }
//     inline size_t size() const { return data_.size(); }
//
//     inline T* data() { return data_.data(); }
//     inline const T* data() const { return data_.data(); }
//
//     void swap(Array2D& other) noexcept {
//         using std::swap;
//         RZ_ASSERT(nx_ == other.nx_ && ny_ == other.ny_, "swap dim!");
//         swap(nx_, other.nx_);
//         swap(ny_, other.ny_);
//         swap(data_, other.data_);
//     }
//
//     // ====== 新增运算符 ======
//     Array2D operator-(const Array2D& rhs) const {
//         RZ_ASSERT(nx_ == rhs.nx_ && ny_ == rhs.ny_, "dimension mismatch");
//         Array2D result(nx_, ny_);
//         for (size_t idx = 0; idx < data_.size(); ++idx) {
//             result.data_[idx] = data_[idx] - rhs.data_[idx];
//         }
//         return result;
//     }
//
//     Array2D& operator-=(const Array2D& rhs) {
//         RZ_ASSERT(nx_ == rhs.nx_ && ny_ == rhs.ny_, "dimension mismatch");
//         for (size_t idx = 0; idx < data_.size(); ++idx) {
//             data_[idx] -= rhs.data_[idx];
//         }
//         return *this;
//     }
//
// private:
//     size_t nx_, ny_;
//     std::vector<T> data_;
// };
//
// template<typename T>
// class Array3D {
// public:
//     Array3D(size_t nx = 0, size_t ny = 0, size_t nz = 0, T init = T{})
//         : nx_(nx), ny_(ny), nz_(nz), data_(nx * ny * nz, init) {}
//
//     // 访问与修改元素
//     inline T& operator()(size_t i, size_t j, size_t k) {
//         RZ_ASSERT(i < nx_ && j < ny_ && k < nz_, "out of range");
//         return data_[i + nx_ * (j + ny_ * k)];
//     }
//
//     inline const T& operator()(size_t i, size_t j, size_t k) const {
//         RZ_ASSERT(i < nx_ && j < ny_ && k < nz_, "out of range");
//         return data_[i + nx_ * (j + ny_ * k)];
//     }
//
//     // 重新设置尺寸并填充值
//     inline void resize(size_t nx, size_t ny, size_t nz, T init = T{}) {
//         nx_ = nx; ny_ = ny; nz_ = nz;
//         data_.assign(nx_ * ny_ * nz_, init);
//     }
//
//     // 填充值
//     inline void fill(T value) { std::fill(data_.begin(), data_.end(), value); }
//
//     // 维度信息
//     inline size_t nx() const { return nx_; }
//     inline size_t ny() const { return ny_; }
//     inline size_t nz() const { return nz_; }
//     inline size_t size() const { return data_.size(); }
//
//     // 数据指针
//     inline T* data() { return data_.data(); }
//     inline const T* data() const { return data_.data(); }
//
//     // 交换（要求维度相同）
//     void swap(Array3D& other) noexcept {
//         using std::swap;
//         RZ_ASSERT(nx_ == other.nx_ && ny_ == other.ny_ && nz_ == other.nz_, "swap dim!");
//         swap(nx_, other.nx_);
//         swap(ny_, other.ny_);
//         swap(nz_, other.nz_);
//         swap(data_, other.data_);
//     }
//
//     // ====== 运算符重载 ======
//     Array3D operator-(const Array3D& rhs) const {
//         RZ_ASSERT(nx_ == rhs.nx_ && ny_ == rhs.ny_ && nz_ == rhs.nz_, "dimension mismatch");
//         Array3D result(nx_, ny_, nz_);
// #pragma omp parallel for schedule(static)
//         for (size_t idx = 0; idx < data_.size(); ++idx) {
//             result.data_[idx] = data_[idx] - rhs.data_[idx];
//         }
//         return result;
//     }
//
//     Array3D& operator-=(const Array3D& rhs) {
//         RZ_ASSERT(nx_ == rhs.nx_ && ny_ == rhs.ny_ && nz_ == rhs.nz_, "dimension mismatch");
// #pragma omp parallel for schedule(static)
//         for (size_t idx = 0; idx < data_.size(); ++idx) {
//             data_[idx] -= rhs.data_[idx];
//         }
//         return *this;
//     }
//     inline void clamp01() {
//         #pragma omp parallel for schedule(static)
//         for (size_t idx = 0; idx < data_.size(); ++idx) {
//             if (data_[idx] < 0.0) data_[idx] = 0.0;
//             else if (data_[idx] > 1.0) data_[idx] = 1.0;
//         }
//     }
//
//     size_t nx_, ny_, nz_;
// private:
//
//     std::vector<T> data_;
// };
//
//
//
// struct _2d_vec {
//     Real vx,vy;
//
//     _2d_vec(Real vx, Real vy) : vx(vx), vy(vy) {}
//
//     _2d_vec operator+(const _2d_vec& rhs) const {
//         return _2d_vec(vx+rhs.vx,vy+rhs.vy);
//     }
//     _2d_vec operator-(const _2d_vec& rhs) const {
//         return _2d_vec(vx-rhs.vx,vy-rhs.vy);
//     }
//     _2d_vec operator*(const Real& rhs) const {
//         return _2d_vec(vx*rhs,vy*rhs);
//     }
//     int x_to_lower() const {
//         return static_cast<int>(std::floor(vx));
//     }
//     int x_to_upper() const {
//         return static_cast<int>(std::ceil(vx));
//     }
//     int y_to_lower() const {
//         return static_cast<int>(std::floor(vy));
//     }
//     int y_to_upper() const {
//         return static_cast<int>(std::ceil(vy));
//     }
//     // Cross product between two vectors
//     static Real cross(const _2d_vec& a, const _2d_vec& b) {
//         return a.vx * b.vy - a.vy * b.vx;
//     }
//     Real norm() const {
//         return std::sqrt(vx * vx + vy * vy);
//     }
//     _2d_vec operator/(const Real& rhs) const {
//         return _2d_vec(vx / rhs, vy / rhs);
//     }
// };
//
//
// struct _3d_vec {
//     Real vx, vy, vz;
//
//     _3d_vec(Real vx = 0.0, Real vy = 0.0, Real vz = 0.0)
//         : vx(vx), vy(vy), vz(vz) {}
//
//     // 向量加法
//     _3d_vec operator+(const _3d_vec& rhs) const {
//         return _3d_vec(vx + rhs.vx, vy + rhs.vy, vz + rhs.vz);
//     }
//
//     // 向量减法
//     _3d_vec operator-(const _3d_vec& rhs) const {
//         return _3d_vec(vx - rhs.vx, vy - rhs.vy, vz - rhs.vz);
//     }
//
//     // 数乘
//     _3d_vec operator*(const Real& rhs) const {
//         return _3d_vec(vx * rhs, vy * rhs, vz * rhs);
//     }
//     // 数乘
//     _3d_vec operator/(const Real& rhs) const {
//         return _3d_vec(vx / rhs, vy / rhs, vz / rhs);
//     }
//     // 向量点积
//     Real dot(const _3d_vec& rhs) const {
//         return vx * rhs.vx + vy * rhs.vy + vz * rhs.vz;
//     }
//
//     // 向量叉积
//     _3d_vec cross(const _3d_vec& rhs) const {
//         return _3d_vec(
//             vy * rhs.vz - vz * rhs.vy,
//             vz * rhs.vx - vx * rhs.vz,
//             vx * rhs.vy - vy * rhs.vx
//         );
//     }
//     _3d_vec opposite() const {
//         return _3d_vec(
//             -vx,-vy,-vz
//         );
//     }
//
//     // 模长
//     Real norm() const {
//         return std::sqrt(vx * vx + vy * vy + vz * vz);
//     }
//     Real squared_norm() const {
//         return vx * vx + vy * vy + vz * vz;
//     }
//
//     // 归一化
//     _3d_vec normalized() const {
//         Real n = norm();
//         return (n > 1e-15) ? _3d_vec(vx / n, vy / n, vz / n) : _3d_vec(0.0, 0.0, 0.0);
//     }
//     // 格式化为 "v vx vy vz" 的形式
//     std::string to_obj_format() const {
//         return "v " + std::to_string(vx) + " " + std::to_string(vy) + " " + std::to_string(vz);
//     }
//
//     // 下取整 / 上取整
//     int x_to_lower() const { return static_cast<int>(std::floor(vx)); }
//     int x_to_upper() const { return static_cast<int>(std::ceil(vx)); }
//     int y_to_lower() const { return static_cast<int>(std::floor(vy)); }
//     int y_to_upper() const { return static_cast<int>(std::ceil(vy)); }
//     int z_to_lower() const { return static_cast<int>(std::floor(vz)); }
//     int z_to_upper() const { return static_cast<int>(std::ceil(vz)); }
//
//     // ======================================================
//     // 排序比较器：先 x，再 y，再 z（严格弱序）
//     // ======================================================
//     static bool sort(const _3d_vec& a, const _3d_vec& b) {
//         if (a.vx < b.vx) return true;
//         if (a.vx > b.vx) return false;
//         if (a.vy < b.vy) return true;
//         if (a.vy > b.vy) return false;
//         return a.vz < b.vz;
//     }
// };
//
//
// struct FLIP_Particles {
//     _2d_vec pos;
//     _2d_vec vec;
//     _2d_vec pic_vec;
//     bool deleted;
//     // 显式默认构造函数
//     FLIP_Particles()
//         : pos(0.0, 0.0), vec(0.0, 0.0),pic_vec(0.0, 0.0),deleted(false) {}
//
//     // 如果你想要直接用参数初始化
//     FLIP_Particles(const _2d_vec& p, const _2d_vec& v)
//         : pos(p), vec(v),pic_vec(0.0, 0.0),deleted(false) {}
// };
//
// struct UGID {
//     int x, y, z;
//     UGID(const int& x_,const int& y_,const int& z_):x(x_),y(y_),z(z_){}
// };
//
// struct FLIP_Particles_3d {
//     _3d_vec pos;
//     _3d_vec old_pos;
//     _3d_vec vec;
//     _3d_vec pic_vec;
//     int contained_in_element_id;
//     bool deleted,need_interpolate;
//     Real temp=0.0,rho=0.0;
//     // 显式默认构造函数
//     FLIP_Particles_3d()
//         : pos(0.0, 0.0,0.0),old_pos(0.0, 0.0,0.0), vec(0.0, 0.0,0.0),
//     pic_vec(0.0, 0.0,0.0),deleted(false),need_interpolate(false)
//     ,contained_in_element_id(-1){}
//
//     // 如果你想要直接用参数初始化
//     FLIP_Particles_3d(const _3d_vec& p, const _3d_vec& v)
//         : pos(p),old_pos(p), vec(v),pic_vec(0.0, 0.0,0.0),deleted(false),need_interpolate(false)
//     ,contained_in_element_id(-1) {}
// };
// struct Source_range {
//     int xmin, xmax, ymin, ymax, zmin, zmax;
//
//     Source_range(int xmin_, int xmax_,
//                  int ymin_, int ymax_,
//                  int zmin_, int zmax_)
//         : xmin(xmin_), xmax(xmax_),
//           ymin(ymin_), ymax(ymax_),
//           zmin(zmin_), zmax(zmax_) {}
// };
//
// struct sphere_source {
//     _3d_vec center;
//     Real radius;
//     _3d_vec init_vec;
//
//     sphere_source(Real xc_, Real yc_, Real zc_, Real r_)
//         : center(xc_, yc_, zc_), radius(r_), init_vec(0.0, 0.0, 0.0) {}
//
//     // ------------------------------------------------------------
//     // 判断点是否在球内
//     // ------------------------------------------------------------
//     inline bool contains(const _3d_vec& p) const
//     {
//         _3d_vec d = p - center;
//         return d.dot(d) <= radius * radius;
//     }
//
//     // ------------------------------------------------------------
//     // 在球体内均匀采样 n 个点
//     // ------------------------------------------------------------
//     std::vector<_3d_vec> sample_points(int n) const
//     {
//         std::vector<_3d_vec> pts;
//         pts.reserve(n);
//
//         static thread_local std::mt19937 rng(123456);
//         std::uniform_real_distribution<Real> uni01(0.0, 1.0);
//
//         for (int i = 0; i < n; ++i)
//         {
//             // --- 均匀体积分布采样 ---
//             const Real u1 = uni01(rng);
//             const Real u2 = uni01(rng);
//             const Real u3 = uni01(rng);
//
//             const Real r  = radius * std::cbrt(u1);
//             const Real phi   = std::acos(2.0 * u2 - 1.0);
//             const Real theta = 2.0 * M_PI * u3;
//
//             _3d_vec offset;
//             offset.vx = r * std::sin(phi) * std::cos(theta);
//             offset.vy = r * std::sin(phi) * std::sin(theta);
//             offset.vz = r * std::cos(phi);
//
//             pts.push_back(center + offset);
//         }
//
//         return pts;
//     }
// };
//
// struct cylinder_source {
//     _3d_vec center;
//     _3d_vec axis;        // unit
//     Real  radius;
//     Real  half_height;
//     _3d_vec init_vec;
//
//     cylinder_source(
//         const _3d_vec& c,
//         const _3d_vec& axis_dir,
//         Real r,
//         Real h
//     )
//         : center(c)
//         , axis(axis_dir)
//         , radius(r)
//         , half_height(h)
//         , init_vec(0.0, 0.0, 0.0)
//     {
//         const Real len = std::sqrt(axis.dot(axis));
//         if (len > 0.0) axis = axis / len;
//     }
//
//     inline bool contains(const _3d_vec& p) const
//     {
//         const _3d_vec d = p - center;
//         const Real t = d.dot(axis);
//         if (std::abs(t) > half_height) return false;
//         const _3d_vec radial = d - axis * t;
//         return radial.dot(radial) <= radius * radius;
//     }
//
//     // ------------------------------------------------------------
//     // 在圆柱内部均匀采样 n 个点
//     // ------------------------------------------------------------
//     std::vector<_3d_vec> sample_points(int n) const
//     {
//         std::vector<_3d_vec> pts;
//         pts.reserve(n);
//
//         // --- 构造正交基 (axis, u, v) ---
//         _3d_vec u;
//         if (std::abs(axis.vx) < 0.9)
//             u = _3d_vec(1, 0, 0);
//         else
//             u = _3d_vec(0, 1, 0);
//
//         u = (u - axis * u.dot(axis));
//         const Real ulen = std::sqrt(u.dot(u));
//         if (ulen > 0.0) u = u / ulen;
//
//         _3d_vec v = axis.cross(u); // 假设你有 cross()
//
//         // --- RNG ---
//         static thread_local std::mt19937 rng(123456);
//         std::uniform_real_distribution<Real> uni01(0.0, 1.0);
//         std::uniform_real_distribution<Real> uni_theta(0.0, 2.0 * M_PI);
//
//         for (int i = 0; i < n; ++i)
//         {
//             const Real t     = (2.0 * uni01(rng) - 1.0) * half_height;
//             const Real rr    = radius * std::sqrt(uni01(rng));
//             const Real theta = uni_theta(rng);
//
//             const Real cx = rr * std::cos(theta);
//             const Real cy = rr * std::sin(theta);
//
//             _3d_vec p = center
//                       + axis * t
//                       + u * cx
//                       + v * cy;
//
//             pts.push_back(p);
//         }
//
//         return pts;
//     }
// };
//
//
// struct IVOCK_Kernel
// {
//     Real _h;
//
//     explicit IVOCK_Kernel(Real h) : _h(h) {}
//
//     // Tent kernel H(r), support in [-1, 1)
//     static inline Real H(Real r)
//     {
//         Real res = 0.0;
//         if (r >= -1.0 && r < 0.0) res = 1.0 + r;
//         if (r >=  0.0 && r < 1.0) res = 1.0 - r;
//         return res;
//     }
//
//     // k(x,y,z) = H(dx/h) H(dy/h) H(dz/h)
//     inline Real weight(Real gx, Real gy, Real gz,
//                          Real px, Real py, Real pz) const
//     {
//         const Real dx = px - gx;
//         const Real dy = py - gy;
//         const Real dz = pz - gz;
//         return H(dx / _h) * H(dy / _h) * H(dz / _h);
//     }
//
//     inline bool in_support(Real gx, Real gy, Real gz,
//                            Real px, Real py, Real pz) const
//     {
//         return (std::fabs(px - gx) < _h &&
//                 std::fabs(py - gy) < _h &&
//                 std::fabs(pz - gz) < _h);
//     }
// };
//
// // Cubic Spline 核函数 (3D) for smoke density
// inline Real cubic_kernel(Real r, Real h) // (1-(r/h)^2)^3
// {
//     if (r >= h) return 0.0;
//     Real q = r / h;
//     Real t = 1.0 - q;
//     return t * t * t;
// }
//
// inline Real desbrun_gascuel_cubic(Real r, Real h)
// {
//     if (r >= h) return 0.0;
//     Real q = r / h;
//     Real t = 1.0 - q;
//     return t * t * t;
// }
//
//
// inline Real monaghan_cubic(Real r, Real h)
// {
//     Real q = r / h;
//     if (q >= 2.0) return 0.0;
//
//     Real alpha = 1.0 / (M_PI * h * h * h); // 3D
//
//     if (q < 1.0)
//     {
//         return alpha * (1.0 - 1.5*q*q + 0.75*q*q*q);
//     }
//     else
//     {
//         Real t = 2.0 - q;
//         return alpha * 0.25 * t * t * t;
//     }
// }
//
//
// static inline Real cubic_kernel_new(Real r, Real h)
// {
//     // q = ||x_p - x_i||^2 / R^2
//     const Real q = (r * r) / (h * h);
//
//     if (q >= 1.0)
//         return 0.0;
//
//     const Real t = 1.0 - q;
//     return t * t * t;   // (1 - q)^3
// }
//
//
// struct Smoke_Particles_3d {
//     _3d_vec old_pos;
//     _3d_vec pos;
//     _3d_vec v;
//     bool deleted;
//     int contained_in_element_id;
//     Smoke_Particles_3d(): pos(0.0, 0.0,0.0),old_pos(0.0, 0.0,0.0),deleted(false),contained_in_element_id(-1){}
//     explicit Smoke_Particles_3d(const _3d_vec& p)
//     : pos(p),old_pos(p),deleted(false),contained_in_element_id(-1) {}
// };
//
//
// struct Trancer {
//     float x,y,z,rest_age;
//     uint32_t id;
// };
//
//
//
//
// class VEM_Monomial_2d {
// public:
//     int p;       // x 方向次数
//     int q;       // y 方向次数
//     Real c;
//
//     VEM_Monomial_2d(int p_degree = 0, int q_degree = 0,
//                  Real c_coeff = 1.0)
//         : p(p_degree), q(q_degree), c(c_coeff){}
//
//     inline void pow_table_2d(Real x, int max_deg, Real* out)const noexcept {
//         out[0] = 1.0;
//         for (int i = 1; i <= max_deg; ++i)
//             out[i] = out[i - 1] * x;
//     }
//     //
//     // // 计算单项式值: c*((x-xc))^p * ((y-yc))^q / hd^(p+q)
//     // Real value(Real x, Real y, Real xc, Real yc, Real hd) const {
//     //     Real dx = x - xc;
//     //     Real dy = y - yc;
//     //     return c * std::pow(dx, p) * std::pow(dy, q) / std::pow(hd, p + q);
//     // }
//     //
//     // // 计算梯度 (d/dx, d/dy)
//     // _2d_vec gradient(Real x, Real y, Real xc, Real yc, Real hd) const {
//     //     Real dx = x - xc;
//     //     Real dy = y - yc;
//     //
//     //     Real d_dx = 0.0, d_dy = 0.0;
//     //     Real denom = std::pow(hd, p + q);
//     //
//     //     if (p > 0)
//     //         d_dx = p * std::pow(dx, p - 1) * std::pow(dy, q) * c / denom;
//     //
//     //     if (q > 0)
//     //         d_dy = q * std::pow(dx, p) * std::pow(dy, q - 1) * c / denom;
//     //
//     //     return _2d_vec{(Real)d_dx, (Real)d_dy};
//     // }
//     inline Real value(Real x, Real y,
//                     Real xc, Real yc,
//                     Real hd) const noexcept
//     {
//         // 基本健壮性假设（debug 下建议断言）
//         // RZ_ASSERT(hd > 0.0, "hd must be positive");
//
//         const Real dx = x - xc;
//         const Real dy = y - yc;
//
//         Real px[4], py[4];        // p,q ≤ 3（可按需要调）
//         Real hpow[8];             // p+q ≤ 6 足够
//
//         pow_table_2d(dx, p, px);
//         pow_table_2d(dy, q, py);
//         pow_table_2d(hd, p + q, hpow);
//
//         return c * px[p] * py[q] / hpow[p + q];
//     }
//     inline _2d_vec gradient(Real x, Real y,
//                             Real xc, Real yc,
//                             Real hd) const noexcept
//     {
//         // RZ_ASSERT(hd > 0.0, "hd must be positive");
//
//         const Real dx = x - xc;
//         const Real dy = y - yc;
//
//         Real px[4], py[4];
//         Real hpow[8];
//
//         pow_table_2d(dx, p, px);
//         pow_table_2d(dy, q, py);
//         pow_table_2d(hd, p + q, hpow);
//
//         const Real base = c / hpow[p + q];
//
//         Real gx = 0.0, gy = 0.0;
//
//         if (p > 0)
//             gx = base * p * px[p - 1] * py[q];
//
//         if (q > 0)
//             gy = base * q * px[p] * py[q - 1];
//
//         return _2d_vec{gx, gy};
//     }
//
//     // ✅ 返回 ∂/∂x 的符号性导数（新的单项式）
//     VEM_Monomial_2d derivative_x() const {
//         if (p == 0)
//             return VEM_Monomial_2d(0, 0, 0.0); // 常数项对 x 求导为 0
//         return VEM_Monomial_2d(
//             p - 1,
//             q,
//             c
//         ).scale(p);    // 系数 p
//     }
//
//     // ✅ 返回 ∂/∂y 的符号性导数（新的单项式）
//     VEM_Monomial_2d derivative_y() const {
//         if (q == 0)
//             return VEM_Monomial_2d(0, 0, 0.0);
//
//         return VEM_Monomial_2d(
//             p,
//             q - 1,
//             c
//         ).scale(q);
//     }
//
//     // ✅ 缩放当前单项式的系数（返回新对象）
//     VEM_Monomial_2d scale(Real s) const {
//         return VEM_Monomial_2d(p, q, c * s);
//     }
//     // ✅ 单项式相乘：返回新的 VEM_Monomial
//     VEM_Monomial_2d operator*(const VEM_Monomial_2d& other) const {
//         return VEM_Monomial_2d(
//             p + other.p,
//             q + other.q,
//             c * other.c
//         );
//     }
//
//     // ✅ 支持累乘写法 a *= b
//     VEM_Monomial_2d& operator*=(const VEM_Monomial_2d& other) {
//         p += other.p;
//         q += other.q;
//         c *= other.c;
//         return *this;
//     }
// };
// // =======================
// //   三维单项式类
// // =======================
// class VEM_Monomial_3d {
// public:
//     int p, q, r;       // 三个方向的次数
//     Real c; // 系数
//
//     // 构造函数
//     VEM_Monomial_3d(int p_degree = 0, int q_degree = 0, int r_degree = 0,
//                    Real c_coeff = 1.0)
//         : p(p_degree), q(q_degree), r(r_degree),
//           c(c_coeff) {}
//
//
//     inline void pow_table(Real x, int max_deg, Real* out)const noexcept {
//         out[0] = 1.0;
//         for (int i = 1; i <= max_deg; ++i)
//             out[i] = out[i - 1] * x;
//     }
//
//     //
//     // // =======================
//     // // 计算单项式值
//     // // =======================
//     // Real value(Real x, Real y, Real z,
//     //              Real xc, Real yc, Real zc,
//     //              Real hd) const {
//     //     Real dx = x - xc;
//     //     Real dy = y - yc;
//     //     Real dz = z - zc;
//     //
//     //     return c * std::pow(dx, p) *
//     //            std::pow(dy, q) *
//     //            std::pow(dz, r) /
//     //            std::pow(hd, p + q + r);
//     // }
//     //
//     // // =======================
//     // // 计算梯度
//     // // =======================
//     // // _3d_vec gradient(Real x, Real y, Real z,
//     // //                  Real xc, Real yc, Real zc,
//     // //                  Real hd) const {
//     // //     Real dx = x - xc;
//     // //     Real dy = y - yc;
//     // //     Real dz = z - zc;
//     // //     Real denom = std::pow(hd, p + q + r);
//     // //
//     // //     Real d_dx = 0.0, d_dy = 0.0, d_dz = 0.0;
//     // //
//     // //     if (p > 0)
//     // //         d_dx = p *
//     // //                std::pow(dx, p - 1) * std::pow(dy, q) * std::pow(dz, r) * c / denom;
//     // //
//     // //     if (q > 0)
//     // //         d_dy = q *
//     // //                std::pow(dx, p) * std::pow(dy, q - 1) * std::pow(dz, r) * c / denom;
//     // //
//     // //     if (r > 0)
//     // //         d_dz = r *
//     // //                std::pow(dx, p) * std::pow(dy, q) * std::pow(dz, r - 1) * c / denom;
//     // //
//     // //     return _3d_vec{(Real)d_dx, (Real)d_dy, (Real)d_dz};
//     // // }
//     // inline _3d_vec gradient(Real x, Real y, Real z,
//     //                     Real xc, Real yc, Real zc,
//     //                     Real hd) const noexcept {
//     //     const Real dx = x - xc;
//     //     const Real dy = y - yc;
//     //     const Real dz = z - zc;
//     //
//     //     auto powi_safe = [](Real base, int exp) noexcept -> Real {
//     //         if (exp < 0) return 0.0;   // 对于负指数，安全返回0
//     //         if (exp == 0) return 1.0;
//     //         Real res = 1.0;
//     //         for (int i = 0; i < exp; ++i) res *= base;
//     //         return res;
//     //     };
//     //
//     //     const Real inv_denom = 1.0 / powi_safe(hd, p + q + r);
//     //     const Real base = c * inv_denom;
//     //
//     //     Real gx = 0.0, gy = 0.0, gz = 0.0;
//     //
//     //     if (p > 0)
//     //         gx = p * base *
//     //              powi_safe(dx, p - 1) *
//     //              powi_safe(dy, q) *
//     //              powi_safe(dz, r);
//     //     if (q > 0)
//     //         gy = q * base *
//     //              powi_safe(dx, p) *
//     //              powi_safe(dy, q - 1) *
//     //              powi_safe(dz, r);
//     //     if (r > 0)
//     //         gz = r * base *
//     //              powi_safe(dx, p) *
//     //              powi_safe(dy, q) *
//     //              powi_safe(dz, r - 1);
//     //
//     //     return _3d_vec{(Real)gx, (Real)gy, (Real)gz};
//     // }
//     inline Real value(Real x, Real y, Real z,
//                         Real xc, Real yc, Real zc,
//                         Real hd) const noexcept
//     {
//         const Real dx = x - xc;
//         const Real dy = y - yc;
//         const Real dz = z - zc;
//
//         Real px[4], py[4], pz[4]; // p,q,r ≤ 3（可按需要调）
//         pow_table(dx, p, px);
//         pow_table(dy, q, py);
//         pow_table(dz, r, pz);
//
//         Real hpow[10]; // p+q+r ≤ 9 足够
//         pow_table(hd, p + q + r, hpow);
//
//         return c * px[p] * py[q] * pz[r] / hpow[p + q + r];
//     }
//
//     inline _3d_vec gradient(Real x, Real y, Real z,
//                         Real xc, Real yc, Real zc,
//                         Real hd) const noexcept
//     {
//         const Real dx = x - xc;
//         const Real dy = y - yc;
//         const Real dz = z - zc;
//
//         Real px[4], py[4], pz[4];
//         pow_table(dx, p, px);
//         pow_table(dy, q, py);
//         pow_table(dz, r, pz);
//
//         Real hpow[10];
//         pow_table(hd, p + q + r, hpow);
//
//         const Real base = c / hpow[p + q + r];
//
//         Real gx = 0.0, gy = 0.0, gz = 0.0;
//
//         if (p > 0)
//             gx = base * p * px[p - 1] * py[q] * pz[r];
//         if (q > 0)
//             gy = base * q * px[p] * py[q - 1] * pz[r];
//         if (r > 0)
//             gz = base * r * px[p] * py[q] * pz[r - 1];
//
//         return _3d_vec{gx, gy, gz};
//     }
//
//
//     // =======================
//     // 对 x 求导：∂/∂x
//     // =======================
//     VEM_Monomial_3d derivative_x() const {
//         if (p == 0)
//             return VEM_Monomial_3d(0, 0, 0, 0.0);
//         return VEM_Monomial_3d(
//             p - 1, q, r,
//             c
//         ).scale(p);
//     }
//
//     // 对 y 求导：∂/∂y
//     VEM_Monomial_3d derivative_y() const {
//         if (q == 0)
//             return VEM_Monomial_3d(0, 0, 0, 0.0);
//         return VEM_Monomial_3d(
//             p, q - 1, r,
//             c
//         ).scale(q);
//     }
//
//     // 对 z 求导：∂/∂z
//     VEM_Monomial_3d derivative_z() const {
//         if (r == 0)
//             return VEM_Monomial_3d(0, 0, 0, 0.0);
//         return VEM_Monomial_3d(
//             p, q, r - 1,
//             c
//         ).scale(r);
//     }
//
//     // =======================
//     // 缩放系数
//     // =======================
//     VEM_Monomial_3d scale(Real s) const {
//         return VEM_Monomial_3d(p, q, r, c * s);
//     }
//
//     // =======================
//     // 单项式相乘
//     // =======================
//     VEM_Monomial_3d operator*(const VEM_Monomial_3d& other) const {
//         return VEM_Monomial_3d(
//             p + other.p,
//             q + other.q,
//             r + other.r,
//             c * other.c
//         );
//     }
//
//     VEM_Monomial_3d& operator*=(const VEM_Monomial_3d& other) {
//         p += other.p;
//         q += other.q;
//         r += other.r;
//         c *= other.c;
//         return *this;
//     }
//
//     // =======================
//     // 打印（调试用）
//     // =======================
//     void print() const {
//         std::cout << "(cx*(x-xc))^" << p
//                   << " * (cy*(y-yc))^" << q
//                   << " * (cz*(z-zc))^" << r << std::endl;
//     }
// };
// class UNGRID {
// public:
//     int nx, ny, nz; // 网格的维度
//     Real grid_size; // 网格单元大小
//     std::vector<std::vector<std::vector<std::vector<std::pair<int,bool>>>>> grid;
//     // ===============================
//     // 默认构造函数（新增）
//     // ===============================
//     UNGRID()
//         : nx(0), ny(0), nz(0), grid_size(0.0) {}
//
//     // 原有构造函数（保持不变）
//     UNGRID(int n)
//         : nx(n), ny(n), nz(n), grid_size(1.0 / (Real)n) {
//         grid.resize(
//             nx,
//             std::vector<std::vector<std::vector<std::pair<int,bool>>>>(
//                 ny,
//                 std::vector<std::vector<std::pair<int,bool>>>(nz)
//             )
//         );
//     }
//
//     // ===============================
//     // resize 接口（新增）
//     // ===============================
//     void resize(int n) {
//         nx = ny = nz = n;
//         grid_size = 1.0 / (Real)n;
//
//         grid.clear();
//         grid.resize(
//             nx,
//             std::vector<std::vector<std::vector<std::pair<int,bool>>>>(
//                 ny,
//                 std::vector<std::vector<std::pair<int,bool>>>(nz)
//             )
//         );
//     }
//
//     // 计算粒子所在的网格单元索引
//     void get_grid_index(const _3d_vec& pos, int& i, int& j, int& k) const {
//         i = static_cast<int>(pos.vx / grid_size);
//         j = static_cast<int>(pos.vy / grid_size);
//         k = static_cast<int>(pos.vz / grid_size);
//
//         i = std::min(std::max(i, 0), nx - 1);
//         j = std::min(std::max(j, 0), ny - 1);
//         k = std::min(std::max(k, 0), nz - 1);
//     }
//
//     // 通过空间坐标访问对应 grid cell 的值
//     std::pair<int,bool> operator()(const _3d_vec& pos) const {
//         int i, j, k;
//         get_grid_index(pos, i, j, k);
//
//         if (grid[i][j][k].empty()) {
//             return std::make_pair(0, false);
//         }
//         return grid[i][j][k][0];
//     }
//
//
//     template <typename ParticleType>
//     void add_particle_to_grid(const ParticleType& particle, int particle_index) {
//         int i, j, k;
//         get_grid_index(particle.pos, i, j, k);
//         grid[i][j][k].push_back(std::make_pair(particle_index, false));
//     }
//
//     template <typename ParticleType>
//     void add_ghost_particle_to_grid(const ParticleType& particle, int particle_index) {
//         int i, j, k;
//         get_grid_index(particle.pos, i, j, k);
//         grid[i][j][k].push_back(std::make_pair(particle_index, true));
//     }
//
//     template <typename ParticleType>
//     void assign_particles_to_grid(const std::vector<ParticleType>& particles) {
//         FUNC_TIMER();
//         grid.clear();
//         grid.resize(
//             nx,
//             std::vector<std::vector<std::vector<std::pair<int,bool>>>>(
//                 ny,
//                 std::vector<std::vector<std::pair<int,bool>>>(nz)
//             )
//         );
//         for (size_t i = 0; i < particles.size(); ++i) {
//             if (particles[i].deleted) continue;
//             add_particle_to_grid(particles[i], i);
//         }
//     }
//
//     template <typename ParticleType>
//     void add_ghost_particles_to_grid(const std::vector<ParticleType>& particles) {
//         FUNC_TIMER();
//         for (size_t i = 0; i < particles.size(); ++i) {
//             add_ghost_particle_to_grid(particles[i], i);
//         }
//     }
//
//     std::vector<std::pair<int,bool>> get_neighbors(const UGID& grid_id, int n) const {
//         std::vector<std::pair<int,bool>> neighbors;
//         for (int dx = -n; dx <= n; ++dx)
//             for (int dy = -n; dy <= n; ++dy)
//                 for (int dz = -n; dz <= n; ++dz) {
//                     int ni = grid_id.x + dx;
//                     int nj = grid_id.y + dy;
//                     int nk = grid_id.z + dz;
//                     if (ni >= 0 && ni < nx && nj >= 0 && nj < ny && nk >= 0 && nk < nz) {
//                         for (auto particle_index : grid[ni][nj][nk]) {
//                             neighbors.push_back(particle_index);
//                         }
//                     }
//                 }
//         return neighbors;
//     }
// };
// class Boundary_UGRID {
// public:
//     int nx, ny, nz;        // grid resolution
//     Real grid_size;      // cell size (1 / n)
//     std::vector<std::vector<std::vector<bool>>> grid;
//
//     // ===============================
//     // 默认构造
//     // ===============================
//     Boundary_UGRID()
//         : nx(0), ny(0), nz(0), grid_size(0.0) {}
//
//     // ===============================
//     // 构造函数
//     // ===============================
//     Boundary_UGRID(int n)
//         : nx(n), ny(n), nz(n), grid_size(1.0 / (Real)n) {
//         grid.resize(
//             nx,
//             std::vector<std::vector<bool>>(
//                 ny,
//                 std::vector<bool>(nz, false)
//             )
//         );
//     }
//
//     // ===============================
//     // resize
//     // ===============================
//     void resize(int n) {
//         nx = ny = nz = n;
//         grid_size = 1.0 / (Real)n;
//
//         grid.clear();
//         grid.resize(
//             nx,
//             std::vector<std::vector<bool>>(
//                 ny,
//                 std::vector<bool>(nz, false)
//             )
//         );
//     }
//
//     // ===============================
//     // pos → grid index
//     // ===============================
//     void get_grid_index(const _3d_vec& pos, int& i, int& j, int& k) const {
//         i = static_cast<int>(pos.vx / grid_size);
//         j = static_cast<int>(pos.vy / grid_size);
//         k = static_cast<int>(pos.vz / grid_size);
//
//         i = std::min(std::max(i, 0), nx - 1);
//         j = std::min(std::max(j, 0), ny - 1);
//         k = std::min(std::max(k, 0), nz - 1);
//     }
//
//     // ===============================
//     // pos 查询（核心接口）
//     // ===============================
//     bool operator()(const _3d_vec& pos) const {
//         int i, j, k;
//         get_grid_index(pos, i, j, k);
//         return grid[i][j][k];
//     }
//
//     // ===============================
//     // index 查询
//     // ===============================
//     bool operator()(int i, int j, int k) const {
//         if (i < 0 || i >= nx || j < 0 || j >= ny || k < 0 || k >= nz)
//             return false;
//         return grid[i][j][k];
//     }
//
//     // ===============================
//     // n-ring 邻域（返回 cell index）
//     // ===============================
//     std::vector<UGID> get_neighbors(const UGID& id, int n) const {
//         std::vector<UGID> neighbors;
//         neighbors.reserve((2*n+1)*(2*n+1)*(2*n+1));
//
//         for (int dx = -n; dx <= n; ++dx)
//             for (int dy = -n; dy <= n; ++dy)
//                 for (int dz = -n; dz <= n; ++dz) {
//
//                     int ni = id.x + dx;
//                     int nj = id.y + dy;
//                     int nk = id.z + dz;
//
//                     if (ni < 0 || ni >= nx ||
//                         nj < 0 || nj >= ny ||
//                         nk < 0 || nk >= nz)
//                         continue;
//
//                     neighbors.emplace_back(ni, nj, nk);
//                 }
//
//         return neighbors;
//     }
// };
//
//
//
// // class VEM_Monomial {
// // public:
// //     int p;       // x 方向次数
// //     int q;       // y 方向次数
// //     Real cx;   // x 系数
// //     Real cy;   // y 系数
// //
// //     VEM_Monomial(int p_degree = 0, int q_degree = 0,
// //                  Real cx_coeff = 1.0, Real cy_coeff = 1.0)
// //         : p(p_degree), q(q_degree), cx(cx_coeff), cy(cy_coeff) {}
// //
// //     // 计算单项式值: (cx*(x-xc))^p * (cy*(y-yc))^q / hd^(p+q)
// //     Real value(Real x, Real y, Real xc, Real yc, Real hd) const {
// //         Real dx = x - xc;
// //         Real dy = y - yc;
// //         return std::pow(cx * dx, p) * std::pow(cy * dy, q) / std::pow(hd, p + q);
// //     }
// //
// //     // 计算梯度 (d/dx, d/dy)
// //     _2d_vec gradient(Real x, Real y, Real xc, Real yc, Real hd) const {
// //         Real dx = x - xc;
// //         Real dy = y - yc;
// //
// //         Real d_dx = 0.0, d_dy = 0.0;
// //         Real denom = std::pow(hd, p + q);
// //
// //         if (p > 0)
// //             d_dx = p * std::pow(cx, p) * std::pow(cy, q) *
// //                    std::pow(dx, p - 1) * std::pow(dy, q) * cx / denom;
// //
// //         if (q > 0)
// //             d_dy = q * std::pow(cx, p) * std::pow(cy, q) *
// //                    std::pow(dx, p) * std::pow(dy, q - 1) * cy / denom;
// //
// //         return _2d_vec{(Real)d_dx, (Real)d_dy};
// //     }
// //     // ✅ 返回 ∂/∂x 的符号性导数（新的单项式）
// //     VEM_Monomial derivative_x() const {
// //         if (p == 0)
// //             return VEM_Monomial(0, 0, 0.0, 0.0); // 常数项对 x 求导为 0
// //
// //         // ∂/∂x [ cx*((x-xc))^p cy*((y-yc))^q ] =
// //         //   p * cx * ((x-xc))^(p-1) * cy * ((y-yc))^q
// //         return VEM_Monomial(
// //             p - 1,
// //             q,
// //             cx,   // 一个来自原 monomial，一个来自导数链式法则的 cx
// //             cy
// //         ).scale(p);    // 系数 p
// //     }
// //
// //     // ✅ 返回 ∂/∂y 的符号性导数（新的单项式）
// //     VEM_Monomial derivative_y() const {
// //         if (q == 0)
// //             return VEM_Monomial(0, 0, 0.0, 0.0);
// //
// //         return VEM_Monomial(
// //             p,
// //             q - 1,
// //             cx,
// //             cy
// //         ).scale(q);
// //     }
// //
// //     // ✅ 缩放当前单项式的系数（返回新对象）
// //     VEM_Monomial scale(Real s) const {
// //         return VEM_Monomial(p, q, cx * s, cy);
// //     }
// //     // ✅ 单项式相乘：返回新的 VEM_Monomial
// //     VEM_Monomial operator*(const VEM_Monomial& other) const {
// //         // cx1*((x-xc))^p1 * cx2*((x-xc))^p2 = (cx1*cx2)^(?)?
// //         // 实际上合并规则是：
// //         //   p_new = p1 + p2
// //         //   q_new = q1 + q2
// //         //   cx_new = cx1 * cx2
// //         //   cy_new = cy1 * cy2
// //         return VEM_Monomial(
// //             p + other.p,
// //             q + other.q,
// //             cx * other.cx,
// //             cy * other.cy
// //         );
// //     }
// //
// //     // ✅ 支持累乘写法 a *= b
// //     VEM_Monomial& operator*=(const VEM_Monomial& other) {
// //         p += other.p;
// //         q += other.q;
// //         cx *= other.cx;
// //         cy *= other.cy;
// //         return *this;
// //     }
// // };

#pragma once

#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <string>
#include <random>
#include <utility>
#include <cstddef>
#include <algorithm>

#include "config.h"

// 只有声明需要的头
#include <iosfwd>

struct CSR64 {
    using ValueType = GlobalReal;
    using ColumnType = int;
    using PointerType = ptrdiff_t;

    PointerType nrows = 0;
    PointerType ncols = 0;

    std::vector<PointerType> row_ptr;
    std::vector<ColumnType> col_idx;
    std::vector<ValueType> values;

    void clear();
};

class FunctionTimer {
public:
    explicit FunctionTimer(const std::string& name);
    ~FunctionTimer();

private:
    std::string name_;
    std::chrono::high_resolution_clock::time_point start_;
};

#define FUNC_TIMER() FunctionTimer __function_timer__(__func__)

class FunctionMemoryTimer {
public:
    struct GpuMemorySnapshot {
        bool available = false;
        size_t used_bytes = 0;
        size_t free_bytes = 0;
        size_t total_bytes = 0;
    };

    explicit FunctionMemoryTimer(const std::string& name);
    ~FunctionMemoryTimer();

private:
    std::string name_;
    std::chrono::high_resolution_clock::time_point start_;
    size_t rss_before_kb_ = 0;
    GpuMemorySnapshot gpu_before_;
};

#define FUNC_MEMORY_TIMER() FunctionMemoryTimer __function_memory_timer__(__func__)

class RZTimer {
public:
    static void mark(int idx);
};

#define RZTIME(idx) RZTimer::mark(idx)

// -------------------------
// 日志辅助：模板必须放头文件
// -------------------------
#include <sstream>
#include <iostream>

template<typename T>
void rz_stream_args(std::ostringstream& oss, T&& arg) {
    oss << std::forward<T>(arg);
}

template<typename T, typename... Args>
void rz_stream_args(std::ostringstream& oss, T&& arg, Args&&... args) {
    oss << std::forward<T>(arg);
    rz_stream_args(oss, std::forward<Args>(args)...);
}

#define RZ_INFO(...)                                                     \
do {                                                                     \
    std::ostringstream _rz_oss;                                          \
    rz_stream_args(_rz_oss, __VA_ARGS__);                                \
    std::cout << "[RZ_INFO] " << _rz_oss.str() << std::endl;             \
} while (0)

#define RZ_DEBUG_INFO()                                                  \
do {                                                                     \
    std::cout << "[RZ_DEBUG] " << __FILE__ << ":" << __LINE__            \
              << std::endl;                                              \
} while (0)

#define RZ_ASSERT(cond, ...)                                                             \
do {                                                                                     \
    if (!(cond)) {                                                                       \
        std::ostringstream _rz_oss;                                                      \
        rz_stream_args(_rz_oss, __VA_ARGS__);                                            \
        std::cerr << "[RZ_ASSERT FAILED] " << __FILE__ << ":" << __LINE__                \
                  << " (" << #cond << ")\n";                                             \
        std::cerr << "[RZ_ASSERT] " << _rz_oss.str() << std::endl;                       \
        std::abort();                                                                    \
    }                                                                                    \
} while (0)

#define RZ_CHECK(cond, ...)                                                              \
do {                                                                                     \
    if (!(cond)) {                                                                       \
        std::ostringstream _rz_oss;                                                      \
        rz_stream_args(_rz_oss, __VA_ARGS__);                                            \
        std::cerr << "[RZ_CHECK FAILED] " << __FILE__ << ":" << __LINE__                 \
                  << " (" << #cond << ")\n";                                             \
        std::cerr << "[RZ_CHECK] " << _rz_oss.str() << std::endl;                        \
    }                                                                                    \
} while (0)

#define RZ_FATAL(...)                                                                    \
do {                                                                                     \
    std::ostringstream _rz_oss;                                                          \
    rz_stream_args(_rz_oss, __VA_ARGS__);                                                \
    std::cerr << "[RZ_FATAL] " << __FILE__ << ":" << __LINE__ << "\n";                  \
    std::cerr << "[RZ_FATAL] " << _rz_oss.str() << std::endl;                           \
    std::abort();                                                                        \
} while (0)

// -------------------------
// 基础数据结构
// -------------------------
struct RGB_Screen {
    int width, height;
    std::vector<unsigned char> pixels;
};

template<typename T>
class Array2D {
public:
    Array2D(size_t nx = 0, size_t ny = 0, T init = T{})
        : nx_(nx), ny_(ny), data_(nx * ny, init) {}

    inline T& operator()(size_t i, size_t j) {
        RZ_ASSERT(i < nx_ && j < ny_, "out of range");
        return data_[i + nx_ * j];
    }

    inline const T& operator()(size_t i, size_t j) const {
        RZ_ASSERT(i < nx_ && j < ny_, "out of range");
        return data_[i + nx_ * j];
    }

    inline void resize(size_t nx, size_t ny, T init = T{}) {
        nx_ = nx; ny_ = ny;
        data_.assign(nx_ * ny_, init);
    }

    inline void fill(T value) { std::fill(data_.begin(), data_.end(), value); }
    inline size_t nx() const { return nx_; }
    inline size_t ny() const { return ny_; }
    inline size_t size() const { return data_.size(); }
    inline T* data() { return data_.data(); }
    inline const T* data() const { return data_.data(); }

    void swap(Array2D& other) noexcept {
        using std::swap;
        RZ_ASSERT(nx_ == other.nx_ && ny_ == other.ny_, "swap dim!");
        swap(nx_, other.nx_);
        swap(ny_, other.ny_);
        swap(data_, other.data_);
    }

    Array2D operator-(const Array2D& rhs) const {
        RZ_ASSERT(nx_ == rhs.nx_ && ny_ == rhs.ny_, "dimension mismatch");
        Array2D result(nx_, ny_);
        for (size_t idx = 0; idx < data_.size(); ++idx) {
            result.data_[idx] = data_[idx] - rhs.data_[idx];
        }
        return result;
    }

    Array2D& operator-=(const Array2D& rhs) {
        RZ_ASSERT(nx_ == rhs.nx_ && ny_ == rhs.ny_, "dimension mismatch");
        for (size_t idx = 0; idx < data_.size(); ++idx) {
            data_[idx] -= rhs.data_[idx];
        }
        return *this;
    }

private:
    size_t nx_, ny_;
    std::vector<T> data_;
};

template<typename T>
class Array3D {
public:
    Array3D(size_t nx = 0, size_t ny = 0, size_t nz = 0, T init = T{})
        : nx_(nx), ny_(ny), nz_(nz), data_(nx * ny * nz, init) {}

    inline T& operator()(size_t i, size_t j, size_t k) {
        RZ_ASSERT(i < nx_ && j < ny_ && k < nz_, "out of range");
        return data_[i + nx_ * (j + ny_ * k)];
    }

    inline const T& operator()(size_t i, size_t j, size_t k) const {
        RZ_ASSERT(i < nx_ && j < ny_ && k < nz_, "out of range");
        return data_[i + nx_ * (j + ny_ * k)];
    }

    inline void resize(size_t nx, size_t ny, size_t nz, T init = T{}) {
        nx_ = nx; ny_ = ny; nz_ = nz;
        data_.assign(nx_ * ny_ * nz_, init);
    }

    inline void fill(T value) { std::fill(data_.begin(), data_.end(), value); }
    inline size_t nx() const { return nx_; }
    inline size_t ny() const { return ny_; }
    inline size_t nz() const { return nz_; }
    inline size_t size() const { return data_.size(); }
    inline T* data() { return data_.data(); }
    inline const T* data() const { return data_.data(); }

    void swap(Array3D& other) noexcept {
        using std::swap;
        RZ_ASSERT(nx_ == other.nx_ && ny_ == other.ny_ && nz_ == other.nz_, "swap dim!");
        swap(nx_, other.nx_);
        swap(ny_, other.ny_);
        swap(nz_, other.nz_);
        swap(data_, other.data_);
    }

    Array3D operator-(const Array3D& rhs) const {
        RZ_ASSERT(nx_ == rhs.nx_ && ny_ == rhs.ny_ && nz_ == rhs.nz_, "dimension mismatch");
        Array3D result(nx_, ny_, nz_);
#pragma omp parallel for schedule(static)
        for (size_t idx = 0; idx < data_.size(); ++idx) {
            result.data_[idx] = data_[idx] - rhs.data_[idx];
        }
        return result;
    }

    Array3D& operator-=(const Array3D& rhs) {
        RZ_ASSERT(nx_ == rhs.nx_ && ny_ == rhs.ny_ && nz_ == rhs.nz_, "dimension mismatch");
#pragma omp parallel for schedule(static)
        for (size_t idx = 0; idx < data_.size(); ++idx) {
            data_[idx] -= rhs.data_[idx];
        }
        return *this;
    }

    inline void clamp01() {
#pragma omp parallel for schedule(static)
        for (size_t idx = 0; idx < data_.size(); ++idx) {
            if (data_[idx] < 0.0) data_[idx] = 0.0;
            else if (data_[idx] > 1.0) data_[idx] = 1.0;
        }
    }

    size_t nx_, ny_, nz_;
private:
    std::vector<T> data_;
};

struct _2d_vec {
    Real vx, vy;
    _2d_vec(Real vx = 0.0, Real vy = 0.0) : vx(vx), vy(vy) {}

    _2d_vec operator+(const _2d_vec& rhs) const { return _2d_vec(vx + rhs.vx, vy + rhs.vy); }
    _2d_vec operator-(const _2d_vec& rhs) const { return _2d_vec(vx - rhs.vx, vy - rhs.vy); }
    _2d_vec operator*(const Real& rhs) const { return _2d_vec(vx * rhs, vy * rhs); }
    _2d_vec operator/(const Real& rhs) const { return _2d_vec(vx / rhs, vy / rhs); }

    int x_to_lower() const { return static_cast<int>(std::floor(vx)); }
    int x_to_upper() const { return static_cast<int>(std::ceil(vx)); }
    int y_to_lower() const { return static_cast<int>(std::floor(vy)); }
    int y_to_upper() const { return static_cast<int>(std::ceil(vy)); }

    static Real cross(const _2d_vec& a, const _2d_vec& b) {
        return a.vx * b.vy - a.vy * b.vx;
    }

    Real norm() const { return std::sqrt(vx * vx + vy * vy); }
};

struct _3d_vec {
    Real vx, vy, vz;
    _3d_vec(Real vx = 0.0, Real vy = 0.0, Real vz = 0.0) : vx(vx), vy(vy), vz(vz) {}

    _3d_vec operator+(const _3d_vec& rhs) const;
    _3d_vec operator-(const _3d_vec& rhs) const;
    _3d_vec operator*(const Real& rhs) const;
    _3d_vec operator/(const Real& rhs) const;
    Real dot(const _3d_vec& rhs) const;
    _3d_vec cross(const _3d_vec& rhs) const;
    _3d_vec opposite() const;
    Real norm() const;
    Real squared_norm() const;
    _3d_vec normalized() const;
    std::string to_obj_format() const;

    int x_to_lower() const;
    int x_to_upper() const;
    int y_to_lower() const;
    int y_to_upper() const;
    int z_to_lower() const;
    int z_to_upper() const;

    static bool sort(const _3d_vec& a, const _3d_vec& b);
};

struct FLIP_Particles {
    _2d_vec pos, vec, pic_vec;
    bool deleted;
    FLIP_Particles();
    FLIP_Particles(const _2d_vec& p, const _2d_vec& v);
};

struct UGID {
    int x, y, z;
    UGID(const int& x_, const int& y_, const int& z_) : x(x_), y(y_), z(z_) {}
};

struct FLIP_Particles_3d {
    _3d_vec pos, old_pos, vec, pic_vec;
    int contained_in_element_id;
    bool deleted, need_interpolate;
    Real temp = 0.0, rho = 0.0;
    FLIP_Particles_3d();
    FLIP_Particles_3d(const _3d_vec& p, const _3d_vec& v);
};

struct Source_range {
    int xmin, xmax, ymin, ymax, zmin, zmax;
    Source_range(int xmin_, int xmax_, int ymin_, int ymax_, int zmin_, int zmax_);
};

struct sphere_source {
    _3d_vec center;
    Real radius;
    _3d_vec init_vec;

    sphere_source(Real xc_, Real yc_, Real zc_, Real r_);
    bool contains(const _3d_vec& p) const;
    std::vector<_3d_vec> sample_points(int n) const;
};

struct cylinder_source {
    _3d_vec center;
    _3d_vec axis;
    Real radius;
    Real half_height;
    _3d_vec init_vec;

    cylinder_source(const _3d_vec& c, const _3d_vec& axis_dir, Real r, Real h);
    bool contains(const _3d_vec& p) const;
    std::vector<_3d_vec> sample_points(int n) const;
};

struct IVOCK_Kernel {
    Real _h;
    explicit IVOCK_Kernel(Real h) : _h(h) {}

    static inline Real H(Real r) {
        Real res = 0.0;
        if (r >= -1.0 && r < 0.0) res = 1.0 + r;
        if (r >=  0.0 && r < 1.0) res = 1.0 - r;
        return res;
    }

    inline Real weight(Real gx, Real gy, Real gz,
                         Real px, Real py, Real pz) const {
        const Real dx = px - gx;
        const Real dy = py - gy;
        const Real dz = pz - gz;
        return H(dx / _h) * H(dy / _h) * H(dz / _h);
    }

    inline bool in_support(Real gx, Real gy, Real gz,
                           Real px, Real py, Real pz) const {
        return (std::fabs(px - gx) < _h &&
                std::fabs(py - gy) < _h &&
                std::fabs(pz - gz) < _h);
    }
};

Real cubic_kernel(Real r, Real h);
Real desbrun_gascuel_cubic(Real r, Real h);
Real monaghan_cubic(Real r, Real h);
Real cubic_kernel_new(Real r, Real h);

struct Smoke_Particles_3d {
    _3d_vec old_pos, pos, v;
    bool deleted;
    int contained_in_element_id;
    Smoke_Particles_3d();
    explicit Smoke_Particles_3d(const _3d_vec& p);
};

struct Trancer {
    float x, y, z, rest_age;
    uint32_t id;
};

class VEM_Monomial_2d {
public:
    int p, q;
    Real c;

    VEM_Monomial_2d(int p_degree = 0, int q_degree = 0, Real c_coeff = 1.0);

    void pow_table_2d(Real x, int max_deg, Real* out) const noexcept;
    Real value(Real x, Real y, Real xc, Real yc, Real hd) const noexcept;
    _2d_vec gradient(Real x, Real y, Real xc, Real yc, Real hd) const noexcept;

    VEM_Monomial_2d derivative_x() const;
    VEM_Monomial_2d derivative_y() const;
    VEM_Monomial_2d scale(Real s) const;
    VEM_Monomial_2d operator*(const VEM_Monomial_2d& other) const;
    VEM_Monomial_2d& operator*=(const VEM_Monomial_2d& other);
};

class VEM_Monomial_3d {
public:
    int p, q, r;
    Real c;

    VEM_Monomial_3d(int p_degree = 0, int q_degree = 0, int r_degree = 0, Real c_coeff = 1.0);

    void pow_table(Real x, int max_deg, Real* out) const noexcept;
    Real value(Real x, Real y, Real z,
                 Real xc, Real yc, Real zc,
                 Real hd) const noexcept;
    _3d_vec gradient(Real x, Real y, Real z,
                     Real xc, Real yc, Real zc,
                     Real hd) const noexcept;

    VEM_Monomial_3d derivative_x() const;
    VEM_Monomial_3d derivative_y() const;
    VEM_Monomial_3d derivative_z() const;
    VEM_Monomial_3d scale(Real s) const;
    VEM_Monomial_3d operator*(const VEM_Monomial_3d& other) const;
    VEM_Monomial_3d& operator*=(const VEM_Monomial_3d& other);
    void print() const;
};

class UNGRID {
public:
    int nx, ny, nz;
    Real grid_size;
    std::vector<std::vector<std::vector<std::vector<std::pair<int,bool>>>>> grid;

    UNGRID();
    explicit UNGRID(int n);
    void resize(int n);
    void get_grid_index(const _3d_vec& pos, int& i, int& j, int& k) const;
    std::pair<int,bool> operator()(const _3d_vec& pos) const;

    template <typename ParticleType>
    void add_particle_to_grid(const ParticleType& particle, int particle_index) {
        int i, j, k;
        get_grid_index(particle.pos, i, j, k);
        grid[i][j][k].push_back(std::make_pair(particle_index, false));
    }

    template <typename ParticleType>
    void add_ghost_particle_to_grid(const ParticleType& particle, int particle_index) {
        int i, j, k;
        get_grid_index(particle.pos, i, j, k);
        grid[i][j][k].push_back(std::make_pair(particle_index, true));
    }

    template <typename ParticleType>
    void assign_particles_to_grid(const std::vector<ParticleType>& particles) {
        FUNC_TIMER();
        grid.clear();
        grid.resize(
            nx,
            std::vector<std::vector<std::vector<std::pair<int,bool>>>>(
                ny,
                std::vector<std::vector<std::pair<int,bool>>>(nz)
            )
        );
        for (size_t i = 0; i < particles.size(); ++i) {
            if (particles[i].deleted) continue;
            add_particle_to_grid(particles[i], i);
        }
    }

    template <typename ParticleType>
    void add_ghost_particles_to_grid(const std::vector<ParticleType>& particles) {
        FUNC_TIMER();
        for (size_t i = 0; i < particles.size(); ++i) {
            add_ghost_particle_to_grid(particles[i], i);
        }
    }

    std::vector<std::pair<int,bool>> get_neighbors(const UGID& grid_id, int n) const;
};

class Boundary_UGRID {
public:
    int nx, ny, nz;
    Real grid_size;
    std::vector<std::vector<std::vector<bool>>> grid;

    Boundary_UGRID();
    explicit Boundary_UGRID(int n);
    void resize(int n);
    void get_grid_index(const _3d_vec& pos, int& i, int& j, int& k) const;
    bool operator()(const _3d_vec& pos) const;
    bool operator()(int i, int j, int k) const;
    std::vector<UGID> get_neighbors(const UGID& id, int n) const;
};
