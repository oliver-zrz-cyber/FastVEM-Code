#pragma once
#include "dfn.h"
#include "vem2d.h"
#include "mac3d.h"
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>
#include <CGAL/Surface_mesh.h>
#include "AbSolver.h"

class BSP_Mesh;
class Split_lcc;
class MiniLcc;
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::Point_3 CGAL_Point_3;
typedef K::Triangle_3 Triangle;
typedef std::list<Triangle>::iterator Iterator;
typedef CGAL::AABB_triangle_primitive<K, Iterator> Primitive;
typedef CGAL::AABB_traits<K, Primitive> AABB_triangle_traits;
typedef CGAL::AABB_tree<AABB_triangle_traits> AABBTree;
typedef CGAL::Surface_mesh<CGAL_Point_3> CGAL_Mesh;

class VEM_Element_3d {
public:
    bool is_U = false;
    bool is_convex = false;
    std::vector<std::vector<int>> polygons;
    std::vector<K::Plane_3> poly_planes;
    std::vector<VEM_Element_2d> elements_2d;
    _3d_vec centroid;
    Real volume,diameter;
    std::vector<int> vertex_local_to_global_dof;
    std::vector<int> edge_local_to_global_dof;
    std::vector<int> polygon_local_to_global_dof;
    // Eigen::MatrixXr project_matrix_star_k2,project_matrix_star_k1,
    // B_k2,D_k2,G_k2,B_k1,D_k1,G_k1,
    // project_matrix_k2,stiffness_matrix_k2,project_matrix_k1,stiffness_matrix_k1,
    // local_LX_k2,local_LY_k2,local_LZ_k2,H_X_k2,H_Y_k2,H_Z_k2,
    // local_LX_k1,local_LY_k1,local_LZ_k1,H_X_k1,H_Y_k1,H_Z_k1;
// 这里 应该 B D G project_matrix_k2 H 都不需要存 都是中间态
    Eigen::MatrixXr project_matrix_star_k2,project_matrix_star_k1,
    local_LX_k2,local_LY_k2,local_LZ_k2,
    local_LX_k1,local_LY_k1,local_LZ_k1;
    Eigen::LocalMatrixXr stiffness_matrix_k2, stiffness_matrix_k1;

    Eigen::VectorXr poly_coeffs_u_k1,poly_coeffs_v_k1,poly_coeffs_w_k1,poly_coeffs_rho_k1,
    poly_coeffs_oldu_k1,poly_coeffs_oldv_k1,poly_coeffs_oldw_k1,
    poly_coeffs_du_k1,poly_coeffs_dv_k1,poly_coeffs_dw_k1;

    bool has_boundary=false;

    VEM_Element_3d()
        : centroid{0.0, 0.0,0.0},
          volume(0.0),
          diameter(0.0)
    {
    }

    // something ..
    std::vector<int> contained_flip_particles;
    std::vector<int> contained_smoke_particles;
    Real integrate_centered_monomial_over_polyhedron(const VEM_Monomial_3d& mono,
                                                       const _3d_vec& C /* (xc,yc,zc) */)const;
    Real integrate_centered_monomial_over_polyhedron_divergence(const VEM_Monomial_3d& mono,
                                                       const _3d_vec& C /* (xc,yc,zc) */)const ;
    void build_up();

    void build_up_k1();

    void save_polygonmesh_to_obj(const std::string& filename) const;

    void save_face_to_obj(const std::vector<int>& polygon,const std::string& filename = "check_face.obj")const;

    bool point_in_elem(const _3d_vec &p,const Real eps = 1e-12) const;

    bool point_in_elem_fast(const _3d_vec& p,const Real eps = 1e-12) const;


    bool point_in_elem_nonconvex(const _3d_vec &p)const;

    void bind_geometry(const std::vector<_3d_vec>& domain_points,
                       const std::vector<Edge>& domain_edges);

    int vertex_count() const;

    int edge_count() const;

    const _3d_vec& vertex_position(int local_vertex_id) const;

    Edge edge(int local_edge_id) const;

    int global_edge_id_from_vertices(int global_v1, int global_v2) const;

    void clear_stiff_k2();

    void clear_stiff_k1();

    void clear_assembly_cache();

    void clear_generate_elements_cache();

    void clear_hierarchy_cache();
private:
    const _3d_vec* domain_points_data_ = nullptr;
    int domain_points_count_ = 0;
    const Edge* domain_edges_data_ = nullptr;
    int domain_edges_count_ = 0;

    static inline Real powi(Real x, int n) {
        if (n == 0) return 1.0;
        Real r = 1.0, b = x;
        int e = n;
        while (e) { if (e & 1) r *= b; b *= b; e >>= 1; }
        return r;
    }

    // 多项式四项（含常数）多项式系数： n!/(a! b! c! d!)，使用 log-gamma 稳定计算
    static inline Real multinom4(int n, int a, int b, int c, int d) {
        // 假定 a+b+c+d == n
        return std::exp(std::lgamma(n + 1.0)
                      - std::lgamma(a + 1.0)
                      - std::lgamma(b + 1.0)
                      - std::lgamma(c + 1.0)
                      - std::lgamma(d + 1.0));
    }

    // 标准 3-simplex 积分：∫ ξ1^A ξ2^B ξ3^C dξ = A!B!C!/(A+B+C+3)!  —— 用 lgamma 稳定计算
    static inline Real simplex_moment(int A, int B, int C) {
        return std::exp(std::lgamma(A + 1.0) + std::lgamma(B + 1.0) + std::lgamma(C + 1.0)
                      - std::lgamma(A + B + C + 4.0));
    }

    // 计算 ∫_{T(a,b,c,d)} (x-xc)^p (y-yc)^q (z-zc)^r dV，
    // 其中这里的 a,b,c,d 是四面体 4 顶点（未中心化），C=(xc,yc,zc) 是中心。
    // 实现时先将 a,b,c,d 全部中心化（减去 C），然后做仿射展开。
    Real integrate_centered_monomial_over_tet(const _3d_vec& a0,
                                                const _3d_vec& b0,
                                                const _3d_vec& c0,
                                                const _3d_vec& d0,
                                                int p, int q, int r,
                                                const _3d_vec& C) const
    {
        // 先将四面体四点中心化
        _3d_vec a{a0.vx - C.vx, a0.vy - C.vy, a0.vz - C.vz};
        _3d_vec b{b0.vx - C.vx, b0.vy - C.vy, b0.vz - C.vz};
        _3d_vec c{c0.vx - C.vx, c0.vy - C.vy, c0.vz - C.vz};
        _3d_vec d{d0.vx - C.vx, d0.vy - C.vy, d0.vz - C.vz};

        // 仿射：X(ξ)=ax + bx1 ξ1 + bx2 ξ2 + bx3 ξ3 等
        const _3d_vec e1 = { b.vx - a.vx, b.vy - a.vy, b.vz - a.vz };
        const _3d_vec e2 = { c.vx - a.vx, c.vy - a.vy, c.vz - a.vz };
        const _3d_vec e3 = { d.vx - a.vx, d.vy - a.vy, d.vz - a.vz };

        // detA
        const Real detA =
            e1.vx * (e2.vy * e3.vz - e2.vz * e3.vy) -
            e1.vy * (e2.vx * e3.vz - e2.vz * e3.vx) +
            e1.vz * (e2.vx * e3.vy - e2.vy * e3.vx);

        const Real J = std::fabs(detA);
        if (J == 0.0) return 0.0; // 退化四面体

        // 常数项（中心化后的 a 点坐标）
        const Real ax = a.vx, ay = a.vy, az = a.vz;
        // 线性系数（A 的三列）
        const Real bx1 = e1.vx, bx2 = e2.vx, bx3 = e3.vx;
        const Real by1 = e1.vy, by2 = e2.vy, by3 = e3.vy;
        const Real bz1 = e1.vz, bz2 = e2.vz, bz3 = e3.vz;

        // 展开 (ax + bx1 ξ1 + bx2 ξ2 + bx3 ξ3)^p 等，并相乘
        Real S = 0.0;

        // X^p 的四项多项式： (bx1 ξ1)^ix (bx2 ξ2)^jx (bx3 ξ3)^kx (ax)^tx，其中 ix+jx+kx+tx=p
        for (int tx = 0; tx <= p; ++tx) {
            const int rx1 = p - tx;
            for (int ix = 0; ix <= rx1; ++ix) {
                const int rx2 = rx1 - ix;
                for (int jx = 0; jx <= rx2; ++jx) {
                    const int kx = rx2 - jx;
                    const Real Cx = multinom4(p, ix, jx, kx, tx)
                                      * powi(ax, tx)
                                      * powi(bx1, ix) * powi(bx2, jx) * powi(bx3, kx);

                    // Y^q
                    for (int ty = 0; ty <= q; ++ty) {
                        const int ry1 = q - ty;
                        for (int iy = 0; iy <= ry1; ++iy) {
                            const int ry2 = ry1 - iy;
                            for (int jy = 0; jy <= ry2; ++jy) {
                                const int ky = ry2 - jy;
                                const Real Cy = multinom4(q, iy, jy, ky, ty)
                                                  * powi(ay, ty)
                                                  * powi(by1, iy) * powi(by2, jy) * powi(by3, ky);

                                // Z^r
                                for (int tz = 0; tz <= r; ++tz) {
                                    const int rz1 = r - tz;
                                    for (int iz = 0; iz <= rz1; ++iz) {
                                        const int rz2 = rz1 - iz;
                                        for (int jz = 0; jz <= rz2; ++jz) {
                                            const int kz = rz2 - jz;
                                            const Real Cz = multinom4(r, iz, jz, kz, tz)
                                                              * powi(az, tz)
                                                              * powi(bz1, iz) * powi(bz2, jz) * powi(bz3, kz);

                                            // ξ 指数
                                            const int A = ix + iy + iz;
                                            const int B = jx + jy + jz;
                                            const int Cc = kx + ky + kz;

                                            // 标准单纯形积分
                                            const Real Iabc = simplex_moment(A, B, Cc);

                                            S += Cx * Cy * Cz * Iabc;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return J * S;
    }

};
class VEM_Domain_3d {
    public:
    std::vector<_3d_vec> points;
    std::vector<std::vector<int>> polygons;
    std::vector<PlaneData> polygons_plane;
    std::vector<Real> level_set;

    std::vector<bool> is_polygon_boundary;
    std::vector<bool> is_obj_boundary;// just temp used for get vels for boundary vertices
    std::vector<bool> is_vertex_boundary;

    std::vector<_3d_vec> vertex_normals,vertex_neumann_vels;
    std::vector<Real> face_neumann_vels;
    std::vector<std::vector<int>> polyhedrons;

    std::vector<VEM_Element_3d> vem_elements;
    std::vector<Edge> edges;

    std::shared_ptr<BSP_Mesh>boundary_mesh;

    bool has_builded_u = false;
    Real U_area = -1;

    Eigen::MatrixXr U2d_project_matrix_star,
    U2d_project_matrix_star_k1;
    Eigen::LocalMatrixXr U2d_stiffness_matrix_k1,
    U2d_stiffness_matrix;

    Eigen::MatrixXr U3d_project_matrix_star_k2,
    U3d_project_matrix_star_k1,U3d_local_LX_k1,U3d_local_LY_k1,U3d_local_LZ_k1
    ,U3d_local_LX_k2,U3d_local_LY_k2,U3d_local_LZ_k2;
    Eigen::LocalMatrixXr U3d_stiffness_matrix_k1,
    U3d_stiffness_matrix_k2;

    Eigen::SparseMatrixR global_stiffness_matrix_k2;
    CSR64 csr_k2,csr_k1;
    AMGCLContext solver_context;
    Eigen::SparseMatrixR global_stiffness_matrix_k1;
    Eigen::SparseMatrixR global_LX_matrix_k1,global_LX_matrix_k2;
    Eigen::SparseMatrixR global_LY_matrix_k1,global_LY_matrix_k2;
    Eigen::SparseMatrixR global_LZ_matrix_k1,global_LZ_matrix_k2;
    Eigen::VectorXr global_rhs_k2,global_p_k2,global_rhs_k1,global_p_k1;
    Eigen::VectorXr k1_dof_u,k1_dof_v,k1_dof_w;
    std::vector<bool> is_Dirichlet_dof;
    std::vector<bool> is_internal_dof;
    std::vector<int> liquid_internal_dofs;
    std::vector<int> liquid_dofs_within_liquid_grids;
    std::vector<int> liquid_global_to_internal_dof;
    std::vector<bool> element_has_liquid;
    Eigen::VectorXr k1_dof_du,k1_dof_dv,k1_dof_dw;
    Eigen::VectorXr k1_dof_u_pmg,k1_dof_v_pmg,k1_dof_w_pmg;
    Eigen::VectorXr k1_dof_density,k1_dof_temp;
    std::list<Triangle> tris;
    AABBTree aabb_tree;

    VEM_UGRID resample_grid;
    Boundary_UGRID boundary_handle_grid;
    void build_boundary_handle_grid();
    std::vector<std::vector<int>> grid_to_elems;
    std::vector<Trancer> tracers;

    std::vector<Trancer> lucy_tracers_1;

    std::vector<Trancer> lucy_tracers_2;

    int grid_res;

    Real _alpha=0.0,_beta=0.0,_decay=0.0;

    bool read_from_file(std::string filename);

    void load_from_lcc(Split_lcc& lcc);

    void load_from_Minilcc(MiniLcc& lcc);

    void load_from_Minilcc_lod(MiniLcc& lcc,const std::vector<int>& lod);

    void init();

    void preprocess_polygons();

    void generate_elements_k1_only();

    void generate_elements(bool generate_2d = true);

    void compute_face_neumann_velocity();

    void assemble_global_stiffness_matrix_k2();


    void assemble_global_stiffness_matrix_k1();

    std::vector<ElementBBox_3d> elem_bboxes;
    void assign_flip_smoke_particles_to_elements_3d(std::vector<FLIP_Particles_3d>& flip_particles,std::vector<Smoke_Particles_3d>&smoke_particles);

    void assign_flip_smoke_particles_to_elements_3d(std::vector<FLIP_Particles_3d>& flip_particles);

    std::vector<int> assign_pts_to_elements_3d(std::vector<_3d_vec>& pts);

    std::vector<int> assign_pts_to_elements_3d_with_BH(std::vector<_3d_vec>& pts,const std::vector<_3d_vec>& old_pts);

    std::vector<int> assign_tracers_to_elements_3d();

    std::vector<int> assign_lucy_tracers1_to_elements_3d();

    std::vector<int> assign_lucy_tracers2_to_elements_3d();

    void build_element_bboxes();

    void link_ele_with_uniform_grid_3d();


    void init_flip_particles(std::vector<FLIP_Particles_3d>& flip_particles,
                             std::vector<FLIP_Particles_3d>& ghost_particles,
                             Real kernel_radius);

    void init_liquid(std::vector<FLIP_Particles_3d>& flip_particles,int n);



    // void PtoG(std::vector<FLIP_Particles_3d>& flip_particles,std::vector<Smoke_Particles_3d>&smoke_particles);
    //
    // void PtoG(std::vector<FLIP_Particles_3d>& flip_particles,int n);

    void PtoG(std::vector<FLIP_Particles_3d>& flip_particles,
        std::vector<FLIP_Particles_3d>& ghost_particles,Real cubk);


    void apply_pressure_k1_poly(Real dt);


    void apply_pressure_k2_poly(Real dt);

    void GtoP_k2(std::vector<FLIP_Particles_3d>& flip_particles,Real flip_alpha,Real dt);

    void GtoP_k2_poly(std::vector<FLIP_Particles_3d>& flip_particles,Real flip_alpha);

    void CFL_GtoPicV(std::vector<FLIP_Particles_3d>& flip_particles, Real flip_alpha) ;

    void GtoP_k1(std::vector<FLIP_Particles_3d>& flip_particles,Real flip_alpha,Real dt);

    void resample_GtoP(std::vector<FLIP_Particles_3d>& flip_particles);

    void inter_rho_to_mg(MAC_Grid3D& mg);

    void flip_advection(std::vector<FLIP_Particles_3d>& flip_particles,const Real& dt);

    void RK_3_advection_k2(std::vector<FLIP_Particles_3d>& flip_particles,const Real& dt);

    void RK_3_advection_liquid(std::vector<FLIP_Particles_3d>& flip_particles,const Real& dt);

    void tracers_RK_3(const Real& dt);

    void flip_adv_skull(std::vector<FLIP_Particles_3d>& flip_particles,const Real& dt);

    void tracer_adv_skull(const Real& dt);

    void tracers_RK_3_lucy(const Real& dt);

    void RK_3_advection_k2(ConcentrationField3D& concentration_field,const Real& dt);

    void smoke_advection(std::vector<Smoke_Particles_3d>&smoke_particles,const Real& dt);

    void smoke_advection_RK2(std::vector<Smoke_Particles_3d> &smoke_particles, const Real &dt);

    void add_external_forces(std::vector<Smoke_Particles_3d>&smoke_particles,const Real& dt,const Real& kernel_radius);

    void add_external_forces(const Real& dt);

    void add_liquid_external_forces(const Real& dt);

    void heat_decay(std::vector<FLIP_Particles_3d>& flip_particles,const Real& dt);

    void heat_decay(const Real& dt);

    void assemble_weakform_rhs_k1(Real dt);

    void assemble_liquid_weakform_rhs_k1(Real dt);

    void liquid_weakform_rhs_k1();

    void assemble_weakform_rhs_k2(Real dt);

    void assemble_weakform_rhs_boundary_k2(Real dt);

    void assemble_weakform_rhs_boundary_k1(Real dt);

    void assemble_strongform_rhs_k1(Real dt);

    void solve_k2(Real dt);

    void solve_k1(Real dt);

    Eigen::SparseMatrixCr build_Ak2();

    Eigen::SparseMatrixR build_Ak1();

    void visualize_solved_pressure_k2(int nx,int ny,int nz);

    void visualize_solved_pressure_k1(int nx,int ny,int nz);

    void visualize_rhs_k1(int nx,int ny,int nz);

    void visualize_rhs_k2(int nx,int ny,int nz);

    Eigen::VectorXr Warm_from_k1(const Eigen::VectorXr& dof_p_k1);

    void Renew_rhs_k1(const Eigen::VectorXr& p_k2, Real dt);

    void save_boundary_to_obj();

    void save_obj_surface_to_obj();

    void save_boundary_cells();

    void build_AABB();

    void direct_AABB(std::string aabbobj);


    void boundary_aware_pos_fix(const std::vector<_3d_vec>&old_poses,std::vector<_3d_vec>&new_poses,const Real dt);

    void Boundary_Handle(Smoke_Particles_3d& particle, AABBTree& aabb_tree,const Real& dt);

    void Boundary_Handle(FLIP_Particles_3d& particle, AABBTree& aabb_tree,const Real& dt);

    void Boundary_Handle(_3d_vec old_pos,_3d_vec& new_pos, AABBTree& aabb_tree,const Real& dt);

    void Boundary_Handle_vel_change(FLIP_Particles_3d& flip,_3d_vec old_pos,_3d_vec& new_pos, AABBTree& aabb_tree,const Real& dt);

    void liquid_boundary_aware_pos_fix(std::vector<FLIP_Particles_3d>& flips,const std::vector<_3d_vec>&old_poses,std::vector<_3d_vec>&new_poses,const Real dt);


    void ensure_polygon_normals_outward(VEM_Element_3d &element);

    void orient_mesh_and_compute_volume(VEM_Element_3d &element);

    void cgal_ccw_orientation(VEM_Element_3d &element);

    void cgal_centroid_volume(VEM_Element_3d &element) ;

    void simple_label_Dirichlet_dofs(const std::vector<FLIP_Particles_3d> &particles);

    void construct_level_set(const std::vector<FLIP_Particles_3d> &particles);

    void save_level_set_to_mesh(const std::string filename);

    void ghostvem_label_Dirichlet_dofs();

    Eigen::SparseMatrixR build_ghost_Ak1();

    void clear_data_for_build();

    void clear_generate_elements_cache(bool trim_allocator = false);

    void clear_hierarchy_build_cache();
};
