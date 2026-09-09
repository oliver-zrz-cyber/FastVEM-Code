#pragma once
#include "dfn.h"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <set>
#include <CGAL/Plane_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
struct ElementBBox {
    Real xmin, xmax, ymin, ymax;
};
struct ElementBBox_3d {
    Real xmin, xmax, ymin, ymax, zmin, zmax;
};

struct Edge {
    int v1, v2;
    Edge(int a, int b) {
        if (a < b) {
            v1 = a;
            v2 = b;
        } else {
            v1 = b;
            v2 = a;
        }
    }

    // 相等比较：无向边
    bool operator==(const Edge& other) const {
        return (v1 == other.v1 && v2 == other.v2) ||
               (v1 == other.v2 && v2 == other.v1);
    }

    // 排序比较：把小的点放前面再比较
    bool operator<(const Edge& other) const {
        int a1 = std::min(v1, v2);
        int a2 = std::max(v1, v2);
        int b1 = std::min(other.v1, other.v2);
        int b2 = std::max(other.v1, other.v2);

        if (a1 != b1) return a1 < b1;
        return a2 < b2;
    }
};

struct PlaneData {
    _3d_vec origin;
    _3d_vec normal;
    _3d_vec u_axis;
    _3d_vec v_axis;
    Real area = 0.0;
    Real diameter = 0.0;
    _3d_vec _3d_centroid = _3d_vec(0,0,0);
    bool is_degenerated = false;
    CGAL::Plane_3<K> cgal_plane;
    _2d_vec _to_2d(const _3d_vec&p) const{
        _3d_vec rel = p - origin;
        Real u = rel.vx * u_axis.vx + rel.vy * u_axis.vy + rel.vz * u_axis.vz;
        Real v = rel.vx * v_axis.vx + rel.vy * v_axis.vy + rel.vz * v_axis.vz;
        return {u,v};
    }
    _3d_vec _to_3d(const _2d_vec& p) const {
        Real x = origin.vx + p.vx * u_axis.vx + p.vy * v_axis.vx;
        Real y = origin.vy + p.vx * u_axis.vy + p.vy * v_axis.vy;
        Real z = origin.vz + p.vx * u_axis.vz + p.vy * v_axis.vz;
        return {_3d_vec{x, y, z}};
    }

};

class VEM_Element_2d {// 这里应该不需要知道谁是 Boundary
public:
    PlaneData plane;// 有了它 这个就是一个3d中的面
    std::vector<_2d_vec> points;
    // Eigen::MatrixXr project_matrix_star,B,D,G,project_matrix,
    // project_matrix_k1,project_matrix_star_k1,B_k1,D_k1,G_k1,stiffness_matrix_k1,
    // stiffness_matrix,local_LX,local_LY,H_X,H_Y;

    Eigen::MatrixXr project_matrix_star,
    project_matrix_star_k1,stiffness_matrix_k1,
    stiffness_matrix,local_LX,local_LY;


    _2d_vec centroid;
    _3d_vec _3d_centroid;
    bool degenerated = false;
    bool build_geometric = true;
    Real area,diameter;
    std::vector<int> local_to_global_dof;
    // std::vector<std::pair<int,int>>temp_volume_dof_to_local_dof;
    std::map<int,int> volume_v_to_local_v;
    std::map<int,int> volume_dof_to_local_dof;
    VEM_Element_2d()
        : centroid{0.0, 0.0},
          area(0.0),
          diameter(0.0)
    {
    }

    // something ..
    std::vector<int> contained_flip_particles;

    // for vem3d

    bool compute_plane_and_projection(std::vector<_3d_vec>& points,const PlaneData& plane_data,const _3d_vec& centroid,std::map<Edge,int>& edge_to_volume_dof);

    std::vector<VEM_Monomial_2d> project_monomial_to_plane(
    const VEM_Monomial_3d &m3d,
    const _3d_vec &center3d  // (xc, yc, zc)
    )const;
    Real polygon_moment_general_2d(const VEM_Monomial_2d& mono,const _2d_vec& moment_centroid,
                                  bool central = true) const ;
    Real polygon_moment_general_2d_advance(const VEM_Monomial_2d& mono,const _2d_vec& moment_centroid,
                                  bool central = true) const ;
    Real polygon_moment_2d_sub_poly(const VEM_Monomial_2d& mono,const _2d_vec& moment_centroid,
                                    const std::vector<_2d_vec>& points,
                                  bool central = true) const ;
    Real polygon_moment_general_3d(const VEM_Monomial_3d& mono,const _3d_vec& momcentroid,
                              bool central = true) const;
    std::vector<VEM_Monomial_2d> expand_centered_monomial(const VEM_Monomial_2d& mono) const;

    bool build_up();


    bool project_only_build_up();

    bool project_only_build_up_k1();
};


class VEM_Domain {
public:
    std::vector<_2d_vec> points;
    std::vector<bool> is_border;
    std::vector<std::vector<int>> polygons;
    std::set<Edge> temp_unique_edges;
    std::vector<Edge> edges;
    std::map<Edge,int> edgetoid_map;
    std::vector<VEM_Element_2d> vem_elements;
    Eigen::SparseMatrixCr global_stiffness_matrix;
    Eigen::SparseMatrixCr global_stiffness_matrix_k1;
    Eigen::SparseMatrixCr global_LX_matrix;
    Eigen::SparseMatrixCr global_LY_matrix;
    Eigen::VectorXr global_rhs,global_p,global_rhs_k1;
    Eigen::VectorXr k1_dof_u,k1_dof_v;
    std::vector<VEM_Monomial_2d> monomials;

    void read_from_file(std::string filename);

    void init();

    void generate_elements();

    void assemble_global_stiffness_matrix();

    void assemble_global_stiffness_matrix_k1();

    void assemble_global_rhs();

    void assemble_global_rhs_k1();

    void assemble_weakform_rhs(Real dt);

    void solve();
    void solve_k1();
    void solve_pic();
    void save_solution_png(const std::string& filename,
                                   int width = 400,
                                   int height = 400);
    void plot_mesh();
    // something maybe should be splited
    std::vector<FLIP_Particles> flip_particles;
    void init_flip_particles();
    void assign_flip_particles_to_elements();
    std::vector<ElementBBox> elem_bboxes;
    void build_element_bboxes();
    void link_ele_with_uniform_grid(std::vector<std::vector<int>>&grid_to_elems,int grid_res);
    void PtoG();
    void GtoP(Real flip_alpha,Real dt);
    void flip_advection(Real dt);

    void uniform_construct(int nx,int ny);
};







