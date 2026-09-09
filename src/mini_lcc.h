#pragma once
#include <deque>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <queue>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include "Planar.h"
using IK = CGAL::Exact_predicates_inexact_constructions_kernel;
using EK = CGAL::Exact_predicates_exact_constructions_kernel;
using IK_to_EK = CGAL::Cartesian_converter<IK, EK>;
using EK_to_IK = CGAL::Cartesian_converter<EK, IK>;
struct NDEdge {
    int v1, v2;
    NDEdge() : v1(-1), v2(-1) {}

    NDEdge(int a, int b) {
        if (a < b) {
            v1 = a;
            v2 = b;
        } else {
            v1 = b;
            v2 = a;
        }
    }

    // 相等比较：无向边
    bool operator==(const NDEdge& other) const {
        return (v1 == other.v1 && v2 == other.v2) ||
               (v1 == other.v2 && v2 == other.v1);
    }

    // 排序比较：把小的点放前面再比较
    bool operator<(const NDEdge& other) const {
        int a1 = std::min(v1, v2);
        int a2 = std::max(v1, v2);
        int b1 = std::min(other.v1, other.v2);
        int b2 = std::max(other.v1, other.v2);

        if (a1 != b1) return a1 < b1;
        return a2 < b2;
    }
};
// Serial BSP cell complex. Deques keep element references valid during insertion.
class MiniLcc{

public:
    std::set<int> vertices_of_cvx(int cid) ;
    std::unordered_set<int> vertices_of_face
    (int fid,std::unordered_map<int,std::pair<int,int>>&edge_split_map) ;
    bool is_edge_between_points(const std::vector<int>& eids, int v1, int v2) ;

    int insert_edge(int v1, int v2);

    std::pair<int,
              int>
    split_and_insert_faces(int planar_id, int face_id, int middle_edge_id,
        std::unordered_map<int,std::pair<int,int>>& edge_split_map) ;


    Real face_area(int fid)  ;


    std::vector<int> partition_face(const std::unordered_set<int>& eids, int planarid) const ;

    Real cvx_volume_fast(const std::vector<int>& cvx) const ;

    Real cvx_volume(const std::vector<int>& cvx) const ;
    void check_and_swap_cvxs(std::pair<std::vector<int>, std::vector<int>>& cvxs_pair,
                             int planar_id) ;
    void fix_cvxs_faces(std::unordered_map<int,std::pair<int,int>>& face_split_map,
        std::unordered_map<int,std::pair<int,int>>& edge_split_map) ;
    void build(const std::vector<Planar>& input_planars,
        int n)
    ;
    std::pair<std::vector<int>, std::vector<int>>
    split_and_insert_cvxs(int planar_id, int cvx_id, int middle_face_id,
        std::unordered_map<int,std::pair<int,int>>& face_split_map,
        std::unordered_map<int,std::pair<int,int>>& edge_split_map);
    EK::Point_3 ek_cvx_center(const std::vector<int>& cvx) const;
    void MultiLevelbuild(const std::vector<Planar>& input_planars,
        int n,int level) ;

    IK::Point_3 cvx_center(const std::vector<int>& cvx) const ;

    void split();
    void partition(int cvx_id);
    void label_cvxs(std::vector<Planar> &planars, Real inside_thred, int nr);
    void GMGlabel_cvxs(std::vector<Planar> &planars, Real inside_thred, int nr);
    void extract_surface(std::string save_path);
    void valid_cvxs();
    void clear();
    void save_lcc(std::string name);
    void save_lcc(std::string name,const std::vector<int>& level_cvxs);
    void save_lcc_compact_points(std::string name,const std::vector<int>& level_cvxs);
    void save_vis_lcc(std::string name);
    void save_vis_lcc(std::string name,const std::vector<int>& level_cvxs);
    void save_lcc_obj(std::string outside_filename, std::string inside_filename);
    void calculate_cut_values();
    void calculate_cut_values_v2();
    void cvxs_to_finest();
    void copy_finest();
    void cvxs_to_finest_v2();
    std::vector<std::vector<int>> generate_lods(int level,int init_n);
    void lod_repair(const std::vector<int>& level_cvxs);
    void lod_repair_v2(std::vector<int>& level_cvxs);
    bool check_faces(const std::unordered_set<int>& fids);
    std::vector<int> merge_face(const std::unordered_set<int>& fids,std::vector<bool> &activate_edges);
    bool check_edges(const std::unordered_set<int>& eids);
    NDEdge merge_edge(const std::unordered_set<int>& eids,std::vector<bool> &activate_vertices);
    bool save_one_cvx_obj(const std::string& filename, int cvx_id);
    std::deque<EK::Point_3> points;
    std::deque<NDEdge> edges;
    std::deque<std::vector<int>> faces;   // face -> edge ids
    std::deque<std::vector<int>> face_to_cvxs;
    std::deque<std::vector<int>> edge_to_faces;
    std::deque<std::vector<int>> cvxs;    // cell -> face ids
    std::deque<std::vector<int>> cvx_inside_planars;
    std::deque<std::vector<Real>> cvx_inside_planars_areas;
    std::deque<int> cvx_partition_poly_index;
    std::deque<bool> cvx_partitioned;//! partitioned = leaf
    std::deque<std::pair<int,int>> cvx_sons;
    std::deque<int> cvx_father;
    std::deque<Real> cvx_cut_values;
    std::deque<bool> cut_value_inside;
    std::deque<bool> cvx_is_inside;
    std::vector<EK_Planar> my_planars;        // one planar per face
    std::queue<int> volumes;
    std::deque<std::vector<int>> faces_copy;   // face -> edge ids
    std::deque<std::vector<int>> cvxs_copy;    // cell -> face ids
};
