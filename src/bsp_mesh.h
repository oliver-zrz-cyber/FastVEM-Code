#pragma once
#include "dfn.h"
#include "Planar.h"
#include <CGAL/Surface_mesh.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Point_set_3.h>
#include <CGAL/Polygon_2.h>

using IK = CGAL::Exact_predicates_inexact_constructions_kernel;
using EK = CGAL::Exact_predicates_exact_constructions_kernel;
using EK_Polygon_mesh = CGAL::Surface_mesh<EK::Point_3>;
using IK_Polygon_mesh = CGAL::Surface_mesh<IK::Point_3>;
using IK_to_EK = CGAL::Cartesian_converter<IK, EK>;
using EK_to_IK = CGAL::Cartesian_converter<EK, IK>;
using Region = std::vector<std::size_t>;
using Regions = std::vector<Region>;
using IK_Point_Set = CGAL::Point_set_3<IK::Point_3>;
using IK_Plane_3 = IK::Plane_3;
using IK_PWN = std::pair<IK::Point_3, IK::Vector_3>;
using IK_Polygon_2 = CGAL::Polygon_2<IK>;
using IK_Triangle_3 = CGAL::Triangle_3<IK>;


class BSP_Mesh {
public:
    EK_Polygon_mesh input_mesh;
    IK_Point_Set point_set;
    CGAL::Bbox_3 box;
    Regions regions;
    std::vector<Planar>planars;
    Real vx=0.0,vy=0.0,vz=0.0;

    void save_AlphaShape(std::string filename, std::vector<Planar> &planars);
    void simulate(Real dt);
    bool read_mesh(std::string mesh_file);
    bool write_obj(std::string mesh_file);
    void sample_points_on_mesh();

    void detect_planars(Real max_distance_to_plane, Real max_accepted_angle, size_t min_region_size);
    void construct_planars_from_input_mesh();
};

