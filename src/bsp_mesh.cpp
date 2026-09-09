#include"bsp_mesh.h"
#include <CGAL/Polygon_mesh_processing/measure.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Alpha_shape_2.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Shape_detection/Region_growing/Region_growing_on_point_set.h>
#include <CGAL/Shape_detection/Region_growing/Region_growing.h>
using Itag = CGAL::Exact_predicates_tag;
using CDT = CGAL::Constrained_Delaunay_triangulation_2<
    IK,
    CGAL::Default,
    Itag>;
using Vb = CGAL::Alpha_shape_vertex_base_2<IK>;
using Fb = CGAL::Alpha_shape_face_base_2<IK>;
using Tds = CGAL::Triangulation_data_structure_2<Vb, Fb>;
using Triangulation_2 = CGAL::Delaunay_triangulation_2<IK, Tds>;
using Alpha_shape_2 = CGAL::Alpha_shape_2<Triangulation_2>;
using IK_Point_map = typename IK_Point_Set::Point_map;
using IK_Normal_map = typename IK_Point_Set::Vector_map;
using Neighbor_query_pc = CGAL::Shape_detection::Point_set::K_neighbor_query<IK, IK_Point_Set, IK_Point_map>;
using Region_type_pc = CGAL::Shape_detection::Point_set::Least_squares_plane_fit_region<IK, IK_Point_Set, IK_Point_map, IK_Normal_map>;
using Region_growing_pc = CGAL::Shape_detection::Region_growing<IK_Point_Set, Neighbor_query_pc, Region_type_pc>;




void BSP_Mesh::simulate(Real dt) {

	const EK::FT dx = EK::FT(dt) * EK::FT(vx);
	const EK::FT dy = EK::FT(dt) * EK::FT(vy);
	const EK::FT dz = EK::FT(dt) * EK::FT(vz);
	Real min_x=1e9,max_x = 0;
	for (auto v : input_mesh.vertices()) {
		const EK::Point_3 p = input_mesh.point(v);
		input_mesh.point(v) = EK::Point_3(p.x() + dx, p.y() + dy, p.z() + dz);
		Real pos_x = CGAL::to_double(input_mesh.point(v).x());
		min_x = std::min(min_x,pos_x);
		max_x = std::max(max_x,pos_x);
	}
	if (min_x<0.008||max_x>0.992) {
		vx = 0.0 - vx;
	}

}

bool BSP_Mesh::read_mesh(std::string mesh_file) {
	CGAL::IO::read_polygon_mesh(mesh_file,input_mesh);
    return true;
}

bool BSP_Mesh::write_obj(std::string mesh_file) {
	return CGAL::IO::write_OBJ(mesh_file,input_mesh);
}

void BSP_Mesh::sample_points_on_mesh() {
    FUNC_MEMORY_TIMER();
    if (point_set.points().size()) {
        box = CGAL::bbox_3(point_set.points().begin(), point_set.points().end());
        return;
    }
    using Point_3 = IK::Point_3;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<Real> dis(0.0, 1.0);
    point_set.add_normal_map();
    Real total_sample = 50000;
    Real total_area = CGAL::to_double(CGAL::Polygon_mesh_processing::area(input_mesh));
    EK_to_IK to_ik;
    for (auto face : input_mesh.faces())
    {
        std::vector<Point_3> tri_coord;
        for (auto point : input_mesh.vertices_around_face(input_mesh.halfedge(face)))
        {
            auto p = input_mesh.point(point);
            tri_coord.push_back(to_ik(p));
        }
        auto normal = CGAL::normal(tri_coord[0], tri_coord[1], tri_coord[2]);
        normal = normal / std::sqrt(normal.squared_length());
        IK::Triangle_3 tri(tri_coord[0], tri_coord[1], tri_coord[2]);
        auto area = std::sqrt(tri.squared_area());
        for (int round = 0; round < std::max(int((area/total_area)*total_sample),1); round++)
        {
            float i = dis(gen), j = dis(gen), k = dis(gen);
            float eps = 1e-6;
            float sum = i + j + k + eps;
            i /= sum, j /= sum, k /= sum;
            auto index = point_set.insert(Point_3(i * tri_coord[0].x() + j * tri_coord[1].x() + k * tri_coord[2].x(),
                i * tri_coord[0].y() + j * tri_coord[1].y() + k * tri_coord[2].y(),
                i * tri_coord[0].z() + j * tri_coord[1].z() + k * tri_coord[2].z()));
            point_set.normal_map()[*index] = normal;
        }
    }

    box = CGAL::bbox_3(point_set.points().begin(), point_set.points().end());
}

void CDTriangulation(CDT &cdt, std::map<IK::Point_2, CDT::Vertex_handle> &p_index, Alpha_shape_2 &as) {
    //insert vertex
    auto vit = as.alpha_shape_vertices_begin();
    while (vit != as.alpha_shape_vertices_end()) {
        CDT::Vertex_handle v = cdt.insert((*vit)->point());
        p_index[(*vit)->point()] = v;
        vit++;
    }

    //insert edge
    auto eit = as.alpha_shape_edges_begin();
    while (eit != as.alpha_shape_edges_end()) {
        switch (as.classify(*eit))
        {
            case Alpha_shape_2::SINGULAR:
                eit++;
                continue;
            default:
                break;
        }
        CDT::Vertex_handle v1 = p_index[as.segment(*eit).source()];
        CDT::Vertex_handle v2 = p_index[as.segment(*eit).target()];

        if (!v1->is_valid() || !v2->is_valid())
            std::cout << "invalid!" << std::endl;

        cdt.insert_constraint(v1, v2);
        eit++;
    }
}


std::list<IK_Triangle_3> Triangles_of_alphaShape(const IK::Plane_3 &plane,
                                                            const std::vector<IK::Point_3> &ps, Real alpha_scale) {
    std::vector<IK_PWN>pwns;
    for (auto& point : ps)
        pwns.push_back(std::make_pair(point, plane.orthogonal_vector()));

    std::vector<IK::Point_2> projected_points(pwns.size());

    for(int k=0;k<pwns.size();++k)
    {
        auto& point_3 = pwns[k].first;
        IK::Point_3 projected = plane.projection(point_3);
        projected_points[k] = plane.to_2d(projected);
    }

    Alpha_shape_2 as(projected_points.begin(), projected_points.end(), IK::FT(10000), Alpha_shape_2::REGULARIZED);
    /******** Todo:use average sense as alpha value**********/
    as.set_alpha(alpha_scale);//"more than 1 components in a alpha";


    CDT cdt;
    std::map<CDT::Point_2, CDT::Vertex_handle> p_index;
    CDTriangulation(cdt, p_index, as);

    auto fit = cdt.finite_faces_begin();
    std::list<IK_Triangle_3> triangles;
    while (fit != cdt.finite_faces_end()) {

        IK::Point_2 p1 = fit->vertex(0)->point();
        IK::Point_2 p2 = fit->vertex(1)->point();
        IK::Point_2 p3 = fit->vertex(2)->point();

        if (CGAL::collinear(p1, p2, p3))
            continue;

        IK::Point_2 center = IK::Point_2(0, 0) + (((p1 - IK::Point_2(0, 0)) + (p2 - IK::Point_2(0, 0)) + (p3 - IK::Point_2(0, 0))) * 1.0 / 3.0);
        int res = as.classify(center);

        if (res == Alpha_shape_2::INTERIOR) {
            triangles.push_back(IK_Triangle_3(plane.to_3d(p1), plane.to_3d(p2), plane.to_3d(p3)));
        }

        fit++;
    }

    return triangles;
}

void BSP_Mesh::detect_planars(Real max_distance_to_plane, Real max_accepted_angle, size_t min_region_size) {
    regions.clear();
    const std::size_t k = 20;
    Neighbor_query_pc neighbor_query(
        point_set,
        k,
        point_set.point_map());
    Region_type_pc region_type(
        point_set,
        max_distance_to_plane, max_accepted_angle, min_region_size,
        point_set.point_map(), point_set.normal_map());
    // Create an instance of the region growing class.
    Region_growing_pc region_growing(
        point_set, neighbor_query, region_type);
    // Run the algorithm.
    std::size_t number_of_regions = 0;
    region_growing.detect(
        boost::make_function_output_iterator([&](const std::vector<std::size_t>& region)
    {
        regions.push_back(region);
    }));
 //-----------------------------------------------------
    IK_to_EK to_ek;
    planars.clear();
    for (auto region : regions)
    {
        if (region.size() < 3)continue;
        Planar planar;
        IK::Plane_3 plane_;
        IK::Vector_3 plane_normal(0, 0, 0);
        for (auto& idx : region)
        {
            planar._3d_Pwns.push_back(std::make_pair(point_set.point(*(point_set.begin() + idx)),
                point_set.normal(*(point_set.begin() + idx))));
        }
        for (auto& pwn : planar._3d_Pwns)planar._3d_Points.push_back(pwn.first);
        for (auto& idx : region)
            plane_normal += point_set.normal_map()[idx]/std::sqrt(point_set.normal_map()[idx].squared_length());
        linear_least_squares_fitting_3(
            planar._3d_Points.begin(),
            planar._3d_Points.end(),
            plane_,
            CGAL::Dimension_tag<0>());
        if (plane_.orthogonal_vector() * plane_normal < 0)
            plane_ = plane_.opposite();
        planar.plane = plane_;
        planar.id = planars.size();
        planars.push_back(planar);
    }


    for (auto& planar : planars)
    {
        for (auto& point_3 : planar._3d_Points)
            planar._ek_3d_Points.push_back(to_ek(point_3));
        planar.ek_plane = to_ek(planar.plane);
        planar.save_alpha = Triangles_of_alphaShape(planar.plane, planar._3d_Points, 0.00001);
        planar.label_alpha = planar.save_alpha;
        for (auto& tri : planar.label_alpha)
            planar.area += std::sqrt(tri.squared_area());
        planar.sample_points_on_label_alpha();
    }
    std::sort(planars.begin(), planars.end());
    save_AlphaShape("detected_planars.ply",planars);
    for (int id=0;id<planars.size();++id)planars[id].id = id;
}
void BSP_Mesh::save_AlphaShape(std::string filename, std::vector<Planar> &planars) {
    IK_Polygon_mesh mesh;
    using Color = CGAL::IO::Color;
    using Vector_3 = IK::Vector_3;
    auto face_color = mesh.add_property_map<IK_Polygon_mesh::Face_index, Color>("f:color", Color(0, 0, 0)).first;
    auto v_normal = mesh.add_property_map<IK_Polygon_mesh::Vertex_index, Vector_3>("v:normal", Vector_3(0, 0, 0)).first;

    for (auto& planar : planars)
    {
        for (auto& tri : planar.save_alpha)
        {
            CGAL::SM_Vertex_index v1 = mesh.add_vertex(tri.vertex(0));
            CGAL::SM_Vertex_index v2 = mesh.add_vertex(tri.vertex(1));
            CGAL::SM_Vertex_index v3 = mesh.add_vertex(tri.vertex(2));
            CGAL::SM_Face_index f = mesh.add_face(v1, v2, v3);
            face_color[f] = planar.color;
            v_normal[v1] = planar.plane.orthogonal_vector();
            v_normal[v2] = planar.plane.orthogonal_vector();
            v_normal[v3] = planar.plane.orthogonal_vector();

        }
    }

    CGAL::IO::write_PLY(filename, mesh);
}
void BSP_Mesh::construct_planars_from_input_mesh() {
        FUNC_MEMORY_TIMER();
    planars.clear();
	regions.clear();
    int planar_id = 0;
    // 遍历网格中的每个面
    for (auto face : input_mesh.faces()) {
        // 获取三角形的三个顶点
        EK_to_IK to_ik;
        std::vector<CGAL::SM_Vertex_index> vertex_indices;
        for (auto point : input_mesh.vertices_around_face(input_mesh.halfedge(face))) {
            vertex_indices.push_back(point);
        }
        RZ_ASSERT(vertex_indices.size() == 3,"");
        // 将三角形的三个顶点添加到 _3d_Points 中
        Planar planar;
        planar._ek_3d_Points.push_back(input_mesh.point(vertex_indices[0]));
        planar._ek_3d_Points.push_back(input_mesh.point(vertex_indices[1]));
        planar._ek_3d_Points.push_back(input_mesh.point(vertex_indices[2]));
        EK::Triangle_3 triangle(planar._ek_3d_Points[0],planar._ek_3d_Points[1],planar._ek_3d_Points[2]);
        planar.ek_plane = triangle.supporting_plane();
        // 计算平面并保存
        IK::Point_3 p0 = to_ik(planar._ek_3d_Points[0]);
        IK::Point_3 p1 = to_ik(planar._ek_3d_Points[1]);
        IK::Point_3 p2 = to_ik(planar._ek_3d_Points[2]);
        RZ_ASSERT(planar.ek_plane.has_on(planar._ek_3d_Points[0]),"");
        IK_Triangle_3 triangle_alpha(p0,p1,p2);
        planar.save_alpha.push_back(triangle_alpha);
        planar.label_alpha = planar.save_alpha;

        planar.id = planar_id++;

        // // 将平面的3D点投影到2D空间
        for (const auto& point : planar._ek_3d_Points) {
            planar._2d_Points.push_back(to_ik(planar.ek_plane.to_2d(point)));
        }
        planar.area = std::sqrt(triangle_alpha.squared_area());
        planar.sample_points_on_label_alpha();
        planars.push_back(planar);
    }
    for (int id=0;id<planars.size();++id)planars[id].id = id;
}
