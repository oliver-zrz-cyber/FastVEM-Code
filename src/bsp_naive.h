#pragma once
#include "dfn.h"
#include "Planar.h"
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Linear_cell_complex_for_combinatorial_map.h>
#include <tbb/concurrent_queue.h>

using IK = CGAL::Exact_predicates_inexact_constructions_kernel;
using EK = CGAL::Exact_predicates_exact_constructions_kernel;
using IK_to_EK = CGAL::Cartesian_converter<IK, EK>;
using EK_to_IK = CGAL::Cartesian_converter<EK, IK>;
using IK_Plane_3 = IK::Plane_3;
using IK_PWN = std::pair<IK::Point_3, IK::Vector_3>;
using IK_Triangle_3 = CGAL::Triangle_3<IK>;
using LLC_Traits = CGAL::Linear_cell_complex_traits<3, EK>;




struct vertex_attributes
{
	int id = -1;
	bool is_removed = false;
};
struct edge_attributes
{
	std::vector<int> belong_to_planars;
	bool is_removed = false;
};
struct CellFacePlane {
	IK::Point_3 p0;
	IK::Vector_3 n; // 指向 cell 外
};
struct face_attributes
{
	int face_id = -1;
	Real area = 0.0;
	int planar_id = -1;
	bool used = false;
	bool link_to_ghost = false;
	bool is_removed = false;
	Real hf=0.0;
	int lock_id = -1;
};
struct polyhedra_attributes
{
	Real inside_probability = 0.0;
	std::map<int, int>hit_planars_map;
	bool is_inside = false;
	int id = -1;
	int tree_node_id = -1;
	int tree_id = -1;
	int leaf_id = -1;
	std::vector<int>inside_planars;
	std::vector<Real>inside_planars_areas;
	std::vector<CellFacePlane>precomputed_planes;
	int partition_poly_index = -1;
	EK::Point_3 center;
	bool is_ghost = false;
	Real hf = 0.0;
};
struct dart_info
{

};
struct Myitem
{
	template<class LCC>
	struct Dart_wrapper
	{
		using Dart_info = dart_info;
		using Vertex_attribute = CGAL::Cell_attribute_with_point< LCC, vertex_attributes>;
		using Edge_attribute = CGAL::Cell_attribute< LCC, edge_attributes>;
		using Face_attributes = CGAL::Cell_attribute< LCC, face_attributes>;
		using Polyhedra_attribute = CGAL::Cell_attribute< LCC, polyhedra_attributes>;
		using Attributes = std::tuple<Vertex_attribute, Edge_attribute, Face_attributes, Polyhedra_attribute>;
	};
};

using zrz_LCC_3 = CGAL::Linear_cell_complex_for_combinatorial_map<3, 3, LLC_Traits, Myitem>;
using Face_attributes = zrz_LCC_3::Attribute_type<2>::type;
using Polyhedra_attribute = zrz_LCC_3::Attribute_type<3>::type;
struct Polyhedra_Merge_functor
{
	Polyhedra_Merge_functor(zrz_LCC_3& alcc) : lcc(alcc)
	{}
	// operator() automatically called after a split.
	void operator()(Polyhedra_attribute& ca1, Polyhedra_attribute& ca2)
	{
		ca1.info() = ca1.info();
	}
private:
	zrz_LCC_3& lcc;
};

struct Polyhedra_Split_functor
{
	Polyhedra_Split_functor(zrz_LCC_3& alcc) : lcc(alcc)
	{}
	// operator() automatically called after a split.
	void operator()(Polyhedra_attribute& ca1, Polyhedra_attribute& ca2)
	{
		auto center1 = lcc.barycenter<3>(ca1.dart());
		auto center2 = lcc.barycenter<3>(ca2.dart());

	}
private:
	zrz_LCC_3& lcc;
};

struct Face_Split_functor
{
	Face_Split_functor(zrz_LCC_3& alcc) : lcc(alcc)
	{}
	// operator() automatically called after a split.
	void operator()(Face_attributes& ca1, Face_attributes& ca2)
	{

	}
private:
	zrz_LCC_3& lcc;
};
using Dart_handle = zrz_LCC_3::Dart_handle;

class Split_lcc// todo : 避免一定无效的划分
{
public:
	Split_lcc() = default;
	Split_lcc(const std::vector<Planar>& planars, int n = 10)
	{
		build(planars, n);
	}

	void build(const std::vector<Planar>& planars, int n = 10);

	void clear();

	int coarse_n = 0;
	int coarse_cell_id_to_tree_id(int x,int y,int z) const{
		return x*coarse_n*coarse_n+y*coarse_n+z;
	}
	void split();
	void PARALLEL_split();
	void push_parallel_volume(Dart_handle cell_3);
	void label_lcc(std::vector<Planar>& planars, Real inside_thred, int nr);
	bool save_lcc(std::string name);
	zrz_LCC_3 lcc;
	std::vector<EK_Planar> my_planars;
	int allocate_face_lock_id();
private:
	float func_time = 0.0;
	float time_p1 = 0.0;
	float time_p2 = 0.0;

	size_t total_partitions = 0;
	void partition(Dart_handle& cell_3);
	void PARALLEL_partition(Dart_handle& cell_3);
	void ensure_face_mutex_exists(int lock_id);
	tbb::concurrent_queue<Dart_handle> PARALLEL_volumes;
	std::deque<std::mutex> face_mutexes_;
	std::mutex face_mutexes_resize_mutex_,global_lcc_mutex;
	void lock_face_locks(const std::vector<int>& input_ids);
	void unlock_face_locks(const std::vector<int>& locked_ids);
	std::vector<int> collect_incident_face_lock_ids(Dart_handle cell_3);

	template<typename Range>
	std::vector<Dart_handle> collect(Range range)
	{
		std::vector<Dart_handle> dhs;
		for (auto dh = range.begin(); dh != range.end(); dh++)
			dhs.push_back(dh);
		return dhs;
	}
	void set_ats(Dart_handle cell_3);

	void precompute_cell_planes(
		const Dart_handle cell_3,
		std::vector<CellFacePlane>& planes);

	Real cell_volume(zrz_LCC_3& lcc, Dart_handle dh);
	Real fdh_area(zrz_LCC_3& lcc, Dart_handle dh) ;
	std::vector<Dart_handle> volumes;

	std::atomic<int>              active_workers{ 0 };
	bool get_three_non_collinear_points(
		const std::vector<IK::Point_3> &pts,
		IK::Point_3 &p0, IK::Point_3 &p1, IK::Point_3 &p2,
		Real eps = 1e-12
	) ;

	bool point_in_cell_3(const IK::Point_3 &p, const Dart_handle cell_3) ;

	bool point_in_cell_fast(
	const IK::Point_3& p,
	const std::vector<CellFacePlane>& planes);

	Real planar_intersect_cell_3_fast(
		const EK_Planar& planar,
		const Dart_handle cell_3);

	Real planar_intersect_cell_3(const EK_Planar&planar,const Dart_handle cell_3);
};

