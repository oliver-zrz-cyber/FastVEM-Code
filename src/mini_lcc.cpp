#include "mini_lcc.h"
#include "dfn.h"
#include <CGAL/Polygon_mesh_processing/repair.h>
#include <CGAL/Polygon_mesh_processing/repair_polygon_soup.h>
#include "Planar.h"
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>
#include <CGAL/Surface_mesh.h>
#include <malloc.h>
using Iterator = std::list<IK_Triangle_3>::iterator;
using Primitive = CGAL::AABB_triangle_primitive<IK, Iterator>;
using AABB_triangle_traits = CGAL::AABB_traits<IK, Primitive>;
using AABBTree = CGAL::AABB_tree<AABB_triangle_traits>;

namespace {
void build_regular_grid_lcc(MiniLcc& lcc, int n, const EK::FT& xmin, const EK::FT& ymin,
    const EK::FT& zmin, const EK::FT& dx, const EK::FT& dy, const EK::FT& dz,
    bool init_hierarchy)
{
    auto vid = [n](int i, int j, int k) -> int {
        return i + (n + 1) * (j + (n + 1) * k);
    };

    for (int k = 0; k <= n; ++k) {
        for (int j = 0; j <= n; ++j) {
            for (int i = 0; i <= n; ++i) {
                lcc.points.emplace_back(
                    xmin + i * dx,
                    ymin + j * dy,
                    zmin + k * dz
                );
            }
        }
    }

    const int edge_x_base = 0;
    const int edge_y_base = edge_x_base + n * (n + 1) * (n + 1);
    const int edge_z_base = edge_y_base + (n + 1) * n * (n + 1);

    auto edge_x = [=](int i, int j, int k) -> int {
        return edge_x_base + i + n * (j + (n + 1) * k);
    };
    auto edge_y = [=](int i, int j, int k) -> int {
        return edge_y_base + i + (n + 1) * (j + n * k);
    };
    auto edge_z = [=](int i, int j, int k) -> int {
        return edge_z_base + i + (n + 1) * (j + (n + 1) * k);
    };

    for (int k = 0; k <= n; ++k) {
        for (int j = 0; j <= n; ++j) {
            for (int i = 0; i < n; ++i) {
                lcc.edges.emplace_back(vid(i, j, k), vid(i + 1, j, k));
            }
        }
    }
    for (int k = 0; k <= n; ++k) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i <= n; ++i) {
                lcc.edges.emplace_back(vid(i, j, k), vid(i, j + 1, k));
            }
        }
    }
    for (int k = 0; k < n; ++k) {
        for (int j = 0; j <= n; ++j) {
            for (int i = 0; i <= n; ++i) {
                lcc.edges.emplace_back(vid(i, j, k), vid(i, j, k + 1));
            }
        }
    }

    const int face_z_base = 0;
    const int face_y_base = face_z_base + n * n * (n + 1);
    const int face_x_base = face_y_base + n * (n + 1) * n;

    auto face_z = [=](int i, int j, int k) -> int {
        return face_z_base + i + n * (j + n * k);
    };
    auto face_y = [=](int i, int j, int k) -> int {
        return face_y_base + i + n * (j + (n + 1) * k);
    };
    auto face_x = [=](int i, int j, int k) -> int {
        return face_x_base + i + (n + 1) * (j + n * k);
    };

    for (int k = 0; k <= n; ++k) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                lcc.faces.push_back({edge_y(i, j, k), edge_x(i, j + 1, k),
                    edge_y(i + 1, j, k), edge_x(i, j, k)});
            }
        }
    }
    for (int k = 0; k < n; ++k) {
        for (int j = 0; j <= n; ++j) {
            for (int i = 0; i < n; ++i) {
                lcc.faces.push_back({edge_x(i, j, k), edge_z(i + 1, j, k),
                    edge_x(i, j, k + 1), edge_z(i, j, k)});
            }
        }
    }
    for (int k = 0; k < n; ++k) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i <= n; ++i) {
                lcc.faces.push_back({edge_z(i, j, k), edge_y(i, j, k + 1),
                    edge_z(i, j + 1, k), edge_y(i, j, k)});
            }
        }
    }

    for (int k = 0; k < n; ++k) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                lcc.cvxs.push_back({
                    face_x(i, j, k), face_x(i + 1, j, k),
                    face_y(i, j, k), face_y(i, j + 1, k),
                    face_z(i, j, k), face_z(i, j, k + 1)
                });
                if (init_hierarchy) {
                    lcc.cvx_sons.push_back(std::make_pair(-1, -1));
                    lcc.cvx_father.push_back(-1);
                }
            }
        }
    }
}
}

bool MiniLcc::save_one_cvx_obj(const std::string& filename, int cvx_id) {
    if (cvx_id < 0 || cvx_id >= static_cast<int>(cvxs.size())) {
        std::cerr << "[save_one_cvx_obj] invalid cvx_id = " << cvx_id << std::endl;
        return false;
    }

    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "[save_one_cvx_obj] failed to open " << filename << std::endl;
        return false;
    }

    const auto& face_ids = cvxs[cvx_id];

    // 收集该 cvx 用到的全局顶点
    std::unordered_set<int> used_vertices;
    for (int fid : face_ids) {
        if (fid < 0 || fid >= static_cast<int>(faces.size())) continue;
        for (int eid : faces[fid]) {
            if (eid < 0 || eid >= static_cast<int>(edges.size())) continue;
            used_vertices.insert(edges[eid].v1);
            used_vertices.insert(edges[eid].v2);
        }
    }

    // 全局点 id -> OBJ 局部点 id（1-based）
    std::unordered_map<int, int> global_to_obj;
    std::vector<int> obj_to_global;
    obj_to_global.reserve(used_vertices.size());

    int obj_vid = 1;
    for (int gid : used_vertices) {
        global_to_obj[gid] = obj_vid++;
        obj_to_global.push_back(gid);
    }

    // 写顶点
    for (int gid : obj_to_global) {
        const auto& p = points[gid];
        out << "v "
            << CGAL::to_double(p.x()) << " "
            << CGAL::to_double(p.y()) << " "
            << CGAL::to_double(p.z()) << "\n";
    }

    // 重建每个 face 的顶点环并写 face
    for (int fid : face_ids) {
        if (fid < 0 || fid >= static_cast<int>(faces.size())) continue;

        const auto& f_edges = faces[fid];
        if (f_edges.size() < 3) continue;

        // 邻接表：vertex -> 相邻顶点
        std::unordered_map<int, std::vector<int>> adj;
        std::unordered_set<int> face_vertices;

        for (int eid : f_edges) {
            if (eid < 0 || eid >= static_cast<int>(edges.size())) continue;
            int a = edges[eid].v1;
            int b = edges[eid].v2;
            adj[a].push_back(b);
            adj[b].push_back(a);
            face_vertices.insert(a);
            face_vertices.insert(b);
        }

        // 检查是否像一个简单环
        bool bad_face = false;
        for (int v : face_vertices) {
            if (adj[v].size() != 2) {
                bad_face = true;
                break;
            }
        }
        if (bad_face) {
            std::cerr << "[save_one_cvx_obj] warning: face " << fid
                      << " is not a simple cycle, skipped.\n";
            continue;
        }

        // 从任意顶点开始绕环
        std::vector<int> loop;
        loop.reserve(face_vertices.size());

        int start = *face_vertices.begin();
        int prev = -1;
        int cur = start;

        while (true) {
            loop.push_back(cur);

            const auto& nbrs = adj[cur];
            int next = -1;
            if (nbrs[0] != prev) next = nbrs[0];
            else next = nbrs[1];

            prev = cur;
            cur = next;

            if (cur == start) break;
            if (static_cast<int>(loop.size()) > static_cast<int>(face_vertices.size()) + 5) {
                std::cerr << "[save_one_cvx_obj] warning: face " << fid
                          << " loop reconstruction failed, skipped.\n";
                bad_face = true;
                break;
            }
        }

        if (bad_face || loop.size() < 3) continue;

        out << "f";
        for (int gid : loop) {
            auto it = global_to_obj.find(gid);
            if (it == global_to_obj.end()) {
                bad_face = true;
                break;
            }
            out << " " << it->second;
        }
        if (!bad_face) out << "\n";
    }

    out.close();
    std::cout << "[save_one_cvx_obj] saved cvx " << cvx_id
              << " to " << filename << std::endl;
    return true;
}

std::set<int> MiniLcc::vertices_of_cvx(int cid) {
	{
		std::set<int> vertices;
		for (auto&faceid:cvxs[cid]) {
			for (auto&edgeid:faces[faceid]) {
				auto edge = edges[edgeid];
				vertices.insert(edge.v1);
				vertices.insert(edge.v2);
			}
		}
		return vertices;
	}
}

std::unordered_set<int> MiniLcc::
vertices_of_face(int fid, std::unordered_map<int, std::pair<int, int>> &edge_split_map){
	// FUNC_TIMER();
	std::unordered_set<int> vertices;
	for (auto&edgeid:faces[fid]) {
		if (!edge_split_map.count(edgeid)) {
			auto edge = edges[edgeid];
			vertices.insert(edge.v1);
			vertices.insert(edge.v2);
		}
		else {
			auto edge1 = edges[edge_split_map[edgeid].first];
			vertices.insert(edge1.v1);
			vertices.insert(edge1.v2);
			auto edge2 = edges[edge_split_map[edgeid].second];
			vertices.insert(edge2.v1);
			vertices.insert(edge2.v2);
		}

	}
	return vertices;
}

bool MiniLcc::is_edge_between_points(const std::vector<int> &eids, int v1, int v2) {
	if (v1 > v2) std::swap(v1, v2);

	for (int eid : eids) {
		const auto& e = edges[eid];
		if (e.v1 == v1 && e.v2 == v2) {
			return true;
		}
	}
	return false;
}

int MiniLcc::insert_edge(int v1, int v2)  {
	edges.emplace_back(v1, v2);
	edge_to_faces.push_back({});
	return edges.size()-1;
}

void MiniLcc::split() {
    FUNC_TIMER();
    if (my_planars.size() == 0) {
        return;
    }
	// construct maps

	edge_to_faces.clear();
	face_to_cvxs.clear();
	edge_to_faces.resize(edges.size());
	face_to_cvxs.resize(faces.size());
	for (int fid=0;fid<faces.size();++fid) {
		for (const auto&eid:faces[fid])
			edge_to_faces[eid].push_back(fid);
	}

	for (int cid=0;cid<cvxs.size();++cid) {
		for (const auto&fid:cvxs[cid])
			face_to_cvxs[fid].push_back(cid);
	}
	//---------------
    while (volumes.size())
    {
        auto cell_3 = volumes.front();
        volumes.pop();
        partition(cell_3);
    }
}

static int total_partitions = 0;
// 注意可能有很多无效的face、edge和 cvx 不要让他们影响了计算速度
void MiniLcc::partition(int cvx_id) {
	// FUNC_TIMER();
	Real intersect_time = 0.0;

	++total_partitions;
	// std::cout<<total_partitions<<"\n";

	RZ_ASSERT(cvx_partitioned[cvx_id]==false,"");
	if (cvx_inside_planars[cvx_id].size()==0 || cvx_partition_poly_index[cvx_id]<0)return;
	auto polys = std::move(cvx_inside_planars[cvx_id]);
	int local_pt_id = cvx_partition_poly_index[cvx_id];
	int planar_id = polys[local_pt_id];
	auto &largest_poly = my_planars[planar_id];
	polys.erase(polys.begin()+local_pt_id);
	bool has_n= false,has_p=false;

	//-------------------------------------------------------------------
	for (const auto &fid:cvxs[cvx_id]) {
		for (const auto&eid:faces[fid]) {
			const auto& p1 = points[edges[eid].v1];
			const auto& p2 = points[edges[eid].v2];
			if (largest_poly.plane.has_on_negative_side(p1))has_n=true;
			else if (largest_poly.plane.has_on_positive_side(p1))has_p=true;
			if (largest_poly.plane.has_on_negative_side(p2))has_n=true;
			else if (largest_poly.plane.has_on_positive_side(p2))has_p=true;
			if (has_n&&has_p)break;
		}
		if (has_n&&has_p)break;
	}
	if (!has_n||!has_p) {
		cvx_partition_poly_index[cvx_id]= cvx_partition_poly_index[cvx_id] - 1;
		cvx_inside_planars[cvx_id] = std::move(polys);
		if(cvx_inside_planars[cvx_id].size())
			volumes.push(cvx_id);
		return;
	}

	std::vector<int> link_cvxs;
	link_cvxs.reserve(64); // 可根据经验调大，减少扩容

	for (auto& face : cvxs[cvx_id]) {
		for (auto& cvxid : face_to_cvxs[face]) {
			if (cvx_partitioned[cvxid]) continue;
			link_cvxs.push_back(cvxid);
		}

	}

	// 去重
	std::sort(link_cvxs.begin(), link_cvxs.end());
	link_cvxs.erase(
		std::unique(link_cvxs.begin(), link_cvxs.end()),
		link_cvxs.end()
	);



	//-------------------------------------------------------------------
	Real tt1 = 0.0;
	auto tt1_t0 = std::chrono::high_resolution_clock::now();
	std::vector<int> inserted_es;
	std::unordered_map<int,std::pair<int,int>> face_split_map;
	std::unordered_map<int,std::pair<int,int>> edge_split_map;

	// 注意这里
	Real tt2 = 0.0;
	auto tt2_t0 = std::chrono::high_resolution_clock::now();
	const auto& cvxfaces = cvxs[cvx_id];
	bool inserted = 0;
	std::set<int> checked_eid;
	for (auto face_id:cvxfaces) {
		std::vector<int> points_on_plane;
		const auto face_vids = vertices_of_face(face_id,edge_split_map);
		RZ_ASSERT(face_vids.size()>=3,"");
		for (auto iter : face_vids) {
			if (largest_poly.plane.has_on(points[iter]))
				points_on_plane.push_back(iter );
		}
		if (points_on_plane.size() == 2) {
			if (!is_edge_between_points(faces[face_id],points_on_plane[0],points_on_plane[1])) {
				// 插入一个边和一个面被切分开
				auto eid = insert_edge(points_on_plane[0],points_on_plane[1]);
				inserted_es.push_back(eid);
				auto fdhs = split_and_insert_faces(planar_id,face_id,eid,edge_split_map);
				face_split_map[face_id] =
					std::make_pair(fdhs.first,fdhs.second);
				if (face_area(fdhs.first)<1e-10||
								face_area(fdhs.second)<1e-10) {
					// points.resize(old_points_size);
					// edges.resize(old_edges_size);
					// faces.resize(old_faces_size);
					cvx_partition_poly_index[cvx_id]=  cvx_partition_poly_index[cvx_id] - 1;
					polys.insert(polys.begin() + local_pt_id, planar_id);
					cvx_inside_planars[cvx_id] = std::move(polys);
					if(cvx_inside_planars[cvx_id].size())
						volumes.push(cvx_id);
					return;
								}
				inserted = true;
			}
			continue;
		}
		if (points_on_plane.size()>2) {
			// RZ_FATAL("ishere?");
			continue;
		}
		int inserted_cnt = 0;

		for (auto eid : faces[face_id])
		{
			const auto& edge = edges[eid];
			if (edge_split_map.count(eid))continue;
			EK::Point_3 point1 = points[edge.v1];
			EK::Point_3 point2 =  points[edge.v2];
			if (largest_poly.plane.has_on(point1)) continue;
			if (largest_poly.plane.has_on(point2)) continue;
			if (largest_poly.plane.has_on_positive_side(point1) == largest_poly.plane.has_on_positive_side(point2))
				continue;
			auto CGALedge = EK::Segment_3(point1, point2);
			if (auto result = CGAL::intersection(CGALedge, largest_poly.plane))
			{
				if (auto* inters_point = boost::get<EK::Point_3>(&*result)) {
					intersect_time+=1.0;
					inserted = true;
					++inserted_cnt;
					int temp_psize = points.size();
					points.push_back(*inters_point);
					points_on_plane.push_back(temp_psize);
					int e1 = insert_edge(edge.v1,temp_psize);
					int e2 = insert_edge(edge.v2,temp_psize);
					inserted_es.push_back(e1);
					inserted_es.push_back(e2);
					edge_split_map[eid] = std::make_pair(e1,e2);
					if (CGAL::squared_distance(*inters_point,point1)<1e-18
						||CGAL::squared_distance(*inters_point,point2)<1e-18) {
						// points.resize(old_points_size);
						// edges.resize(old_edges_size);
						// faces.resize(old_faces_size);
						cvx_partition_poly_index[cvx_id]=  cvx_partition_poly_index[cvx_id] - 1;
						polys.insert(polys.begin() + local_pt_id, planar_id);
						cvx_inside_planars[cvx_id] = std::move(polys);
						if(cvx_inside_planars[cvx_id].size())
							volumes.push(cvx_id);
						return;
						}
				}
			}
		}
		if (points_on_plane.size() < 2) {
			RZ_ASSERT(inserted_cnt==0,inserted_cnt);
			continue;
		}
		if (!points_on_plane.empty()) {
			RZ_ASSERT(points_on_plane.size() == 2,points_on_plane.size());
			RZ_ASSERT(my_planars[planar_id].plane.has_on(points[points_on_plane[0]]),"");
			RZ_ASSERT(my_planars[planar_id].plane.has_on(points[points_on_plane[1]]),"");
			auto eid = insert_edge(points_on_plane[0],points_on_plane[1]);
			inserted_es.push_back(eid);
			auto fdhs = split_and_insert_faces(planar_id,face_id,eid,edge_split_map);

			face_split_map[face_id] =
				std::make_pair(fdhs.first,fdhs.second);

			if (face_area(fdhs.first)<1e-10||
											face_area(fdhs.second)<1e-10)
			{
				// points.resize(old_points_size);
				// edges.resize(old_edges_size);
				// faces.resize(old_faces_size);
				cvx_partition_poly_index[cvx_id]=  cvx_partition_poly_index[cvx_id] - 1;
				polys.insert(polys.begin() + local_pt_id, planar_id);
				cvx_inside_planars[cvx_id] = std::move(polys);
				if(cvx_inside_planars[cvx_id].size())
					volumes.push(cvx_id);
				return;
			}
		}
	}
	auto tt2_t1 = std::chrono::high_resolution_clock::now();
	tt2 += std::chrono::duration<Real>(tt2_t1 - tt2_t0).count();


	// split volume
	// 新的面 用 edge表示
	std::unordered_set<int> eids;
	for (auto fid:cvxfaces) {
		for (auto eid:faces[fid]) {
			if (!edge_split_map.count(eid))
				eids.insert(eid);
			else {
				eids.insert(edge_split_map[eid].first);
				eids.insert(edge_split_map[eid].second);
			}
		}
	}
	for (const auto&it:inserted_es) {
		eids.insert(it);
	}
	auto nface = partition_face(eids,planar_id);
	faces.push_back(nface);
	face_to_cvxs.push_back({});

	auto new_cvxs = split_and_insert_cvxs(planar_id,cvx_id,
		faces.size()-1,face_split_map,edge_split_map);

	if (cvx_volume_fast(new_cvxs.first)<1e-10||cvx_volume_fast(new_cvxs.second)<1e-10) {
		// points.resize(old_points_size);
		// edges.resize(old_edges_size);
		// faces.resize(old_faces_size);
		cvx_partition_poly_index[cvx_id]=  cvx_partition_poly_index[cvx_id] - 1;
		polys.insert(polys.begin() + local_pt_id, planar_id);
		cvx_inside_planars[cvx_id] = std::move(polys);
		if(cvx_inside_planars[cvx_id].size())
			volumes.push(cvx_id);
		return;
	}
	auto tt1_t1 = std::chrono::high_resolution_clock::now();
	tt1 += std::chrono::duration<Real>(tt1_t1 - tt1_t0).count();
	Real tt0 = 0.0;
	auto t0 = std::chrono::high_resolution_clock::now();
	check_and_swap_cvxs(new_cvxs,planar_id);

	int neg_cvx_id = cvxs.size();
	cvxs.push_back(new_cvxs.first);
	int pos_cvx_id = cvxs.size();
	cvxs.push_back(new_cvxs.second);
	cvx_partitioned.push_back(false);
	cvx_partitioned.push_back(false);
	if (cvx_sons.size()) {
		cvx_sons.push_back({-1,-1});
		cvx_sons.push_back({-1,-1});
		cvx_father.push_back(cvx_id);
		cvx_father.push_back(cvx_id);
	}
	cvx_inside_planars.push_back({});
	cvx_inside_planars.push_back({});
	cvx_inside_planars_areas.push_back({});
	cvx_inside_planars_areas.push_back({});
	cvx_partition_poly_index.push_back(-1);
	cvx_partition_poly_index.push_back(-1);
	if (cvx_sons.size()) {
		cvx_sons[cvx_id] = std::make_pair(neg_cvx_id,pos_cvx_id);
	}
	for (auto& planar_id : polys)
	{
		const auto &planar = my_planars[planar_id];
		bool flag_pos = false;
		bool flag_neg = false;
		if (planar.id == -2)flag_pos = true,flag_neg = true;
		for (auto& _3d_point : planar._3d_Points)
		{
			if (largest_poly.plane.has_on_positive_side(_3d_point))
				flag_pos = true;
			else if (largest_poly.plane.has_on_negative_side(_3d_point))
				flag_neg = true;
		}
		if (flag_pos)
			cvx_inside_planars[pos_cvx_id].push_back(planar_id);
		if(flag_neg)
			cvx_inside_planars[neg_cvx_id].push_back(planar_id);
	}

	cvx_partition_poly_index[pos_cvx_id]
		= cvx_inside_planars[pos_cvx_id].size() - 1;
	cvx_partition_poly_index[neg_cvx_id]
		= cvx_inside_planars[neg_cvx_id].size() - 1;

	if (cvx_inside_planars[pos_cvx_id].size() > 0) {
		volumes.push(pos_cvx_id);
	}
	if (cvx_inside_planars[neg_cvx_id].size() > 0) {
		volumes.push(neg_cvx_id);
	}

	//更新 cvxs  faces ....
	fix_cvxs_faces(face_split_map,edge_split_map);


	//更新 map 先删除和新的 两个 cvx有关的face、edge的联系 再用 所有的邻接 包含这俩进行一次更新
	for (auto&face:cvxs[neg_cvx_id]) {
		face_to_cvxs[face].clear();
		for (auto& edge:faces[face])
			edge_to_faces[edge].clear();
	}
	for (auto&face:cvxs[pos_cvx_id]) {
		face_to_cvxs[face].clear();
		for (auto& edge:faces[face])
			edge_to_faces[edge].clear();
	}

	link_cvxs.push_back(neg_cvx_id);
	link_cvxs.push_back(pos_cvx_id);

	// 或许分为 link cvxs 与 link faces ?


	for (auto&cvxid:link_cvxs) {
		for (const auto&fid:cvxs[cvxid]){
			face_to_cvxs[fid].emplace_back(cvxid);
			for (const auto&eid:faces[fid])
				edge_to_faces[eid].emplace_back(fid);
		}
	}
	cvx_partitioned[cvx_id] = true;
	auto t1 = std::chrono::high_resolution_clock::now();
	tt0 += std::chrono::duration<Real>(t1 - t0).count();

	// RZ_INFO("tt0: ",tt0*1000,"ms");
	// RZ_INFO("tt1: ",tt1*1000,"ms");
	// RZ_INFO("tt2: ",tt2*1000,"ms");
}

Real MiniDray(const std::vector <IK::Point_3>& centers, AABBTree& tree, std::vector<IK::Vector_3>& rays)
{
	float r = 0;
	int valid_ray_count = 1;
	for (int i = 0; i < rays.size(); i++)
	{
		for (int j = 0; j < centers.size(); ++j)
		{
			const IK::Point_3& t_point = centers[j];
			IK::Ray_3 ray_query(t_point, t_point + rays[i]);
			if (auto res = tree.first_intersected_primitive(ray_query))
			{
				Iterator triangle = *res;
				if (triangle->supporting_plane().has_on_negative_side(t_point)) {
					r++; //inside
				}
			}
			valid_ray_count++;
		}
	}
	r = r / (float)valid_ray_count;
	return r;

}
void MiniLcc::label_cvxs(std::vector<Planar> &planars, Real inside_thred, int nr) {
	FUNC_TIMER();
	std::list<IK_Triangle_3> alpha_shape;
	for (auto& planar : planars)
		alpha_shape.insert(alpha_shape.end(), planar.label_alpha.begin(), planar.label_alpha.end());
	std::deque<std::vector<int>> valid_cvxs;
	for (int i=0;i<cvxs.size();++i)
		if (!cvx_partitioned[i])
			valid_cvxs.push_back(cvxs[i]);
	cvxs.swap(valid_cvxs);

	cvx_is_inside.assign(cvxs.size(),false);
	// // // rays
	std::vector<IK::Vector_3> rays;
	Real pi = 2 * acos(0.0);
	Real gold = 3 - sqrt(5);
	int samples = nr;

	for (int i = 0; i < nr; i++) {
		Real z = 1 - (Real(i) / Real(samples - 1)) * 2;
		Real theta = pi * i * gold;
		Real x = cos(theta) * sqrt(1 - z * z);
		Real y = sin(theta) * sqrt(1 - z * z);
		rays.push_back(IK::Vector_3(x, y, z));
	}
	//centers

	// centers
	std::vector<std::vector<IK::Point_3>> cell_cs;
	std::vector<int> ids(cvxs.size());
	std::vector<Real> in_value(cvxs.size(), 0.0);

	for (int cid = 0; cid < (int)cvxs.size(); ++cid)
	{
		std::vector<IK::Point_3> centers;
		centers.push_back(cvx_center(cvxs[cid]));
		cell_cs.push_back(std::move(centers));
		ids[cid] = cid;
	}
	auto tree = AABBTree(alpha_shape.begin(), alpha_shape.end());

	std::cout << cell_cs.size() << " cells\n";
	for (int i : ids) {
		in_value[i] = MiniDray(cell_cs[i], tree, rays);
	}


	for (int cid = 0; cid < (int)cvxs.size(); ++cid)
		cvx_is_inside[cid] = (in_value[cid] > inside_thred);
	cvx_partitioned.assign(cvx_is_inside.size(),false);
}

void MiniLcc::GMGlabel_cvxs(std::vector<Planar> &planars, Real inside_thred, int nr) {
	FUNC_TIMER();
	std::list<IK_Triangle_3> alpha_shape;
	for (auto& planar : planars)
		alpha_shape.insert(alpha_shape.end(), planar.label_alpha.begin(), planar.label_alpha.end());
	std::deque<std::vector<int>> valid_cvxs;
	std::unordered_map<int,int>index_map;
	for (int i=0;i<cvxs.size();++i)
		if (!cvx_partitioned[i]) {
			index_map[valid_cvxs.size()] = i;
			valid_cvxs.push_back(cvxs[i]);
		}



	cvx_is_inside.assign(cvxs.size(),false);
	// // rays
	std::vector<IK::Vector_3> rays;
	Real pi = 2 * acos(0.0);
	Real gold = 3 - sqrt(5);
	int samples = nr;

	for (int i = 0; i < nr; i++) {
		Real z = 1 - (Real(i) / Real(samples - 1)) * 2;
		Real theta = pi * i * gold;
		Real x = cos(theta) * sqrt(1 - z * z);
		Real y = sin(theta) * sqrt(1 - z * z);
		rays.push_back(IK::Vector_3(x, y, z));
	}
	//centers

	// centers
	std::vector<std::vector<IK::Point_3>> cell_cs;
	std::vector<int> ids(valid_cvxs.size());
	std::vector<Real> in_value(valid_cvxs.size(), 0.0);

	for (int cid = 0; cid < (int)valid_cvxs.size(); ++cid)
	{
		std::vector<IK::Point_3> centers;
		centers.push_back(cvx_center(valid_cvxs[cid]));
		cell_cs.push_back(std::move(centers));
		ids[cid] = cid;
	}
	auto tree = AABBTree(alpha_shape.begin(), alpha_shape.end());

	std::cout << cell_cs.size() << " cells\n";
	for (int i : ids) {
		in_value[i] = MiniDray(cell_cs[i], tree, rays);
	}


	for (int cid = 0; cid < (int)valid_cvxs.size(); ++cid)
		cvx_is_inside[index_map[cid]] = (in_value[cid] > inside_thred);
}

void MiniLcc::extract_surface(std::string save_path) {
	using CGAL::Polygon_mesh_processing::repair_polygon_soup;
	using CGAL::Polygon_mesh_processing::orient_polygon_soup;
	using CGAL::Polygon_mesh_processing::polygon_soup_to_polygon_mesh;

	EK_to_IK to_inexact;

	// 1) 统计哪些 face 被 inside cvx 引用
	std::map<int, int> face_count;
	for (int cid = 0; cid < (int)cvxs.size(); ++cid) {
		if (!cvx_is_inside[cid]) continue;
		for (int fid : cvxs[cid]) {
			RZ_ASSERT(fid >= 0 && fid < (int)faces.size(), "face id out of range");
			face_count[fid]++;
		}
	}

	// 2) 恢复一个 face 的有序顶点环
	auto ordered_vertices_of_face = [&](int fid) -> std::vector<int> {
		RZ_ASSERT(fid >= 0 && fid < (int)faces.size(), "face id out of range");

		const auto& eids = faces[fid];
		RZ_ASSERT(eids.size() >= 3, "face has fewer than 3 edges");

		std::map<int, std::vector<int>> adj;
		for (int eid : eids) {
			RZ_ASSERT(eid >= 0 && eid < (int)edges.size(), "edge id out of range");
			const auto& e = edges[eid];
			adj[e.v1].push_back(e.v2);
			adj[e.v2].push_back(e.v1);
		}

		for (const auto& kv : adj) {
			RZ_ASSERT(kv.second.size() == 2,
				"face is not a simple polygon (vertex degree != 2)");
		}

		std::vector<int> loop;
		loop.reserve(adj.size());

		int start = adj.begin()->first;
		int prev = -1;
		int cur = start;

		while (true) {
			loop.push_back(cur);

			const auto& nbrs = adj[cur];
			RZ_ASSERT(nbrs.size() == 2, "invalid face adjacency");

			int nxt = (nbrs[0] == prev ? nbrs[1] : nbrs[0]);

			prev = cur;
			cur = nxt;

			if (cur == start) break;

			RZ_ASSERT((int)loop.size() <= (int)adj.size() + 1,
				"loop reconstruction failed");
		}

		RZ_ASSERT(loop.size() >= 3, "reconstructed face loop has fewer than 3 vertices");
		return loop;
	};

	// 3) polygon soup
	std::vector<IK::Point_3> soup_points;
	std::vector<std::vector<size_t>> soup_polygons;

	for (const auto& kv : face_count) {
		int fid = kv.first;
		int cnt = kv.second;

		// 只出现一次的 inside face 才是 surface
		if (cnt != 1) continue;

		auto loop = ordered_vertices_of_face(fid);
		if (loop.size() < 3) continue;

		std::vector<size_t> polygon;
		polygon.reserve(loop.size());

		for (int vid : loop) {
			RZ_ASSERT(vid >= 0 && vid < (int)points.size(), "vertex id out of range");
			polygon.push_back(soup_points.size());
			soup_points.push_back(to_inexact(points[vid]));
		}

		soup_polygons.push_back(std::move(polygon));
	}

	RZ_ASSERT(!soup_polygons.empty(), "no surface polygons extracted");

	std::cout << "surface polygons: " << soup_polygons.size() << "\n";
	std::cout << "repair polygon soup\n";
	repair_polygon_soup(soup_points, soup_polygons);

	std::cout << "orient polygon soup\n";
	orient_polygon_soup(soup_points, soup_polygons);

	std::cout << "polygon soup to mesh\n";
	CGAL::Surface_mesh<IK::Point_3> mesh;
	polygon_soup_to_polygon_mesh(soup_points, soup_polygons, mesh);

	std::cout << "write mesh: " << save_path << "\n";
	bool ok = CGAL::IO::write_polygon_mesh(save_path, mesh,
		CGAL::parameters::stream_precision(17));
	RZ_ASSERT(ok, "failed to write polygon mesh");
}

void MiniLcc::valid_cvxs() {
	std::deque<std::vector<int>> valid_cvxs;
	for (int i=0;i<cvxs.size();++i)
		if (!cvx_partitioned[i])
			valid_cvxs.push_back(cvxs[i]);
	cvxs.swap(valid_cvxs);
}

void MiniLcc::clear() {
    FUNC_MEMORY_TIMER();
	std::deque<EK::Point_3>().swap(points);
	std::deque<NDEdge>().swap(edges);

	std::deque<std::vector<int>>().swap(faces);
	std::deque<std::vector<int>>().swap(face_to_cvxs);
	std::deque<std::vector<int>>().swap(edge_to_faces);
	std::deque<std::vector<int>>().swap(cvxs);
	std::deque<std::vector<int>>().swap(cvx_inside_planars);

	std::deque<std::vector<Real>>().swap(cvx_inside_planars_areas);

	std::deque<int>().swap(cvx_partition_poly_index);
	std::deque<bool>().swap(cvx_partitioned);
	std::deque<std::pair<int, int>>().swap(cvx_sons);
	std::deque<int>().swap(cvx_father);
	std::deque<Real>().swap(cvx_cut_values);
	std::deque<bool>().swap(cut_value_inside);

	std::deque<bool>().swap(cvx_is_inside);
	std::vector<EK_Planar>().swap(my_planars);

	{
		std::queue<int> empty;
		std::swap(volumes, empty);
	}

	std::deque<std::vector<int>>().swap(faces_copy);
	std::deque<std::vector<int>>().swap(cvxs_copy);

	malloc_trim(0);
}

// 假设在 MiniLcc 类内实现
void MiniLcc::save_lcc(std::string filename)
{
    // ------------------------------------------------------------
    // 你需要根据自己的 NDEdge 定义，改这个函数里取两个端点的方式
    // 例如：
    //   return { edges[eid].v1, edges[eid].v2 };
    // 或 return { edges[eid].first, edges[eid].second };
    // ------------------------------------------------------------
    auto get_edge_vertices = [&](int eid) -> std::pair<int,int>
    {
        const auto& e = edges[eid];

        // ===== 请按你的 NDEdge 实际字段修改这里 =====
        return { e.v1, e.v2 };
        // ============================================
    };

    auto reorder_face_edges_to_vertex_loop =
        [&](const std::vector<int>& face_edge_ids) -> std::vector<int>
    {
        std::vector<int> loop;
        if (face_edge_ids.empty()) return loop;

        // 1) 收集本面涉及的边和点，并建立“点 -> 邻接点”关系
        std::unordered_map<int, std::vector<int>> vertex_neighbors;
        std::set<int> used_vertices;

        for (int eid : face_edge_ids) {
            if (eid < 0 || eid >= (int)edges.size()) {
                throw std::runtime_error("save_lcc: face contains invalid edge id.");
            }

            auto [a, b] = get_edge_vertices(eid);
            if (a < 0 || a >= (int)points.size() || b < 0 || b >= (int)points.size()) {
                throw std::runtime_error("save_lcc: edge contains invalid vertex id.");
            }

            vertex_neighbors[a].push_back(b);
            vertex_neighbors[b].push_back(a);
            used_vertices.insert(a);
            used_vertices.insert(b);
        }

        // 2) 简单多边形检查：每个点度数应为 2
        for (int v : used_vertices) {
            if (vertex_neighbors[v].size() != 2) {
                std::cerr << "[save_lcc] invalid face: vertex degree != 2, vertex = "
                          << v << ", degree = " << vertex_neighbors[v].size() << std::endl;
                throw std::runtime_error("save_lcc: face is not a simple polygon.");
            }
        }

        // 3) 从第一条边开始，沿环走一圈，恢复有序顶点
        auto [v0, v1] = get_edge_vertices(face_edge_ids[0]);

        loop.reserve(face_edge_ids.size());
        loop.push_back(v0);

        int prev = v0;
        int curr = v1;

        // 面有 m 条边，则应得到 m 个顶点
        for (int step = 1; step < (int)face_edge_ids.size(); ++step) {
            loop.push_back(curr);

            const auto& nbrs = vertex_neighbors[curr];
            int next = -1;

            if (nbrs[0] == prev) next = nbrs[1];
            else if (nbrs[1] == prev) next = nbrs[0];
            else {
                throw std::runtime_error("save_lcc: broken face loop while reconstructing polygon.");
            }

            prev = curr;
            curr = next;
        }

        // 4) 闭环检查：最后应回到起点 v0
        if (curr != v0) {
            std::cerr << "[save_lcc] face loop does not close. expected "
                      << v0 << ", got " << curr << std::endl;
            throw std::runtime_error("save_lcc: reconstructed face loop does not close.");
        }

        return loop;
    };

    // ------------------------------------------------------------
    // 收集要导出的 cell
    // 这里默认：
    //   - 跳过被继续分裂过的 cell: cvx_partitioned[cid] == true
    //   - 跳过 inside cell: cvx_is_inside[cid] == true
    // 若你的语义不同，可自行改掉
    // ------------------------------------------------------------
    std::vector<int> export_cell_ids;
    export_cell_ids.reserve(cvxs.size());

    for (int cid = 0; cid < (int)cvxs.size(); ++cid) {
        bool skip = false;

        if (cid < (int)cvx_partitioned.size() && cvx_partitioned[cid]) {
            skip = true;
        }
        if (cid < (int)cvx_is_inside.size() && cvx_is_inside[cid]) {
            skip = true;
        }

        if (!skip) {
            export_cell_ids.push_back(cid);
        }
    }

    // ------------------------------------------------------------
    // Points: 直接输出全部全局点，cell/face 都引用全局点 id
    // 这样最稳，不需要再压缩点编号
    // ------------------------------------------------------------
    const int num_points = (int)points.size();
    const int num_cells  = (int)export_cell_ids.size();

    // 每个 cell 的 face 顶点环
    std::vector<std::vector<std::vector<int>>> cells_faces;
    cells_faces.reserve(num_cells);

    for (int cid : export_cell_ids) {
        const auto& cell_face_ids = cvxs[cid];
        std::vector<std::vector<int>> one_cell_faces;
        one_cell_faces.reserve(cell_face_ids.size());

        for (int fid : cell_face_ids) {
            if (fid < 0 || fid >= (int)faces.size()) {
                throw std::runtime_error("save_lcc: cell contains invalid face id.");
            }

            const auto& face_edge_ids = faces[fid];
            auto vertex_loop = reorder_face_edges_to_vertex_loop(face_edge_ids);

            if (vertex_loop.size() < 3) {
                throw std::runtime_error("save_lcc: face has fewer than 3 vertices.");
            }

            one_cell_faces.push_back(std::move(vertex_loop));
        }

        cells_faces.push_back(std::move(one_cell_faces));
    }

    // ------------------------------------------------------------
    // 生成 VTK UnstructuredGrid / POLYHEDRON 所需编码
    // ------------------------------------------------------------
    const int VTK_POLYHEDRON = 42;

    std::vector<int> connectivity;
    std::vector<int> offsets;
    std::vector<unsigned char> types(num_cells, static_cast<unsigned char>(VTK_POLYHEDRON));
    std::vector<int> faces_flat;
    std::vector<int> faceoffsets;

    connectivity.reserve(1024);
    offsets.reserve(num_cells);
    faces_flat.reserve(1024);
    faceoffsets.reserve(num_cells);

    int connectivity_offset = 0;
    int face_offset = 0;

    for (const auto& cell_faces : cells_faces) {
        // connectivity / offsets
        // 对 POLYHEDRON，connectivity 通常为该 cell 所有 face 的顶点索引串接
        for (const auto& face : cell_faces) {
            for (int vid : face) {
                connectivity.push_back(vid);
            }
            connectivity_offset += (int)face.size();
        }
        offsets.push_back(connectivity_offset);

        // faces / faceoffsets
        // 格式: [numFaces, n0, face0..., n1, face1..., ...]
        faces_flat.push_back((int)cell_faces.size());
        ++face_offset;

        for (const auto& face : cell_faces) {
            faces_flat.push_back((int)face.size());
            ++face_offset;

            for (int vid : face) {
                faces_flat.push_back(vid);
                ++face_offset;
            }
        }
        faceoffsets.push_back(face_offset);
    }

    // ------------------------------------------------------------
    // 写 VTU
    // ------------------------------------------------------------
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "[save_lcc] failed to open file: " << filename << std::endl;

    }

    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    out << "  <UnstructuredGrid>\n";
    out << "    <Piece NumberOfPoints=\"" << num_points
        << "\" NumberOfCells=\"" << num_cells << "\">\n";

    // Points
    out << "      <Points>\n";
    out << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    out << std::setprecision(17);
    for (const auto& p : points) {
        out << "          " << p.x() << " " << p.y() << " " << p.z() << "\n";
    }
    out << "        </DataArray>\n";
    out << "      </Points>\n";

    // Cells
    out << "      <Cells>\n";

    out << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    out << "          ";
    for (int x : connectivity) out << x << " ";
    out << "\n";
    out << "        </DataArray>\n";

    out << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    out << "          ";
    for (int x : offsets) out << x << " ";
    out << "\n";
    out << "        </DataArray>\n";

    out << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    out << "          ";
    for (unsigned int x : types) out << x << " ";
    out << "\n";
    out << "        </DataArray>\n";

    out << "        <DataArray type=\"Int32\" Name=\"faces\" format=\"ascii\">\n";
    out << "          ";
    for (int x : faces_flat) out << x << " ";
    out << "\n";
    out << "        </DataArray>\n";

    out << "        <DataArray type=\"Int32\" Name=\"faceoffsets\" format=\"ascii\">\n";
    out << "          ";
    for (int x : faceoffsets) out << x << " ";
    out << "\n";
    out << "        </DataArray>\n";

    out << "      </Cells>\n";

    // CellData
    out << "      <CellData Scalars=\"cell_id\">\n";
    out << "        <DataArray type=\"Int32\" Name=\"cell_id\" format=\"ascii\">\n";
    out << "          ";
    for (int i = 0; i < num_cells; ++i) out << i << " ";
    out << "\n";
    out << "        </DataArray>\n";
    out << "      </CellData>\n";

    out << "    </Piece>\n";
    out << "  </UnstructuredGrid>\n";
    out << "</VTKFile>\n";

    out.close();

    std::cout << "✅ VTU file saved to: " << filename << std::endl;
    std::cout << "   points = " << num_points << ", cells = " << num_cells << std::endl;


}

void MiniLcc::save_lcc(std::string filename,const std::vector<int>& level_cvxs) {
	    // ------------------------------------------------------------
    // 你需要根据自己的 NDEdge 定义，改这个函数里取两个端点的方式
    // 例如：
    //   return { edges[eid].v1, edges[eid].v2 };
    // 或 return { edges[eid].first, edges[eid].second };
    // ------------------------------------------------------------
    auto get_edge_vertices = [&](int eid) -> std::pair<int,int>
    {
        const auto& e = edges[eid];

        // ===== 请按你的 NDEdge 实际字段修改这里 =====
        return { e.v1, e.v2 };
        // ============================================
    };

    auto reorder_face_edges_to_vertex_loop =
        [&](const std::vector<int>& face_edge_ids) -> std::vector<int>
    {
        std::vector<int> loop;
        if (face_edge_ids.empty()) return loop;

        // 1) 收集本面涉及的边和点，并建立“点 -> 邻接点”关系
        std::unordered_map<int, std::vector<int>> vertex_neighbors;
        std::set<int> used_vertices;

        for (int eid : face_edge_ids) {
            if (eid < 0 || eid >= (int)edges.size()) {
                throw std::runtime_error("save_lcc: face contains invalid edge id.");
            }

            auto [a, b] = get_edge_vertices(eid);
            if (a < 0 || a >= (int)points.size() || b < 0 || b >= (int)points.size()) {
                throw std::runtime_error("save_lcc: edge contains invalid vertex id.");
            }

            vertex_neighbors[a].push_back(b);
            vertex_neighbors[b].push_back(a);
            used_vertices.insert(a);
            used_vertices.insert(b);
        }

        // 2) 简单多边形检查：每个点度数应为 2
        for (int v : used_vertices) {
            if (vertex_neighbors[v].size() != 2) {
                std::cerr << "[save_lcc] invalid face: vertex degree != 2, vertex = "
                          << v << ", degree = " << vertex_neighbors[v].size() << std::endl;
                throw std::runtime_error("save_lcc: face is not a simple polygon.");
            }
        }

        // 3) 从第一条边开始，沿环走一圈，恢复有序顶点
        auto [v0, v1] = get_edge_vertices(face_edge_ids[0]);

        loop.reserve(face_edge_ids.size());
        loop.push_back(v0);

        int prev = v0;
        int curr = v1;

        // 面有 m 条边，则应得到 m 个顶点
        for (int step = 1; step < (int)face_edge_ids.size(); ++step) {
            loop.push_back(curr);

            const auto& nbrs = vertex_neighbors[curr];
            int next = -1;

            if (nbrs[0] == prev) next = nbrs[1];
            else if (nbrs[1] == prev) next = nbrs[0];
            else {
                throw std::runtime_error("save_lcc: broken face loop while reconstructing polygon.");
            }

            prev = curr;
            curr = next;
        }

        // 4) 闭环检查：最后应回到起点 v0
        if (curr != v0) {
            std::cerr << "[save_lcc] face loop does not close. expected "
                      << v0 << ", got " << curr << std::endl;
            throw std::runtime_error("save_lcc: reconstructed face loop does not close.");
        }

        return loop;
    };

    // ------------------------------------------------------------
    // 收集要导出的 cell
    // 这里默认：
    //   - 跳过被继续分裂过的 cell: cvx_partitioned[cid] == true
    //   - 跳过 inside cell: cvx_is_inside[cid] == true
    // 若你的语义不同，可自行改掉
    // ------------------------------------------------------------
	std::vector<int> export_cell_ids;
	for (auto cid:level_cvxs)if (cvx_is_inside[cid])continue;
	else export_cell_ids.push_back(cid);


    // ------------------------------------------------------------
    // Points: 直接输出全部全局点，cell/face 都引用全局点 id
    // 这样最稳，不需要再压缩点编号
    // ------------------------------------------------------------
    const int num_points = (int)points.size();
    const int num_cells  = (int)export_cell_ids.size();

    // 每个 cell 的 face 顶点环
    std::vector<std::vector<std::vector<int>>> cells_faces;
    cells_faces.reserve(num_cells);

    for (int cid : export_cell_ids) {
        const auto& cell_face_ids = cvxs[cid];
        std::vector<std::vector<int>> one_cell_faces;
        one_cell_faces.reserve(cell_face_ids.size());

        for (int fid : cell_face_ids) {
            if (fid < 0 || fid >= (int)faces.size()) {
                throw std::runtime_error("save_lcc: cell contains invalid face id.");
            }

            const auto& face_edge_ids = faces[fid];
            auto vertex_loop = reorder_face_edges_to_vertex_loop(face_edge_ids);

            if (vertex_loop.size() < 3) {
                throw std::runtime_error("save_lcc: face has fewer than 3 vertices.");
            }

            one_cell_faces.push_back(std::move(vertex_loop));
        }

        cells_faces.push_back(std::move(one_cell_faces));
    }

    // ------------------------------------------------------------
    // 生成 VTK UnstructuredGrid / POLYHEDRON 所需编码
    // ------------------------------------------------------------
    const int VTK_POLYHEDRON = 42;

    std::vector<int> connectivity;
    std::vector<int> offsets;
    std::vector<unsigned char> types(num_cells, static_cast<unsigned char>(VTK_POLYHEDRON));
    std::vector<int> faces_flat;
    std::vector<int> faceoffsets;

    connectivity.reserve(1024);
    offsets.reserve(num_cells);
    faces_flat.reserve(1024);
    faceoffsets.reserve(num_cells);

    int connectivity_offset = 0;
    int face_offset = 0;

    for (const auto& cell_faces : cells_faces) {
        // connectivity / offsets
        // 对 POLYHEDRON，connectivity 通常为该 cell 所有 face 的顶点索引串接
        for (const auto& face : cell_faces) {
            for (int vid : face) {
                connectivity.push_back(vid);
            }
            connectivity_offset += (int)face.size();
        }
        offsets.push_back(connectivity_offset);

        // faces / faceoffsets
        // 格式: [numFaces, n0, face0..., n1, face1..., ...]
        faces_flat.push_back((int)cell_faces.size());
        ++face_offset;

        for (const auto& face : cell_faces) {
            faces_flat.push_back((int)face.size());
            ++face_offset;

            for (int vid : face) {
                faces_flat.push_back(vid);
                ++face_offset;
            }
        }
        faceoffsets.push_back(face_offset);
    }

    // ------------------------------------------------------------
    // 写 VTU
    // ------------------------------------------------------------
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "[save_lcc] failed to open file: " << filename << std::endl;

    }

    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    out << "  <UnstructuredGrid>\n";
    out << "    <Piece NumberOfPoints=\"" << num_points
        << "\" NumberOfCells=\"" << num_cells << "\">\n";

    // Points
    out << "      <Points>\n";
    out << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    out << std::setprecision(17);
    for (const auto& p : points) {
        out << "          " << p.x() << " " << p.y() << " " << p.z() << "\n";
    }
    out << "        </DataArray>\n";
    out << "      </Points>\n";

    // Cells
    out << "      <Cells>\n";

    out << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    out << "          ";
    for (int x : connectivity) out << x << " ";
    out << "\n";
    out << "        </DataArray>\n";

    out << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    out << "          ";
    for (int x : offsets) out << x << " ";
    out << "\n";
    out << "        </DataArray>\n";

    out << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    out << "          ";
    for (unsigned int x : types) out << x << " ";
    out << "\n";
    out << "        </DataArray>\n";

    out << "        <DataArray type=\"Int32\" Name=\"faces\" format=\"ascii\">\n";
    out << "          ";
    for (int x : faces_flat) out << x << " ";
    out << "\n";
    out << "        </DataArray>\n";

    out << "        <DataArray type=\"Int32\" Name=\"faceoffsets\" format=\"ascii\">\n";
    out << "          ";
    for (int x : faceoffsets) out << x << " ";
    out << "\n";
    out << "        </DataArray>\n";

    out << "      </Cells>\n";

    // CellData
    out << "      <CellData Scalars=\"cell_id\">\n";
    out << "        <DataArray type=\"Int32\" Name=\"cell_id\" format=\"ascii\">\n";
    out << "          ";
    for (int i = 0; i < num_cells; ++i) out << i << " ";
    out << "\n";
    out << "        </DataArray>\n";
    out << "      </CellData>\n";

    out << "    </Piece>\n";
    out << "  </UnstructuredGrid>\n";
    out << "</VTKFile>\n";

    out.close();

    std::cout << "✅ VTU file saved to: " << filename << std::endl;
    std::cout << "   points = " << num_points << ", cells = " << num_cells << std::endl;
}

void MiniLcc::save_lcc_compact_points(std::string name,const std::vector<int>& level_cvxs) {
    FUNC_TIMER();

    // ------------------------------------------------------------
    // 你需要根据自己的 NDEdge 定义，改这里
    // ------------------------------------------------------------
    auto get_edge_vertices = [&](int eid) -> std::pair<int, int>
    {
        const auto& e = edges[eid];
        return { e.v1, e.v2 };
    };

    // ------------------------------------------------------------
    // 根据 face 的 edge 环，恢复有序顶点环
    // ------------------------------------------------------------
    auto reorder_face_edges_to_vertex_loop =
        [&](const std::vector<int>& face_edge_ids) -> std::vector<int>
    {
        std::vector<int> loop;
        if (face_edge_ids.empty()) return loop;

        std::unordered_map<int, std::vector<int>> vertex_neighbors;
        std::set<int> used_vertices;

        for (int eid : face_edge_ids) {
            if (eid < 0 || eid >= (int)edges.size()) {
                throw std::runtime_error("save_lcc_compact_points: invalid edge id in face.");
            }

            auto [a, b] = get_edge_vertices(eid);
            if (a < 0 || a >= (int)points.size() || b < 0 || b >= (int)points.size()) {
                throw std::runtime_error("save_lcc_compact_points: invalid vertex id in edge.");
            }

            vertex_neighbors[a].push_back(b);
            vertex_neighbors[b].push_back(a);
            used_vertices.insert(a);
            used_vertices.insert(b);
        }

        for (int v : used_vertices) {
            if (vertex_neighbors[v].size() != 2) {
                std::cerr << "[save_lcc_compact_points] invalid face: vertex degree != 2, vertex = "
                          << v << ", degree = " << vertex_neighbors[v].size() << std::endl;
                throw std::runtime_error("save_lcc_compact_points: face is not a simple polygon.");
            }
        }

        auto [v0, v1] = get_edge_vertices(face_edge_ids[0]);

        loop.reserve(face_edge_ids.size());
        loop.push_back(v0);

        int prev = v0;
        int curr = v1;

        for (int step = 1; step < (int)face_edge_ids.size(); ++step) {
            loop.push_back(curr);

            const auto& nbrs = vertex_neighbors[curr];
            int next = -1;

            if (nbrs[0] == prev) next = nbrs[1];
            else if (nbrs[1] == prev) next = nbrs[0];
            else {
                throw std::runtime_error("save_lcc_compact_points: broken face loop.");
            }

            prev = curr;
            curr = next;
        }

        if (curr != v0) {
            std::cerr << "[save_lcc_compact_points] face loop does not close. expected "
                      << v0 << ", got " << curr << std::endl;
            throw std::runtime_error("save_lcc_compact_points: reconstructed loop does not close.");
        }

        return loop;
    };

    // ------------------------------------------------------------
    // 收集要导出的 cell
    // ------------------------------------------------------------
    std::vector<int> export_cell_ids;
    export_cell_ids.reserve(level_cvxs.size());

    for (int cid : level_cvxs) {
        if (cvx_is_inside[cid]) continue;
        export_cell_ids.push_back(cid);
    }

    const int num_cells = (int)export_cell_ids.size();

    // ------------------------------------------------------------
    // 第一步：先提取每个 cell 的 face 顶点环（仍然用全局点编号）
    // ------------------------------------------------------------
    std::vector<std::vector<std::vector<int>>> cells_faces_global;
    cells_faces_global.reserve(num_cells);

    // 收集真正使用到的全局点
    std::vector<int> used_global_points;
    used_global_points.reserve(1024);

    std::vector<uint8_t> point_mark(points.size(), 0);
	std::map<int,int> global_edge_cnt;
    for (int cid : export_cell_ids) {
        const auto& cell_face_ids = cvxs[cid];
        std::vector<std::vector<int>> one_cell_faces;
        one_cell_faces.reserve(cell_face_ids.size());
    	std::map<int,int> edge_cnt;
        for (int fid : cell_face_ids) {
            if (fid < 0 || fid >= (int)faces.size()) {
                throw std::runtime_error("save_lcc_compact_points: invalid face id in cell.");
            }

            const auto& face_edge_ids = faces[fid];

        	for (auto edge:face_edge_ids)edge_cnt[edge]++;

            auto vertex_loop = reorder_face_edges_to_vertex_loop(face_edge_ids);

            if (vertex_loop.size() < 3) {
                throw std::runtime_error("save_lcc_compact_points: face has fewer than 3 vertices.");
            }

            for (int gvid : vertex_loop) {
                if (!point_mark[gvid]) {
                    point_mark[gvid] = 1;
                    used_global_points.push_back(gvid);
                }
            }

            one_cell_faces.push_back(std::move(vertex_loop));
        }
    	for (auto&kv:edge_cnt) {
    		if (kv.second!=2) {
    			for (auto&nkv:edge_cnt) {
    				std::cout<<nkv.second<<" ";
    			}
    			for (auto&nkv:edge_cnt) {
    				if (nkv.second == 1)
    					RZ_INFO(edges[nkv.first].v1," ",edges[nkv.first].v2);
    			}
    			RZ_FATAL("");
    		}
    	}
    	for (auto&kv:edge_cnt)global_edge_cnt[kv.first]++;
        cells_faces_global.push_back(std::move(one_cell_faces));
    }
	std::vector<EK::Point_3>ps;
	for (auto&kv:global_edge_cnt) {
		if (kv.second <= 2) {
			ps.push_back(points[edges[kv.first].v1]);
			ps.push_back(points[edges[kv.first].v2]);
		}
	}
	std::string outname = name;
	size_t pos = outname.find_last_of('.');
	if (pos != std::string::npos)
		outname = outname.substr(0, pos);
	outname += ".xyz";

	std::ofstream fout(outname);
	for (auto &p : ps) {
		fout << p.x() << " " << p.y() << " " << p.z() << "\n";
	}
	fout.close();


    // ------------------------------------------------------------
    // 第二步：建立全局点 -> 局部点 重编号
    // ------------------------------------------------------------
    std::vector<int> global_to_local(points.size(), -1);
    for (int new_vid = 0; new_vid < (int)used_global_points.size(); ++new_vid) {
        int old_vid = used_global_points[new_vid];
        global_to_local[old_vid] = new_vid;
    }

    const int num_points = (int)used_global_points.size();

    // ------------------------------------------------------------
    // 第三步：把每个 face 的顶点环从全局编号映射到局部编号
    // ------------------------------------------------------------
    std::vector<std::vector<std::vector<int>>> cells_faces_local;
    cells_faces_local.reserve(num_cells);

    for (const auto& cell_faces : cells_faces_global) {
        std::vector<std::vector<int>> one_cell_faces;
        one_cell_faces.reserve(cell_faces.size());

        for (const auto& face_loop_global : cell_faces) {
            std::vector<int> face_loop_local;
            face_loop_local.reserve(face_loop_global.size());

            for (int old_vid : face_loop_global) {
                int new_vid = global_to_local[old_vid];
                if (new_vid < 0) {
                    throw std::runtime_error("save_lcc_compact_points: point remap failed.");
                }
                face_loop_local.push_back(new_vid);
            }

            one_cell_faces.push_back(std::move(face_loop_local));
        }

        cells_faces_local.push_back(std::move(one_cell_faces));
    }

    // ------------------------------------------------------------
    // 生成 VTK UnstructuredGrid / POLYHEDRON 所需编码
    // ------------------------------------------------------------
    const int VTK_POLYHEDRON = 42;

    std::vector<int> connectivity;
    std::vector<int> offsets;
    std::vector<unsigned char> types(num_cells, static_cast<unsigned char>(VTK_POLYHEDRON));
    std::vector<int> faces_flat;
    std::vector<int> faceoffsets;

    connectivity.reserve(1024);
    offsets.reserve(num_cells);
    faces_flat.reserve(1024);
    faceoffsets.reserve(num_cells);

    int connectivity_offset = 0;
    int face_offset = 0;

    for (const auto& cell_faces : cells_faces_local) {
        for (const auto& face : cell_faces) {
            for (int vid : face) {
                connectivity.push_back(vid);
            }
            connectivity_offset += (int)face.size();
        }
        offsets.push_back(connectivity_offset);

        faces_flat.push_back((int)cell_faces.size());
        ++face_offset;

        for (const auto& face : cell_faces) {
            faces_flat.push_back((int)face.size());
            ++face_offset;

            for (int vid : face) {
                faces_flat.push_back(vid);
                ++face_offset;
            }
        }
        faceoffsets.push_back(face_offset);
    }

    // ------------------------------------------------------------
    // 写 VTU
    // ------------------------------------------------------------
    std::ofstream out(name);
    if (!out.is_open()) {
        std::cerr << "[save_lcc_compact_points] failed to open file: " << name << std::endl;
        return;
    }

    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    out << "  <UnstructuredGrid>\n";
    out << "    <Piece NumberOfPoints=\"" << num_points
        << "\" NumberOfCells=\"" << num_cells << "\">\n";

    // Points：只输出 used_global_points
    out << "      <Points>\n";
    out << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    out << std::setprecision(17);

    for (int old_vid : used_global_points) {
        const auto& p = points[old_vid];
        out << "          " << p.x() << " " << p.y() << " " << p.z() << "\n";
    }

    out << "        </DataArray>\n";
    out << "      </Points>\n";

    // Cells
    out << "      <Cells>\n";

    out << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    out << "          ";
    for (int x : connectivity) out << x << " ";
    out << "\n";
    out << "        </DataArray>\n";

    out << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    out << "          ";
    for (int x : offsets) out << x << " ";
    out << "\n";
    out << "        </DataArray>\n";

    out << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    out << "          ";
    for (unsigned int x : types) out << x << " ";
    out << "\n";
    out << "        </DataArray>\n";

    out << "        <DataArray type=\"Int32\" Name=\"faces\" format=\"ascii\">\n";
    out << "          ";
    for (int x : faces_flat) out << x << " ";
    out << "\n";
    out << "        </DataArray>\n";

    out << "        <DataArray type=\"Int32\" Name=\"faceoffsets\" format=\"ascii\">\n";
    out << "          ";
    for (int x : faceoffsets) out << x << " ";
    out << "\n";
    out << "        </DataArray>\n";

    out << "      </Cells>\n";

    // CellData
    out << "      <CellData Scalars=\"cell_id\">\n";
    out << "        <DataArray type=\"Int32\" Name=\"cell_id\" format=\"ascii\">\n";
    out << "          ";
    for (int i = 0; i < num_cells; ++i) out << i << " ";
    out << "\n";
    out << "        </DataArray>\n";
    out << "      </CellData>\n";

    out << "    </Piece>\n";
    out << "  </UnstructuredGrid>\n";
    out << "</VTKFile>\n";

    out.close();

    std::cout << "✅ Compact VTU file saved to: " << name << std::endl;
    std::cout << "   used points = " << num_points
              << ", cells = " << num_cells << std::endl;
}

void MiniLcc::save_vis_lcc(std::string filename)
{
    auto get_edge_vertices = [&](int eid) -> std::pair<int,int>
    {
        const auto& e = edges[eid];
        return { e.v1, e.v2 };
    };

    auto reorder_face_edges_to_vertex_loop =
        [&](const std::vector<int>& face_edge_ids) -> std::vector<int>
    {
        std::vector<int> loop;
        if (face_edge_ids.empty()) return loop;

        std::unordered_map<int, std::vector<int>> vertex_neighbors;
        std::set<int> used_vertices;

        for (int eid : face_edge_ids) {
            if (eid < 0 || eid >= (int)edges.size()) {
                throw std::runtime_error("save_vis_lcc: face contains invalid edge id.");
            }

            auto [a, b] = get_edge_vertices(eid);
            if (a < 0 || a >= (int)points.size() || b < 0 || b >= (int)points.size()) {
                throw std::runtime_error("save_vis_lcc: edge contains invalid vertex id.");
            }

            vertex_neighbors[a].push_back(b);
            vertex_neighbors[b].push_back(a);
            used_vertices.insert(a);
            used_vertices.insert(b);
        }

        for (int v : used_vertices) {
            if (vertex_neighbors[v].size() != 2) {
                std::cerr << "[save_vis_lcc] invalid face: vertex degree != 2, vertex = "
                          << v << ", degree = " << vertex_neighbors[v].size() << std::endl;
                throw std::runtime_error("save_vis_lcc: face is not a simple polygon.");
            }
        }

        auto [v0, v1] = get_edge_vertices(face_edge_ids[0]);

        loop.reserve(face_edge_ids.size());
        loop.push_back(v0);

        int prev = v0;
        int curr = v1;

        for (int step = 1; step < (int)face_edge_ids.size(); ++step) {
            loop.push_back(curr);

            const auto& nbrs = vertex_neighbors[curr];
            int next = -1;

            if (nbrs[0] == prev) next = nbrs[1];
            else if (nbrs[1] == prev) next = nbrs[0];
            else {
                throw std::runtime_error("save_vis_lcc: broken face loop while reconstructing polygon.");
            }

            prev = curr;
            curr = next;
        }

        if (curr != v0) {
            std::cerr << "[save_vis_lcc] face loop does not close. expected "
                      << v0 << ", got " << curr << std::endl;
            throw std::runtime_error("save_vis_lcc: reconstructed face loop does not close.");
        }

        return loop;
    };

    // ------------------------------------------------------------
    // 自动收集要导出的 cell
    // 逻辑与 save_lcc 类似：
    // 1) 跳过被继续划分过的 cell
    // 2) 跳过 inside cell
    // 这样就不需要外部传入 lod / level_cvxs
    // ------------------------------------------------------------
    std::vector<int> export_cell_ids;
    export_cell_ids.reserve(cvxs.size());

    for (int cid = 0; cid < (int)cvxs.size(); ++cid) {
        bool skip = false;

        if (cid < (int)cvx_partitioned.size() && cvx_partitioned[cid]) {
            skip = true;
        }

        // 这里你有两种可能：
        // 1. 用 cut_value_inside
        // 2. 用 cvx_is_inside
        // 取决于你当前 MiniLcc 真正维护的是哪个
        if (cid < (int)cut_value_inside.size() && cut_value_inside[cid]) {
            skip = true;
        }
        // 如果你实际用的是 cvx_is_inside，就改成下面这句：
        // if (cid < (int)cvx_is_inside.size() && cvx_is_inside[cid]) skip = true;

        if (!skip) {
            export_cell_ids.push_back(cid);
        }
    }

    const int num_cells = (int)export_cell_ids.size();

    // 每个 cell 的每个 face，保存“展开后的局部点编号”
    std::vector<std::vector<std::vector<int>>> cells_faces;
    cells_faces.reserve(num_cells);

    // 导出点：不复用
    std::vector<EK::Point_3> vis_points;
    vis_points.reserve(1024);

    // 记录每个导出点对应原始 global vertex id
    std::vector<int> vis_point_to_global_vid;
    vis_point_to_global_vid.reserve(1024);

    for (int cid : export_cell_ids) {
        if (cid < 0 || cid >= (int)cvxs.size()) {
            throw std::runtime_error("save_vis_lcc: invalid cell id.");
        }

        const auto& cell_face_ids = cvxs[cid];
        std::vector<std::vector<int>> one_cell_faces;
        one_cell_faces.reserve(cell_face_ids.size());

        for (int fid : cell_face_ids) {
            if (fid < 0 || fid >= (int)faces.size()) {
                throw std::runtime_error("save_vis_lcc: cell contains invalid face id.");
            }

            const auto& face_edge_ids = faces[fid];
            auto global_vertex_loop = reorder_face_edges_to_vertex_loop(face_edge_ids);

            if (global_vertex_loop.size() < 3) {
                throw std::runtime_error("save_vis_lcc: face has fewer than 3 vertices.");
            }

            // 每个 face 的每个顶点 occurrence 都新建一个点
            std::vector<int> face_local_ids;
            face_local_ids.reserve(global_vertex_loop.size());

            for (int gvid : global_vertex_loop) {
                if (gvid < 0 || gvid >= (int)points.size()) {
                    throw std::runtime_error("save_vis_lcc: global vertex id out of range.");
                }

                int new_vid = (int)vis_points.size();
                vis_points.push_back(points[gvid]);
                vis_point_to_global_vid.push_back(gvid);
                face_local_ids.push_back(new_vid);
            }

            one_cell_faces.push_back(std::move(face_local_ids));
        }

        cells_faces.push_back(std::move(one_cell_faces));
    }

    const int num_points = (int)vis_points.size();

    // ------------------------------------------------------------
    // 生成 VTK UnstructuredGrid / POLYHEDRON 编码
    // ------------------------------------------------------------
    const int VTK_POLYHEDRON = 42;

    std::vector<int> connectivity;
    std::vector<int> offsets;
    std::vector<unsigned char> types(num_cells, static_cast<unsigned char>(VTK_POLYHEDRON));
    std::vector<int> faces_flat;
    std::vector<int> faceoffsets;

    connectivity.reserve(1024);
    offsets.reserve(num_cells);
    faces_flat.reserve(1024);
    faceoffsets.reserve(num_cells);

    int connectivity_offset = 0;
    int face_offset = 0;

    for (const auto& cell_faces : cells_faces) {
        for (const auto& face : cell_faces) {
            for (int vid : face) {
                connectivity.push_back(vid);
            }
            connectivity_offset += (int)face.size();
        }
        offsets.push_back(connectivity_offset);

        faces_flat.push_back((int)cell_faces.size());
        ++face_offset;

        for (const auto& face : cell_faces) {
            faces_flat.push_back((int)face.size());
            ++face_offset;

            for (int vid : face) {
                faces_flat.push_back(vid);
                ++face_offset;
            }
        }
        faceoffsets.push_back(face_offset);
    }

    // ------------------------------------------------------------
    // 写 VTU
    // ------------------------------------------------------------
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "[save_vis_lcc] failed to open file: " << filename << std::endl;
        return;
    }

    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    out << "  <UnstructuredGrid>\n";
    out << "    <Piece NumberOfPoints=\"" << num_points
        << "\" NumberOfCells=\"" << num_cells << "\">\n";

    out << "      <Points>\n";
    out << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    out << std::setprecision(17);
    for (const auto& p : vis_points) {
        out << "          " << p.x() << " " << p.y() << " " << p.z() << "\n";
    }
    out << "        </DataArray>\n";
    out << "      </Points>\n";

    out << "      <Cells>\n";

    out << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    out << "          ";
    for (int x : connectivity) out << x << " ";
    out << "\n";
    out << "        </DataArray>\n";

    out << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    out << "          ";
    for (int x : offsets) out << x << " ";
    out << "\n";
    out << "        </DataArray>\n";

    out << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    out << "          ";
    for (unsigned int x : types) out << x << " ";
    out << "\n";
    out << "        </DataArray>\n";

    out << "        <DataArray type=\"Int32\" Name=\"faces\" format=\"ascii\">\n";
    out << "          ";
    for (int x : faces_flat) out << x << " ";
    out << "\n";
    out << "        </DataArray>\n";

    out << "        <DataArray type=\"Int32\" Name=\"faceoffsets\" format=\"ascii\">\n";
    out << "          ";
    for (int x : faceoffsets) out << x << " ";
    out << "\n";
    out << "        </DataArray>\n";

    out << "      </Cells>\n";

    out << "      <CellData Scalars=\"cell_id\">\n";
    out << "        <DataArray type=\"Int32\" Name=\"cell_id\" format=\"ascii\">\n";
    out << "          ";
    for (int i = 0; i < num_cells; ++i) out << export_cell_ids[i] << " ";
    out << "\n";
    out << "        </DataArray>\n";
    out << "      </CellData>\n";

    out << "      <PointData Scalars=\"global_vid\">\n";
    out << "        <DataArray type=\"Int32\" Name=\"global_vid\" format=\"ascii\">\n";
    out << "          ";
    for (int x : vis_point_to_global_vid) out << x << " ";
    out << "\n";
    out << "        </DataArray>\n";
    out << "      </PointData>\n";

    out << "    </Piece>\n";
    out << "  </UnstructuredGrid>\n";
    out << "</VTKFile>\n";

    out.close();

    std::cout << "✅ VIS VTU file saved to: " << filename << std::endl;
    std::cout << "   duplicated points = " << num_points
              << ", cells = " << num_cells << std::endl;
}

void MiniLcc::save_vis_lcc(std::string filename, const std::vector<int> &level_cvxs) {
    auto get_edge_vertices = [&](int eid) -> std::pair<int,int>
    {
        const auto& e = edges[eid];
        return { e.v1, e.v2 };
    };

    auto reorder_face_edges_to_vertex_loop =
        [&](const std::vector<int>& face_edge_ids) -> std::vector<int>
    {
        std::vector<int> loop;
        if (face_edge_ids.empty()) return loop;

        std::unordered_map<int, std::vector<int>> vertex_neighbors;
        std::set<int> used_vertices;

        for (int eid : face_edge_ids) {
            if (eid < 0 || eid >= (int)edges.size()) {
                throw std::runtime_error("save_vis_lcc: face contains invalid edge id.");
            }

            auto [a, b] = get_edge_vertices(eid);
            if (a < 0 || a >= (int)points.size() || b < 0 || b >= (int)points.size()) {
                throw std::runtime_error("save_vis_lcc: edge contains invalid vertex id.");
            }

            vertex_neighbors[a].push_back(b);
            vertex_neighbors[b].push_back(a);
            used_vertices.insert(a);
            used_vertices.insert(b);
        }

        for (int v : used_vertices) {
            if (vertex_neighbors[v].size() != 2) {
                std::cerr << "[save_vis_lcc] invalid face: vertex degree != 2, vertex = "
                          << v << ", degree = " << vertex_neighbors[v].size() << std::endl;
                throw std::runtime_error("save_vis_lcc: face is not a simple polygon.");
            }
        }

        auto [v0, v1] = get_edge_vertices(face_edge_ids[0]);

        loop.reserve(face_edge_ids.size());
        loop.push_back(v0);

        int prev = v0;
        int curr = v1;

        for (int step = 1; step < (int)face_edge_ids.size(); ++step) {
            loop.push_back(curr);

            const auto& nbrs = vertex_neighbors[curr];
            int next = -1;

            if (nbrs[0] == prev) next = nbrs[1];
            else if (nbrs[1] == prev) next = nbrs[0];
            else {
                throw std::runtime_error("save_vis_lcc: broken face loop while reconstructing polygon.");
            }

            prev = curr;
            curr = next;
        }

        if (curr != v0) {
            std::cerr << "[save_vis_lcc] face loop does not close. expected "
                      << v0 << ", got " << curr << std::endl;
            throw std::runtime_error("save_vis_lcc: reconstructed face loop does not close.");
        }

        return loop;
    };

    // ------------------------------------------------------------
    // 收集要导出的 cell
    // ------------------------------------------------------------
    std::vector<int> export_cell_ids;
    for (int cid : level_cvxs) {
        if (cut_value_inside[cid]) continue;
        export_cell_ids.push_back(cid);
    }

    const int num_cells = (int)export_cell_ids.size();

    // 每个 cell 的每个 face，保存“展开后的局部点编号”
    std::vector<std::vector<std::vector<int>>> cells_faces;
    cells_faces.reserve(num_cells);

    // 导出用点数组：这里不复用点
    std::vector<EK::Point_3> vis_points;
    vis_points.reserve(1024);

    // 可选调试：记录每个导出点对应的原始全局点 id
    std::vector<int> vis_point_to_global_vid;
    vis_point_to_global_vid.reserve(1024);

    for (int cid : export_cell_ids) {
        if (cid < 0 || cid >= (int)cvxs.size()) {
            throw std::runtime_error("save_vis_lcc: invalid cell id.");
        }

        const auto& cell_face_ids = cvxs[cid];
        std::vector<std::vector<int>> one_cell_faces;
        one_cell_faces.reserve(cell_face_ids.size());

        for (int fid : cell_face_ids) {
            if (fid < 0 || fid >= (int)faces.size()) {
                throw std::runtime_error("save_vis_lcc: cell contains invalid face id.");
            }

            const auto& face_edge_ids = faces[fid];
            auto global_vertex_loop = reorder_face_edges_to_vertex_loop(face_edge_ids);

            if (global_vertex_loop.size() < 3) {
                throw std::runtime_error("save_vis_lcc: face has fewer than 3 vertices.");
            }

            // 关键：同一个 face 内的每个顶点 occurrence 都新建一个点 id
            std::vector<int> face_local_ids;
            face_local_ids.reserve(global_vertex_loop.size());

            for (int gvid : global_vertex_loop) {
                if (gvid < 0 || gvid >= (int)points.size()) {
                    throw std::runtime_error("save_vis_lcc: global vertex id out of range.");
                }

                int new_vid = (int)vis_points.size();
                vis_points.push_back(points[gvid]);
                vis_point_to_global_vid.push_back(gvid);
                face_local_ids.push_back(new_vid);
            }

            one_cell_faces.push_back(std::move(face_local_ids));
        }

        cells_faces.push_back(std::move(one_cell_faces));
    }

    const int num_points = (int)vis_points.size();

    // ------------------------------------------------------------
    // 生成 VTK UnstructuredGrid / POLYHEDRON 所需编码
    // ------------------------------------------------------------
    const int VTK_POLYHEDRON = 42;

    std::vector<int> connectivity;
    std::vector<int> offsets;
    std::vector<unsigned char> types(num_cells, static_cast<unsigned char>(VTK_POLYHEDRON));
    std::vector<int> faces_flat;
    std::vector<int> faceoffsets;

    connectivity.reserve(1024);
    offsets.reserve(num_cells);
    faces_flat.reserve(1024);
    faceoffsets.reserve(num_cells);

    int connectivity_offset = 0;
    int face_offset = 0;

    for (const auto& cell_faces : cells_faces) {
        for (const auto& face : cell_faces) {
            for (int vid : face) {
                connectivity.push_back(vid);
            }
            connectivity_offset += (int)face.size();
        }
        offsets.push_back(connectivity_offset);

        faces_flat.push_back((int)cell_faces.size());
        ++face_offset;

        for (const auto& face : cell_faces) {
            faces_flat.push_back((int)face.size());
            ++face_offset;

            for (int vid : face) {
                faces_flat.push_back(vid);
                ++face_offset;
            }
        }
        faceoffsets.push_back(face_offset);
    }

    // ------------------------------------------------------------
    // 写 VTU
    // ------------------------------------------------------------
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "[save_vis_lcc] failed to open file: " << filename << std::endl;
        return;
    }

    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    out << "  <UnstructuredGrid>\n";
    out << "    <Piece NumberOfPoints=\"" << num_points
        << "\" NumberOfCells=\"" << num_cells << "\">\n";

    // Points：完全展开，不复用
    out << "      <Points>\n";
    out << "        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    out << std::setprecision(17);
    for (const auto& p : vis_points) {
        out << "          " << p.x() << " " << p.y() << " " << p.z() << "\n";
    }
    out << "        </DataArray>\n";
    out << "      </Points>\n";

    // Cells
    out << "      <Cells>\n";

    out << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    out << "          ";
    for (int x : connectivity) out << x << " ";
    out << "\n";
    out << "        </DataArray>\n";

    out << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    out << "          ";
    for (int x : offsets) out << x << " ";
    out << "\n";
    out << "        </DataArray>\n";

    out << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    out << "          ";
    for (unsigned int x : types) out << x << " ";
    out << "\n";
    out << "        </DataArray>\n";

    out << "        <DataArray type=\"Int32\" Name=\"faces\" format=\"ascii\">\n";
    out << "          ";
    for (int x : faces_flat) out << x << " ";
    out << "\n";
    out << "        </DataArray>\n";

    out << "        <DataArray type=\"Int32\" Name=\"faceoffsets\" format=\"ascii\">\n";
    out << "          ";
    for (int x : faceoffsets) out << x << " ";
    out << "\n";
    out << "        </DataArray>\n";

    out << "      </Cells>\n";

    // CellData
    out << "      <CellData Scalars=\"cell_id\">\n";
    out << "        <DataArray type=\"Int32\" Name=\"cell_id\" format=\"ascii\">\n";
    out << "          ";
    for (int i = 0; i < num_cells; ++i) out << i << " ";
    out << "\n";
    out << "        </DataArray>\n";
    out << "      </CellData>\n";

    // 可选：写一个 PointData，记录每个展开点来自哪个原始 global vid
    out << "      <PointData Scalars=\"global_vid\">\n";
    out << "        <DataArray type=\"Int32\" Name=\"global_vid\" format=\"ascii\">\n";
    out << "          ";
    for (int x : vis_point_to_global_vid) out << x << " ";
    out << "\n";
    out << "        </DataArray>\n";
    out << "      </PointData>\n";

    out << "    </Piece>\n";
    out << "  </UnstructuredGrid>\n";
    out << "</VTKFile>\n";

    out.close();

    std::cout << "✅ VIS VTU file saved to: " << filename << std::endl;
    std::cout << "   duplicated points = " << num_points << ", cells = " << num_cells << std::endl;
}
//
// void MiniLcc::save_lcc_obj(std::string filename)
// {
// 	EK_to_IK to_ik;
// 	std::vector<IK::Point_3>ik_points;
// 	for (auto&p:points)ik_points.push_back(to_ik(p));
//
//     auto get_edge_vertices = [&](int eid) -> std::pair<int,int>
//     {
//         const auto& e = edges[eid];
//         return { e.v1, e.v2 };
//     };
//
//     auto reorder_face_edges_to_vertex_loop =
//         [&](const std::vector<int>& face_edge_ids) -> std::vector<int>
//     {
//         std::vector<int> loop;
//         if (face_edge_ids.empty()) return loop;
//
//         std::unordered_map<int, std::vector<int>> vertex_neighbors;
//         std::set<int> used_vertices;
//
//         for (int eid : face_edge_ids) {
//             if (eid < 0 || eid >= (int)edges.size()) {
//                 throw std::runtime_error("save_lcc_obj: face contains invalid edge id.");
//             }
//
//             auto [a, b] = get_edge_vertices(eid);
//             if (a < 0 || a >= (int)ik_points.size() || b < 0 || b >= (int)ik_points.size()) {
//                 throw std::runtime_error("save_lcc_obj: edge contains invalid vertex id.");
//             }
//
//             vertex_neighbors[a].push_back(b);
//             vertex_neighbors[b].push_back(a);
//             used_vertices.insert(a);
//             used_vertices.insert(b);
//         }
//
//         for (int v : used_vertices) {
//             if (vertex_neighbors[v].size() != 2) {
//                 std::cerr << "[save_lcc_obj] invalid face: vertex degree != 2, vertex = "
//                           << v << ", degree = " << vertex_neighbors[v].size() << std::endl;
//                 throw std::runtime_error("save_lcc_obj: face is not a simple polygon.");
//             }
//         }
//
//         auto [v0, v1] = get_edge_vertices(face_edge_ids[0]);
//
//         loop.reserve(face_edge_ids.size());
//         loop.push_back(v0);
//
//         int prev = v0;
//         int curr = v1;
//
//         for (int step = 1; step < (int)face_edge_ids.size(); ++step) {
//             loop.push_back(curr);
//
//             const auto& nbrs = vertex_neighbors[curr];
//             int next = -1;
//
//             if (nbrs[0] == prev) next = nbrs[1];
//             else if (nbrs[1] == prev) next = nbrs[0];
//             else {
//                 throw std::runtime_error("save_lcc_obj: broken face loop while reconstructing polygon.");
//             }
//
//             prev = curr;
//             curr = next;
//         }
//
//         if (curr != v0) {
//             std::cerr << "[save_lcc_obj] face loop does not close. expected "
//                       << v0 << ", got " << curr << std::endl;
//             throw std::runtime_error("save_lcc_obj: reconstructed face loop does not close.");
//         }
//
//         return loop;
//     };
//
//     auto point_to_vec = [&](const IK::Point_3& p) -> Eigen::Vector3d
//     {
//         return Eigen::Vector3d(p.x(), p.y(), p.z());
//     };
//
//     auto compute_polygon_centroid =
//         [&](const std::vector<int>& vids) -> Eigen::Vector3d
//     {
//         Eigen::Vector3d c(0.0, 0.0, 0.0);
//         for (int vid : vids) {
//             c += point_to_vec(ik_points[vid]);
//         }
//         c /= (Real)vids.size();
//         return c;
//     };
//
//     // 用 Newell 方法求多边形法向，适合一般 3D polygon
//     auto compute_polygon_normal =
//         [&](const std::vector<int>& vids) -> Eigen::Vector3d
//     {
//         Eigen::Vector3d n(0.0, 0.0, 0.0);
//         const int m = (int)vids.size();
//         for (int i = 0; i < m; ++i) {
//             const auto& p0 = ik_points[vids[i]];
//             const auto& p1 = ik_points[vids[(i + 1) % m]];
//             n[0] += (p0.y() - p1.y()) * (p0.z() + p1.z());
//             n[1] += (p0.z() - p1.z()) * (p0.x() + p1.x());
//             n[2] += (p0.x() - p1.x()) * (p0.y() + p1.y());
//         }
//         return n;
//     };
//
//     auto compute_cell_centroid =
//         [&](int cid) -> Eigen::Vector3d
//     {
//         std::set<int> unique_vids;
//         for (int fid : cvxs[cid]) {
//             if (fid < 0 || fid >= (int)faces.size()) {
//                 throw std::runtime_error("save_lcc_obj: invalid face id in cell.");
//             }
//             for (int eid : faces[fid]) {
//                 auto [a, b] = get_edge_vertices(eid);
//                 unique_vids.insert(a);
//                 unique_vids.insert(b);
//             }
//         }
//
//         if (unique_vids.empty()) {
//             throw std::runtime_error("save_lcc_obj: empty cell.");
//         }
//
//         Eigen::Vector3d c(0.0, 0.0, 0.0);
//         for (int vid : unique_vids) {
//             c += point_to_vec(ik_points[vid]);
//         }
//         c /= (Real)unique_vids.size();
//         return c;
//     };
//
//     // ------------------------------------------------------------
//     // 自动收集导出的 cell
//     // 和 save_lcc 一样，跳过 partitioned 和 inside
//     // 你如果实际 inside 标记不是 cvx_is_inside，就改成自己的变量
//     // ------------------------------------------------------------
//     std::vector<int> export_cell_ids;
//     export_cell_ids.reserve(cvxs.size());
//
//     for (int cid = 0; cid < (int)cvxs.size(); ++cid) {
//         bool skip = false;
//
//         if (cid < (int)cvx_partitioned.size() && cvx_partitioned[cid]) {
//             skip = true;
//         }
//         if (cid < (int)cvx_is_inside.size() && cvx_is_inside[cid]) {
//             skip = true;
//         }
//         // 如果你实际用的是 cut_value_inside，就换成：
//         // if (cid < (int)cut_value_inside.size() && cut_value_inside[cid]) skip = true;
//
//         if (!skip) {
//             export_cell_ids.push_back(cid);
//         }
//     }
//
//     std::ofstream out(filename);
//     if (!out.is_open()) {
//         std::cerr << "[save_lcc_obj] failed to open file: " << filename << std::endl;
//         return;
//     }
//
//     out << "# save_lcc_obj generated by MiniLcc\n";
//
//     // OBJ 顶点编号从 1 开始
//     int obj_vid = 1;
//     int exported_face_cnt = 0;
//     int exported_cell_cnt = 0;
//
//     for (int cid : export_cell_ids) {
//         if (cid < 0 || cid >= (int)cvxs.size()) {
//             throw std::runtime_error("save_lcc_obj: invalid cell id.");
//         }
//
//         const Eigen::Vector3d cell_centroid = compute_cell_centroid(cid);
//
//         out << "\n";
//         out << "o cell_" << cid << "\n";
//
//         for (int fid : cvxs[cid]) {
//             if (fid < 0 || fid >= (int)faces.size()) {
//                 throw std::runtime_error("save_lcc_obj: cell contains invalid face id.");
//             }
//
//             std::vector<int> loop = reorder_face_edges_to_vertex_loop(faces[fid]);
//             if ((int)loop.size() < 3) {
//                 throw std::runtime_error("save_lcc_obj: face has fewer than 3 vertices.");
//             }
//
//             Eigen::Vector3d face_centroid = compute_polygon_centroid(loop);
//             Eigen::Vector3d face_normal   = compute_polygon_normal(loop);
//
//             // 希望法向朝外：
//             // outward direction ~ face_centroid - cell_centroid
//             Eigen::Vector3d outward_dir = face_centroid - cell_centroid;
//
//             // 若当前法向与 outward_dir 点积 < 0，说明朝内，需要翻转
//             if (face_normal.dot(outward_dir) < 0.0) {
//                 std::reverse(loop.begin(), loop.end());
//             }
//
//             // 每个 face occurrence 都重新写点，不复用
//             std::vector<int> face_obj_ids;
//             face_obj_ids.reserve(loop.size());
//
//             for (int gvid : loop) {
//                 if (gvid < 0 || gvid >= (int)points.size()) {
//                     throw std::runtime_error("save_lcc_obj: global vertex id out of range.");
//                 }
//
//                 const auto& p = points[gvid];
//                 out << "v " << std::setprecision(17)
//                     << p.x() << " " << p.y() << " " << p.z() << "\n";
//
//                 face_obj_ids.push_back(obj_vid);
//                 ++obj_vid;
//             }
//
//             out << "f";
//             for (int id : face_obj_ids) {
//                 out << " " << id;
//             }
//             out << "\n";
//
//             ++exported_face_cnt;
//         }
//
//         ++exported_cell_cnt;
//     }
//
//     out.close();
//
//     std::cout << "✅ OBJ file saved to: " << filename << std::endl;
//     std::cout << "   cells = " << exported_cell_cnt
//               << ", faces = " << exported_face_cnt
//               << ", duplicated points = " << (obj_vid - 1) << std::endl;
// }

// void MiniLcc::save_lcc_obj(std::string filename)
// {
// 	EK_to_IK to_ik;
// 	std::vector<IK::Point_3> ik_points;
// 	for (auto &p : points) ik_points.push_back(to_ik(p));
//
//     auto get_edge_vertices = [&](int eid) -> std::pair<int,int>
//     {
//         const auto& e = edges[eid];
//         return { e.v1, e.v2 };
//     };
//
//     auto reorder_face_edges_to_vertex_loop =
//         [&](const std::vector<int>& face_edge_ids) -> std::vector<int>
//     {
//         std::vector<int> loop;
//         if (face_edge_ids.empty()) return loop;
//
//         std::unordered_map<int, std::vector<int>> vertex_neighbors;
//         std::set<int> used_vertices;
//
//         for (int eid : face_edge_ids) {
//             if (eid < 0 || eid >= (int)edges.size()) {
//                 throw std::runtime_error("save_lcc_obj: face contains invalid edge id.");
//             }
//
//             auto [a, b] = get_edge_vertices(eid);
//             if (a < 0 || a >= (int)ik_points.size() || b < 0 || b >= (int)ik_points.size()) {
//                 throw std::runtime_error("save_lcc_obj: edge contains invalid vertex id.");
//             }
//
//             vertex_neighbors[a].push_back(b);
//             vertex_neighbors[b].push_back(a);
//             used_vertices.insert(a);
//             used_vertices.insert(b);
//         }
//
//         for (int v : used_vertices) {
//             if (vertex_neighbors[v].size() != 2) {
//                 std::cerr << "[save_lcc_obj] invalid face: vertex degree != 2, vertex = "
//                           << v << ", degree = " << vertex_neighbors[v].size() << std::endl;
//                 throw std::runtime_error("save_lcc_obj: face is not a simple polygon.");
//             }
//         }
//
//         auto [v0, v1] = get_edge_vertices(face_edge_ids[0]);
//
//         loop.reserve(face_edge_ids.size());
//         loop.push_back(v0);
//
//         int prev = v0;
//         int curr = v1;
//
//         for (int step = 1; step < (int)face_edge_ids.size(); ++step) {
//             loop.push_back(curr);
//
//             const auto& nbrs = vertex_neighbors[curr];
//             int next = -1;
//
//             if (nbrs[0] == prev) next = nbrs[1];
//             else if (nbrs[1] == prev) next = nbrs[0];
//             else {
//                 throw std::runtime_error("save_lcc_obj: broken face loop while reconstructing polygon.");
//             }
//
//             prev = curr;
//             curr = next;
//         }
//
//         if (curr != v0) {
//             std::cerr << "[save_lcc_obj] face loop does not close. expected "
//                       << v0 << ", got " << curr << std::endl;
//             throw std::runtime_error("save_lcc_obj: reconstructed face loop does not close.");
//         }
//
//         return loop;
//     };
//
//     auto point_to_vec = [&](const IK::Point_3& p) -> Eigen::Vector3d
//     {
//         return Eigen::Vector3d(p.x(), p.y(), p.z());
//     };
//
//     auto compute_polygon_centroid =
//         [&](const std::vector<int>& vids) -> Eigen::Vector3d
//     {
//         Eigen::Vector3d c(0.0, 0.0, 0.0);
//         for (int vid : vids) {
//             c += point_to_vec(ik_points[vid]);
//         }
//         c /= (Real)vids.size();
//         return c;
//     };
//
//     auto compute_polygon_normal =
//         [&](const std::vector<int>& vids) -> Eigen::Vector3d
//     {
//         Eigen::Vector3d n(0.0, 0.0, 0.0);
//         const int m = (int)vids.size();
//         for (int i = 0; i < m; ++i) {
//             const auto& p0 = ik_points[vids[i]];
//             const auto& p1 = ik_points[vids[(i + 1) % m]];
//             n[0] += (p0.y() - p1.y()) * (p0.z() + p1.z());
//             n[1] += (p0.z() - p1.z()) * (p0.x() + p1.x());
//             n[2] += (p0.x() - p1.x()) * (p0.y() + p1.y());
//         }
//         return n;
//     };
//
//     auto compute_cell_centroid =
//         [&](int cid) -> Eigen::Vector3d
//     {
//         std::set<int> unique_vids;
//         for (int fid : cvxs[cid]) {
//             if (fid < 0 || fid >= (int)faces.size()) {
//                 throw std::runtime_error("save_lcc_obj: invalid face id in cell.");
//             }
//             for (int eid : faces[fid]) {
//                 auto [a, b] = get_edge_vertices(eid);
//                 unique_vids.insert(a);
//                 unique_vids.insert(b);
//             }
//         }
//
//         if (unique_vids.empty()) {
//             throw std::runtime_error("save_lcc_obj: empty cell.");
//         }
//
//         Eigen::Vector3d c(0.0, 0.0, 0.0);
//         for (int vid : unique_vids) {
//             c += point_to_vec(ik_points[vid]);
//         }
//         c /= (Real)unique_vids.size();
//         return c;
//     };
//
//     std::vector<int> export_cell_ids;
//     export_cell_ids.reserve(cvxs.size());
//
//     for (int cid = 0; cid < (int)cvxs.size(); ++cid) {
//         bool skip = false;
//
//         if (cid < (int)cvx_partitioned.size() && cvx_partitioned[cid]) {
//             skip = true;
//         }
//         if (cid < (int)cvx_is_inside.size() && cvx_is_inside[cid]) {
//             skip = true;
//         }
//
//         if (!skip) {
//             export_cell_ids.push_back(cid);
//         }
//     }
//
//     std::ofstream out(filename);
//     if (!out.is_open()) {
//         std::cerr << "[save_lcc_obj] failed to open file: " << filename << std::endl;
//         return;
//     }
//
//     out << "# save_lcc_obj generated by MiniLcc\n";
//
//     int obj_vid = 1;
//     int exported_face_cnt = 0;
//     int exported_cell_cnt = 0;
//
//     for (int cid : export_cell_ids) {
//         if (cid < 0 || cid >= (int)cvxs.size()) {
//             throw std::runtime_error("save_lcc_obj: invalid cell id.");
//         }
//
//         const Eigen::Vector3d cell_centroid = compute_cell_centroid(cid);
//
//         out << "\n";
//         out << "o cell_" << cid << "\n";
//
//         // 关键改动：
//         // 同一个 cell 内复用顶点；不同 cell 之间不复用
//         std::unordered_map<int, int> cell_global_to_obj_vid;
//
//         for (int fid : cvxs[cid]) {
//             if (fid < 0 || fid >= (int)faces.size()) {
//                 throw std::runtime_error("save_lcc_obj: cell contains invalid face id.");
//             }
//
//             std::vector<int> loop = reorder_face_edges_to_vertex_loop(faces[fid]);
//             if ((int)loop.size() < 3) {
//                 throw std::runtime_error("save_lcc_obj: face has fewer than 3 vertices.");
//             }
//
//             Eigen::Vector3d face_centroid = compute_polygon_centroid(loop);
//             Eigen::Vector3d face_normal   = compute_polygon_normal(loop);
//             Eigen::Vector3d outward_dir   = face_centroid - cell_centroid;
//
//             if (face_normal.dot(outward_dir) < 0.0) {
//                 std::reverse(loop.begin(), loop.end());
//             }
//
//             std::vector<int> face_obj_ids;
//             face_obj_ids.reserve(loop.size());
//
//             for (int gvid : loop) {
//                 if (gvid < 0 || gvid >= (int)points.size()) {
//                     throw std::runtime_error("save_lcc_obj: global vertex id out of range.");
//                 }
//
//                 auto it = cell_global_to_obj_vid.find(gvid);
//                 if (it == cell_global_to_obj_vid.end()) {
//                     const auto& p = points[gvid];
//                     out << "v " << std::setprecision(17)
//                         << p.x() << " " << p.y() << " " << p.z() << "\n";
//
//                     cell_global_to_obj_vid[gvid] = obj_vid;
//                     face_obj_ids.push_back(obj_vid);
//                     ++obj_vid;
//                 } else {
//                     face_obj_ids.push_back(it->second);
//                 }
//             }
//
//             out << "f";
//             for (int id : face_obj_ids) {
//                 out << " " << id;
//             }
//             out << "\n";
//
//             ++exported_face_cnt;
//         }
//
//         ++exported_cell_cnt;
//     }
//
//     out.close();
//
//     std::cout << "✅ OBJ file saved to: " << filename << std::endl;
//     std::cout << "   cells = " << exported_cell_cnt
//               << ", faces = " << exported_face_cnt
//               << ", vertices = " << (obj_vid - 1) << std::endl;
// }

void MiniLcc::save_lcc_obj(std::string outside_filename, std::string inside_filename)
{
	EK_to_IK to_ik;
	std::vector<IK::Point_3> ik_points;
	for (auto &p : points) ik_points.push_back(to_ik(p));

    auto get_edge_vertices = [&](int eid) -> std::pair<int,int>
    {
        const auto& e = edges[eid];
        return { e.v1, e.v2 };
    };

    auto reorder_face_edges_to_vertex_loop =
        [&](const std::vector<int>& face_edge_ids) -> std::vector<int>
    {
        std::vector<int> loop;
        if (face_edge_ids.empty()) return loop;

        std::unordered_map<int, std::vector<int>> vertex_neighbors;
        std::set<int> used_vertices;

        for (int eid : face_edge_ids) {
            if (eid < 0 || eid >= (int)edges.size()) {
                throw std::runtime_error("save_lcc_obj: face contains invalid edge id.");
            }

            auto [a, b] = get_edge_vertices(eid);
            if (a < 0 || a >= (int)ik_points.size() || b < 0 || b >= (int)ik_points.size()) {
                throw std::runtime_error("save_lcc_obj: edge contains invalid vertex id.");
            }

            vertex_neighbors[a].push_back(b);
            vertex_neighbors[b].push_back(a);
            used_vertices.insert(a);
            used_vertices.insert(b);
        }

        for (int v : used_vertices) {
            if (vertex_neighbors[v].size() != 2) {
                std::cerr << "[save_lcc_obj] invalid face: vertex degree != 2, vertex = "
                          << v << ", degree = " << vertex_neighbors[v].size() << std::endl;
                throw std::runtime_error("save_lcc_obj: face is not a simple polygon.");
            }
        }

        auto [v0, v1] = get_edge_vertices(face_edge_ids[0]);

        loop.reserve(face_edge_ids.size());
        loop.push_back(v0);

        int prev = v0;
        int curr = v1;

        for (int step = 1; step < (int)face_edge_ids.size(); ++step) {
            loop.push_back(curr);

            const auto& nbrs = vertex_neighbors[curr];
            int next = -1;

            if (nbrs[0] == prev) next = nbrs[1];
            else if (nbrs[1] == prev) next = nbrs[0];
            else {
                throw std::runtime_error("save_lcc_obj: broken face loop while reconstructing polygon.");
            }

            prev = curr;
            curr = next;
        }

        if (curr != v0) {
            std::cerr << "[save_lcc_obj] face loop does not close. expected "
                      << v0 << ", got " << curr << std::endl;
            throw std::runtime_error("save_lcc_obj: reconstructed face loop does not close.");
        }

        return loop;
    };

    auto point_to_vec = [&](const IK::Point_3& p) -> Eigen::Vector3d
    {
        return Eigen::Vector3d(p.x(), p.y(), p.z());
    };

    auto compute_polygon_centroid =
        [&](const std::vector<int>& vids) -> Eigen::Vector3d
    {
        Eigen::Vector3d c(0.0, 0.0, 0.0);
        for (int vid : vids) {
            c += point_to_vec(ik_points[vid]);
        }
        c /= (Real)vids.size();
        return c;
    };

    auto compute_polygon_normal =
        [&](const std::vector<int>& vids) -> Eigen::Vector3d
    {
        Eigen::Vector3d n(0.0, 0.0, 0.0);
        const int m = (int)vids.size();
        for (int i = 0; i < m; ++i) {
            const auto& p0 = ik_points[vids[i]];
            const auto& p1 = ik_points[vids[(i + 1) % m]];
            n[0] += (p0.y() - p1.y()) * (p0.z() + p1.z());
            n[1] += (p0.z() - p1.z()) * (p0.x() + p1.x());
            n[2] += (p0.x() - p1.x()) * (p0.y() + p1.y());
        }
        return n;
    };

    auto compute_cell_centroid =
        [&](int cid) -> Eigen::Vector3d
    {
        std::set<int> unique_vids;
        for (int fid : cvxs[cid]) {
            if (fid < 0 || fid >= (int)faces.size()) {
                throw std::runtime_error("save_lcc_obj: invalid face id in cell.");
            }
            for (int eid : faces[fid]) {
                auto [a, b] = get_edge_vertices(eid);
                unique_vids.insert(a);
                unique_vids.insert(b);
            }
        }

        if (unique_vids.empty()) {
            throw std::runtime_error("save_lcc_obj: empty cell.");
        }

        Eigen::Vector3d c(0.0, 0.0, 0.0);
        for (int vid : unique_vids) {
            c += point_to_vec(ik_points[vid]);
        }
        c /= (Real)unique_vids.size();
        return c;
    };

    auto write_cells_to_obj =
        [&](const std::string& filename,
            const std::vector<int>& cell_ids,
            bool flip_to_outside)
    {
        std::ofstream out(filename);
        if (!out.is_open()) {
            std::cerr << "[save_lcc_obj] failed to open file: " << filename << std::endl;
            return;
        }

        out << "# save_lcc_obj generated by MiniLcc\n";

        int obj_vid = 1;
        int exported_face_cnt = 0;
        int exported_cell_cnt = 0;

        for (int cid : cell_ids) {
            if (cid < 0 || cid >= (int)cvxs.size()) {
                throw std::runtime_error("save_lcc_obj: invalid cell id.");
            }

            const Eigen::Vector3d cell_centroid = compute_cell_centroid(cid);

            out << "\n";
            out << "o cell_" << cid << "\n";

            // 同一个 cell 内复用顶点；不同 cell 不复用
            std::unordered_map<int, int> cell_global_to_obj_vid;

            for (int fid : cvxs[cid]) {
                if (fid < 0 || fid >= (int)faces.size()) {
                    throw std::runtime_error("save_lcc_obj: cell contains invalid face id.");
                }

                std::vector<int> loop = reorder_face_edges_to_vertex_loop(faces[fid]);
                if ((int)loop.size() < 3) {
                    throw std::runtime_error("save_lcc_obj: face has fewer than 3 vertices.");
                }

                Eigen::Vector3d face_centroid = compute_polygon_centroid(loop);
                Eigen::Vector3d face_normal   = compute_polygon_normal(loop);
                Eigen::Vector3d outward_dir   = face_centroid - cell_centroid;

                // outside 文件：法向朝 cell 外
                // inside 文件：法向朝 cell 内（即和 outside 相反）
                Real s = face_normal.dot(outward_dir);

                if (flip_to_outside) {
                    if (s < 0.0) std::reverse(loop.begin(), loop.end());
                } else {
                    if (s > 0.0) std::reverse(loop.begin(), loop.end());
                }

                std::vector<int> face_obj_ids;
                face_obj_ids.reserve(loop.size());

                for (int gvid : loop) {
                    if (gvid < 0 || gvid >= (int)points.size()) {
                        throw std::runtime_error("save_lcc_obj: global vertex id out of range.");
                    }

                    auto it = cell_global_to_obj_vid.find(gvid);
                    if (it == cell_global_to_obj_vid.end()) {
                        const auto& p = points[gvid];
                        out << "v " << std::setprecision(17)
                            << p.x() << " " << p.y() << " " << p.z() << "\n";

                        cell_global_to_obj_vid[gvid] = obj_vid;
                        face_obj_ids.push_back(obj_vid);
                        ++obj_vid;
                    } else {
                        face_obj_ids.push_back(it->second);
                    }
                }

                out << "f";
                for (int id : face_obj_ids) {
                    out << " " << id;
                }
                out << "\n";

                ++exported_face_cnt;
            }

            ++exported_cell_cnt;
        }

        out.close();

        std::cout << "✅ OBJ file saved to: " << filename << std::endl;
        std::cout << "   cells = " << exported_cell_cnt
                  << ", faces = " << exported_face_cnt
                  << ", vertices = " << (obj_vid - 1) << std::endl;
    };

    std::vector<int> outside_cell_ids;
    std::vector<int> inside_cell_ids;
    outside_cell_ids.reserve(cvxs.size());
    inside_cell_ids.reserve(cvxs.size());

    for (int cid = 0; cid < (int)cvxs.size(); ++cid) {
        if (cid < (int)cvx_partitioned.size() && cvx_partitioned[cid]) {
            continue;
        }

        bool is_inside = (cid < (int)cvx_is_inside.size() && cvx_is_inside[cid]);
        if (is_inside) inside_cell_ids.push_back(cid);
        else outside_cell_ids.push_back(cid);
    }

    // outside: 面朝 cell 外
    write_cells_to_obj(outside_filename, outside_cell_ids, true);

    // inside: 面朝 cell 内（和 outside 反向）
    write_cells_to_obj(inside_filename, inside_cell_ids, false);
}

void MiniLcc::calculate_cut_values() {
	cvx_cut_values.assign(cvxs.size(),0.0);
	cut_value_inside.assign(cvxs.size(),true);
	// cvx_is_inside.assign(cvxs.size(),false);
	// return;
	for (int cid=0;cid<cvxs.size();++cid) {
		if (!cvx_partitioned[cid])
			cut_value_inside[cid] = cvx_is_inside[cid];
	}

	for (int cid=0;cid<cvxs.size();++cid) {
		if (cvx_partitioned[cid])continue;
		int f = cvx_father[cid];
		if (cut_value_inside[cid] == true)continue;
		while (f!=-1) {
			cut_value_inside[f] = false;
			f = cvx_father[f];
		}
	}
	//
	for (int cid=0;cid<cvxs.size();++cid) {
		auto sons = cvx_sons[cid];
		if (sons.first==-1)continue;
		if (cut_value_inside[cid] == cut_value_inside[sons.first] &&
			cut_value_inside[cid] == cut_value_inside[sons.second])
			continue;


		if (cut_value_inside[cid] != cut_value_inside[sons.first]) {
			cvx_cut_values[cid] = cvx_volume_fast(cvxs[sons.first]);
		}
		else
			cvx_cut_values[cid] = cvx_volume_fast(cvxs[sons.second]);
	}
	cvx_is_inside.clear();
	for (auto&inside:cut_value_inside)cvx_is_inside.push_back(inside);

}

void MiniLcc::calculate_cut_values_v2() {
    FUNC_MEMORY_TIMER();
	const int n = static_cast<int>(cvxs.size());
	cvx_cut_values.assign(n,0.0);
	cut_value_inside.assign(n,true);

	std::vector<Real> volumes(n,0.0);
	for (int cid=0;cid<n;++cid) {
		volumes[cid] = cvx_volume_fast(cvxs[cid]);
	}

	const Real thod = 1e-6;
	std::vector<Real> outside_leaf_volumes(n,0.0);
	std::vector<char> inside_flags(n,1);

	for (int cid=n-1;cid>=0;--cid) {
		const auto sons = cvx_sons[cid];
		if (sons.first==-1) {
			const bool leaf_inside = cid < static_cast<int>(cvx_is_inside.size()) && cvx_is_inside[cid];
			inside_flags[cid] = leaf_inside ? 1 : 0;
			outside_leaf_volumes[cid] = leaf_inside ? 0.0 : volumes[cid];
			continue;
		}

		outside_leaf_volumes[cid] = outside_leaf_volumes[sons.first] + outside_leaf_volumes[sons.second];
			const Real ratio = outside_leaf_volumes[cid] / std::max<Real>(volumes[cid], 1e-300);
		inside_flags[cid] = ratio > thod ? 0 : 1;
	}

	for (int cid=0;cid<n;++cid) {
		cut_value_inside[cid] = inside_flags[cid] != 0;
		const auto sons = cvx_sons[cid];
		if (sons.first==-1)continue;
		if (inside_flags[cid] == inside_flags[sons.first] &&
			inside_flags[cid] == inside_flags[sons.second])
			continue;

		if (inside_flags[cid] != inside_flags[sons.first])
			cvx_cut_values[cid] = volumes[sons.first];
		else
			cvx_cut_values[cid] = volumes[sons.second];
	}

	cvx_is_inside.clear();
	for (int cid=0;cid<n;++cid)cvx_is_inside.push_back(inside_flags[cid] != 0);
}

void MiniLcc::cvxs_to_finest() {
	FUNC_TIMER();
	int ecnt=0,fcnt=0;
	for (int cid=0;cid<cvxs.size();++cid) {
		if (cvx_partitioned[cid])continue;
		fcnt+=cvxs[cid].size();
		for (auto&fid:cvxs[cid])
			ecnt+=faces[fid].size();
	}
	RZ_INFO(ecnt," ",fcnt);


	std::vector<std::vector<int>> cvx_leaves;
	cvx_leaves.resize(cvxs.size(),{});
	for (int cid=0;cid<cvxs.size();++cid) {
		if (cvx_partitioned[cid])continue;
		int f = cvx_father[cid];
		while (f!=-1) {
			cvx_leaves[f].push_back(cid);
			f = cvx_father[f];
		}
	}
	for (int cid=0;cid<cvxs.size();++cid) {
		if (!cvx_partitioned[cid])continue;
		cvxs[cid].clear();
		std::vector<int> fs;
		for (auto&leaf_cid:cvx_leaves[cid]) {
			for (auto&fid:cvxs[leaf_cid])
				fs.push_back(fid);
		}

		// 排序
		std::sort(fs.begin(), fs.end());

		// 结果
		std::vector<int> fs_unique_once;
		fs_unique_once.reserve(fs.size());

		for (int i = 0; i < (int)fs.size(); ) {
			int j = i + 1;
			while (j < (int)fs.size() && fs[j] == fs[i]) ++j;

			int count = j - i;

			// 断言：最多出现两次
			RZ_ASSERT(count <= 2, "face appears more than twice");

			// 只保留出现一次的
			if (count == 1) {
				fs_unique_once.push_back(fs[i]);
			}

			i = j;
		}
		cvxs[cid].swap(fs_unique_once);
	}


}

void MiniLcc::copy_finest() {
	cvxs_copy = cvxs;
	faces_copy = faces;
}

void MiniLcc::cvxs_to_finest_v2() {
	FUNC_TIMER();
	cvxs = cvxs_copy;
	faces = faces_copy;

	std::vector<std::vector<int>> cvx_leaves;
	cvx_leaves.resize(cvxs.size(),{});
	for (int cid=0;cid<cvxs.size();++cid) {
		if (cvx_partitioned[cid])continue;
		int f = cvx_father[cid];
		while (f!=-1) {
			cvx_leaves[f].push_back(cid);
			f = cvx_father[f];
		}
	}
	for (int cid=0;cid<cvxs.size();++cid) {
		if (!cvx_partitioned[cid])continue;
		cvxs[cid].clear();
		std::vector<int> fs;
		for (auto&leaf_cid:cvx_leaves[cid]) {
			for (auto&fid:cvxs[leaf_cid])
				fs.push_back(fid);
		}

		// 排序
		std::sort(fs.begin(), fs.end());

		// 结果
		std::vector<int> fs_unique_once;
		fs_unique_once.reserve(fs.size());

		for (int i = 0; i < (int)fs.size(); ) {
			int j = i + 1;
			while (j < (int)fs.size() && fs[j] == fs[i]) ++j;

			int count = j - i;

			// 断言：最多出现两次
			RZ_ASSERT(count <= 2, "face appears more than twice");

			// 只保留出现一次的
			if (count == 1) {
				fs_unique_once.push_back(fs[i]);
			}

			i = j;
		}
		cvxs[cid].swap(fs_unique_once);
	}
}

void MiniLcc::lod_repair_v2(std::vector<int> &level_cvxs) {
	FUNC_TIMER();
	std::vector<int> level_cvxs_outside;
	for (auto&cid:level_cvxs)if (!cvx_is_inside[cid])level_cvxs_outside.push_back(cid);
	level_cvxs.swap(level_cvxs_outside);
	std::vector<std::vector<int>> face_to_cvxs(faces.size());
	std::vector<std::vector<int>> edge_to_faces(edges.size());
	std::vector<char> active_edges(edges.size(), false);

	auto push_unique = [](std::vector<int>& dst, int value) {
		if (std::find(dst.begin(), dst.end(), value) == dst.end())
			dst.push_back(value);
	};

	auto erase_value = [](std::vector<int>& dst, int value) {
		dst.erase(std::remove(dst.begin(), dst.end(), value), dst.end());
	};

	auto check_faces_vec = [&](const std::vector<int>& fids) {
		if (fids.empty()) return true;
		if (fids.size() != 2) return false;
		std::vector<int> vids;
		vids.reserve(fids.size() * 8);
		for (int fid : fids) {
			RZ_ASSERT(fid >= 0 && fid < (int)faces.size(), "invalid fid");
			for (int eid : faces[fid]) {
				RZ_ASSERT(eid >= 0 && eid < (int)edges.size(), "invalid eid");
				vids.push_back(edges[eid].v1);
				vids.push_back(edges[eid].v2);
			}
		}
		if (vids.empty()) return true;
		std::sort(vids.begin(), vids.end());
		vids.erase(std::unique(vids.begin(), vids.end()), vids.end());
		if (vids.size() < 3) return true;

		const EK::Point_3& p0 = points[vids[0]];
		int i1 = -1;
		for (int i = 1; i < (int)vids.size(); ++i) {
			if (points[vids[i]] != p0) {
				i1 = i;
				break;
			}
		}
		if (i1 == -1) return true;

		const EK::Point_3& p1 = points[vids[i1]];
		EK::Line_3 line(p0, p1);
		EK::Plane_3 plane;
		bool found_plane = false;
		for (int i = i1 + 1; i < (int)vids.size(); ++i) {
			const EK::Point_3& p = points[vids[i]];
			if (!line.has_on(p)) {
				plane = EK::Plane_3(p0, p1, p);
				found_plane = true;
				break;
			}
		}
		if (!found_plane) return true;
		for (int vid : vids) {
			if (!plane.has_on(points[vid])) return false;
		}
		return true;
	};

	auto merge_face_vec = [&](const std::vector<int>& fids) {
		RZ_ASSERT(fids.size() == 2, fids.size());
		std::vector<int> eids;
		for (int fid : fids) {
			RZ_ASSERT(fid >= 0 && fid < (int)faces.size(), "invalid fid");
			for (int eid : faces[fid]) {
				RZ_ASSERT(eid >= 0 && eid < (int)edges.size(), "invalid eid");
				eids.push_back(eid);
			}
		}
		if (eids.empty()) return std::vector<int>{};
		std::sort(eids.begin(), eids.end());

		std::vector<int> merged_face;
		merged_face.reserve(eids.size());
		for (int i = 0; i < (int)eids.size(); ) {
			int j = i + 1;
			while (j < (int)eids.size() && eids[j] == eids[i]) ++j;
			int cnt = j - i;
			RZ_ASSERT(cnt <= 2, "edge appears more than twice");
			if (cnt == 1) {
				merged_face.push_back(eids[i]);
			} else {
				active_edges[eids[i]] = false;
			}
			i = j;
		}
		return merged_face;
	};

	for (const auto& cid:level_cvxs)
	{
		for (auto&fid:cvxs[cid]) {
			push_unique(face_to_cvxs[fid], cid);
			for (auto&eid:faces[fid]) {
				active_edges[eid] = true;
				push_unique(edge_to_faces[eid], fid);
			}
		}
	}

	// merge faces ~ remove edges ~ new faces ~ edge_to_faces ~ vertex_to_edges（后面统一更新）;

	std::deque<int> face_candidates;
	std::vector<char> face_candidate_dead(edge_to_faces.size(), false);
	for (int eid = 0; eid < (int)edge_to_faces.size(); ++eid) {
		if (active_edges[eid]) face_candidates.push_back(eid);
	}

	while (!face_candidates.empty()) {
		const int candidate_eid = face_candidates.front();
		face_candidates.pop_front();
		if (candidate_eid < 0 || candidate_eid >= (int)edge_to_faces.size()) continue;
		if (!active_edges[candidate_eid] || face_candidate_dead[candidate_eid]) continue;

		const auto old_fids = edge_to_faces[candidate_eid];
		if (!check_faces_vec(old_fids)) {
			face_candidate_dead[candidate_eid] = true;
			continue;
		}

		auto new_face = merge_face_vec(old_fids);
		const int nfid = faces.size();
		face_to_cvxs.emplace_back();
		for (auto&old_fid:old_fids) {
			for (auto&cid:face_to_cvxs[old_fid])
				push_unique(face_to_cvxs[nfid], cid);
		}
		faces.push_back(new_face);
		for (int eid : new_face) {
			auto& inc_faces = edge_to_faces[eid];
			for (int ofid : old_fids) {
				erase_value(inc_faces, ofid);
			}
			push_unique(inc_faces, nfid);
			face_candidates.push_back(eid);
		}

		std::unordered_map<int,int>cvx_counts;
		for (auto&old_fid:old_fids) {
			for (auto& cid:face_to_cvxs[old_fid])
				cvx_counts[cid]++;
		}
		std::vector<int> todo_cids;
		for (auto& cvx_count:cvx_counts)if (cvx_count.second>=2) {
			todo_cids.push_back(cvx_count.first);
		}else if (cvx_count.second==1)RZ_FATAL("cvx_count.second==1");
		RZ_ASSERT(todo_cids.size()<=2,todo_cids.size());
		for (const auto& cid : todo_cids) {
			auto& cell_faces = cvxs[cid];
			bool touched = false;

			std::vector<int> new_cell_faces;
			new_cell_faces.reserve(cell_faces.size());

			for (int fid : cell_faces) {
				if (std::find(old_fids.begin(), old_fids.end(), fid) != old_fids.end()) {
					touched = true;
				} else {
					new_cell_faces.push_back(fid);
				}
			}
			if (touched) {
				new_cell_faces.push_back(nfid);
				cell_faces.swap(new_cell_faces);
			}
		}
	}

	// // // merge edges ~ remove vertices

	std::vector<std::vector<int>>vertex_to_edges(points.size());
	edge_to_faces.assign(edges.size(), {});
	std::vector<char> active_vertices(points.size(), false);
	for (const auto& cid:level_cvxs) {
		for (const auto&fid:cvxs[cid]) {
			for (const auto&eid:faces[fid]) {
				push_unique(edge_to_faces[eid], fid);
				push_unique(vertex_to_edges[edges[eid].v1], eid);
				push_unique(vertex_to_edges[edges[eid].v2], eid);
				active_vertices[edges[eid].v1] = true;
				active_vertices[edges[eid].v2] = true;
			}
		}
	}

//---------------------------------------
	auto check_edges_vec = [&](const std::vector<int>& eids) {
		if (eids.empty()) return true;
		std::vector<int> vids;
		vids.reserve(eids.size() * 2);
		for (int eid : eids) {
			RZ_ASSERT(eid >= 0 && eid < (int)edges.size(), "invalid eid");
			const auto& e = edges[eid];
			vids.push_back(e.v1);
			vids.push_back(e.v2);
		}
		std::sort(vids.begin(), vids.end());
		vids.erase(std::unique(vids.begin(), vids.end()), vids.end());
		if (vids.size() <= 2) return true;

		const EK::Point_3& p0 = points[vids[0]];
		int i1 = -1;
		for (int i = 1; i < (int)vids.size(); ++i) {
			if (points[vids[i]] != p0) {
				i1 = i;
				break;
			}
		}
		if (i1 == -1) return true;

		const EK::Point_3& p1 = points[vids[i1]];
		EK::Line_3 line(p0, p1);
		for (int vid : vids) {
			if (!line.has_on(points[vid])) return false;
		}
		return true;
	};

	auto merge_edge_vec = [&](const std::vector<int>& eids) {
		RZ_ASSERT(eids.size() == 2, eids.size());
		const int eid0 = eids[0];
		const int eid1 = eids[1];
		RZ_ASSERT(eid0 >= 0 && eid0 < (int)edges.size(), "invalid eid0");
		RZ_ASSERT(eid1 >= 0 && eid1 < (int)edges.size(), "invalid eid1");

		const auto& e0 = edges[eid0];
		const auto& e1 = edges[eid1];
		int shared = -1;
		int a = -1, b = -1;
		if (e0.v1 == e1.v1) {
			shared = e0.v1;
			a = e0.v2;
			b = e1.v2;
		}
		else if (e0.v1 == e1.v2) {
			shared = e0.v1;
			a = e0.v2;
			b = e1.v1;
		}
		else if (e0.v2 == e1.v1) {
			shared = e0.v2;
			a = e0.v1;
			b = e1.v2;
		}
		else if (e0.v2 == e1.v2) {
			shared = e0.v2;
			a = e0.v1;
			b = e1.v1;
		}

		RZ_ASSERT(shared != -1, "two edges do not share a vertex");
		RZ_ASSERT(a != -1 && b != -1, "invalid merged edge endpoints");
		RZ_ASSERT(a != b, "merged edge degenerates to a point");
		active_vertices[shared] = false;
		NDEdge ans;
		ans.v1 = a;
		ans.v2 = b;
		return ans;
	};

	std::deque<int> vertex_candidates;
	std::vector<char> vertex_candidate_dead(vertex_to_edges.size(), false);
	for (int vid = 0; vid < (int)vertex_to_edges.size(); ++vid) {
		if (active_vertices[vid]) vertex_candidates.push_back(vid);
	}

	while (!vertex_candidates.empty()) {
		const int candidate_vid = vertex_candidates.front();
		vertex_candidates.pop_front();
		if (candidate_vid < 0 || candidate_vid >= (int)vertex_to_edges.size()) continue;
		if (!active_vertices[candidate_vid] || vertex_candidate_dead[candidate_vid]) continue;

		const auto old_eids = vertex_to_edges[candidate_vid];
		std::unordered_map<int,int>face_cnt;
		for (auto&oeid:old_eids) {
			for (auto&fid:edge_to_faces[oeid])
				face_cnt[fid]++;
		}
		bool edge_cross_cell = false;
		for (auto&temp:face_cnt)
			if (temp.second!=2)
				edge_cross_cell = true;
		if (edge_cross_cell) {
			vertex_candidate_dead[candidate_vid] = true;
			continue;
		}

		if (!check_edges_vec(old_eids)) {
			vertex_candidate_dead[candidate_vid] = true;
			continue;
		}

		auto new_edge = merge_edge_vec(old_eids);
		const int neid = edges.size();
		edges.push_back(new_edge);
		edge_to_faces.emplace_back();
		for (auto&oeid:old_eids)
			for (auto&fid:edge_to_faces[oeid])
				push_unique(edge_to_faces[neid], fid);
		std::vector<int>nvids;
		nvids.push_back(new_edge.v1);
		nvids.push_back(new_edge.v2);

		for (int vid : nvids) {
			auto& inc_edges = vertex_to_edges[vid];
			for (int oeid : old_eids) {
				erase_value(inc_edges, oeid);
			}
			push_unique(inc_edges, neid);
			vertex_candidates.push_back(vid);
		}
		// 再统一修改这些 face：旧 edge -> 新 edge
		for (int fid : edge_to_faces[neid]) {
			auto& face_edges = faces[fid];
			std::vector<int> new_face_edges;
			new_face_edges.reserve(face_edges.size());
			bool replaced = false;
			int cnt = 0;
			for (int eid : face_edges) {
				if (std::find(old_eids.begin(), old_eids.end(), eid) != old_eids.end()) {
					replaced = true;
					++cnt;
				} else {
					new_face_edges.push_back(eid);
				}
			}
			if (replaced) {
				new_face_edges.push_back(neid);
				face_edges.swap(new_face_edges);
			}
			RZ_ASSERT(replaced&&cnt==2,"");
		}
	}
}

std::vector<std::vector<int>> MiniLcc::generate_lods(int level,int init_n) {
	FUNC_TIMER();
    FUNC_MEMORY_TIMER();
	std::vector<std::vector<int>> ans;
	std::vector<int> cvx_lods;
	cvx_lods.assign(cvxs.size(),level-1);
	int cnt = init_n*init_n*init_n*7;
	int begin = 0;
	//这里关系注意 因为有过程！
	for (int i=0;i<level-1;++i) {
		for (int k=0;k<cnt;++k) {
			RZ_ASSERT((begin+k)<cvx_lods.size(),"");
			cvx_lods[begin+k] = i;
		}
		begin += cnt;
		cnt *= 8;
	}

	// 根据 cut-value 生成 lods
	std::vector<Real> acc;
	for (int l=0;l<level;++l) {
		acc.push_back(1-(level - l - 1)*0.2 + 1e-10);
	}
	Real current_acc = 1.0;
	Real model_volume = 0;
	for (int cid=0;cid<cvxs.size();++cid) {
		if (!cvx_partitioned[cid]&&cvx_is_inside[cid])
			model_volume += cvx_volume_fast(cvxs[cid]);
	}


	std::vector<bool> activate_cvxs;
	for (int cid=0;cid<cvxs.size();++cid) {
		if (cvx_partitioned[cid])activate_cvxs.push_back(false);
		else activate_cvxs.push_back(true);
	}

	for (int l=level-1;l>=0;--l) {
		std::vector<int> level_cvxs;

		//用一个最小堆 存储 std::pair<Real,std::pair<int,int>> loss\tree_id\node_id
		struct Compare {
			bool operator()(const std::pair<Real, int>& a, const std::pair<Real, int>& b) {
				return a.first > b.first;  // 比较 Real 值，最小堆
			}
		};
		std::priority_queue<std::pair<Real,
			int>, std::vector<std::pair<Real, int>>, Compare> heap;
		for (int cid=0;cid<cvxs.size();++cid) {
				if (activate_cvxs[cid])continue;// important
				if (cvx_lods[cid]<l) continue;
				int son_cid1 = cvx_sons[cid].first;
				int son_cid2 = cvx_sons[cid].second;
				if (!activate_cvxs[son_cid1]||!activate_cvxs[son_cid2]) continue;
				heap.push({cvx_cut_values[cid],cid});
		}
		while (current_acc>acc[l] && !heap.empty()) {
			auto to_erase = heap.top();
			heap.pop();
			Real min_loss = to_erase.first;
			int min_cid = to_erase.second;
			if (min_cid == -1)break;
			current_acc -= min_loss/model_volume;
			int son_cid1 = cvx_sons[min_cid].first;
			int son_cid2 = cvx_sons[min_cid].second;
			activate_cvxs[son_cid1] = false;
			activate_cvxs[son_cid2] = false;
			activate_cvxs[min_cid] = true;

			int father_cid = cvx_father[min_cid];
			if (father_cid == -1)continue;

			if (cvx_lods[father_cid]<l) continue;
			int son1 = cvx_sons[father_cid].first;
			int son2 = cvx_sons[father_cid].second;
			if (!activate_cvxs[son1]||!activate_cvxs[son2]) continue;
			heap.push({cvx_cut_values[father_cid],father_cid});

		}
		for (int cid=0;cid<cvxs.size();++cid) {
			if (activate_cvxs[cid])
				level_cvxs.push_back(cid);
		}
		ans.push_back(level_cvxs);
	}
	std::reverse(ans.begin(),ans.end());
	return ans;
}

void MiniLcc::lod_repair(const std::vector<int> &level_cvxs) {
	FUNC_TIMER();
	std::unordered_map<int,std::unordered_set<int>>face_to_cvxs;
	std::unordered_map<int,std::unordered_set<int>>edge_to_faces;
	std::vector<bool> active_edges;
	active_edges.resize(edges.size(),false);

	for (const auto& cid:level_cvxs)
	{
		for (auto&fid:cvxs[cid]) {
			face_to_cvxs[fid].insert(cid);
			for (auto&eid:faces[fid]) {
				active_edges[eid] = true;
				edge_to_faces[eid].insert(fid);
			}
		}
	}
	//锁住所有叶子节点的面 不能合并
	for (const auto& cid:level_cvxs) {
		if (cvx_partitioned[cid])continue;
		for (auto&fid:cvxs[cid]) {
			for (auto&eid:faces[fid]) {
				active_edges[eid] = false;
			}
		}
	}

	for (auto it = edge_to_faces.begin(); it != edge_to_faces.end(); ) {
		const auto &kv = *it;
		if (!active_edges[kv.first]) {
			it = edge_to_faces.erase(it);
			continue;
		}
		++it;
	}
	// merge faces ~ remove edges ~ new faces ~ edge_to_faces ~ vertex_to_edges（后面统一更新）;

	bool flag = true;
	while (flag) {
		flag=false;
		for (auto it = edge_to_faces.begin(); it != edge_to_faces.end(); )
		{
			const auto &kv = *it;
			if (!active_edges[kv.first]) {
				it = edge_to_faces.erase(it);
				continue;
			}
			if (check_faces(kv.second))
			{
				++it;
				// 这个边可以去除 这个面需要合并 edge to faces更新
				auto new_face = merge_face(kv.second,active_edges);
				int nfid = faces.size();
				for (auto&old_fid:kv.second) {
					for (auto&cid:face_to_cvxs[old_fid])
						face_to_cvxs[nfid].insert(cid);
				}
				faces.push_back(new_face);
				for (int eid : new_face) {
					auto eit = edge_to_faces.find(eid);
					if (eit==edge_to_faces.end())continue;
					auto& inc_faces = eit->second;
					for (int ofid : kv.second) {
						inc_faces.erase(ofid);
					}
					inc_faces.insert(nfid);
				}

				std::unordered_map<int,int>cvx_counts;
				for (auto&old_fid:kv.second) {
					for (auto& cid:face_to_cvxs[old_fid])
						cvx_counts[cid]++;
				}
				std::vector<int> todo_cids;
				for (auto& cvx_count:cvx_counts)if (cvx_count.second>=2) {
					todo_cids.push_back(cvx_count.first);
				}else if (cvx_count.second==1)RZ_FATAL("cvx_count.second==1");
				RZ_ASSERT(todo_cids.size()<=2,todo_cids.size());
				for (const auto& cid : todo_cids) {
					auto& cell_faces = cvxs[cid];
					bool touched = false;

					std::vector<int> new_cell_faces;
					new_cell_faces.reserve(cell_faces.size());

					for (int fid : cell_faces) {
						if (kv.second.find(fid) != kv.second.end()) {
							touched = true;
						} else {
							new_cell_faces.push_back(fid);
						}
					}

					if (touched) {
						RZ_ASSERT(cvx_partitioned[cid],"");
						new_cell_faces.push_back(nfid);
						cell_faces.swap(new_cell_faces);
					}
				}

				flag = true;
				break;
			}
			else {
				it = edge_to_faces.erase(it);
			}
		}
	}

	// // // merge edges ~ remove vertices
	//这里不能只看 level_cvxs中的 锁住！
	//应该是 既属于不能动的 fine level面 又属于当前的面则这个面不能变 （因为我们变的是面 所以这里细粒度到面）
	std::unordered_map<int,std::unordered_set<int>>vertex_to_edges;
	edge_to_faces.clear();
	std::vector<bool> active_vertices;
	active_vertices.resize(points.size(),false);
	std::unordered_set<int> level_fid;
	for (const auto& cid:level_cvxs) {
		for (const auto&fid:cvxs[cid]) {
			level_fid.insert(fid);
			for (const auto&eid:faces[fid]) {
				edge_to_faces[eid].insert(fid);
				vertex_to_edges[edges[eid].v1].insert(eid);
				vertex_to_edges[edges[eid].v2].insert(eid);
				active_vertices[edges[eid].v1] = true;
				active_vertices[edges[eid].v2] = true;
			}
		}
	}
	std::unordered_set<int> global_lock_fids;
	for (int cid = 0;cid<cvxs.size();++cid) {
		if (cvx_partitioned[cid])continue;
		for (const auto&fid:cvxs[cid]) {
			global_lock_fids.insert(fid);
		}
	}
	for (auto&cid:level_cvxs) {
		for (const auto&fid:cvxs[cid]) {
			if (!global_lock_fids.count(fid))continue;
			for (const auto&eid:faces[fid]) {
				active_vertices[edges[eid].v1] = false;
				active_vertices[edges[eid].v2] = false;
			}
		}
	}
//---------------------------------------
	flag = true;
	while (flag) {
		flag=false;
		for (auto it = vertex_to_edges.begin(); it != vertex_to_edges.end(); )
		{
			const auto& kv = *it;
			if (!active_vertices[kv.first]) {
				it = vertex_to_edges.erase(it);
				continue;
			}
			std::unordered_map<int,int>face_cnt;
			for (auto&oeid:kv.second) {
				for (auto&fid:edge_to_faces[oeid])
					face_cnt[fid]++;
			}
			bool edge_cross_cell = false;
			for (auto&temp:face_cnt)
				if (temp.second!=2)
					edge_cross_cell = true;
			if (edge_cross_cell) {
				it = vertex_to_edges.erase(it);
				continue;
			}

			if (check_edges(kv.second))
			{
				++it;
				auto new_edge = merge_edge(kv.second,active_vertices);
				int neid = edges.size();
				edges.push_back(new_edge);
				for (auto&oeid:kv.second)
					for (auto&fid:edge_to_faces[oeid])
						edge_to_faces[neid].insert(fid);
				std::vector<int>nvids;
				nvids.push_back(new_edge.v1);
				nvids.push_back(new_edge.v2);

				for (int vid : nvids) {
					auto vit = vertex_to_edges.find(vid);
					if (vit==vertex_to_edges.end())continue;
					auto& inc_edges = vit->second;
					for (int oeid : kv.second) {
						inc_edges.erase(oeid);
					}
					inc_edges.insert(neid);
				}
				// 再统一修改这些 face：旧 edge -> 新 edge
				for (int fid : edge_to_faces[neid]) {
					auto& face_edges = faces[fid];
					std::vector<int> new_face_edges;
					new_face_edges.reserve(face_edges.size());
					bool replaced = false;
					int cnt = 0;
					for (int eid : face_edges) {
						if (kv.second.find(eid) != kv.second.end()) {
							replaced = true;
							++cnt;
						} else {
							new_face_edges.push_back(eid);
						}
					}
					if (replaced) {
						new_face_edges.push_back(neid);
						face_edges.swap(new_face_edges);
					}
					RZ_ASSERT(replaced&&cnt==2,"");
				}

				flag = true;
				break;
			}
			else {
				it = vertex_to_edges.erase(it);
			}
		}
	}


}

bool MiniLcc::check_faces(const std::unordered_set<int> &fids) {
	if (fids.empty()) return true;
	if (fids.size()!=2)return false;
	std::vector<int> vids;
	vids.reserve(fids.size() * 8);

	// 收集所有 face 上的顶点
	for (int fid : fids) {
		RZ_ASSERT(fid >= 0 && fid < (int)faces.size(), "invalid fid");

		for (int eid : faces[fid]) {
			RZ_ASSERT(eid >= 0 && eid < (int)edges.size(), "invalid eid");
			vids.push_back(edges[eid].v1);
			vids.push_back(edges[eid].v2);
		}
	}

	if (vids.empty()) return true;

	// 去重
	std::sort(vids.begin(), vids.end());
	vids.erase(std::unique(vids.begin(), vids.end()), vids.end());

	// 少于3个点，默认共面
	if (vids.size() < 3) return true;

	const EK::Point_3& p0 = points[vids[0]];

	// 找到与 p0 不同的第二个点
	int i1 = -1;
	for (int i = 1; i < (int)vids.size(); ++i) {
		if (points[vids[i]] != p0) {
			i1 = i;
			break;
		}
	}
	if (i1 == -1) return true; // 所有点重合

	const EK::Point_3& p1 = points[vids[i1]];
	EK::Line_3 line(p0, p1);

	// 找到第一个不在该直线上的点，构造平面
	bool found_plane = false;
	EK::Plane_3 plane;

	for (int i = i1 + 1; i < (int)vids.size(); ++i) {
		const EK::Point_3& p = points[vids[i]];
		if (!line.has_on(p)) {
			plane = EK::Plane_3(p0, p1, p);
			found_plane = true;
			break;
		}
	}

	// 所有点共线 => 也算合法
	if (!found_plane) return true;

	// 检查所有点是否都严格在该平面上
	for (int vid : vids) {
		if (!plane.has_on(points[vid])) {
			return false;
		}
	}

	return true;
}

std::vector<int> MiniLcc::merge_face(const std::unordered_set<int> &fids,std::vector<bool> &activate_edges) {
	RZ_ASSERT(fids.size()==2,fids.size());
	std::vector<int> eids;
	for (int fid : fids) {
		RZ_ASSERT(fid >= 0 && fid < (int)faces.size(), "invalid fid");
		for (int eid : faces[fid]) {
			RZ_ASSERT(eid >= 0 && eid < (int)edges.size(), "invalid eid");
			eids.push_back(eid);
		}
	}

	if (eids.empty()) return {};

	std::sort(eids.begin(), eids.end());

	std::vector<int> merged_face;
	merged_face.reserve(eids.size());

	for (int i = 0; i < (int)eids.size(); ) {
		int j = i + 1;
		while (j < (int)eids.size() && eids[j] == eids[i]) ++j;

		int cnt = j - i;

		// 合并共面的若干子面时，一条边最多应出现两次
		RZ_ASSERT(cnt <= 2, "edge appears more than twice");

		if (cnt == 1) {
			merged_face.push_back(eids[i]);
		}else {
			activate_edges[eids[i]]=false;
		}

		i = j;
	}

	return merged_face;
}

bool MiniLcc::check_edges(const std::unordered_set<int> &eids) {
	if (eids.empty()) return true;

	std::vector<int> vids;
	vids.reserve(eids.size() * 2);

	// 收集所有端点
	for (int eid : eids) {
		RZ_ASSERT(eid >= 0 && eid < (int)edges.size(), "invalid eid");
		const auto& e = edges[eid];
		vids.push_back(e.v1);
		vids.push_back(e.v2);
	}

	// 去重
	std::sort(vids.begin(), vids.end());
	vids.erase(std::unique(vids.begin(), vids.end()), vids.end());

	// 少于2个点 => 必然共线
	if (vids.size() <= 2) return true;

	const EK::Point_3& p0 = points[vids[0]];

	// 找到与 p0 不同的第二个点
	int i1 = -1;
	for (int i = 1; i < (int)vids.size(); ++i) {
		if (points[vids[i]] != p0) {
			i1 = i;
			break;
		}
	}

	// 所有点重合
	if (i1 == -1) return true;

	const EK::Point_3& p1 = points[vids[i1]];
	EK::Line_3 line(p0, p1);

	// 检查所有点是否在该直线上（精确）
	for (int vid : vids) {
		if (!line.has_on(points[vid])) {
			return false;
		}
	}

	return true;
}

NDEdge MiniLcc::merge_edge(const std::unordered_set<int> &eids,std::vector<bool> &activate_vertices) {
	RZ_ASSERT(eids.size() == 2, eids.size());

	auto it = eids.begin();
	int eid0 = *it++;
	int eid1 = *it;

	RZ_ASSERT(eid0 >= 0 && eid0 < (int)edges.size(), "invalid eid0");
	RZ_ASSERT(eid1 >= 0 && eid1 < (int)edges.size(), "invalid eid1");

	const auto& e0 = edges[eid0];
	const auto& e1 = edges[eid1];
	int shared = -1;
	int a = -1, b = -1;

	// 找公共端点 + 两个外侧端点
	if (e0.v1 == e1.v1) {
		shared = e0.v1;
		a = e0.v2;
		b = e1.v2;
	}
	else if (e0.v1 == e1.v2) {
		shared = e0.v1;
		a = e0.v2;
		b = e1.v1;
	}
	else if (e0.v2 == e1.v1) {
		shared = e0.v2;
		a = e0.v1;
		b = e1.v2;
	}
	else if (e0.v2 == e1.v2) {
		shared = e0.v2;
		a = e0.v1;
		b = e1.v1;
	}

	RZ_ASSERT(shared != -1, "two edges do not share a vertex");
	RZ_ASSERT(a != -1 && b != -1, "invalid merged edge endpoints");
	RZ_ASSERT(a != b, "merged edge degenerates to a point");
	activate_vertices[shared] = false;
	NDEdge ans;
	ans.v1 = a;
	ans.v2 = b;
	return ans;
}


 std::pair<int,
              int>
    MiniLcc::split_and_insert_faces(int planar_id, int face_id, int middle_edge_id,
        std::unordered_map<int,std::pair<int,int>>& edge_split_map) {
        // FUNC_TIMER();
        RZ_ASSERT(planar_id >= 0 && planar_id < (int)my_planars.size(),"");
        RZ_ASSERT(face_id >= 0 && face_id < (int)faces.size(),"");
        RZ_ASSERT(middle_edge_id >= 0 && middle_edge_id < (int)edges.size(),"");

        const auto& plane = my_planars[planar_id].plane;
        const auto& old_face = faces[face_id];

        std::vector<int> face_neg;
        std::vector<int> face_pos;


        auto point_side = [&](const EK::Point_3& p) -> int {
            if (plane.has_on_positive_side(p)) return 1;
            if (plane.has_on_negative_side(p)) return -1;
            return 0;
        };
        std::vector<int> eids;
        for (auto eid : old_face) {
            if (!edge_split_map.count(eid))
                eids.push_back(eid);
            else {
                eids.push_back(edge_split_map[eid].first);
                eids.push_back(edge_split_map[eid].second);
            }
        }
        for (auto eid : eids) {
            RZ_ASSERT(eid >= 0 && eid < (int)edges.size(),"");

            const auto& e = edges[eid];
            const auto& p1 = points[e.v1];
            const auto& p2 = points[e.v2];

            int s1 = point_side(p1);
            int s2 = point_side(p2);

            // 这条边整体归属哪一侧
            // 按你的前提：除了 middle_edge 外，其他 edge 应该只在某一侧
            if ((s1 <= 0 && s2 <= 0) && (s1 < 0 || s2 < 0)) {
                // 至少一个端点在负侧，且没有端点在正侧
                face_neg.push_back(eid);
            }
            else if ((s1 >= 0 && s2 >= 0) && (s1 > 0 || s2 > 0)) {
                // 至少一个端点在正侧，且没有端点在负侧
                face_pos.push_back(eid);
            }
            else {
                RZ_FATAL("");
            }
        }

        // middle edge 同时属于两个新 face
        face_neg.push_back(middle_edge_id);
        face_pos.push_back(middle_edge_id);

        // 插入并返回 iterator
        const int id0 = (int)faces.size();
        faces.push_back(std::move(face_neg));
        face_to_cvxs.push_back({});
        const int id1 = (int)faces.size();
        faces.push_back(std::move(face_pos));
        face_to_cvxs.push_back({});
        return {id0, id1};
    }

    Real MiniLcc::face_area(int fid)  {
        const auto eids = faces[fid];
        if (eids.size() < 3) return 0.0;

        // 1) 构建顶点邻接
        std::map<int, std::vector<int>> adj;

        for (int eid : eids) {
            const auto& e = edges[eid];
            adj[e.v1].push_back(e.v2);
            adj[e.v2].push_back(e.v1);
        }

        // 每个顶点应该恰好连接两个边（简单多边形）
        for (const auto& kv : adj) {
            RZ_ASSERT(kv.second.size() == 2,
                "face is not a valid simple polygon (vertex degree != 2) :",kv.second.size());
        }

        // 2) 恢复顶点环
        std::vector<int> loop;
        loop.reserve(adj.size());

        int start = adj.begin()->first;
        int prev = -1;
        int cur = start;

        while (true) {
            loop.push_back(cur);

            const auto& nbrs = adj[cur];

            int nxt;
            if (nbrs.size() == 1) {
                nxt = nbrs[0];
            } else {
                nxt = (nbrs[0] == prev ? nbrs[1] : nbrs[0]);
            }

            prev = cur;
            cur = nxt;

            if (cur == start) break;

            RZ_ASSERT((int)loop.size() <= (int)adj.size() + 1,
                "loop reconstruction failed (likely non-manifold or broken face)");
        }

        if (loop.size() < 3) return 0.0;

        // 3) 三角扇面积
        const auto& p0 = points[loop[0]];
        EK::Vector_3 sum(0, 0, 0);

        for (int i = 1; i + 1 < (int)loop.size(); ++i) {
            const auto& p1 = points[loop[i]];
            const auto& p2 = points[loop[i + 1]];

            sum = sum + CGAL::cross_product(p1 - p0, p2 - p0);
        }

        Real area = 0.5 * std::sqrt(CGAL::to_double(sum.squared_length()));

        RZ_ASSERT(std::isfinite(area), "face area is NaN or Inf");

        return area;
    }


    std::vector<int> MiniLcc::partition_face(const std::unordered_set<int>& eids, int planarid) const {
        RZ_ASSERT(planarid >= 0 && planarid < (int)my_planars.size(), "planar id out of range");

        const auto& plane = my_planars[planarid].plane;
        std::vector<int> face_eids;
        face_eids.reserve(eids.size());

        for (int eid : eids) {
            const auto& e = edges[eid];
            const auto& p1 = points[e.v1];
            const auto& p2 = points[e.v2];

            // 只有整条边都落在该平面上，才认为它属于这个 partition face
            if (plane.has_on(p1) && plane.has_on(p2)) {
                face_eids.push_back(eid);
            }
        }

        RZ_ASSERT(face_eids.size() >= 3, "partition face has fewer than 3 coplanar edges");

        return face_eids;
    }

    std::pair<std::vector<int>, std::vector<int>>
    MiniLcc::split_and_insert_cvxs(int planar_id, int cvx_id, int middle_face_id,
        std::unordered_map<int,std::pair<int,int>>& face_split_map,
        std::unordered_map<int,std::pair<int,int>>& edge_split_map) {// speeded

        RZ_ASSERT(planar_id >= 0 && planar_id < (int)my_planars.size(), "planar_id out of range");
        RZ_ASSERT(cvx_id >= 0 && cvx_id < (int)cvxs.size(), "cvx_id out of range");
        RZ_ASSERT(middle_face_id >= 0 && middle_face_id < (int)faces.size(), "middle_face_id out of range");

        const auto& plane = my_planars[planar_id].plane;
        const auto& old_cvx = cvxs[cvx_id];

        std::vector<int> cvx_neg;
        std::vector<int> cvx_pos;


        auto point_side = [&](const EK::Point_3& p) -> int {
            if (plane.has_on_positive_side(p)) return 1;
            if (plane.has_on_negative_side(p)) return -1;
            return 0;
        };
        std::vector<int> fids;
        for (auto&fid:old_cvx) {
            auto find = face_split_map.find(fid);
            if (find == face_split_map.end())
                fids.push_back(fid);
            else {
                fids.push_back(find->second.first);
                fids.push_back(find->second.second);
            }
        }
        for (int fid : fids) {
            RZ_ASSERT(fid >= 0 && fid < (int)faces.size(), "face id in cvx out of range");

            bool has_n = false;
            bool has_p = false;
            for (auto&eid:faces[fid]) {
                int s = point_side(points[edges[eid].v1]);
                if (s < 0) {
                    has_n = true;
                    break;
                }
                if (s > 0) {
                    has_p = true;
                    break;
                }

                s = point_side(points[edges[eid].v2]);
                if (s < 0) {
                    has_n = true;
                    break;
                }
                if (s>0) {
                    has_p = true;
                    break;
                }
            }

            if (has_n) {
                cvx_neg.push_back(fid);
            }
            if (has_p) {
                cvx_pos.push_back(fid);
            }
        }

        // 中间面同时属于两个新 cvx
        cvx_neg.push_back(middle_face_id);
        cvx_pos.push_back(middle_face_id);

        // 可选一致性检查：一个闭凸 polyhedron 被单平面切分后，
        // 两个子 cvx 至少都应有 4 个面（tetra 的下界）
        RZ_ASSERT(cvx_neg.size() >= 4, "negative cvx has too few faces");
        RZ_ASSERT(cvx_pos.size() >= 4, "positive cvx has too few faces");

        return {cvx_neg, cvx_pos};
    }

    Real MiniLcc::cvx_volume_fast(const std::vector<int>& cvx) const {
        // FUNC_TIMER();
        RZ_ASSERT(!cvx.empty(), "cvx is empty");
        EK_to_IK to_ik;
        // ---------- 1) 体中心：用唯一顶点平均 ----------
        static thread_local std::vector<int> vertex_mark;
        static thread_local int vertex_stamp = 1;
        ++vertex_stamp;

        if ((int)vertex_mark.size() < (int)points.size()) {
            vertex_mark.resize(points.size(), 0);
        }
        if (vertex_stamp == std::numeric_limits<int>::max()) {
            std::fill(vertex_mark.begin(), vertex_mark.end(), 0);
            vertex_stamp = 1;
        }

        std::vector<int> vids;
        vids.reserve(cvx.size() * 8);

        for (int fid : cvx) {
            for (int eid : faces[fid]) {
                const auto& e = edges[eid];

                if (vertex_mark[e.v1] != vertex_stamp) {
                    vertex_mark[e.v1] = vertex_stamp;
                    vids.push_back(e.v1);
                }
                if (vertex_mark[e.v2] != vertex_stamp) {
                    vertex_mark[e.v2] = vertex_stamp;
                    vids.push_back(e.v2);
                }
            }
        }

        RZ_ASSERT(vids.size() >= 4, "a 3D convex cell must have at least 4 vertices");

        Real cvx_x = 0.0, cvx_y = 0.0, cvx_z = 0.0;
        for (int vid : vids) {
            const auto p = to_ik(points[vid]);
            cvx_x += p.x();
            cvx_y += p.y();
            cvx_z += p.z();
        }

        const Real inv_nv = 1.0 / (Real)vids.size();
        cvx_x *= inv_nv;
        cvx_y *= inv_nv;
        cvx_z *= inv_nv;

        const IK::Point_3 cvx_c(cvx_x, cvx_y, cvx_z);

        // ---------- 2) face center + edge tetra ----------
        Real volume = 0.0;

        for (int fid : cvx) {
            const auto& eids = faces[fid];
            // face center: face唯一顶点平均
            ++vertex_stamp;
            if (vertex_stamp == std::numeric_limits<int>::max()) {
                std::fill(vertex_mark.begin(), vertex_mark.end(), 0);
                vertex_stamp = 1;
            }

            std::vector<int> fvids;
            fvids.reserve(eids.size() * 2);

            for (int eid : eids) {
                const auto& e = edges[eid];

                if (vertex_mark[e.v1] != vertex_stamp) {
                    vertex_mark[e.v1] = vertex_stamp;
                    fvids.push_back(e.v1);
                }
                if (vertex_mark[e.v2] != vertex_stamp) {
                    vertex_mark[e.v2] = vertex_stamp;
                    fvids.push_back(e.v2);
                }
            }

            RZ_ASSERT(fvids.size() >= 3, "face has fewer than 3 unique vertices");

            Real fx = 0.0, fy = 0.0, fz = 0.0;
            for (int vid : fvids) {
                const auto p = to_ik(points[vid]);
                fx += p.x();
                fy += p.y();
                fz += p.z();
            }
            const Real inv_nf = 1.0 / (Real)fvids.size();
            fx *= inv_nf;
            fy *= inv_nf;
            fz *= inv_nf;

            const IK::Point_3 face_c(fx, fy, fz);

            Real face_contrib = 0.0;
            for (int eid : eids) {
                const auto& e = edges[eid];
                const auto p0 = to_ik(points[e.v1]);
                const auto p1 = to_ik(points[e.v2]);

                const IK::Vector_3 a = face_c - cvx_c;
                const IK::Vector_3 b = p0 - cvx_c;
                const IK::Vector_3 d = p1 - cvx_c;

                const auto triple = CGAL::scalar_product(CGAL::cross_product(a, b), d);
                face_contrib += std::abs(triple) / 6.0;
            }

            volume += face_contrib;
        }

        RZ_ASSERT(std::isfinite(volume), "cvx volume is NaN or Inf");
        return volume;
    }

    Real MiniLcc::cvx_volume(const std::vector<int>& cvx) const {
        RZ_ASSERT(!cvx.empty(), "cvx is empty");

        // 1) 收集该 cvx 的所有唯一顶点
        std::set<int> vset;
        for (int fid : cvx) {
            for (int eid : faces[fid]) {
                const auto& e = edges[eid];

                vset.insert(e.v1);
                vset.insert(e.v2);
            }
        }
        RZ_ASSERT(vset.size() >= 4, "a 3D convex cell must have at least 4 vertices");

        // 2) 用顶点平均值构造一个内部参考点
        Real cx = 0.0, cy = 0.0, cz = 0.0;
        for (int vid : vset) {
            cx += CGAL::to_double(points[vid].x());
            cy += CGAL::to_double(points[vid].y());
            cz += CGAL::to_double(points[vid].z());
        }
        cx /= (Real)vset.size();
        cy /= (Real)vset.size();
        cz /= (Real)vset.size();

        EK::Point_3 c(cx, cy, cz);

        // 局部函数：从一个 face 恢复有序顶点环
        auto ordered_vertices_of_face = [&](int fid) -> std::vector<int> {
            const auto& eids = faces[fid];
            RZ_ASSERT(eids.size() >= 3, "face has fewer than 3 edges");

            std::map<int, std::vector<int>> adj;
            for (int eid : eids) {
                const auto& e = edges[eid];
                adj[e.v1].push_back(e.v2);
                adj[e.v2].push_back(e.v1);
            }

            for (const auto& kv : adj) {
                RZ_ASSERT(kv.second.size() == 2,
                          "face is not a simple polygon (vertex degree != 2)");
            }

            std::vector<int> loop;
            loop.reserve(adj.size());

            int start = adj.begin()->first;
            int prev = -1;
            int cur = start;

            while (true) {
                loop.push_back(cur);

                const auto& nbrs = adj[cur];
                int nxt = (nbrs[0] == prev ? nbrs[1] : nbrs[0]);

                prev = cur;
                cur = nxt;

                if (cur == start) break;

                RZ_ASSERT((int)loop.size() <= (int)adj.size() + 1,
                          "loop reconstruction failed");
            }

            RZ_ASSERT(loop.size() >= 3, "reconstructed face loop has fewer than 3 vertices");
            return loop;
        };

        // 3) 三角扇 + 内部点四面体分解
        Real volume = 0.0;

        for (int fid : cvx) {
            auto loop = ordered_vertices_of_face(fid);
            const auto& p0 = points[loop[0]];

            for (int i = 1; i + 1 < (int)loop.size(); ++i) {
                const auto& p1 = points[loop[i]];
                const auto& p2 = points[loop[i + 1]];

                EK::Vector_3 a = p0 - c;
                EK::Vector_3 b = p1 - c;
                EK::Vector_3 d = p2 - c;

                auto triple = CGAL::scalar_product(CGAL::cross_product(a, b), d);
                Real tet_vol = std::abs(CGAL::to_double(triple)) / 6.0;
                volume += tet_vol;
            }
        }

        RZ_ASSERT(std::isfinite(volume), "cvx volume is NaN or Inf");
        return volume;
    }

    void MiniLcc::check_and_swap_cvxs(std::pair<std::vector<int>, std::vector<int>>& cvxs_pair,
                             int planar_id) {
        // FUNC_TIMER();
        RZ_ASSERT(planar_id >= 0 && planar_id < (int)my_planars.size(),
                  "planar_id out of range");

        const auto& plane = my_planars[planar_id].plane;

        // ---- 计算一个 cvx 的 centroid ----
        auto compute_centroid = [&](const std::vector<int>& cvx) -> EK::Point_3 {
            RZ_ASSERT(!cvx.empty(), "cvx is empty");

            std::set<int> vset;
            for (int fid : cvx) {
                for (int eid : faces[fid]) {
                    const auto& e = edges[eid];
                    vset.insert(e.v1);
                    vset.insert(e.v2);
                }
            }

            RZ_ASSERT(!vset.empty(), "cvx has no vertices");

            Real cx = 0, cy = 0, cz = 0;
            for (int vid : vset) {
                const auto& p = points[vid];
                cx += CGAL::to_double(p.x());
                cy += CGAL::to_double(p.y());
                cz += CGAL::to_double(p.z());
            }

            cx /= (Real)vset.size();
            cy /= (Real)vset.size();
            cz /= (Real)vset.size();

            return EK::Point_3(cx, cy, cz);
        };

        auto side = [&](const EK::Point_3& p) -> int {
            if (plane.has_on_positive_side(p)) return 1;
            if (plane.has_on_negative_side(p)) return -1;
            return 0;
        };

        auto c0 = compute_centroid(cvxs_pair.first);

        int s0 = side(c0);


        // ---- 如果顺序反了就 swap ----
        // 目标：first = negative, second = positive
        if (s0 > 0 ) {
            std::swap(cvxs_pair.first, cvxs_pair.second);
        }
    }
    void MiniLcc::fix_cvxs_faces(std::unordered_map<int,std::pair<int,int>>& face_split_map,
        std::unordered_map<int,std::pair<int,int>>& edge_split_map) {
        // FUNC_TIMER();
        std::unordered_set<int> cvxs_to_renew;
        for (auto&face_pair:face_split_map) {
            for (auto cvxid:face_to_cvxs[face_pair.first]) {
                if (cvx_partitioned[cvxid])continue;
                cvxs_to_renew.insert(cvxid);
            }
        }
        for (auto cvxid:cvxs_to_renew) {
            auto& cvx = cvxs[cvxid];
            std::vector<int> new_fids;
            new_fids.reserve(cvx.size() * 2);
            for (int fid : cvx) {
                auto it = face_split_map.find(fid);
                if (it == face_split_map.end()) {
                    new_fids.push_back(fid);
                } else {
                    new_fids.push_back(it->second.first);
                    new_fids.push_back(it->second.second);
                }
            }
            cvx = std::move(new_fids);
        }

        std::unordered_set<int> faces_to_renew;
        for (auto&edge_pair:edge_split_map)
        {
            for (auto faceid:edge_to_faces[edge_pair.first]) {
                faces_to_renew.insert(faceid);
            }
        }

        for (auto faceid:faces_to_renew) {
            auto& face = faces[faceid];
            std::vector<int> new_eids;
            new_eids.reserve(face.size() * 2);

            for (int eid : face) {
                auto it = edge_split_map.find(eid);
                if (it == edge_split_map.end()) {
                    new_eids.push_back(eid);
                } else {
                    new_eids.push_back(it->second.first);
                    new_eids.push_back(it->second.second);
                }
            }
            face = std::move(new_eids);
        }
    }
    void MiniLcc::build(const std::vector<Planar>& input_planars,
        int n)
    {
        assert(n > 0);
        FUNC_TIMER();


        clear();

        EK::FT xmin = 0.0,  ymin = 0.0,  zmin = 0.0;
        EK::FT xmax = 1.0,  ymax = 1.0,  zmax = 1.0;

        const EK::FT dx = (xmax - xmin) / n;
        const EK::FT dy = (ymax - ymin) / n;
        const EK::FT dz = (zmax - zmin) / n;

        build_regular_grid_lcc(*this, n, xmin, ymin, zmin, dx, dy, dz, false);

        IK_to_EK to_ek;
        for (const auto& planar : input_planars) {
            EK_Planar p;
            p.plane = planar.ek_plane;
            p._3d_Points = planar._ek_3d_Points;
            p.sampled_points = planar.sampled_3d_Points;
            p.area = planar.area;
            p.id = my_planars.size();
            my_planars.push_back(p);
        }
        cvx_inside_planars.clear();
        cvx_inside_planars.resize(cvxs.size());

        cvx_inside_planars_areas.clear();
        cvx_inside_planars_areas.resize(cvxs.size());

        cvx_partition_poly_index.clear();
        cvx_partition_poly_index.resize(cvxs.size(), -1);

        cvx_partitioned.clear();
        cvx_partitioned.resize(cvxs.size(), false);
        EK_to_IK to_ik;


        auto cid_of = [n](int x, int y, int z) -> int {
            return x + n * (y + n * z);
        };

        for (int pid = 0; pid < static_cast<int>(my_planars.size()); ++pid) {
            const auto& planar = my_planars[pid];

            int minx = n + 1, miny = n + 1, minz = n + 1;
            int maxx = -1,    maxy = -1,    maxz = -1;

            // 先根据 planar 顶点求它覆盖到的 cell index 范围
            for (const auto& p : planar._3d_Points) {
                auto ikp = to_ik(p);

                int x = static_cast<int>(std::floor(CGAL::to_double((ikp.x() - xmin) / dx)));
                int y = static_cast<int>(std::floor(CGAL::to_double((ikp.y() - ymin) / dy)));
                int z = static_cast<int>(std::floor(CGAL::to_double((ikp.z() - zmin) / dz)));

                minx = std::min(minx, x);
                miny = std::min(miny, y);
                minz = std::min(minz, z);
                maxx = std::max(maxx, x);
                maxy = std::max(maxy, y);
                maxz = std::max(maxz, z);
            }

            minx = std::max(minx, 0);
            miny = std::max(miny, 0);
            minz = std::max(minz, 0);
            maxx = std::min(maxx, n - 1);
            maxy = std::min(maxy, n - 1);
            maxz = std::min(maxz, n - 1);

            if (minx > maxx || miny > maxy || minz > maxz) continue;

            for (int x = minx; x <= maxx; ++x) {
                for (int y = miny; y <= maxy; ++y) {
                    for (int z = minz; z <= maxz; ++z) {
                        int cid = cid_of(x, y, z);
                        auto verts = vertices_of_cvx(cid);

                        bool has_n = false, has_p = false;
                        for (const auto& v : verts) {
                            if (planar.plane.has_on_negative_side(points[v])) has_n = true;
                            if (planar.plane.has_on_positive_side(points[v])) has_p = true;
                        }
                        if (!has_n || !has_p) continue;
                        cvx_inside_planars[cid].push_back(pid);
                        cvx_inside_planars_areas[cid].push_back(0.5 * planar.area);
                    }
                }
            }
        }

        // 对每个 cvx 内的 planar 按 area 排序
        for (int cid = 0; cid < static_cast<int>(cvxs.size()); ++cid) {

            std::vector<std::pair<Real, int>> sort_buf;
            sort_buf.reserve(cvx_inside_planars[cid].size());
            for (int i = 0; i < static_cast<int>(cvx_inside_planars[cid].size()); ++i) {
                sort_buf.emplace_back(
                    cvx_inside_planars_areas[cid][i],
                    cvx_inside_planars[cid][i]
                );
            }
            for (int i=0;i<static_cast<int>(cvx_inside_planars[cid].size());++i) {
                const auto &largest_poly = my_planars[cvx_inside_planars[cid][i]];

                for (int j=0;j<static_cast<int>(cvx_inside_planars[cid].size());++j) {
                    if (i==j)continue;
                    const auto &planar = my_planars[cvx_inside_planars[cid][j]];
                    bool flag_pos = false;
                    bool flag_neg = false;
                    for (auto& _3d_point : planar._3d_Points)
                    {
                        if (largest_poly.plane.has_on_positive_side(_3d_point))
                            flag_pos = true;
                        else if (largest_poly.plane.has_on_negative_side(_3d_point))
                            flag_neg = true;
                    }
                    if (flag_neg && flag_pos)sort_buf[i].first-=1;
                }
            }
            std::sort(sort_buf.begin(), sort_buf.end());
            cvx_inside_planars[cid].clear();
            cvx_inside_planars_areas[cid].clear();
            for (const auto& pr : sort_buf) {
                cvx_inside_planars[cid].push_back(pr.second);
                cvx_inside_planars_areas[cid].push_back(pr.first);
            }
            if (!cvx_inside_planars[cid].empty()) {
                cvx_partition_poly_index[cid] =
                    static_cast<int>(cvx_inside_planars[cid].size()) - 1;
                volumes.push(cid);
            }

        }
        // RZ_INFO(volumes.size());
        // exit(0);
    }


    void MiniLcc::MultiLevelbuild(const std::vector<Planar>& input_planars,
        int n,int level) {
        FUNC_MEMORY_TIMER();
        assert(n > 0);
        FUNC_TIMER();

        EK::FT xmin = 0.0,  ymin = 0.0,  zmin = 0.0;
        EK::FT xmax = 1.0,  ymax = 1.0,  zmax = 1.0;
        clear();

        EK::FT dx = (xmax - xmin) / n;
        EK::FT dy = (ymax - ymin) / n;
        EK::FT dz = (zmax - zmin) / n;

        build_regular_grid_lcc(*this, n, xmin, ymin, zmin, dx, dy, dz, true);
        cvx_inside_planars.clear();
        cvx_inside_planars.resize(cvxs.size());

        cvx_inside_planars_areas.clear();
        cvx_inside_planars_areas.resize(cvxs.size());

        cvx_partition_poly_index.clear();
        cvx_partition_poly_index.resize(cvxs.size(), -1);

        cvx_partitioned.clear();
        cvx_partitioned.resize(cvxs.size(), false);
        {//每个cell插入 3个平面 然后 split 以上 进行 level-1 次
            IK_to_EK to_ek;
            for (int i=0;i<level-1;++i) {
                for (int cid=0;cid<cvxs.size();++cid) {
                    if (cvx_partitioned[cid])continue;
                    const auto& cvx = cvxs[cid];
                    auto center = ek_cvx_center(cvx);
                    // x 方向
                    EK::Vector_3 nx = EK::Vector_3(1,0,0);
                    EK_Planar px;
                    px.plane = EK::Plane_3(center,nx);
                    px.id = -2;
                    // y 方向
                    EK::Vector_3 ny = EK::Vector_3(0,1,0);
                    EK_Planar py;
                    py.plane = EK::Plane_3(center,ny);
                    py.id = -2;
                    // z方向
                    EK::Vector_3 nz = EK::Vector_3(0,0,1);
                    EK_Planar pz;
                    pz.plane = EK::Plane_3(center,nz);
                    pz.id = -2;

                    cvx_inside_planars[cid].push_back(my_planars.size());
                    my_planars.push_back(px);
                    cvx_inside_planars[cid].push_back(my_planars.size());
                    my_planars.push_back(py);
                    cvx_inside_planars[cid].push_back(my_planars.size());
                    my_planars.push_back(pz);

                    cvx_partition_poly_index[cid] = cvx_inside_planars[cid].size() - 1;
                    volumes.push(cid);
                }
                // partition
                split();
            }
        }


        // exit(0);
        IK_to_EK to_ek;
        for (const auto& planar : input_planars) {
            EK_Planar p;
            p.plane = planar.ek_plane;
            p._3d_Points = planar._ek_3d_Points;
            p.sampled_points = planar.sampled_3d_Points;
            p.area = planar.area;
            p.id = my_planars.size();
            my_planars.push_back(p);
        }

        EK_to_IK to_ik;
        int begin = 0;
        for (int i=0;i<level-1;++i) {
            begin+=n*n*n*7;
            n*=2;
        }
        dx = (xmax - xmin) / n;
        dy = (ymax - ymin) / n;
        dz = (zmax - zmin) / n;
        std::unordered_map<int,int>cid_map;
        for (int cid=0;cid<cvxs.size();++cid) {
            if (cvx_partitioned[cid])continue;
            auto ikp = ek_cvx_center(cvxs[cid]);
            int x = static_cast<int>(std::floor(CGAL::to_double((ikp.x() - xmin) / dx)));
            int y = static_cast<int>(std::floor(CGAL::to_double((ikp.y() - ymin) / dy)));
            int z = static_cast<int>(std::floor(CGAL::to_double((ikp.z() - zmin) / dz)));

            cid_map[x + n * (y + n * z)] = cid;
        }
        auto cid_of = [n,&cid_map](int x, int y, int z) -> int {
            RZ_ASSERT(cid_map.count(x + n * (y + n * z)),"");
            return cid_map[x + n * (y + n * z)];
        };

        for (int pid = 0; pid < static_cast<int>(my_planars.size()); ++pid) {
            const auto& planar = my_planars[pid];

            int minx = n + 1, miny = n + 1, minz = n + 1;
            int maxx = -1,    maxy = -1,    maxz = -1;

            // 先根据 planar 顶点求它覆盖到的 cell index 范围
            for (const auto& p : planar._3d_Points) {
                int x = static_cast<int>(std::floor(CGAL::to_double((p.x() - xmin) / dx)));
                int y = static_cast<int>(std::floor(CGAL::to_double((p.y() - ymin) / dy)));
                int z = static_cast<int>(std::floor(CGAL::to_double((p.z() - zmin) / dz)));

                minx = std::min(minx, x);
                miny = std::min(miny, y);
                minz = std::min(minz, z);
                maxx = std::max(maxx, x);
                maxy = std::max(maxy, y);
                maxz = std::max(maxz, z);
            }

            minx = std::max(minx, 0);
            miny = std::max(miny, 0);
            minz = std::max(minz, 0);
            maxx = std::min(maxx, n - 1);
            maxy = std::min(maxy, n - 1);
            maxz = std::min(maxz, n - 1);

            if (minx > maxx || miny > maxy || minz > maxz) continue;

            for (int x = minx; x <= maxx; ++x) {
                for (int y = miny; y <= maxy; ++y) {
                    for (int z = minz; z <= maxz; ++z) {
                        int cid = cid_of(x, y, z);
                        RZ_ASSERT(cid<cvxs.size(),"");
                        auto verts = vertices_of_cvx(cid);

                        bool has_n = false, has_p = false;
                        for (const auto& v : verts) {
                            RZ_ASSERT(v<points.size(),"");
                            if (planar.plane.has_on_negative_side(points[v])) has_n = true;
                            if (planar.plane.has_on_positive_side(points[v])) has_p = true;
                        }
                        if (!has_n || !has_p) continue;
                        cvx_inside_planars[cid].push_back(pid);
                        RZ_ASSERT(cid<cvx_inside_planars.size(),"");
                        cvx_inside_planars_areas[cid].push_back(0.5 * planar.area);
                        RZ_ASSERT(cid<cvx_inside_planars_areas.size(),"");
                    }
                }
            }
        }

        // 对每个 cvx 内的 planar 按 area 排序
        for (int cid = 0; cid < static_cast<int>(cvxs.size()); ++cid) {

            std::vector<std::pair<Real, int>> sort_buf;
            sort_buf.reserve(cvx_inside_planars[cid].size());
            for (int i = 0; i < static_cast<int>(cvx_inside_planars[cid].size()); ++i) {
                sort_buf.emplace_back(
                    cvx_inside_planars_areas[cid][i],
                    cvx_inside_planars[cid][i]
                );
            }
            for (int i=0;i<static_cast<int>(cvx_inside_planars[cid].size());++i) {
                const auto &largest_poly = my_planars[cvx_inside_planars[cid][i]];

                for (int j=0;j<static_cast<int>(cvx_inside_planars[cid].size());++j) {
                    if (i==j)continue;
                    const auto &planar = my_planars[cvx_inside_planars[cid][j]];
                    bool flag_pos = false;
                    bool flag_neg = false;
                    for (auto& _3d_point : planar._3d_Points)
                    {
                        if (largest_poly.plane.has_on_positive_side(_3d_point))
                            flag_pos = true;
                        else if (largest_poly.plane.has_on_negative_side(_3d_point))
                            flag_neg = true;
                    }
                    if (flag_neg && flag_pos)sort_buf[i].first-=1;
                }
            }
            std::sort(sort_buf.begin(), sort_buf.end());
            cvx_inside_planars[cid].clear();
            cvx_inside_planars_areas[cid].clear();
            for (const auto& pr : sort_buf) {
                cvx_inside_planars[cid].push_back(pr.second);
                cvx_inside_planars_areas[cid].push_back(pr.first);
            }
            if (!cvx_inside_planars[cid].empty()) {
                cvx_partition_poly_index[cid] =
                    static_cast<int>(cvx_inside_planars[cid].size()) - 1;
                volumes.push(cid);
            }

        }
    }

    IK::Point_3 MiniLcc::cvx_center(const std::vector<int>& cvx) const {
        RZ_ASSERT(!cvx.empty(), "cvx is empty");

        std::set<int> vset;
        for (int fid : cvx) {
            for (int eid : faces[fid]) {

                const auto& e = edges[eid];
                vset.insert(e.v1);
                vset.insert(e.v2);
            }
        }

        RZ_ASSERT(!vset.empty(), "cvx has no vertices");

        Real cx = 0.0, cy = 0.0, cz = 0.0;
        for (int vid : vset) {
            const auto& p = points[vid];
            cx += CGAL::to_double(p.x());
            cy += CGAL::to_double(p.y());
            cz += CGAL::to_double(p.z());
        }

        cx /= (Real)vset.size();
        cy /= (Real)vset.size();
        cz /= (Real)vset.size();

        return IK::Point_3(cx, cy, cz);
    }

    EK::Point_3 MiniLcc::ek_cvx_center(const std::vector<int>& cvx) const {
        RZ_ASSERT(!cvx.empty(), "cvx is empty");

        std::set<int> vset;
        for (int fid : cvx) {
            for (int eid : faces[fid]) {

                const auto& e = edges[eid];
                vset.insert(e.v1);
                vset.insert(e.v2);
            }
        }

        RZ_ASSERT(!vset.empty(), "cvx has no vertices");
        std::vector<EK::Point_3>ps;

        for (int vid : vset) {
            const auto& p = points[vid];
            ps.push_back(p);
        }

        return CGAL::centroid(ps.begin(),ps.end());
    }
