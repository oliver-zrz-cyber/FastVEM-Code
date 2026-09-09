#include "bsp_naive.h"
#include <execution>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>
#include <CGAL/centroid.h>
using Iterator = std::list<IK_Triangle_3>::iterator;
using Primitive = CGAL::AABB_triangle_primitive<IK, Iterator>;
using AABB_triangle_traits = CGAL::AABB_traits<IK, Primitive>;
using AABBTree = CGAL::AABB_tree<AABB_triangle_traits>;

static std::atomic<int> g_next_face_lock_id{0};
static std::atomic<int> pending_tasks_{0};
int Split_lcc::allocate_face_lock_id() {
	return g_next_face_lock_id.fetch_add(1, std::memory_order_relaxed);
}
int peek_next_face_lock_id()  {
	return g_next_face_lock_id.load(std::memory_order_relaxed);
}

void Split_lcc::build(const std::vector<Planar> &planars, int n) {
		// Initialize the split and merge functors
		lcc.onsplit_functor<3>() = Polyhedra_Split_functor(lcc);
		lcc.onsplit_functor<2>() = Face_Split_functor(lcc);
		lcc.onmerge_functor<3>() = Polyhedra_Merge_functor(lcc);
		// Set bounds for the grid
		Real xmin = 0.0, ymin = 0.0, zmin = 0.0;
		Real xmax = 1.0, ymax = 1.0, zmax = 1.0;
		// 3D array to store the dart_handle of each cell
		std::vector<std::vector<std::vector<Dart_handle>>> cells(n, std::vector<std::vector<Dart_handle>>(n, std::vector<Dart_handle>(n)));
		// Loop to create an n * n * n grid of cells
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < n; ++j) {
				for (int k = 0; k < n; ++k) {
					// Define the eight corners for the current cell in the 3D grid
					EK::Point_3 p0 = {xmin + i * (xmax - xmin) / n, ymin + j * (ymax - ymin) / n, zmin + k * (zmax - zmin) / n};
					EK::Point_3 p1 = {xmin + (i + 1) * (xmax - xmin) / n, ymin + j * (ymax - ymin) / n, zmin + k * (zmax - zmin) / n};
					EK::Point_3 p2 = {xmin + (i + 1) * (xmax - xmin) / n, ymin + (j + 1) * (ymax - ymin) / n, zmin + k * (zmax - zmin) / n};
					EK::Point_3 p3 = {xmin + i * (xmax - xmin) / n, ymin + (j + 1) * (ymax - ymin) / n, zmin + k * (zmax - zmin) / n};
					EK::Point_3 p4 = {xmin + i * (xmax - xmin) / n, ymin + (j + 1) * (ymax - ymin) / n, zmin + (k + 1) * (zmax - zmin) / n};
					EK::Point_3 p5 = {xmin + i * (xmax - xmin) / n, ymin + j * (ymax - ymin) / n, zmin + (k + 1) * (zmax - zmin) / n};
					EK::Point_3 p6 = {xmin + (i + 1) * (xmax - xmin) / n, ymin + j  * (ymax - ymin) / n, zmin + (k + 1) * (zmax - zmin) / n};
					EK::Point_3 p7 = {xmin + (i + 1) * (xmax - xmin) / n, ymin + (j + 1) * (ymax - ymin) / n, zmin + (k + 1) * (zmax - zmin) / n};

					// Create the cell in LCC
					auto cell_dh = lcc.make_hexahedron(p0, p1, p2, p3, p4, p5, p6, p7);
					lcc.set_attribute<3>(cell_dh, lcc.create_attribute<3>());
					lcc.info<3>(cell_dh).center = lcc.barycenter<3>(cell_dh);

					// For each face, add attributes
					for (auto fdh : collect(lcc.one_dart_per_incident_cell<2, 3>(cell_dh))) {
						lcc.set_attribute<2>(fdh, lcc.create_attribute<2>());
					}

					// Store the dart_handle in the 3D array
					cells[i][j][k] = cell_dh;
					precompute_cell_planes(cell_dh,lcc.info<3>(cell_dh).precomputed_planes);
				}
			}
		}

		// Optionally add ghost cells at the boundary and sew them
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < n; ++j) {
				for (int k = 0; k < n; ++k) {
					Dart_handle cell_dh = cells[i][j][k];

					// Only add ghost cells and sew boundary faces
					if (i == 0 || j == 0 || k == 0 || i == n - 1 || j == n - 1 || k == n - 1) {

						if (i == 0) {
							lcc.info<2>(lcc.beta<2>(cell_dh)).link_to_ghost = true;
						}
						if (j == 0) {
							lcc.info<2>(cell_dh).link_to_ghost = true;
						}
						if (k == 0) {
							lcc.info<2>(lcc.beta<0, 2>(cell_dh)).link_to_ghost = true;
						}
						if (i == n - 1) {
							lcc.info<2>(lcc.beta<1, 1, 2>(cell_dh)).link_to_ghost = true;
						}
						if (j == n - 1) {
							lcc.info<2>(lcc.beta<2, 1, 1, 2>(cell_dh)).link_to_ghost = true;
						}
						if (k == n - 1) {
							lcc.info<2>(lcc.beta<1, 2>(cell_dh)).link_to_ghost = true;
						}
					}
				}
			}
		}
		// // // // Perform sewing between adjacent cells
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < n; ++j) {
				for (int k = 0; k < n; ++k) {
					Dart_handle cell_dh = cells[i][j][k];

					// Sew the corresponding faces between adjacent cells in the x direction (right neighbor)
					if (i < n - 1) {
						// Sew right side of cell (i, j, k) with left side of cell (i + 1, j, k)
						lcc.sew<3>(lcc.beta<1, 1, 2>(cell_dh), lcc.beta<2>(cells[i + 1][j][k]));  // Right with Left
					}
					// // Sew the corresponding faces between adjacent cells in the y direction (top neighbor)
					if (j < n - 1) {
						// Sew back side of cell (i, j, k) with front side of cell (i, j + 1, k)
						lcc.sew<3>(lcc.beta<2, 1, 1, 2>(cell_dh), (cells[i][j + 1][k]));
					}
					// // Sew the corresponding faces between adjacent cells in the z direction (back neighbor)
					if (k < n - 1) {
						lcc.sew<3>(lcc.beta<1,2>(cell_dh), lcc.beta<0,2>(cells[i][j][k + 1]));
					}
				}
			}
		}
		// --- push back planars and sort inside each  ...
		IK_to_EK to_ek;
		for (const auto& planar : planars) {
			EK_Planar p;
			p.plane = planar.ek_plane;
			p._3d_Points = planar._ek_3d_Points;
			p.sampled_points = planar.sampled_3d_Points;
			p.area = planar.area;
			p.id = my_planars.size();
			my_planars.push_back(p);
		}

		EK_to_IK to_ik;
		for (int pid=0;pid<my_planars.size();++pid) {
			int minx=n+1,miny=n+1,minz=n+1;
			int maxx=-1,maxy =-1,maxz =-1;
			for (auto&p:my_planars[pid]._3d_Points) {
				auto ikp = to_ik(p);
				int x = std::floor(ikp.x()*n);
				int y = std::floor(ikp.y()*n);
				int z = std::floor(ikp.z()*n);
				minx = std::min(minx,x);
				miny = std::min(miny,y);
				minz = std::min(minz,z);
				maxx = std::max(x,maxx);
				maxy = std::max(y,maxy);
				maxz = std::max(z,maxz);
			}
			minx = std::max(minx,0);
			miny = std::max(miny,0);
			minz = std::max(minz,0);
			maxx = std::min(maxx,n-1);
			maxy = std::min(maxy,n-1);
			maxz = std::min(maxz,n-1);
			for (int x=minx;x<=maxx;++x)
				for (int y=miny;y<=maxy;++y)
					for (int z=minz;z<=maxz;++z) {
						auto& cell_3 = cells[x][y][z];
						// if (my_planars[pid].sampled_points.size()<10000) {
						// 	Real ratio = planar_intersect_cell_3_fast(my_planars[pid],cell_3);
						// 	if (ratio>1e-3) {
						// 		lcc.info<3>(cell_3).inside_planars.push_back(pid);
						// 		lcc.info<3>(cell_3).inside_planars_areas.push_back(ratio*my_planars[pid].area);
						// 	}
						// }else {
							bool has_n= false,has_p=false;
							auto vertices = collect(lcc.one_dart_per_incident_cell<0, 3>(cell_3));
							for (auto v:vertices) {
								if (my_planars[pid].plane.has_on_negative_side(lcc.point(v)))has_n=true;
								if (my_planars[pid].plane.has_on_positive_side(lcc.point(v)))has_p=true;
							}
							if (!has_n||!has_p) {
								continue;
							}
							lcc.info<3>(cell_3).inside_planars.push_back(pid);
							lcc.info<3>(cell_3).inside_planars_areas.push_back(0.5*my_planars[pid].area);
						// }
					}
		}

		// sort planars inside each cell
		for (auto cell:collect(lcc.one_dart_per_cell<3>())) {
			const auto& planarids = lcc.info<3>(cell).inside_planars;
			std::vector<std::pair<Real,int>> sort_func;
			for (int i=0;i<planarids.size();++i) {
				sort_func.push_back(
					std::make_pair(lcc.info<3>(cell).inside_planars_areas[i],planarids[i]));
			}
			for (int i=0;i<planarids.size();++i) {
			    const auto &largest_poly = my_planars[planarids[i]];

			    for (int j=0;j<planarids.size();++j) {
			        if (i==j)continue;
			        const auto &planar = my_planars[planarids[j]];
			        bool flag_pos = false;
			        bool flag_neg = false;
			        for (auto& _3d_point : planar._3d_Points)
			        {
			            if (largest_poly.plane.has_on_positive_side(_3d_point))
			                flag_pos = true;
			            else if (largest_poly.plane.has_on_negative_side(_3d_point))
			                flag_neg = true;
			        }
			        if (flag_neg && flag_pos)sort_func[i].first-=1;
			    }
			}
			std::sort(sort_func.begin(), sort_func.end());
			lcc.info<3>(cell).inside_planars.clear();
			lcc.info<3>(cell).inside_planars_areas.clear();
			for (auto&pair:sort_func) {
				lcc.info<3>(cell).inside_planars.push_back(pair.second);
			}
		}

		for (auto cell:collect(lcc.one_dart_per_cell<3>())) {
			if (lcc.info<3>(cell).inside_planars.size()) {
				lcc.info<3>(cell).partition_poly_index = lcc.info<3>(cell).inside_planars.size() - 1;
				volumes.push_back(cell);
			}
		}
		// RZ_INFO(volumes.size());
	}

void Split_lcc::clear() {
	// 清空 LCC（删除所有 darts / attributes / cells）
	lcc.clear();

	// 清空几何数据
	my_planars.clear();
	volumes.clear();

	// 重置统计信息
	total_partitions = 0;

	func_time = 0.0f;
	time_p1 = 0.0f;
	time_p2 = 0.0f;

	active_workers.store(0);

	coarse_n = 0;
}

void Split_lcc::split() {
    FUNC_TIMER();
    if (my_planars.size() == 0) {
        lcc.info<3>(volumes.back()).is_inside = true;
        for (auto& cell_3 : collect(lcc.one_dart_per_cell<3>()))
            if (lcc.info<3>(cell_3).is_ghost)
                lcc.info<3>(cell_3).is_inside = false;
        return;
    }

    while (volumes.size())
    {
        auto cell_3 = volumes.back();
        volumes.pop_back();
        partition(cell_3);
    }
}

void Split_lcc::push_parallel_volume(Dart_handle cell_3) {
	PARALLEL_volumes.push(cell_3);
	pending_tasks_.fetch_add(1, std::memory_order_relaxed);
}
void Split_lcc::PARALLEL_split() {
	FUNC_TIMER();
	RZ_INFO("begin");
	//首先是对于所有的初始 fdh 申请 lock id
	for (auto&fdh:collect(lcc.one_dart_per_cell<2>())) {
		lcc.info<2>(fdh).lock_id = allocate_face_lock_id();
		ensure_face_mutex_exists(lcc.info<2>(fdh).lock_id);
	}
	// ---------------------------------------------------------------
	if (my_planars.size() == 0) {
		auto all_cells = collect(lcc.one_dart_per_cell<3>());
		if (!all_cells.empty()) {
			lcc.info<3>(all_cells.front()).is_inside = true;
		}
		for (auto& cell_3 : all_cells) {
			if (lcc.info<3>(cell_3).is_ghost)
				lcc.info<3>(cell_3).is_inside = false;
		}
		return;
	}

	// 清空可能残留的并发队列
	{
		Dart_handle tmp;
		while (PARALLEL_volumes.try_pop(tmp)) {}
	}
	pending_tasks_.store(0, std::memory_order_relaxed);

	// 将当前已有 volumes 中的任务全部搬到并发队列
	while (!volumes.empty()) {
		Dart_handle cell_3 = volumes.back();
		volumes.pop_back();
		PARALLEL_volumes.push(cell_3);
		pending_tasks_.fetch_add(1, std::memory_order_relaxed);
	}

	unsigned int num_threads = std::thread::hardware_concurrency();
	if (num_threads == 0) num_threads = 8;
	RZ_INFO("mutex size: ",face_mutexes_.size());
	std::vector<std::thread> workers;
	workers.reserve(num_threads);
	RZ_INFO(num_threads);
	for (unsigned int tid = 0; tid < num_threads; ++tid) {
		workers.emplace_back([this]() {
			while (true) {
				Dart_handle cell_3;
				if (PARALLEL_volumes.try_pop(cell_3)) {
					PARALLEL_partition(cell_3);  // 这里只是并行调度，暂时不做 face locking
					pending_tasks_.fetch_sub(1, std::memory_order_relaxed);
				}
				else {
					if (pending_tasks_.load(std::memory_order_relaxed) == 0)
						break;
					std::this_thread::yield();
				}
			}
		});
	}
	RZ_INFO("here");
	for (auto& th : workers) {
		th.join();
	}
}

Real Dray(const std::vector <IK::Point_3>& centers, AABBTree& tree, std::vector<IK::Vector_3>& rays)
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
void Split_lcc::label_lcc(std::vector<Planar> &planars, Real inside_thred, int nr) {
	std::list<IK_Triangle_3> alpha_shape;
	for (auto& planar : planars)
		alpha_shape.insert(alpha_shape.end(), planar.label_alpha.begin(), planar.label_alpha.end());

	// rays
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
	EK_to_IK to_ik;
	std::vector<std::vector<IK::Point_3>>cell_3_cs;
	std::vector<int> ids;
	auto vec_vdh = collect(lcc.one_dart_per_cell<3>());
	std::vector<float>in_value(vec_vdh.size());

	for (auto& cell_3 : vec_vdh)lcc.info<3>(cell_3).is_inside = false;

	// {
	// 	for (auto& cell_3 : vec_vdh)lcc.info<3>(cell_3).is_inside = true;
	// 	for (auto& cell_3 : vec_vdh)
	// 		if (cell_volume(lcc,cell_3)<1e-8)
	// 			lcc.info<3>(cell_3).is_inside = false;
	//
	// 	return;
	// }


	for (int k = 0; k < vec_vdh.size(); ++k)
	{
		auto& cell_3 = vec_vdh[k];
		std::vector<IK::Point_3>centers;
		centers.push_back(to_ik(lcc.barycenter<3>(cell_3)));
		cell_3_cs.push_back(centers);
		ids.push_back(ids.size());
	}
	auto tree = AABBTree(alpha_shape.begin(), alpha_shape.end());

	std::cout << cell_3_cs.size() << "cells \n";
	std::for_each(std::execution::par, ids.begin(), ids.end(),// todo: speedup using GPU
		[&](int i) {
		in_value[i] = Dray(cell_3_cs[i], tree, rays);
	}
	);
	for (int k = 0; k < vec_vdh.size(); ++k)
	{
		if (lcc.info<3>(vec_vdh[k]).is_ghost)
		{
			lcc.info<3>(vec_vdh[k]).inside_probability = 0.0;
			lcc.info<3>(vec_vdh[k]).is_inside = false;
			continue;
		}
		lcc.info<3>(vec_vdh[k]).inside_probability = in_value[k];
		if (in_value[k] > inside_thred)
			lcc.info<3>(vec_vdh[k]).is_inside = true;
		else lcc.info<3>(vec_vdh[k]).is_inside = false;
	}
}

bool Split_lcc::save_lcc(std::string filename) {
	//    // 提取顶点和单元（面）
	bool vis = true;
    std::vector<EK::Point_3> vertices;  // 存储顶点
    std::vector<std::vector<std::vector<int>>> cells;  // 每个单元的面连接信息

    // 收集所有唯一的顶点
	auto v0dhs = collect(lcc.one_dart_per_cell<0>());
	for (auto dh : v0dhs)
		lcc.info<0>(dh).id = -1;
	int vertex_index = 0;
	if (!vis) {
		auto cdhs = collect(lcc.one_dart_per_cell<3>());
		for (auto cdh : cdhs) {
			if (lcc.info<3>(cdh).is_ghost) continue;
			if (lcc.info<3>(cdh).is_inside)continue;
			for (auto vdh : collect(lcc.one_dart_per_incident_cell<0, 3>(cdh))) {
				if (lcc.info<0>(vdh).id == -1) {
					lcc.info<0>(vdh).id = vertex_index++;  // 为每个顶点分配唯一ID
					vertices.push_back(lcc.point(vdh));  // 存储顶点
				}
			}
		}
	}
	std::cout<<"here\n";
	//
	// std::cout<<"vertices size: "<<vertices.size()<<std::endl;
	// int vertex_index = 0;
    // 收集多面体单元数据
    for (auto dh : collect(lcc.one_dart_per_cell<3>())) {
        if (lcc.info<3>(dh).is_ghost) continue;  // 跳过ghost单元
    	if (lcc.info<3>(dh).is_inside)continue;
        std::vector<std::vector<int>> faces;

    	for (auto fdh : collect(lcc.one_dart_per_incident_cell<2, 3>(dh))) {
    		auto face_start = fdh;
    		auto face_dart = face_start;
    		std::vector<int> face_connectivity;
    		do {
    			int vertex_id=-1;
    			if (!vis)
    				vertex_id = lcc.info<0>(face_dart).id;
    			else {
    				vertex_id = vertices.size();
    				vertices.push_back(lcc.point(face_dart));
    			}

    			if (vertex_id == -1) {
    				std::cerr<<"vertex_id = -1\n";
    				exit(0);
    			}
    			face_connectivity.push_back(vertex_id);  // 存储每个面的顶点ID
    			face_dart = lcc.beta<1>(face_dart);  // 遍历面上的顶点
    		} while (face_dart != face_start);  // 完成一个面上的顶点遍历
    		faces.push_back(face_connectivity);
    	}
		cells.push_back(faces);
    }
	std::cout<<"here1\n";
   // === Polyhedron 编码 ===
    const int VTK_POLYHEDRON = 42;
    std::vector<int> connectivity;
    for (const auto& cell_faces : cells) {
        for (const auto& face : cell_faces) {
            connectivity.insert(connectivity.end(), face.begin(), face.end());
        }
    }

    // 动态计算偏移量
    std::vector<int> offsets = {0};
    for (const auto& cell_faces : cells) {
        int cell_face_count = 0;
        for (const auto& face : cell_faces) {
            cell_face_count += face.size();  // 累计每个面上的顶点数量
        }
        offsets.push_back(offsets.back() + cell_face_count);  // 每个面有多少顶点，动态计算偏移量
    }

    // 单元类型
    std::vector<int> types(cells.size(), VTK_POLYHEDRON);

    // 编码面信息
    std::vector<int> faces_flat;
    std::vector<int> faceoffsets;
    int current_face_offset = 0;
    for (const auto& cell_faces : cells) {
        std::vector<int> cell_flat = {static_cast<int>(cell_faces.size())};  // 面数
        for (const auto& face : cell_faces) {
            cell_flat.push_back(static_cast<int>(face.size()));  // 每个面的顶点数
            cell_flat.insert(cell_flat.end(), face.begin(), face.end());  // 添加面顶点ID
        }
        faces_flat.insert(faces_flat.end(), cell_flat.begin(), cell_flat.end());
        current_face_offset += cell_flat.size();
        faceoffsets.push_back(current_face_offset);
    }

    // === 写入 .vtu 文件 ===
    std::ofstream out(filename);
    out << "<?xml version=\"1.0\"?>\n";
    out << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    out << "<UnstructuredGrid>\n";
    out << "<Piece NumberOfPoints=\"" << vertices.size() << "\" NumberOfCells=\"" << cells.size() << "\">\n";

    // 写入顶点数据
    out << "<Points>\n";
    out << "<DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
	out << std::setprecision(17);
    for (const auto& p : vertices) {
        out << p.x() << " " << p.y() << " " << p.z() << "\n";
    }
    out << "</DataArray>\n";
    out << "</Points>\n";

    // 写入单元数据
    out << "<Cells>\n";
    out << "<DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
    for (const auto& vertex : connectivity) {
        out << vertex << " ";
    }
    out << "\n</DataArray>\n";

    // 写入偏移量数据
    out << "<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
    for (const auto& offset : offsets) {
        out << offset << " ";
    }
    out << "\n</DataArray>\n";

    // 写入单元类型（VTK_POLYHEDRON）
    out << "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n";
    for (const auto& type : types) {
        out << type << " ";
    }
    out << "\n</DataArray>\n";

    // 写入面数据
    out << "<DataArray type=\"Int32\" Name=\"faces\" format=\"ascii\">\n";
    for (const auto& face : faces_flat) {
        out << face << " ";
    }
    out << "\n</DataArray>\n";

    // 写入面偏移量数据
    out << "<DataArray type=\"Int32\" Name=\"faceoffsets\" format=\"ascii\">\n";
    for (const auto& face_offset : faceoffsets) {
        out << face_offset << " ";
    }
    out << "\n</DataArray>\n";

    out << "</Cells>\n";

    // 添加 CellData，用于可视化（Cell ID）
    out << "<CellData Scalars=\"cell_id\">\n";
    out << "<DataArray type=\"Int32\" Name=\"cell_id\" format=\"ascii\">\n";
    for (int i = 0; i < cells.size(); ++i) {
        out << i << " ";
    }
    out << "\n</DataArray>\n";
    out << "</CellData>\n";

    out << "</Piece>\n";
    out << "</UnstructuredGrid>\n";
    out << "</VTKFile>\n";

    out.close();
    std::cout << "✅ VTU 文件已保存: " << filename << std::endl;
}

void Split_lcc::partition(Dart_handle &cell_3) {
    auto start_time = clock();
	// ++total_partitions;
	// std::cout<<total_partitions<<"\n";

	const int tree_id = lcc.info<3>(cell_3).tree_id;
	const int tree_node_id = lcc.info<3>(cell_3).tree_node_id;

	if (lcc.info<3>(cell_3).inside_planars.size()==0 || lcc.info<3>(cell_3).partition_poly_index<0)return;
	auto polys = std::move(lcc.info<3>(cell_3).inside_planars);
	int pt_id = lcc.info<3>(cell_3).partition_poly_index;
	int deleted_polyid = polys[pt_id];

	auto largest_poly = my_planars[polys[pt_id]];
	polys.erase(polys.begin()+pt_id);
	bool has_n= false,has_p=false;
	auto vertices = collect(lcc.one_dart_per_incident_cell<0, 3>(cell_3));
	for (auto v:vertices) {
		if (largest_poly.plane.has_on_negative_side(lcc.point(v)))has_n=true;
		if (largest_poly.plane.has_on_positive_side(lcc.point(v)))has_p=true;
	}

	if (!has_n||!has_p) {
		lcc.info<3>(cell_3).partition_poly_index= lcc.info<3>(cell_3).partition_poly_index - 1;
		lcc.info<3>(cell_3).inside_planars = std::move(polys);
		if(lcc.info<3>(cell_3).inside_planars.size())
			volumes.push_back(cell_3);
		return;
	}
	std::vector<Dart_handle> inserted_pdhs;
	std::vector<Dart_handle> inserted_fdhs;
	zrz_LCC_3::Dart_handle new_fdh = lcc.null_dart_handle;
	auto faces = collect(lcc.one_dart_per_incident_cell<2, 3>(cell_3));
	bool inserted = 0;
	for (auto fdh : faces)
	{
		bool cut_edge = false;
		bool link_to_ghost = lcc.info<2>(fdh).link_to_ghost;
		auto new_dhs = std::vector<zrz_LCC_3::Dart_handle>{};
		auto pdhs = collect(lcc.darts_of_orbit<1>(fdh)); //one side
		RZ_ASSERT(pdhs.size() >= 3,"pdhs.size() >= 3");
		for (auto iter : pdhs) {
			EK::Point_3 p = lcc.point(iter);
			if (largest_poly.plane.has_on(p))
				new_dhs.push_back({ iter });
		}
		if (new_dhs.size() == 2) {
			if (!(lcc.next(new_dhs[0]) == new_dhs[1] || lcc.next(new_dhs[1]) == new_dhs[0])) {
				new_fdh = lcc.insert_cell_1_in_cell_2(new_dhs[0], new_dhs[1]);
				inserted_fdhs.push_back(new_fdh);
				if (fdh_area(lcc,new_fdh)<1e-10||fdh_area(lcc,lcc.beta<2>(new_fdh))<1e-10) {
					for (auto new_fdh:inserted_fdhs)
						lcc.remove_cell<1>(new_fdh);
					for (auto new_pdh:inserted_pdhs)
						lcc.remove_cell<0>(new_pdh);
					lcc.info<3>(cell_3).partition_poly_index= lcc.info<3>(cell_3).partition_poly_index - 1;
					polys.insert(polys.begin() + pt_id, deleted_polyid);
					lcc.info<3>(cell_3).inside_planars = std::move(polys);
					if(lcc.info<3>(cell_3).inside_planars.size())
						volumes.push_back(cell_3);
					return;
				}
				lcc.info<2>(new_fdh).link_to_ghost = link_to_ghost;
				lcc.info<2>(lcc.beta<2>(new_fdh)).link_to_ghost = link_to_ghost;
				inserted = true;
				RZ_ASSERT(new_fdh != lcc.null_dart_handle,"cut_edge1");
			}
			else {
				if (lcc.next(new_dhs[1]) == new_dhs[0])new_fdh = new_dhs[1];
				else new_fdh = new_dhs[0];
				RZ_ASSERT(new_fdh != lcc.null_dart_handle,"cut_edge2");
			}
			continue;
		}
		if (new_dhs.size() > 2) {// 这里需要找一个dart 使得其next也在平面上

			// find two
			for (int i=0;i<new_dhs.size();++i) {
				for (int j=0;j<new_dhs.size();++j) {
					if (i==j)continue;
					if (lcc.next(new_dhs[i]) == new_dhs[j]) {
						new_fdh = new_dhs[i];
						break;
					}
					if (lcc.next(new_dhs[j]) == new_dhs[i]) {
						new_fdh = new_dhs[j];
						break;
					}
				}
			}
			continue;
		}
		int inserted_cnt = 0;
		for (auto pdh : pdhs)
		{
			EK::Point_3 point1 = lcc.point(pdh);
			EK::Point_3 point2 = lcc.point(lcc.next(pdh));
			if (largest_poly.plane.has_on(point1)) continue;
			if (largest_poly.plane.has_on(point2)) continue;
			if (largest_poly.plane.has_on_positive_side(point1) == largest_poly.plane.has_on_positive_side(point2))
				continue;
			auto edge = EK::Segment_3(point1, point2);
			if (auto result = CGAL::intersection(edge, largest_poly.plane))
			{
				if (auto* inters_point = boost::get<EK::Point_3>(&*result)) {
					inserted = true;
					++inserted_cnt;
					auto new_pdh = lcc.insert_point_in_cell<1>(pdh, *inters_point);
					new_dhs.push_back(new_pdh);
					inserted_pdhs.push_back(new_pdh);
					if (CGAL::squared_distance(*inters_point,point1)<1e-18
						||CGAL::squared_distance(*inters_point,point2)<1e-18) {
						for (auto new_fdh:inserted_fdhs)
							lcc.remove_cell<1>(new_fdh);
						for (auto new_pdh:inserted_pdhs)
							lcc.remove_cell<0>(new_pdh);
						lcc.info<3>(cell_3).partition_poly_index= lcc.info<3>(cell_3).partition_poly_index - 1;
						polys.insert(polys.begin() + pt_id, deleted_polyid);
						lcc.info<3>(cell_3).inside_planars = std::move(polys);
						if(lcc.info<3>(cell_3).inside_planars.size())
							volumes.push_back(cell_3);
						return;
					}
				}
			}
		}
		if (new_dhs.size() < 2) {
			RZ_ASSERT(inserted_cnt==0,inserted_cnt);
			continue;
		}

		if (!cut_edge&&!new_dhs.empty()) {
			RZ_ASSERT(new_dhs.size() == 2,new_dhs.size());
			RZ_ASSERT(new_dhs[0] != lcc.null_dart_handle,"");
			RZ_ASSERT(new_dhs[1] != lcc.null_dart_handle,"");
			new_fdh = lcc.insert_cell_1_in_cell_2(new_dhs[0], new_dhs[1]);
			inserted_fdhs.push_back(new_fdh);
			if (fdh_area(lcc,new_fdh)<1e-10||fdh_area(lcc,lcc.beta<2>(new_fdh))<1e-10) {
				for (auto new_fdh:inserted_fdhs)
					lcc.remove_cell<1>(new_fdh);
				for (auto new_pdh:inserted_pdhs)
					lcc.remove_cell<0>(new_pdh);
				lcc.info<3>(cell_3).partition_poly_index= lcc.info<3>(cell_3).partition_poly_index - 1;
				polys.insert(polys.begin() + pt_id, deleted_polyid);
				lcc.info<3>(cell_3).inside_planars = std::move(polys);
				if(lcc.info<3>(cell_3).inside_planars.size())
					volumes.push_back(cell_3);
				return;
			}
			lcc.info<2>(new_fdh).link_to_ghost = link_to_ghost;
			lcc.info<2>(lcc.beta<2>(new_fdh)).link_to_ghost = link_to_ghost;
			RZ_ASSERT(new_fdh != lcc.null_dart_handle,"!cut_edge");
		}
	}

	// split volume
	RZ_ASSERT(new_fdh != lcc.null_dart_handle,"new_fdh == lcc.null_dart_handle");

	std::vector< zrz_LCC_3::Dart_handle > new_face_path;
	auto next_dh = new_fdh;
	do {
		// all face contain lcc.next
		auto out_dhs = collect(lcc.darts_of_cell<0, 2>(lcc.next(next_dh)));
		for (auto Dh : out_dhs)
		{
			if (lcc.beta<2>(Dh) == next_dh) continue;
			if (largest_poly.plane.has_on(lcc.point(lcc.next(Dh))))
			{
				new_face_path.push_back(Dh);
				break;
			}
		}
		RZ_ASSERT(next_dh != new_face_path.back(),"");
		next_dh = new_face_path.back();
	} while (next_dh != new_fdh);

	RZ_ASSERT(new_face_path.size() >= 3,"");
	if (!lcc.is_insertable_cell_2_in_cell_3(new_face_path.begin(), new_face_path.end()))
	{
		std::cout << "fatal error\n";
		return;
	}

	auto new_volume = lcc.insert_cell_2_in_cell_3(new_face_path.begin(), new_face_path.end());

	if (cell_volume(lcc,new_volume)<1e-10||cell_volume(lcc,lcc.beta<3>(new_volume))<1e-10) {
		lcc.remove_cell<2>(new_volume);
		for (auto new_fdh:inserted_fdhs)
			lcc.remove_cell<1>(new_fdh);
		for (auto new_pdh:inserted_pdhs)
			lcc.remove_cell<0>(new_pdh);
		lcc.info<3>(cell_3).partition_poly_index= lcc.info<3>(cell_3).partition_poly_index - 1;
		polys.insert(polys.begin() + pt_id, deleted_polyid);
		lcc.info<3>(cell_3).inside_planars = std::move(polys);
		if(lcc.info<3>(cell_3).inside_planars.size())
			volumes.push_back(cell_3);
		return;
	}


	lcc.set_attribute<2>(new_volume, lcc.create_attribute<2>());
	lcc.info<2>(new_volume).planar_id = largest_poly.id;
	// lcc.info<2>(new_volume).area = area_of_fdh(lcc, new_volume, largest_poly.plane);
	lcc.set_attribute<3>(new_volume, lcc.create_attribute<3>());
	lcc.set_attribute<3>(lcc.beta<3>(new_volume), lcc.create_attribute<3>());

	auto center_1 = lcc.barycenter<3>(new_volume);
	if (!largest_poly.plane.has_on_positive_side(center_1))
		new_volume = lcc.beta<3>(new_volume);

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
			lcc.info<3>(new_volume).inside_planars.push_back(planar_id);
		if(flag_neg)
			lcc.info<3>(lcc.beta<3>(new_volume)).inside_planars.push_back(planar_id);
	}

	lcc.info<3>(new_volume).partition_poly_index
		= lcc.info<3>(new_volume).inside_planars.size() - 1;
	lcc.info<3>(lcc.beta<3>(new_volume)).partition_poly_index
		= lcc.info<3>(lcc.beta<3>(new_volume)).inside_planars.size() - 1;

	if (lcc.info<3>(new_volume).inside_planars.size() > 0)
		volumes.push_back(new_volume);
	if (lcc.info<3>(lcc.beta<3>(new_volume)).inside_planars.size() > 0)
		volumes.push_back(lcc.beta<3>(new_volume));

	auto end_time = clock();
	func_time += (end_time - start_time) / 1000.0;
}
void Split_lcc::ensure_face_mutex_exists(int lock_id) {
	if (lock_id < (int)face_mutexes_.size()) return;

	std::lock_guard<std::mutex> guard(face_mutexes_resize_mutex_);
	while (lock_id >= (int)face_mutexes_.size()) {
		face_mutexes_.emplace_back();
	}
}

void Split_lcc::lock_face_locks(const std::vector<int> &input_ids) {
	// 1. 合法性检查
	for (int id : input_ids) {
		RZ_ASSERT(id >= 0, "invalid face lock_id");
	}

	// 2. 按统一顺序加锁：升序
	for (int id : input_ids) {
		RZ_ASSERT(id<face_mutexes_.size(),"here!");
		face_mutexes_[id].lock();
	}
}

void Split_lcc::unlock_face_locks(const std::vector<int> &locked_ids) {
	for (auto it = locked_ids.rbegin(); it != locked_ids.rend(); ++it) {
		int id = *it;
		RZ_ASSERT(id >= 0, "invalid face lock_id when unlocking");
		RZ_ASSERT(id<face_mutexes_.size(),"here!");
		face_mutexes_[id].unlock();
	}
}
std::vector<int> Split_lcc::collect_incident_face_lock_ids(Dart_handle cell_3) {
	std::vector<int> ids;
	const auto& faces =
		collect(lcc.one_dart_per_incident_cell<2, 3>(cell_3));
	ids.reserve(faces.size());

	for (auto fdh : faces) {
		int id = lcc.info<2>(fdh).lock_id;
		RZ_ASSERT(id >= 0, "face lock_id is not initialized");
		ids.push_back(id);
	}

	std::sort(ids.begin(), ids.end());
	ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
	return ids;
}
void Split_lcc::PARALLEL_partition(Dart_handle &cell_3) {
	// 首先是得到所有的锁
	RZ_INFO("1");
	std::vector<int> to_lock;
	{
		// std::lock_guard<std::mutex> g(global_lcc_mutex);
		to_lock = collect_incident_face_lock_ids(cell_3);
	}
	RZ_INFO("2");
	lock_face_locks(to_lock);
	RZ_INFO("3");
	// 然后是 划分

	// 然后是释放所有的锁
	unlock_face_locks(to_lock);
	RZ_INFO("4");
}


void Split_lcc::set_ats(Dart_handle cell_3)
	{
		lcc.set_attribute<3>(cell_3, lcc.create_attribute<3>());
		for(auto&cell_2:collect(lcc.one_dart_per_incident_cell<2,3>(cell_3)))
			lcc.set_attribute<2>(cell_2, lcc.create_attribute<2>());
		for (auto& cell_1 : collect(lcc.one_dart_per_incident_cell<1, 3>(cell_3)))
			lcc.set_attribute<1>(cell_1, lcc.create_attribute<1>());
	}



	void Split_lcc::precompute_cell_planes(
		const Dart_handle cell_3,
		std::vector<CellFacePlane>& planes)
	{
		EK_to_IK to_ik;
		planes.clear();

		IK::Point_3 cell_center = to_ik(lcc.barycenter<3>(cell_3));

		for (auto fdh : collect(lcc.one_dart_per_incident_cell<2,3>(cell_3))) {

			std::vector<IK::Point_3> face_pts;
			for (auto dh : collect(lcc.darts_of_orbit<1>(fdh))) {
				face_pts.push_back(to_ik(lcc.point(dh)));
			}
			if (face_pts.size() < 3) continue;

			IK::Point_3 p0, p1, p2;
			if (!get_three_non_collinear_points(face_pts, p0, p1, p2))
				continue;

			IK::Vector_3 n = CGAL::cross_product(p1 - p0, p2 - p0);

			// 保证法向朝外
			if ((cell_center - p0) * n > 0) {
				n = -n;
			}

			planes.push_back({p0, n});
		}
	}

	Real Split_lcc::cell_volume(zrz_LCC_3& lcc, Dart_handle dh)
	{
		EK::Point_3 center = lcc.barycenter<3>(dh);
		EK::FT volume = 0;
		for (auto fdh = lcc.one_dart_per_incident_cell<2, 3>(dh).begin();
			fdh != lcc.one_dart_per_incident_cell<2, 3>(dh).end();++fdh) {
			auto pdhs = collect(lcc.darts_of_orbit<1>(fdh));
			auto size = pdhs.size();
			auto p0 = lcc.point(pdhs[0]);
			for (int i = 1; i < size - 1; i++) {
				auto p1 = lcc.point(pdhs[i]);
				auto p2 = lcc.point(pdhs[i + 1]);
				volume += CGAL::volume(p0, p1, p2, center);
			}
			}
		return CGAL::to_double(volume);
	}
	Real Split_lcc::fdh_area(zrz_LCC_3& lcc, Dart_handle dh) {
		Real area = 0;
		EK_to_IK to_ik;
		std::vector<IK::Point_3>points;
		for (auto&pdh:collect(lcc.darts_of_orbit<1>(dh))) {
			points.push_back(to_ik(lcc.point(pdh)));
		}
		IK::Point_3 centroid = CGAL::centroid(points.begin(), points.end());
		RZ_ASSERT(points.size()>2,"");
		for (int k=0;k<points.size();++k) {
			IK_Triangle_3 tri(points[k],points[(k+1)%points.size()],centroid);
			area += std::sqrt(tri.squared_area());
		}
		return area;
	}

	bool Split_lcc::get_three_non_collinear_points(
		const std::vector<IK::Point_3> &pts,
		IK::Point_3 &p0, IK::Point_3 &p1, IK::Point_3 &p2,
		Real eps
	) {
		const int n = pts.size();
		if (n < 3) return false;

		p0 = pts[0];
		for (int i = 1; i < n; ++i) {
			for (int j = i + 1; j < n; ++j) {
				IK::Vector_3 v1 = pts[i] - p0;
				IK::Vector_3 v2 = pts[j] - p0;
				IK::Vector_3 cp = CGAL::cross_product(v1, v2);
				if (cp.squared_length() > eps) {
					p1 = pts[i];
					p2 = pts[j];
					return true;
				}
			}
		}
		return false;
	}

	bool Split_lcc::point_in_cell_3(const IK::Point_3 &p, const Dart_handle cell_3) {
		const Real eps = 1e-12;
		EK_to_IK to_ik;
		// cell barycenter 用于判断法向朝向（inside / outside）
		IK::Point_3 cell_center = to_ik(lcc.barycenter<3>(cell_3));

		// 遍历 cell 的所有面
		for (auto fdh : collect(lcc.one_dart_per_incident_cell<2,3>(cell_3))) {

			// 收集该面的所有点
			std::vector<IK::Point_3> face_pts;
			for (auto dh : collect(lcc.darts_of_orbit<1>(fdh))) {
				face_pts.push_back(to_ik(lcc.point(dh)));
			}

			RZ_ASSERT(face_pts.size() >= 3, "Face has < 3 vertices");

			// 找 3 个不共线点
			IK::Point_3 p0, p1, p2;
			bool ok = get_three_non_collinear_points(face_pts, p0, p1, p2);
			if (!ok)continue;

			// 构造平面
			IK::Vector_3 n = CGAL::cross_product(p1 - p0, p2 - p0);

			// 确定法向朝向：必须指向 cell 外
			if ((cell_center - p0) * n > 0) {
				n = -n;
			}

			// 半空间测试
			Real side = (p - p0) * n;

			// 如果在外侧，则不在 cell 内
			if (side > eps) {
				return false;
			}
		}
		return true;
	}

	bool Split_lcc::point_in_cell_fast(
	const IK::Point_3& p,
	const std::vector<CellFacePlane>& planes)
	{
		const Real eps = 1e-9;

		for (const auto& fp : planes) {
			if ((p - fp.p0) * fp.n > eps)
				return false;
		}
		return true;
	}

	Real Split_lcc::planar_intersect_cell_3_fast(
		const EK_Planar& planar,
		const Dart_handle cell_3)
	{
		const int N = (int)planar.sampled_points.size();
		// CoLOD_ASSERT(N > 50, N);

		const auto& planes = lcc.info<3>(cell_3).precomputed_planes;
		RZ_ASSERT(planes.size() > 4, planes.size());

		int inside_cnt = 0;

#pragma omp parallel for reduction(+:inside_cnt) schedule(static)
		for (int i = 0; i < N; ++i) {
			if (point_in_cell_fast(planar.sampled_points[i], planes)) {
				inside_cnt += 1;
			}
		}

		return Real(inside_cnt) / Real(N);
	}

	Real Split_lcc::planar_intersect_cell_3(const EK_Planar&planar,const Dart_handle cell_3) {//todo: speed up
		RZ_ASSERT(planar.sampled_points.size()>=30,planar.sampled_points.size());
		std::vector<bool> is_in(planar.sampled_points.size(),false);
		for (int i=0;i<planar.sampled_points.size();++i) {
			if (point_in_cell_3(planar.sampled_points[i],cell_3)) {
				is_in[i] = true;
			}
		}
		Real ans = 0.0;
		for (int i=0;i<is_in.size();++i) if (is_in[i]) ans += 1.0;
		ans /= is_in.size();
		return ans;
	}