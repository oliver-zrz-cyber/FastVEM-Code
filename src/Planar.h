#pragma once
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Polygon_2.h>
#include <random>

#include "config.h"

using IK = CGAL::Exact_predicates_inexact_constructions_kernel;
using EK = CGAL::Exact_predicates_exact_constructions_kernel;
using IK_Plane_3 = IK::Plane_3;
using IK_PWN = std::pair<IK::Point_3, IK::Vector_3>;
using IK_Polygon_2 = CGAL::Polygon_2<IK>;
using IK_Triangle_3 = CGAL::Triangle_3<IK>;
struct Planar
{
    IK_Plane_3 plane{};
    EK::Plane_3 ek_plane;
    std::vector<IK_PWN> _3d_Pwns;
    std::vector<IK::Point_3> _3d_Points;
    std::vector<IK::Point_3> sampled_3d_Points;
    std::vector<EK::Point_3> _ek_3d_Points;
    std::vector<IK::Point_2> _2d_Points;
    IK_Polygon_2 _convex_2d;


    Real area = 0.0;
    Real score = 0.0;

    std::list<IK_Triangle_3> save_alpha;
    std::list<IK_Triangle_3> label_alpha;


    Real cost = 0.0;
    CGAL::IO::Color color;
    int id = -1;
    int decomposed_id = -1;
    int group_id = -1;
    bool is_checked_for_merge = false;
    bool is_lod_0 = false;
    bool operator<(const Planar& other)const {
        return area < other.area;
    }
    void sample_points_on_label_alpha() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<Real> dis(0.0, 1.0);
        for (auto tri : label_alpha)
        {
            auto area = std::sqrt(tri.squared_area());
            int sample_num = std::ceil(area/1e-8);
            sample_num = std::max(sample_num,30);
            // sample_num = std::min(10000, sample_num);
            for (int round = 0; round < sample_num; round++)
            {
                float i = dis(gen), j = dis(gen), k = dis(gen);
                float eps = 1e-6;
                float sum = i + j + k + eps;
                i /= sum, j /= sum, k /= sum;
                sampled_3d_Points.push_back(IK::Point_3(i * tri[0].x() + j * tri[1].x() + k * tri[2].x(),
                    i * tri[0].y() + j * tri[1].y() + k * tri[2].y(),
                    i * tri[0].z() + j * tri[1].z() + k * tri[2].z()));
            }
        }

    }
};

struct EK_Planar {
    int id = -1;
    EK::Plane_3 plane;
    std::vector<EK::Point_3>_3d_Points;
    std::vector<IK::Point_3> sampled_points;
    Real area = -1.0;
    int min_lod = -1;
};
