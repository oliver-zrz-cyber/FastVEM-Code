#pragma once
#include "dfn.h"

void render_loop_3d(RGB_Screen& screen);

void render_loop_3d_VEM(RGB_Screen &screen,std::string domain,std::string save_name);


void render_loop_3d_VEM(RGB_Screen &screen,std::string domain,std::string save_path,
    const std::string save_name,
    const sphere_source& source,
    int FULL_CFL = 2,Real kernel_radius = 1/200.0,std::string aabb_path="");

void render_loop_3d_VEM(std::string mesh_file,std::string save_path,
    const std::string save_name,
    const sphere_source& source,
    int FULL_CFL = 2,Real kernel_radius = 1/200.0,int n= 80, std::string aabb_path="");




void render_loop_3d_VEM(std::string mesh_file,std::string save_path,
                        const std::string save_name,
                        const sphere_source& source,
                        int FULL_CFL = 2,int init_n=6, int level=4, std::string aabb_path="");




// void render_loop_3d_VEM(RGB_Screen &screen,std::string domain,std::string save_name,
//     const std::string save_path,
//     const sphere_source& source,
//     int FULL_CFL = 2,Real kernel_radius = 1/200.0,std::string aabb_path="");



void render_loop_3d_VEM(RGB_Screen &screen,std::vector<std::string> domains);

