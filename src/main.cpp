#include <filesystem>
#include <iostream>
#include <string>
#include "sim.h"
#include "sim3d.h"
#include "liquidSim3d.h"
#include "convergence.h"

// in flip or pic only value on dofs are really valid!
int main(int argc,char* argv[]) {
    constexpr int legacy_argument_count = 10;
    if (argc < legacy_argument_count + 1) {
        std::cerr << "Usage: VEMFLUID <10 legacy arguments> "
                  << "[--solver gmg|amg] [--resolution N]\n";
        return 2;
    }

    std::string solver = "gmg";
    int resolution = 64;
    for (int i = legacy_argument_count + 1; i < argc; i += 2) {
        if (i + 1 >= argc) {
            std::cerr << "Missing value for " << argv[i] << '\n';
            return 2;
        }
        const std::string option = argv[i];
        if (option == "--solver") {
            solver = argv[i + 1];
        } else if (option == "--resolution") {
            try {
                resolution = std::stoi(argv[i + 1]);
            } catch (const std::exception&) {
                std::cerr << "Invalid resolution: " << argv[i + 1] << '\n';
                return 2;
            }
        } else {
            std::cerr << "Unknown option: " << option << '\n';
            return 2;
        }
    }
    if (resolution <= 0 || (solver != "gmg" && solver != "amg")) {
        std::cerr << "Solver must be gmg or amg and resolution must be positive.\n";
        return 2;
    }
    RGB_Screen screen;
    screen.width = 900;
    screen.height = 900;
    screen.pixels.resize(screen.width * screen.height * 3, 0); // 初始化为黑色
    // render_loop(screen);
    // if (argc<2)RZ_FATAL("you need a domain!0____0");
    // render_loop_vem(screen,argv[1]);
    // For debug ------------------------------------------------------------
    // VEM_Domain vem_domain;
    // vem_domain.read_from_file(argv[1]);
    // vem_domain.init();
    // vem_domain.generate_elements();
    // vem_domain.assemble_global_stiffness_matrix();
    // vem_domain.assemble_global_rhs();
    // vem_domain.solve();
    // vem_domain.assemble_global_stiffness_matrix_k1();
    // vem_domain.assemble_global_rhs_k1();
    // vem_domain.solve_k1();
    // // vem_domain.assemble_weakform_rhs();
    // vem_domain.save_solution_png("/home/lhw/桌面/VortexCoupling/PIC/RzPIC/outputs/solution2.png");

    //-----------------------------------------3d-----------------
    // render_loop_3d(screen);
    std::filesystem::path p(argv[2]);
    std::filesystem::path folder(argv[7]);
    sphere_source source(std::stod(argv[3]),std::stod(argv[4]),
    std::stod(argv[5]),std::stod(argv[6]));
    std::cout<< folder.string() << " " << p.stem().string();
    
    if (solver == "amg") {
        const Real kernel_radius = 1.0 / static_cast<Real>(resolution);
        render_loop_3d_VEM(argv[10], folder.string(), p.stem().string(), source,
            std::stoi(argv[8]), kernel_radius, resolution, argv[9]);
    } else {
        // The hierarchy starts at 8 and doubles at each level.
        int init_n = 8;
        int level = 1;
        int finest_resolution = init_n;
        while (finest_resolution < resolution) {
            finest_resolution *= 2;
            ++level;
        }
        if (finest_resolution != resolution) {
            std::cerr << "GMG resolution must be 8, 16, 32, 64, ...; got "
                      << resolution << '\n';
            return 2;
        }
        render_loop_3d_VEM(argv[10], folder.string(), p.stem().string(), source,
            std::stoi(argv[8]), init_n, level, argv[9]);
    }



    return 0;
}
