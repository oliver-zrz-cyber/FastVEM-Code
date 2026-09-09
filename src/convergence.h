#pragma once
#include <string>
#include "mac.h"
#include "vem2d.h"
class MAC_DB {
    MAC_Grid grid;
public:
    void convergence_test_simple(const std::string& case_name = "all",
                                 const std::string& geometry_name = "all");
    void convergence_test_ghost_fluid(const std::string& case_name = "all",
                                      const std::string& geometry_name = "all");
};


class VEM_DB {
    VEM_Domain vem_grid;
public:
    void convergence_test_simple(const std::string& case_name = "all",
                                 const std::string& geometry_name = "all");
    void convergence_test_simple_points_based_boundary(const std::string& case_name = "all",
                                                       const std::string& geometry_name = "all");
    void convergence_test_fraction_fluid(const std::string& case_name = "all",
                                         const std::string& geometry_name = "all");
    void convergence_test_ghost_fluid(const std::string& case_name = "all",
                                      const std::string& geometry_name = "all");
    void convergence_test_Nitsche_fluid();
};
