// #include "PCG.h"
// #include <Eigen/Sparse>
// #include <iostream>
// #include <vector>
//
// int main() {
//     constexpr int n = 1000;
//
//     // 1. 构建 3 对角 SPD 矩阵
//     Eigen::SparseMatrix<double, Eigen::RowMajor> A(n, n);
//     std::vector<Eigen::Triplet<double>> T;
//     T.reserve(3 * n);
//
//     for (int i = 0; i < n; ++i) {
//         T.emplace_back(i, i, 2.0);
//         if (i > 0)     T.emplace_back(i, i - 1, -1.0);
//         if (i < n - 1) T.emplace_back(i, i + 1, -1.0);
//     }
//
//     A.setFromTriplets(T.begin(), T.end());
//     A.makeCompressed();
//
//     // 2. RHS 和初值
//     std::vector<double> rhs(n, 1.1);
//     std::vector<double> x(n, 0.0);
//
//     // 3. 构建 PCG
//     PCG pcg;
//     PCG::params prm;
//     prm.maxiter = 1000;
//     prm.tol     = 1e-5;
//     prm.verbose = true;
//     pcg.build(A,prm);
//
//
//
//
//     // 4. Identity preconditioner
//     IdentityPrecond P;
//
//     // 5. Solve
//     auto [iters, relres] = pcg(P, rhs.data(), x.data());
//
//     std::cout << "CG finished in " << iters
//               << " iterations, relres = " << relres << std::endl;
//
//     // 6. Compute and output residual r = rhs - A * x
//     Eigen::VectorXd x_eig(n), rhs_eig(n);
//     for (int i = 0; i < n; ++i) {
//         x_eig[i]   = x[i];
//         rhs_eig[i] = rhs[i];
//     }
//
//     Eigen::VectorXd r = rhs_eig - A * x_eig;
//     std::cout << "||rhs||_2 = " << rhs_eig.norm() << std::endl;
//     std::cout << "||r||_2 = " << r.norm() << std::endl;
//     std::cout << "||r||_2 / ||rhs||_2 = "
//               << r.norm() / rhs_eig.norm() << std::endl;
//
//
//     // // ================= Eigen CG for comparison =================
//     // Eigen::ConjugateGradient<
//     //     Eigen::SparseMatrix<double, Eigen::RowMajor>,
//     //     Eigen::Lower | Eigen::Upper
//     // > eigen_cg;
//     //
//     // eigen_cg.setMaxIterations(450);
//     // eigen_cg.setTolerance(prm.tol);
//     // Eigen::VectorXd x_eigen(n);
//     // Eigen::VectorXd rhs_vec(n);
//     // for (int i = 0; i < n; ++i) {
//     //     rhs_vec[i] = rhs[i];
//     // }
//     //
//     // x_eigen.setZero();
//     //
//     // // Solve
//     // eigen_cg.compute(A);
//     // x_eigen = eigen_cg.solve(rhs_vec);
//     //
//     // // Residual
//     // Eigen::VectorXd r_eigen = rhs_vec - A * x_eigen;
//     //
//     // // Output Eigen CG info
//     // std::cout << "\n===== Eigen ConjugateGradient =====" << std::endl;
//     // std::cout << "Eigen CG iterations = "
//     //           << eigen_cg.iterations() << std::endl;
//     // std::cout << "Eigen CG reported relres = "
//     //           << eigen_cg.error() << std::endl;
//     // std::cout << "Eigen CG true ||r||_2 = "
//     //           << r_eigen.norm() << std::endl;
//     // std::cout << "Eigen CG true ||r||_2 / ||rhs||_2 = "
//     //           << r_eigen.norm() / rhs_vec.norm() << std::endl;
//
//     return 0;
// }
