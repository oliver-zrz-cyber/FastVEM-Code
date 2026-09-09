#include "Visualization.h"
#include <vtkSmartPointer.h>
#include <vtkImageData.h>
#include <vtkDoubleArray.h>
#include <vtkPointData.h>
#include <vtkXMLImageDataWriter.h>
#include <string>
#include "stb_image_write.h"
#include <set>
// 保存压力场到 .vti 文件
void save_pressure_vti(const int nx,
                        const int ny,
                        const int nz,
                        const Real dx,
                       const Eigen::VectorXr& pressure,
                       const std::string& filename)
{

    vtkSmartPointer<vtkImageData> image = vtkSmartPointer<vtkImageData>::New();
    image->SetDimensions(nx, ny, nz);
    image->SetSpacing(dx, dx, dx);
    image->SetOrigin(0.0, 0.0, 0.0);

    vtkSmartPointer<vtkDoubleArray> pArray = vtkSmartPointer<vtkDoubleArray>::New();
    pArray->SetName("pressure");
    pArray->SetNumberOfComponents(1);
    pArray->SetNumberOfTuples(nx * ny * nz);

    // 将 Eigen::VectorXr 拷贝到 VTK array
    for (vtkIdType i = 0; i < pressure.size(); ++i) {
        pArray->SetValue(i, pressure[i]);
    }

    image->GetPointData()->SetScalars(pArray);

    vtkSmartPointer<vtkXMLImageDataWriter> writer =
        vtkSmartPointer<vtkXMLImageDataWriter>::New();
    writer->SetFileName(filename.c_str());
    writer->SetInputData(image);
    writer->Write();
}
bool isSymmetric(const Eigen::SparseMatrixR& A) {

    Eigen::SparseMatrixR A_transpose = A.transpose();
    Eigen::SparseMatrixR diff = A - A_transpose;
    RZ_INFO("max symmetric norm:",diff.norm(),"  A norm: ",A.norm());
    RZ_INFO("A Mean : ",A.cwiseAbs().sum()/A.size());
    GlobalReal maxVal = std::numeric_limits<GlobalReal>::lowest();  // 设置初始最大值为最小值
    int max_col,max_row;
    for (int k = 0; k < A.outerSize(); ++k) {
        for (typename Eigen::SparseMatrixR::InnerIterator it(A, k); it; ++it) {
            RZ_ASSERT(k == it.row(),k,it.row(),it.col());
            if (std::abs(it.value())>maxVal)max_row = it.row(),max_col = it.col();
            maxVal = std::max(maxVal, std::abs(it.value()));
        }
    }
    RZ_INFO("A max:",maxVal," ",max_row," ",max_col);
    RZ_INFO("A max:",maxVal);
    GlobalReal max_abs_diff = 0.0;
    for (int k = 0; k < A.outerSize(); ++k)
        for (Eigen::SparseMatrixR::InnerIterator it(A, k); it; ++it) {
            int i = it.row(), j = it.col();
            max_abs_diff = std::max(max_abs_diff,std::abs(A.coeff(i, j) - A.coeff(j, i)));
        }
    RZ_INFO("max symmetric:",max_abs_diff);
    for (int k = 0; k < A.outerSize(); ++k)
        for (Eigen::SparseMatrixR::InnerIterator it(A, k); it; ++it) {
            int i = it.row(), j = it.col();
            if (i != j && std::abs(A.coeff(i, j) - A.coeff(j, i)) > 1e-5)
                return false;
        }
    return true;
}

// void visualize_A_sparsity(const Eigen::SparseMatrixR &A,
//                           const std::string &filename) {
//     const int rows = A.rows();
//     const int cols = A.cols();
//
//     if (!isSymmetric(A))RZ_INFO("visualize_A_sparsity not Symmetric");
//     // return;
//     // 如果矩阵为空，则返回
//     if (rows == 0 || cols == 0) {
//         std::cerr << "[visualize_A_sparsity] Empty matrix.\n";
//         return;
//     }
//
//     // 计算总的元素数量
//     size_t total = static_cast<size_t>(rows) * cols;
//     std::cout << "Matrix size: " << rows << " x " << cols
//               << ", nonzeros = " << A.nonZeros() << std::endl;
//
//     // 如果矩阵太大，打印警告
//     if (total > 2e8) {  // 如果矩阵元素大于 2e8，认为是超大矩阵
//         std::cerr << "[visualize_A_sparsity] Too large: "
//                   << rows << "x" << cols << " = " << total << " pixels\n";
//     }
//     const int img_size = 2000;
//     // 下采样因子，确保生成 2000x2000 的图像
//     int downsample_factor_row = std::max(1, rows / img_size + 1);  // 最大 2000x2000 的图像
//     int downsample_factor_col = std::max(1, cols / img_size + 1);
//     // 创建图像数据，初始化为白色（255）
//     std::vector<unsigned char> img(img_size * img_size, 255);
//
//     // 遍历矩阵并将非零元素映射到下采样图像
//     // for (int r = 0; r < rows; r += 1) {
//     //     for (int c = 0; c < cols; c += 1) {
//     //         if (A.coeff(r, c) != 0) {
//     //             // 将非零元素设置为黑色（0）
//     //             img[(r / downsample_factor_row) * img_size + (c / downsample_factor_col)] = 0;
//     //         }
//     //     }
//     // }
//     std::set<int> used_coarse_dofs;
//     for (int k = 0; k < A.outerSize(); ++k) {
//         for (Eigen::SparseMatrixR::InnerIterator it(A, k); it; ++it) {
//             int r = it.row();
//             int c = it.col();
//             used_coarse_dofs.insert(it.col());
//             img[(r / downsample_factor_row) * img_size + (c / downsample_factor_col)] = 0;
//         }
//     }
//     RZ_INFO("no zero coarse dofs:",used_coarse_dofs.size());
//
//     std::cout << "[visualize_A_sparsity] Finished iterating nonzeros.\n";
//
//     // 保存为 PNG 图像，行列数为 2000x2000
//     if (!stbi_write_png(filename.c_str(), img_size, img_size,
//         1, img.data(), img_size)) {
//         std::cerr << "[visualize_A_sparsity] Failed to write PNG: " << filename << std::endl;
//     } else {
//         std::cout << "[visualize_A_sparsity] Saved to " << filename << std::endl;
//     }
// }
//
// void visualize_A_sparsity(const Eigen::SparseMatrixR &A,
//                           const std::string &filename) {
//     const int rows = A.rows();
//     const int cols = A.cols();
//
//     if (!isSymmetric(A)) RZ_INFO("visualize_A_sparsity not Symmetric");
//
//     if (rows == 0 || cols == 0) {
//         std::cerr << "[visualize_A_sparsity] Empty matrix.\n";
//         return;
//     }
//
//     size_t total = static_cast<size_t>(rows) * cols;
//     std::cout << "Matrix size: " << rows << " x " << cols
//               << ", nonzeros = " << A.nonZeros() << std::endl;
//
//     if (total > 2e8) {
//         std::cerr << "[visualize_A_sparsity] Too large: "
//                   << rows << "x" << cols << " = " << total << " pixels\n";
//     }
//
//     const int img_size = 2000;
//
//     int downsample_factor_row = std::max(1, rows / img_size + 1);
//     int downsample_factor_col = std::max(1, cols / img_size + 1);
//
//
//     // RGB image: 3 channels
//     std::vector<unsigned char> img(img_size * img_size * 3, 255); // white background
//
//     std::set<int> used_coarse_dofs;
//     for (int k = 0; k < A.outerSize(); ++k) {
//         for (Eigen::SparseMatrixR::InnerIterator it(A, k); it; ++it) {
//             int r = it.row();
//             int c = it.col();
//
//             int rr = r / downsample_factor_row;
//             int cc = c / downsample_factor_col;
//             if (rr < 0 || rr >= img_size || cc < 0 || cc >= img_size)
//                 continue;
//
//             used_coarse_dofs.insert(c);
//
//             int idx = (rr * img_size + cc) * 3;
//
//             // Blue color: (0, 114, 189)
//             img[idx + 0] = 0;    // R
//             img[idx + 1] = 114;  // G
//             img[idx + 2] = 189;  // B
//         }
//     }
//
//     RZ_INFO("no zero coarse dofs:", used_coarse_dofs.size());
//
//     std::cout << "[visualize_A_sparsity] Finished iterating nonzeros.\n";
//
//     // Write RGB PNG
//     if (!stbi_write_png(filename.c_str(),
//                         img_size, img_size,
//                         3,               // RGB
//                         img.data(),
//                         img_size * 3)) { // stride
//         std::cerr << "[visualize_A_sparsity] Failed to write PNG: " << filename << std::endl;
//     } else {
//         std::cout << "[visualize_A_sparsity] Saved to " << filename << std::endl;
//     }
// }
void visualize_A_sparsity(const Eigen::SparseMatrixR &A,
                          const std::string &filename) {
    const int rows = A.rows();
    const int cols = A.cols();

    if (!isSymmetric(A)) RZ_INFO("visualize_A_sparsity not Symmetric");

    if (rows == 0 || cols == 0) {
        std::cerr << "[visualize_A_sparsity] Empty matrix.\n";
        return;
    }

    size_t total = static_cast<size_t>(rows) * cols;
    std::cout << "Matrix size: " << rows << " x " << cols
              << ", nonzeros = " << A.nonZeros() << std::endl;

    if (total > static_cast<size_t>(2e8)) {
        std::cerr << "[visualize_A_sparsity] Too large: "
                  << rows << "x" << cols << " = " << total << " pixels\n";
    }

    const int img_size = 2000;

    // 使用 Real 缩放，而不是整数 downsample
    const Real scale_row = static_cast<Real>(img_size) / static_cast<Real>(rows);
    const Real scale_col = static_cast<Real>(img_size) / static_cast<Real>(cols);

    // RGB image: 3 channels, white background
    std::vector<unsigned char> img(static_cast<size_t>(img_size) * img_size * 3, 255);

    std::set<int> used_coarse_dofs;
    for (int k = 0; k < A.outerSize(); ++k) {
        for (Eigen::SparseMatrixR::InnerIterator it(A, k); it; ++it) {
            const int r = it.row();
            const int c = it.col();

            int rr = static_cast<int>(std::floor(r * scale_row));
            int cc = static_cast<int>(std::floor(c * scale_col));

            // 防止最后一个元素由于浮点误差映射到 img_size
            rr = std::max(0, std::min(rr, img_size - 1));
            cc = std::max(0, std::min(cc, img_size - 1));

            used_coarse_dofs.insert(c);

            const size_t idx = (static_cast<size_t>(rr) * img_size + cc) * 3;

            // Blue color: (0, 114, 189)
            img[idx + 0] = 0;    // R
            img[idx + 1] = 114;  // G
            img[idx + 2] = 189;  // B
        }
    }

    RZ_INFO("no zero coarse dofs:", used_coarse_dofs.size());

    std::cout << "[visualize_A_sparsity] Finished iterating nonzeros.\n";

    if (!stbi_write_png(filename.c_str(),
                        img_size,
                        img_size,
                        3,
                        img.data(),
                        img_size * 3)) {
        std::cerr << "[visualize_A_sparsity] Failed to write PNG: " << filename << std::endl;
    } else {
        std::cout << "[visualize_A_sparsity] Saved to " << filename << std::endl;
    }
}
