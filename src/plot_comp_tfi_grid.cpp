#include "hdg/plot_comp_tfi_grid.h" // Include your specific header
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib> // Required for std::system

// Internal helper function (only visible in this file)
static void save_csv(const std::string& filename, const Eigen::MatrixXd& mat) {
    std::ofstream file(filename);
    if (file.is_open()) {
        const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
        file << mat.format(CSVFormat);
        file.close();
    } else {
        std::cerr << "Error: Unable to save " << filename << std::endl;
    }
}

void plot_comp_tfi_grid(
    const Eigen::MatrixXd& Phi_tfi, 
    const Eigen::MatrixXd& Psi_tfi, 
    const Eigen::MatrixXd& X, 
    const Eigen::MatrixXd& Y, 
    double x_p, double y_p, 
    double phi_val, double psi_val
) {
    std::cout << "Exporting grid data for Python plotting..." << std::endl;

    // 1. Export Data
    save_csv("plot_data_phi.csv", Phi_tfi);
    save_csv("plot_data_psi.csv", Psi_tfi);
    save_csv("plot_data_x.csv", X);
    save_csv("plot_data_y.csv", Y);

    Eigen::MatrixXd point_data(1, 4);
    point_data << x_p, y_p, phi_val, psi_val;
    save_csv("plot_data_point.csv", point_data);

    // 2. Call Python
    // Assumes 'plot_comp_tfi_grid.py' is in your build folder
    std::cout << "Launching plot_comp_tfi_grid.py..." << std::endl;
    int res = std::system("python plot_comp_tfi_grid.py");

    if (res != 0) {
        std::cerr << "Warning: Python script returned error code " << res << std::endl;
    }
}
