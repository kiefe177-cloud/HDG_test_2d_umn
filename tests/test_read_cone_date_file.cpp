#include <iostream>
#include <cassert>
#include <fstream>
#include <vector>
#include "hdg/read_cone_dat_file.h"

// Helper to save matrix to CSV
void save_matrix_to_csv(const std::string& filename, const Eigen::MatrixXd& mat) {
    std::ofstream file(filename);
    if (file.is_open()) {
        // High precision for floating point accuracy
        const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
        file << mat.format(CSVFormat);
        file.close();
        std::cout << "Saved: " << filename << std::endl;
    } else {
        std::cerr << "Could not write to " << filename << std::endl;
    }
}

int main(){
    // 1. Define the path relative to the build folder
    std::string filepath = "../grids/test_grid.dat";

    std::cout << "Attempting to read: " << filepath << "..." << std::endl;

    try {
        // 2. Call the function
        ConeData result = read_cone_dat_file(filepath);

        // 3. Print Dimensions
        std::cout << "Successfully read file." << std::endl;
        std::cout << "Dimensions: " << result.nx << " x " << result.ny << std::endl;

        // 4. Validation
        if (result.nx != 5 || result.ny != 5){
            std::cerr << "Dimension Error! Expected 5x5, got " 
                      << result.nx << "x" << result.ny << std::endl;
            return 1;
        }

        // 5. Check Corners and Structure (Assuming X varies by row, Y by col)
        // Check Origin (0,0)
        assert(result.x(0,0) == 0 && "Error at x(0,0)");
        assert(result.y(0,0) == 0 && "Error at y(0,0)");

        // Check Diagonal (4,4)
        assert(result.x(4,4) == 4 && "Error at x(4,4)");
        assert(result.y(4,4) == 4 && "Error at y(4,4)");

        // Check off-axis behavior
        // If x(row, col) = row, then x(4,0) should be 4
        assert(result.x(4,0) == 4 && "Error at x(4,0)"); 
        
        // If y(row, col) = col, then y(0,4) should be 4
        assert(result.y(0,4) == 4 && "Error at y(0,4)");

        std::cout << "[PASS] Grid values match expected logic." << std::endl;

        // 6. Save outputs for Python plotting
        save_matrix_to_csv(std::string(OUTPUT_DIR_TEST) + "/grid_x_tfi.csv", result.x);
        save_matrix_to_csv(std::string(OUTPUT_DIR_TEST) + "/grid_y_tfi.csv", result.y);
        save_matrix_to_csv(std::string(OUTPUT_DIR_TEST) + "/grid_z_tfi.csv", result.z);
        
    } catch (const std::exception& e){
        std::cerr << "Test Failed: " << e.what() << std::endl;
        std::cerr << "Ensure '" << filepath << "' exists and is formatted correctly." << std::endl;
        return 1;
    }

    return 0;
}
