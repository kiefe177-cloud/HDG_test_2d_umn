#pragma once
#include <Eigen/Dense>
#include <string>
#include <vector>

// Struct to hold all grid outputs
struct ConeData{
    Eigen::MatrixXd x;      // Grid X
    Eigen::MatrixXd y;      // Grid Y
    Eigen::MatrixXd z;      // Grid Z
    int nx;                 // Number points in X
    int ny;                 // Number points in Y
    Eigen::MatrixXd coords; // Raw Nx3 list of coordinates
};


// Reads in .dat file and outputs {X,Y,Z} matrices and nx and nt
ConeData read_cone_dat_file(const std::string& filename);
