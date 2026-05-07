#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>

// Element-face connectivity for an AMR mesh.
struct Msh
{
    Eigen::MatrixXi FtoE;   // 2 x Nf   — face -> [left elem, right elem]
    Eigen::MatrixXi EtoF;   // 12 x Ne  — element -> face indices
    struct Elem
    {
        int p;
        Eigen::MatrixXd x;
        Eigen::MatrixXd y;
        Eigen::SparseMatrix<double> J;
        Eigen::SparseMatrix<double> Ja11;
        Eigen::SparseMatrix<double> Ja12;
        Eigen::SparseMatrix<double> Ja21;
        Eigen::SparseMatrix<double> Ja22;
        Eigen::MatrixXd w1d;
        Eigen::MatrixXd r;
        Eigen::MatrixXd invr;
        
    };  // Ne      
    struct Face
    {
        int p;
        Eigen::MatrixXd x;
        Eigen::MatrixXd y;
        Eigen::SparseMatrix<double> nx;
        Eigen::SparseMatrix<double> ny;
        Eigen::SparseMatrix<double> nn;
        Eigen::MatrixXd w1d;
        Eigen::MatrixXd Jf;
        Eigen::MatrixXd r;
        Eigen::MatrixXd invr;
    };  // Nf

    std::vector<Elem> elem;
    std::vector<Face> face;
};