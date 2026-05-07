#pragma once
#include <Eigen/Dense>

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
        Eigen::MatrixXd J;
        Eigen::MatrixXd Ja11;
        Eigen::MatrixXd Ja12;
        Eigen::MatrixXd Ja21;
        Eigen::MatrixXd Ja22;
        Eigen::MatrixXd w1d;
        Eigen::MatrixXd r;
        Eigen::MatrixXd invr;
        
    };  // Ne      
    struct Face
    {
        int p;
        Eigen::MatrixXd x;
        Eigen::MatrixXd y;
        Eigen::MatrixXd J;
        Eigen::MatrixXd nx;
        Eigen::MatrixXd ny;
        Eigen::MatrixXd nn;
        Eigen::MatrixXd w1d;
        Eigen::MatrixXd Jf;
        Eigen::MatrixXd r;
        Eigen::MatrixXd invr;
    };  // Nf

    std::vector<Elem> elem;
    std::vector<Face> face;
};