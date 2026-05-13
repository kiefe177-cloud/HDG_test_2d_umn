#include <iostream>
#include <cassert>
#include <Eigen/Sparse>
#include <Eigen/Dense>

// Include your headers
#include "hdg/sem2.h"
#include "hdg/sem1.h"

// Required for Kronecker Product
#include <unsupported/Eigen/KroneckerProduct>

using namespace Eigen;

int main(){
    int p = 2; // Nodes = p+1 = 3. Total 2D nodes = 9.
    Sem2Op op = sem2(p);
    Sem1Op op1 = sem1(p); // Used to verify results

    double tol = 1e-4;

    // 1. Verify Mass Matrix M
    assert(op.M.nonZeros() == 9 && "Mass Matrix M should have 9 non-zero entries for p=2");

    SparseMatrix<double> expected_M(9,9);

    expected_M.insert(0,0) = 0.1111; // 1/9
    expected_M.insert(1,1) = 0.4444; // 4/9
    expected_M.insert(2,2) = 0.1111; // 1/9
    expected_M.insert(3,3) = 0.4444; // 4/9
    expected_M.insert(4,4) = 1.7778; // 16/9 (Center node)
    expected_M.insert(5,5) = 0.4444; // 4/9
    expected_M.insert(6,6) = 0.1111; // 1/9
    expected_M.insert(7,7) = 0.4444; // 4/9
    expected_M.insert(8,8) = 0.1111; // 1/9

    assert(op.M.isApprox(expected_M, tol) && "Mass Matrix M values mismatch");
    std::cout << "[PASS] Mass Matrix M" << std::endl;

    // 2. Verify Stiffness Matrices S
    SparseMatrix<double> expected_Sx = Eigen::kroneckerProduct(MatrixXd(op1.M), op1.S);
    SparseMatrix<double> expected_Sy = Eigen::kroneckerProduct(op1.S, MatrixXd(op1.M));

    // Verify Sx (S[0])
    assert(op.S[0].isApprox(expected_Sx, tol) && "Stiffness Matrix Sx mismatch");
    
    // Verify Sy (S[1])
    assert(op.S[1].isApprox(expected_Sy, tol) && "Stiffness Matrix Sy mismatch");

    std::cout << "[PASS] Stiffness Matrices S" << std::endl;

    // 3. Verify Boundary Operators L and T
    // Left Boundary Operator L
    SparseMatrix<double> expected_L_1(9,3);
    expected_L_1.insert(0,0) = 0.3333;
    expected_L_1.insert(3,1) = 1.3333;
    expected_L_1.insert(6,2) = 0.3333;

    SparseMatrix<double> expected_L_2(9,3);
    expected_L_2.insert(2,0) = 0.3333;
    expected_L_2.insert(5,1) = 1.3333;
    expected_L_2.insert(8,2) = 0.3333;

    SparseMatrix<double> expected_L_3(9,3);
    expected_L_3.insert(0,0) = 0.3333;
    expected_L_3.insert(1,1) = 1.3333;
    expected_L_3.insert(2,2) = 0.3333;

    SparseMatrix<double> expected_L_4(9,3);
    expected_L_4.insert(6,0) = 0.3333;
    expected_L_4.insert(7,1) = 1.3333;
    expected_L_4.insert(8,2) = 0.3333;

  
    MatrixXd expected_L_5(9,3);
    expected_L_5 << 0.4583, 0.5000,-0.1250,
                  0.0000, 0.0000, 0.0000,
                  0.0000, 0.0000, 0.0000,
                 -0.2500, 1.0000, 0.5833,
                  0.0000, 0.0000, 0.0000,
                  0.0000, 0.0000, 0.0000,
                  0.1250,-0.1667,-0.1250,
                  0.0000, 0.0000, 0.0000,
                  0.0000, 0.0000, 0.0000;

    MatrixXd expected_L_6(9,3);
    expected_L_6 <<0.0000, 0.0000, 0.0000,
                  0.0000, 0.0000, 0.0000,
                  0.4583, 0.5000,-0.1250,
                  0.0000, 0.0000, 0.0000,
                  0.0000, 0.0000, 0.0000,
                 -0.2500, 1.0000, 0.5833,
                  0.0000, 0.0000, 0.0000,
                  0.0000, 0.0000, 0.0000,
                  0.1250,-0.1667,-0.1250;


    MatrixXd expected_L_7(9,3);
    expected_L_7 << 0.4583, 0.5000,-0.1250,
                 -0.2500, 1.0000, 0.5833,
                  0.1250,-0.1667,-0.1250,
                  0.0000, 0.0000, 0.0000,
                  0.0000, 0.0000, 0.0000,
                  0.0000, 0.0000, 0.0000,
                  0.0000, 0.0000, 0.0000,
                  0.0000, 0.0000, 0.0000,
                  0.0000, 0.0000, 0.0000;

    MatrixXd expected_L_8(9,3);
    expected_L_8 <<0.0000, 0.0000, 0.0000,
                  0.0000, 0.0000, 0.0000,
                  0.0000, 0.0000, 0.0000,
                  0.0000, 0.0000, 0.0000,
                  0.0000, 0.0000, 0.0000,
                  0.0000, 0.0000, 0.0000,
                  0.4583, 0.5000,-0.1250,
                 -0.2500, 1.0000, 0.5833,
                  0.1250,-0.1667,-0.1250;


    MatrixXd expected_L_9(9,3);
    expected_L_9 << -0.1250, -0.1667,0.1250,
                  0.0000, 0.0000, 0.0000,
                  0.0000, 0.0000, 0.0000,
                  0.5833, 1.0000,-0.2500,
                  0.0000, 0.0000, 0.0000,
                  0.0000, 0.0000, 0.0000,
                 -0.1250, 0.5000, 0.4583,
                  0.0000, 0.0000, 0.0000,
                  0.0000, 0.0000, 0.0000;

    MatrixXd expected_L_10(9,3);
    expected_L_10 << 0.0000, 0.0000, 0.0000,
                  0.0000, 0.0000, 0.0000,
                 -0.1250,-0.1667, 0.1250,
                  0.0000, 0.0000, 0.0000,
                  0.0000, 0.0000, 0.0000,
                  0.5833, 1.0000,-0.2500,
                  0.0000, 0.0000, 0.0000,
                  0.0000, 0.0000, 0.0000,
                 -0.1250, 0.5000, 0.4583;

    MatrixXd expected_L_11(9,3);
    expected_L_11 << -0.1250, -0.1667, 0.1250,
                  0.5833, 1.0000,-0.2500,
                 -0.1250, 0.5000, 0.4583,
                  0.0000, 0.0000, 0.0000,
                  0.0000, 0.0000, 0.0000,
                  0.0000, 0.0000, 0.0000,
                  0.0000, 0.0000, 0.0000,
                  0.0000, 0.0000, 0.0000,
                  0.0000, 0.0000, 0.0000;

    MatrixXd expected_L_12(9,3);
    expected_L_12 << 0.0000, 0.0000, 0.0000,
                  0.0000, 0.0000, 0.0000,
                  0.0000, 0.0000, 0.0000,
                  0.0000, 0.0000, 0.0000,
                  0.0000, 0.0000, 0.0000,
                  0.0000, 0.0000, 0.0000,
                 -0.1250,-0.1667, 0.1250,
                  0.5833, 1.0000,-0.2500,
                 -0.1250, 0.5000, 0.4583;



    assert(op.L[0].isApprox(expected_L_1, tol) && "Boundary Operator L_1 mismatch");
    assert(op.L[1].isApprox(expected_L_2, tol) && "Boundary Operator L_2 mismatch");
    assert(op.L[2].isApprox(expected_L_3, tol) && "Boundary Operator L_3 mismatch");
    assert(op.L[3].isApprox(expected_L_4, tol) && "Boundary Operator L_4 mismatch");
    assert(op.L[4].isApprox(expected_L_5, tol) && "Boundary Operator L_5 mismatch");
    assert(op.L[5].isApprox(expected_L_6, tol) && "Boundary Operator L_6 mismatch");
    assert(op.L[6].isApprox(expected_L_7, tol) && "Boundary Operator L_7 mismatch");
    assert(op.L[7].isApprox(expected_L_8, tol) && "Boundary Operator L_8 mismatch");
    assert(op.L[8].isApprox(expected_L_9, tol) && "Boundary Operator L_9 mismatch");
    assert(op.L[9].isApprox(expected_L_10, tol) && "Boundary Operator L_10 mismatch");
    assert(op.L[10].isApprox(expected_L_11, tol) && "Boundary Operator L_11 mismatch");
    assert(op.L[11].isApprox(expected_L_12, tol) && "Boundary Operator L_12 mismatch");
    std::cout << "[PASS] Boundary Operators L" << std::endl;

    // Verify Trace Operator T
    SparseMatrix<double> expected_T_1(3,9);
    expected_T_1.insert(0,0) = 1.0;
    expected_T_1.insert(1,3) = 1.0;
    expected_T_1.insert(2,6) = 1.0;

    SparseMatrix<double> expected_T_2(3,9);
    expected_T_2.insert(0,2) = 1.0;
    expected_T_2.insert(1,5) = 1.0;
    expected_T_2.insert(2,8) = 1.0;

    SparseMatrix<double> expected_T_3(3,9);
    expected_T_3.insert(0,0) = 1.0;
    expected_T_3.insert(1,1) = 1.0;
    expected_T_3.insert(2,2) = 1.0;

    SparseMatrix<double> expected_T_4(3,9);
    expected_T_4.insert(0,6) = 1.0;
    expected_T_4.insert(1,7) = 1.0;
    expected_T_4.insert(2,8) = 1.0;

    MatrixXd expected_T_5(3,9);
    expected_T_5 << 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                    0.3750, 0.0000, 0.0000, 0.7500, 0.0000, 0.0000,-0.1250, 0.0000, 0.0000,
                    0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;

    MatrixXd expected_T_6(3,9);
    expected_T_6 << 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                    0.0000, 0.0000, 0.3750, 0.0000, 0.0000, 0.7500, 0.0000, 0.0000,-0.1250,
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000;

    MatrixXd expected_T_7(3,9);
    expected_T_7 << 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                    0.3750, 0.7500,-0.1250, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                    0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;

    MatrixXd expected_T_8(3,9);
    expected_T_8 << 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000,
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.3750, 0.7500,-0.1250,
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000;

    MatrixXd expected_T_9(3,9);
    expected_T_9 << 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                   -0.1250, 0.0000, 0.0000, 0.7500, 0.0000, 0.0000, 0.3750, 0.0000, 0.0000,
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000;

    MatrixXd expected_T_10(3,9);
    expected_T_10 << 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000,
                     0.0000, 0.0000,-0.1250, 0.0000, 0.0000, 0.7500, 0.0000, 0.0000, 0.3750,
                     0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000;
                     
    MatrixXd expected_T_11(3,9);
    expected_T_11 << 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                    -0.1250, 0.7500, 0.3750, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                     0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000;

    MatrixXd expected_T_12(3,9);
    expected_T_12 << 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000,
                     0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,-0.1250, 0.7500, 0.3750,
                     0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000;

    assert(op.T[0].isApprox(expected_T_1, tol) && "Trace Operator T_1 mismatch");
    assert(op.T[1].isApprox(expected_T_2, tol) && "Trace Operator T_2 mismatch");
    assert(op.T[2].isApprox(expected_T_3, tol) && "Trace Operator T_3 mismatch");
    assert(op.T[3].isApprox(expected_T_4, tol) && "Trace Operator T_4 mismatch");
    assert(op.T[4].isApprox(expected_T_5, tol) && "Trace Operator T_5 mismatch");
    assert(op.T[5].isApprox(expected_T_6, tol) && "Trace Operator T_6 mismatch");
    assert(op.T[6].isApprox(expected_T_7, tol) && "Trace Operator T_7 mismatch");
    assert(op.T[7].isApprox(expected_T_8, tol) && "Trace Operator T_8 mismatch");
    assert(op.T[8].isApprox(expected_T_9, tol) && "Trace Operator T_9 mismatch");
    assert(op.T[9].isApprox(expected_T_10, tol) && "Trace Operator T_10 mismatch");
    assert(op.T[10].isApprox(expected_T_11, tol) && "Trace Operator T_11 mismatch");
    assert(op.T[11].isApprox(expected_T_12, tol) && "Trace Operator T_12 mismatch");
    std::cout << "[PASS] Trace Operators T_1 through T_12" << std::endl;
    
    return 0;
}
