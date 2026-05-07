#include "hdg/calc_grads_strong.h"
#include "hdg/multidof_ops.h"
#include "hdg/GLDeriv.h"
#include "hdg/mesh_utils.h"

#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <unsupported/Eigen/KroneckerProduct>

#include <cassert>
#include <stdexcept>
#include <vector>

Grad_Variables calc_grads_strong(const Eigen::MatrixXd& bU,
                                 const Eigen::MatrixXd& bU_f,
                                 const Msh& msh,
                                 const int nvar,
                                 const Sem1Op& op1,
                                 const Sem2Op& op2)
{
    // ---- Sizes ----
    const int p   = msh.elem[0].p;
    const int n1d = p + 1;                 // 1D nodes per face
    const int npe = n1d * n1d;             // 2D nodes per element
    const int Ne  = static_cast<int>(msh.elem.size());
    const int Nf  = static_cast<int>(msh.FtoE.cols());
    const int Me  = nvar * npe;            // dofs per element across all variables
    const int Mf  = nvar * n1d;            // dofs per face across all variables

    const int faces_per_elem = msh.EtoF.rows();

    // ---- Multi-dof operators (M for the volume mass, T[f] for face traces) ----
    MultidofOps ops = multidof_ops(nvar, op1, op2);

    // ---- Identity factors for kron expansion ----
    Eigen::SparseMatrix<double> I(nvar, nvar);
    I.setIdentity();
    Eigen::SparseMatrix<double> Ip(n1d, n1d);
    Ip.setIdentity();

    // ---- 1D and 2D reference derivative operators (sparse) ----
    Eigen::MatrixXd Dr_dense = GLDeriv(p);
    Eigen::SparseMatrix<double> Dr = Dr_dense.sparseView();

    Eigen::SparseMatrix<double> D_xi  = Eigen::kroneckerProduct(Ip, Dr);
    Eigen::SparseMatrix<double> D_eta = Eigen::kroneckerProduct(Dr, Ip);

    Eigen::SparseMatrix<double> D_xi_var  = Eigen::kroneckerProduct(I, D_xi);
    Eigen::SparseMatrix<double> D_eta_var = Eigen::kroneckerProduct(I, D_eta);

    // ---- Assemble QA, QBx, QBy element by element ----
    std::vector<Eigen::Triplet<double>> trips_QA;
    std::vector<Eigen::Triplet<double>> trips_QBx;
    std::vector<Eigen::Triplet<double>> trips_QBy;

    // Rough estimate: each Me-row block has ~Me*n1d nonzeros from D operators.
    // Over-reserving is harmless; under-reserving just causes vector regrowth.
    const long est_nnz_per_block = static_cast<long>(Me) * n1d * nvar;
    trips_QA .reserve(Ne * est_nnz_per_block);
    trips_QBx.reserve(Ne * est_nnz_per_block);
    trips_QBy.reserve(Ne * est_nnz_per_block);

    for (int i1 = 0; i1 < Ne; ++i1) {
        // Per-element metric matrices, expanded across variables via I (x) ...
        Eigen::SparseMatrix<double> J    = Eigen::kroneckerProduct(I, msh.elem[i1].J);
        Eigen::SparseMatrix<double> Ja11 = Eigen::kroneckerProduct(I, msh.elem[i1].Ja11);
        Eigen::SparseMatrix<double> Ja12 = Eigen::kroneckerProduct(I, msh.elem[i1].Ja12);
        Eigen::SparseMatrix<double> Ja21 = Eigen::kroneckerProduct(I, msh.elem[i1].Ja21);
        Eigen::SparseMatrix<double> Ja22 = Eigen::kroneckerProduct(I, msh.elem[i1].Ja22);

        const int j1 = Me * i1;   // 0-based block start (row and col)

        // Mass matrix scaled by local Jacobian
        Eigen::SparseMatrix<double> qa_block = J * ops.M;

        // Pointwise gradient assembly (negative sign matches MATLAB strong form)
        Eigen::SparseMatrix<double> qbx_block = -ops.M * (Ja11 * D_xi_var + Ja21 * D_eta_var);
        Eigen::SparseMatrix<double> qby_block = -ops.M * (Ja12 * D_xi_var + Ja22 * D_eta_var);

        // Stamp each block into the global triplet list with offset (j1, j1).
        for (int k = 0; k < qa_block.outerSize(); ++k)
            for (Eigen::SparseMatrix<double>::InnerIterator it(qa_block, k); it; ++it)
                trips_QA.emplace_back(j1 + it.row(), j1 + it.col(), it.value());

        for (int k = 0; k < qbx_block.outerSize(); ++k)
            for (Eigen::SparseMatrix<double>::InnerIterator it(qbx_block, k); it; ++it)
                trips_QBx.emplace_back(j1 + it.row(), j1 + it.col(), it.value());

        for (int k = 0; k < qby_block.outerSize(); ++k)
            for (Eigen::SparseMatrix<double>::InnerIterator it(qby_block, k); it; ++it)
                trips_QBy.emplace_back(j1 + it.row(), j1 + it.col(), it.value());
    }

    Eigen::SparseMatrix<double> QA (Ne * Me, Ne * Me);
    Eigen::SparseMatrix<double> QBx(Ne * Me, Ne * Me);
    Eigen::SparseMatrix<double> QBy(Ne * Me, Ne * Me);

    QA .setFromTriplets(trips_QA .begin(), trips_QA .end());
    QBx.setFromTriplets(trips_QBx.begin(), trips_QBx.end());
    QBy.setFromTriplets(trips_QBy.begin(), trips_QBy.end());

    // ---- Assemble the face reconstruction operator QF ----
    std::vector<Eigen::Triplet<double>> trips_QF;
    // Each interior face contributes ~2 * Mf * Me trace entries; boundaries half that.
    const long est_nnz_per_face = static_cast<long>(Mf) * Me;
    trips_QF.reserve(2 * Nf * est_nnz_per_face);

    // Helper: copy every non-zero of T_block into QF with offsets (k1, j1) and a weight.
    auto stamp_T_block = [&](int k1_offset,
                             int j1_offset,
                             double weight,
                             const Eigen::SparseMatrix<double>& T_block)
    {
        for (int k = 0; k < T_block.outerSize(); ++k)
            for (Eigen::SparseMatrix<double>::InnerIterator it(T_block, k); it; ++it)
                trips_QF.emplace_back(k1_offset + it.row(),
                                      j1_offset + it.col(),
                                      weight * it.value());
    };

    for (int i1 = 0; i1 < Nf; ++i1) {
        const int e1 = msh.FtoE(0, i1);
        const int e2 = msh.FtoE(1, i1);
        const int k1 = Mf * i1;

        if (e1 < 0) {
            // Boundary face: trace from e2 only
            int f2 = find_local_face(msh.EtoF, e2, i1, faces_per_elem);
            assert(f2 >= 0 && "face not found in element e2");
            stamp_T_block(k1, Me * e2, 1.0, ops.T[f2]);
        }
        else if (e2 < 0) {
            // Boundary face: trace from e1 only
            int f1 = find_local_face(msh.EtoF, e1, i1, faces_per_elem);
            assert(f1 >= 0 && "face not found in element e1");
            stamp_T_block(k1, Me * e1, 1.0, ops.T[f1]);
        }
        else {
            // Interior face: average of both element traces
            int f1 = find_local_face(msh.EtoF, e1, i1, faces_per_elem);
            int f2 = find_local_face(msh.EtoF, e2, i1, faces_per_elem);
            assert(f1 >= 0 && f2 >= 0);
            stamp_T_block(k1, Me * e1, 0.5, ops.T[f1]);
            stamp_T_block(k1, Me * e2, 0.5, ops.T[f2]);
        }
    }

    Eigen::SparseMatrix<double> QF(Nf * Mf, Ne * Me);
    QF.setFromTriplets(trips_QF.begin(), trips_QF.end());

    // ---- Solve for the volume gradients ----
    // Flatten bU column-major into a long vector (matches MATLAB's reshape(bU,[],1)).
    Eigen::VectorXd bU_flat = Eigen::Map<const Eigen::VectorXd>(bU.data(), bU.size());

    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(QA);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("calc_grads_strong: QA factorization failed");
    }

    Eigen::VectorXd rhs_x = -(QBx * bU_flat);
    Eigen::VectorXd rhs_y = -(QBy * bU_flat);

    Eigen::VectorXd bUx_flat = solver.solve(rhs_x);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("calc_grads_strong: QA solve (x) failed");
    }
    Eigen::VectorXd bUy_flat = solver.solve(rhs_y);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("calc_grads_strong: QA solve (y) failed");
    }

    // ---- Project the volume gradients onto the faces ----
    Eigen::VectorXd bUx_f_flat = QF * bUx_flat;
    Eigen::VectorXd bUy_f_flat = QF * bUy_flat;

    // ---- Reshape back to multi-variable matrix layout ----
    Grad_Variables out;
    out.bUx   = Eigen::Map<Eigen::MatrixXd>(bUx_flat.data(),   Me, Ne);
    out.bUy   = Eigen::Map<Eigen::MatrixXd>(bUy_flat.data(),   Me, Ne);
    out.bUz   = Eigen::MatrixXd::Zero(Me, Ne);
    out.bUx_f = Eigen::Map<Eigen::MatrixXd>(bUx_f_flat.data(), Mf, Nf);
    out.bUy_f = Eigen::Map<Eigen::MatrixXd>(bUy_f_flat.data(), Mf, Nf);
    out.bUz_f = Eigen::MatrixXd::Zero(Mf, Nf);

    return out;
}