#include "hdg/flux_split.h"
#include "hdg/sd.h"
#include <unsupported/Eigen/KroneckerProduct>

namespace{
static Eigen::SparseMatrix<double> sparse_block(
    const std::vector<std::vector<Eigen::SparseMatrix<double>>>& blocks)
{
    const int br = blocks.size();
    const int bc = blocks[0].size();

    std::vector<int> row_offsets(br + 1, 0);
    std::vector<int> col_offsets(bc + 1, 0);
    for (int i = 0; i < br; ++i) row_offsets[i + 1] = row_offsets[i] + blocks[i][0].rows();
    for (int j = 0; j < bc; ++j) col_offsets[j + 1] = col_offsets[j] + blocks[0][j].cols();

    std::vector<Eigen::Triplet<double>> trips;
    int nnz_total = 0;
    for (auto& row : blocks) for (auto& B : row) nnz_total += B.nonZeros();
    trips.reserve(nnz_total);

    for (int i = 0; i < br; ++i) {
        for (int j = 0; j < bc; ++j) {
            const auto& B = blocks[i][j];
            const int r0 = row_offsets[i], c0 = col_offsets[j];
            for (int k = 0; k < B.outerSize(); ++k)
                for (Eigen::SparseMatrix<double>::InnerIterator it(B, k); it; ++it)
                    trips.emplace_back(r0 + it.row(), c0 + it.col(), it.value());
        }
    }

    Eigen::SparseMatrix<double> out(row_offsets[br], col_offsets[bc]);
    out.setFromTriplets(trips.begin(), trips.end());
    return out;
}

}
flux_split_jacobians flux_split(const Eigen::VectorXd& U,
                                const Eigen::SparseMatrix<double>& nx,
                                const Eigen::SparseMatrix<double>& ny,
                                const Eigen::SparseMatrix<double>& nn,
                                const SimulationParams& params,
                                const int nvar)
{
    using SparseD   = Eigen::SparseMatrix<double>;
    double gam = params.gam;

    double eps = 0.01;
    Eigen::VectorXd nx_diag = nx.diagonal();   // dense vector of diagonal entries
    Eigen::VectorXd ny_diag = ny.diagonal();   // dense vector of diagonal entries
    
    Eigen::VectorXd nn_diag = nn.diagonal();

    Eigen::VectorXd nx_vec = nx_diag.array() / nn_diag.array();
    Eigen::VectorXd ny_vec = ny_diag.array() / nn_diag.array();

    int N = static_cast<int>(U.size()) / nvar;

    // ---- Identity factors for kron expansion ----
    SparseD I(N, N);
    I.setIdentity();
    SparseD Z(N,N);

    Eigen::VectorXd rho = U.segment(0*N,N);
    Eigen::VectorXd rhou = U.segment(1*N,N);
    Eigen::VectorXd rhov = U.segment(2*N,N);
    Eigen::VectorXd rhow = U.segment(3*N,N);
    Eigen::VectorXd E = U.segment(4*N,N);

    Eigen::VectorXd u = (rhou.array() / rho.array()).matrix();
    Eigen::VectorXd v = (rhov.array() / rho.array()).matrix();
    Eigen::VectorXd w = (rhow.array() / rho.array()).matrix();

    Eigen::VectorXd q2 = (u.array().square() + v.array().square() + w.array().square());
    Eigen::VectorXd p = (gam - 1.0) * (E.array() - 0.5 * rho.array()*q2.array());
    Eigen::VectorXd a = (gam * p.array()/rho.array()).sqrt();

    Eigen::VectorXd u2 = (nx_vec.array() * u.array() + ny_vec.array() * v.array());

    const double gm1 = gam - 1.0;
    Eigen::VectorXd inv_rho = rho.array().inverse();
    Eigen::VectorXd inv_a   = a.array().inverse();

    auto sd_scaled = [&](double s, const Eigen::VectorXd& v){
        return sd((s * v.array()).matrix());
    };

    SparseD dVdU = sparse_block({
        { sd_scaled(0.5*gm1,q2),                        sd_scaled(-gm1,u), sd_scaled(-gm1,v), sd_scaled(-gm1,w), gm1*I},
        { -sd((u.array() * inv_rho.array()).matrix()),  sd(inv_rho),       Z,                 Z,                 Z},
        { -sd((v.array() * inv_rho.array()).matrix()),  Z,                 sd(inv_rho),       Z,                 Z},
        { -sd((w.array() * inv_rho.array()).matrix()),  Z,                 Z,                 sd(inv_rho),       Z},
        { I,                                            Z,                 Z,                 Z,                 Z},
    });

    SparseD dUdV = sparse_block({
        { Z,        Z,          Z,          Z,          I},
        { Z,        sd(rho),    Z,          Z,          sd(u)},
        { Z,        Z,          sd(rho),    Z,          sd(v)},
        { Z,        Z,          Z,          sd(rho),    sd(w)},
        { I/gm1,    sd(rhou),   sd(rhov),   sd(rhow),   sd_scaled(0.5,q2)},
    });

    SparseD iC = sparse_block({
        {0.5*I, sd_scaled(0.5, (rho.array() * a.array() * nx_vec.array()).matrix()),    sd_scaled(0.5, (rho.array() * a.array() * ny_vec.array()).matrix()),    Z, Z},
        {0.5*I, sd_scaled(-0.5, (rho.array() * a.array() * nx_vec.array()).matrix()),   sd_scaled(-0.5, (rho.array() * a.array() * ny_vec.array()).matrix()),   Z, Z},
        {Z,     sd(-ny_vec),                                                            sd(nx_vec),                                                             Z, Z},
        {Z,     Z,                                                                      Z,                                                                      I, Z},
        {-I,    Z,                                                                      Z,                                                                      Z, sd(a.array().square())},
    });

    SparseD C = sparse_block({
        {I,                                                           I,                                                                       Z,                     Z,Z},
        {sd((nx_vec.array()*inv_rho.array()*inv_a.array()).matrix()), sd_scaled(-1.0,(nx_vec.array()*inv_rho.array()*inv_a.array()).matrix()), sd_scaled(-1.0,ny_vec),Z,Z},
        {sd((ny_vec.array()*inv_rho.array()*inv_a.array()).matrix()), sd_scaled(-1.0,(ny_vec.array()*inv_rho.array()*inv_a.array()).matrix()), sd_scaled(1.0,nx_vec), Z,Z},
        {Z,                                                           Z,                                                                       Z,                     I,Z},
        {sd(inv_a.array().square().matrix()),                         sd(inv_a.array().square().matrix()),                                     Z,                     Z,sd(inv_a.array().square().matrix())},
    });
    
    Eigen::VectorXd L1 = u2 + a;
    Eigen::VectorXd L2 = u2 - a;
    Eigen::VectorXd L3 = u2;

    Eigen::VectorXd t1 = (L1.array().square() + (eps*a.array()).square()).sqrt().matrix();
    Eigen::VectorXd t2 = (L2.array().square() + (eps*a.array()).square()).sqrt().matrix();
    Eigen::VectorXd t3 = (L3.array().square() + (eps*a.array()).square()).sqrt().matrix();

    Eigen::VectorXd L1p = 0.5*(L1 + t1);
    Eigen::VectorXd L2p = 0.5*(L2 + t2);
    Eigen::VectorXd L3p = 0.5*(L3 + t3);

    Eigen::VectorXd L1m = 0.5*(L1 - t1);
    Eigen::VectorXd L2m = 0.5*(L2 - t2);
    Eigen::VectorXd L3m = 0.5*(L3 - t3);

    SparseD Dp = sparse_block({
        {sd(L1p), Z, Z, Z, Z},
        {Z, sd(L2p), Z, Z, Z},
        {Z, Z, sd(L3p), Z, Z},
        {Z, Z, Z, sd(L3p), Z},  
        {Z, Z, Z, Z,       sd(L3p)}, 
    });

    SparseD Dm = sparse_block({
        {sd(L1m), Z, Z, Z, Z},
        {Z, sd(L2m), Z, Z, Z},
        {Z, Z, sd(L3m), Z, Z},
        {Z, Z, Z, sd(L3m), Z},  
        {Z, Z, Z, Z,       sd(L3m)}, 
    });

    SparseD I_nvar(nvar, nvar);
    I_nvar.setIdentity();

    SparseD Ap = Eigen::kroneckerProduct(I_nvar, nn)*(dUdV*C*Dp*iC*dVdU);
    SparseD Am = Eigen::kroneckerProduct(I_nvar, nn)*(dUdV*C*Dm*iC*dVdU);

    flux_split_jacobians out;
    out.Ap = Ap;
    out.Am = Am;

    out.Am.prune(0.0, 1e-14);
    out.Ap.prune(0.0, 1e-14);

    return out;
}