#include "hdg/mesh_test_grad.h"
#include "hdg/amr_refine.h"
#include "hdg/amr_dump.h"
#include "hdg/msh_amr_curv.h"

Msh msh_test_grad(int p, const PhysMesh& phys_msh, const std::string& amr_dump_path) {
    // Predicates take a leaf bounding box [x0, y0, x2, y2] = Eigen::Vector4d.
    // Indices: x(0)=x0, x(1)=y0, x(2)=x2, x(3)=y2.
    auto c0 = [](const Eigen::Vector4d&  )    { return true; };
    auto c1 = [](const Eigen::Vector4d& x)    { return ((x(0) == 0.0) || (x(2) == 0.0))
                                                    && ((x(1) == 0.0) || (x(3) == 0.0)); };
    auto c2 = [](const Eigen::Vector4d& x)    { return (x(2) >  0.50) && (x(3) > -0.75); };
    auto c3 = [](const Eigen::Vector4d& x)    { return (x(1) > -0.80) && (x(1) <  -0.6); };
    auto c4 = [](const Eigen::Vector4d& x)    { return  x(1) < -0.40; };
    auto c5 = [](const Eigen::Vector4d& x)    { return  x(1) < -0.25; };
    auto c6 = [](const Eigen::Vector4d& x)    { return  x(1) <  0.50; };
    auto c7 = [](const Eigen::Vector4d& x)    { return  x(1) <  0.00; };
    auto c8 = [](const Eigen::Vector4d& x)    { return  x(1) == -1.00; };
    auto c9 = [](const Eigen::Vector4d& x)    { return  x(1) < -0.375; };

    // Initial domain: a single leaf covering [-1,1] x [-1,1].
    AmrNode amr{AmrLeaf(-1.0, -1.0, 1.0, 1.0)};

    amr = amr_refine(c0, amr);
    amr = amr_refine(c0, amr);
    amr = amr_refine(c7, amr);
    amr = amr_refine(c5, amr);
    //amr = amr_refine(c9, amr);

    if (!amr_dump_path.empty()) {
        dump_amr_leaves_csv(amr_dump_path, amr);
    }
    return msh_amr_curv(p, amr, phys_msh);
}