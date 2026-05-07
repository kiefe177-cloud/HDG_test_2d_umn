/*

msh_connect

GLQuad

ndgrid

amr_flatten

calcgeoms

*/
#include "hdg/msh_amr_curv.h"
#include "hdg/msh_connect.h"
#include "hdg/GLDeriv.h"
#include "hdg/GLQuad.h"
#include "hdg/ndgrid.h"
#include "hdg/amr_flatten.h"
#include "hdg/calcgeom_struct.h"
#include "hdg/calc_geom_iso_tfi.h"
#include "hdg/vec.h"
#include "hdg/sd.h"
#include <fstream>

void write_csv(const Eigen::MatrixXi& M, const std::string& path) {
    std::ofstream f(path);
    for (int i = 0; i < M.rows(); ++i) {
        for (int j = 0; j < M.cols(); ++j) {
            f << M(i, j);
            if (j + 1 < M.cols()) f << ",";
        }
        f << "\n";
    }
}


Msh msh_amr_curv(const int p, AmrNode& amr,const PhysMesh& phys_msh){

    Msh msh;
    calc_geom_output out_parent;
    calc_geom_output out_sw;
    calc_geom_output out_se;
    calc_geom_output out_nw;
    calc_geom_output out_ne;

    Connect connect = msh_connect(amr);

    msh.FtoE = connect.FtoE;
    msh.EtoF = connect.EtoF;

    //write_csv(msh.EtoF, "EtoF_cpp.csv");
    //write_csv(msh.FtoE, "FtoE_cpp.csv");

    Eigen::VectorXd r,w;

    GLQuad(p,r,w);
    Eigen::MatrixXd D = GLDeriv(p);

    auto[xi,eta] = ndgrid(r,r);

    // std::cout << "r:\n" << r.transpose() << "\n";
    // std::cout << "xi:\n" << xi << "\n";
    // std::cout << "eta:\n" << eta << "\n";

    Eigen::MatrixXd elem_matrix = amr_flatten(amr);

    //printf("Test");

    int Ne = elem_matrix.rows();
    int Nf = msh.FtoE.cols();

    msh.elem.resize(Ne);
    msh.face.resize(Nf);

    for (int ie = 0; ie < Ne; ++ie){

        double phi_min = elem_matrix(ie,0);
        double psi_min = elem_matrix(ie,1);
        double phi_max = elem_matrix(ie,2);
        double psi_max = elem_matrix(ie,3);
        
        //std::cout << "elem_matrix size: " << elem_matrix.rows() << " x " << elem_matrix.cols() << std::endl;
        //std::cout << "ie = " << ie << std::endl;
        // calcgeom_iso_tfi
        out_parent = calc_geom_iso_tfi(p,xi,eta,elem_matrix.row(ie),phys_msh);

        // std::cout << "Out_parent J1" << out_parent.Jf_1 << "\n";
        // std::cout << "Out_parent J2" << out_parent.Jf_2 << "\n";
        // std::cout << "Out_parent J3" << out_parent.Jf_3 << "\n";
        // std::cout << "Out_parent J4" << out_parent.Jf_4 << "\n";
   
        msh.elem[ie].p = p;
        msh.elem[ie].x = out_parent.x1;
        msh.elem[ie].y = out_parent.y1;
        msh.elem[ie].J = sd(vec(out_parent.J));
        msh.elem[ie].Ja11 = sd(vec(out_parent.Ja11));
        msh.elem[ie].Ja12 = sd(vec(out_parent.Ja12));
        msh.elem[ie].Ja21 = sd(vec(out_parent.Ja21));
        msh.elem[ie].Ja22 = sd(vec(out_parent.Ja22));
        msh.elem[ie].w1d = w;
        
        

        double phi_c = (phi_max + phi_min)/2;
        double psi_c = (psi_max + psi_min)/2;

        Eigen::Vector4d elem_sw(phi_min, psi_min, phi_c, psi_c);
        Eigen::Vector4d elem_se(phi_c, psi_min, phi_max, psi_c);
        Eigen::Vector4d elem_nw(phi_min, psi_c, phi_c, psi_max);
        Eigen::Vector4d elem_ne(phi_c, psi_c, phi_max, psi_max);

        out_sw = calc_geom_iso_tfi(p,xi,eta,elem_sw,phys_msh);
        out_se = calc_geom_iso_tfi(p,xi,eta,elem_se,phys_msh);
        out_nw = calc_geom_iso_tfi(p,xi,eta,elem_nw,phys_msh);
        out_ne = calc_geom_iso_tfi(p,xi,eta,elem_ne,phys_msh);

        // std::cout << "Out_sw J1 " << out_sw.Jf_1 << "\n";
        // std::cout << "Out_sw J2 " << out_sw.Jf_2 << "\n";
        // std::cout << "Out_sw J3 " << out_sw.Jf_3 << "\n";
        // std::cout << "Out_sw J4 " << out_sw.Jf_4 << "\n";

        
        // std::cout << "Out_se J1 " << out_se.Jf_1 << "\n";
        // std::cout << "Out_se J2 " << out_se.Jf_2 << "\n";
        // std::cout << "Out_se J3 " << out_se.Jf_3 << "\n";
        // std::cout << "Out_se J4 " << out_se.Jf_4 << "\n";

        // std::cout << "Out_nw J1 " << out_nw.Jf_1 << "\n";
        // std::cout << "Out_nw J2 " << out_nw.Jf_2 << "\n";
        // std::cout << "Out_nw J3 " << out_nw.Jf_3 << "\n";
        // std::cout << "Out_nw J4 " << out_nw.Jf_4 << "\n";

        
        // std::cout << "Out_ne J1 " << out_ne.Jf_1 << "\n";
        // std::cout << "Out_ne J2 " << out_ne.Jf_2 << "\n";
        // std::cout << "Out_ne J3 " << out_ne.Jf_3 << "\n";
        // std::cout << "Out_ne J4 " << out_ne.Jf_4 << "\n";
   

        // Left face x-lo
        if (msh.EtoF(0,ie) >= 0){
            int f1 = msh.EtoF(0,ie);
            msh.face[f1].p = p;
            msh.face[f1].x = vec(out_parent.x1.row(0));
            msh.face[f1].y = vec(out_parent.y1.row(0));
            msh.face[f1].nx = sd(vec(out_parent.Ja11.row(0)));
            msh.face[f1].ny = sd(vec(out_parent.Ja12.row(0)));
            msh.face[f1].nn = sd(vec((out_parent.Ja11.row(0).array().square() + out_parent.Ja12.row(0).array().square()).sqrt()));
            msh.face[f1].w1d = w;
            msh.face[f1].Jf = vec(out_parent.Jf_1);
        } else {
            int f1 = msh.EtoF(4,ie);
            int f2 = msh.EtoF(8,ie);
            // Face 5 - sw
            msh.face[f1].p = p;
            msh.face[f1].x = vec(out_sw.x1.row(0));
            msh.face[f1].y = vec(out_sw.y1.row(0));
            msh.face[f1].nx = sd(vec(out_sw.Ja11.row(0)));
            msh.face[f1].ny = sd(vec(out_sw.Ja12.row(0)));
            msh.face[f1].nn = sd(vec((out_sw.Ja11.row(0).array().square() + out_sw.Ja12.row(0).array().square()).sqrt()));
            msh.face[f1].w1d = w;
            msh.face[f1].Jf = vec(out_sw.Jf_1);
            // Face 9 - nw
            msh.face[f2].p = p;
            msh.face[f2].x = vec(out_nw.x1.row(0));
            msh.face[f2].y = vec(out_nw.y1.row(0));
            msh.face[f2].nx = sd(vec(out_nw.Ja11.row(0)));
            msh.face[f2].ny = sd(vec(out_nw.Ja12.row(0)));
            msh.face[f2].nn = sd(vec((out_nw.Ja11.row(0).array().square() + out_nw.Ja12.row(0).array().square()).sqrt()));
            msh.face[f2].w1d = w;
            msh.face[f2].Jf = vec(out_nw.Jf_1);
        }

        // Right face x-hi
        if (msh.EtoF(1,ie) >= 0){
            int f1 = msh.EtoF(1,ie);
            msh.face[f1].p = p;
            msh.face[f1].x = vec(out_parent.x1.row(p));
            msh.face[f1].y = vec(out_parent.y1.row(p));
            msh.face[f1].nx = sd(vec(out_parent.Ja11.row(p)));
            msh.face[f1].ny = sd(vec(out_parent.Ja12.row(p)));
            msh.face[f1].nn = sd(vec((out_parent.Ja11.row(p).array().square() + out_parent.Ja12.row(p).array().square()).sqrt()));
            msh.face[f1].w1d = w;
            msh.face[f1].Jf = vec(out_parent.Jf_2);
        } else {
            int f1 = msh.EtoF(5,ie);
            int f2 = msh.EtoF(9,ie);
            // Face 6 - se
            msh.face[f1].p = p;
            msh.face[f1].x = vec(out_se.x1.row(p));
            msh.face[f1].y = vec(out_se.y1.row(p));
            msh.face[f1].nx = sd(vec(out_se.Ja11.row(p)));
            msh.face[f1].ny = sd(vec(out_se.Ja12.row(p)));
            msh.face[f1].nn = sd(vec((out_se.Ja11.row(p).array().square() + out_se.Ja12.row(p).array().square()).sqrt()));
            msh.face[f1].w1d = w;
            msh.face[f1].Jf = vec(out_se.Jf_2);
            // Face 10 - ne
            msh.face[f2].p = p;
            msh.face[f2].x = vec(out_ne.x1.row(p));
            msh.face[f2].y = vec(out_ne.y1.row(p));
            msh.face[f2].nx = sd(vec(out_ne.Ja11.row(p)));
            msh.face[f2].ny = sd(vec(out_ne.Ja12.row(p)));
            msh.face[f2].nn = sd(vec((out_ne.Ja11.row(p).array().square() + out_ne.Ja12.row(p).array().square()).sqrt()));
            msh.face[f2].w1d = w;
            msh.face[f2].Jf = vec(out_ne.Jf_2);
        }

        // Bottom face y-lo
        if (msh.EtoF(2,ie) >= 0){
            int f1 = msh.EtoF(2,ie);
            msh.face[f1].p = p;
            msh.face[f1].x = vec(out_parent.x1.col(0));
            msh.face[f1].y = vec(out_parent.y1.col(0));
            msh.face[f1].nx = sd(vec(out_parent.Ja21.col(0)));
            msh.face[f1].ny = sd(vec(out_parent.Ja22.col(0)));
            msh.face[f1].nn = sd(vec((out_parent.Ja21.col(0).array().square() + out_parent.Ja22.col(0).array().square()).sqrt()));
            msh.face[f1].w1d = w;
            msh.face[f1].Jf = vec(out_parent.Jf_3);
        } else {
            int f1 = msh.EtoF(6,ie);
            int f2 = msh.EtoF(10,ie);
            // Face 7 - sw
            msh.face[f1].p = p;
            msh.face[f1].x = vec(out_sw.x1.col(0));
            msh.face[f1].y = vec(out_sw.y1.col(0));
            msh.face[f1].nx = sd(vec(out_sw.Ja21.col(0)));
            msh.face[f1].ny = sd(vec(out_sw.Ja22.col(0)));
            msh.face[f1].nn = sd(vec((out_sw.Ja21.col(0).array().square() + out_sw.Ja22.col(0).array().square()).sqrt()));
            msh.face[f1].w1d = w;
            msh.face[f1].Jf = vec(out_sw.Jf_3);
            // Face 11 - se
            msh.face[f2].p = p;
            msh.face[f2].x = vec(out_se.x1.col(0));
            msh.face[f2].y = vec(out_se.y1.col(0));
            msh.face[f2].nx = sd(vec(out_se.Ja21.col(0)));
            msh.face[f2].ny = sd(vec(out_se.Ja22.col(0)));
            msh.face[f2].nn = sd(vec((out_se.Ja21.col(0).array().square() + out_se.Ja22.col(0).array().square()).sqrt()));
            msh.face[f2].w1d = w;
            msh.face[f2].Jf = vec(out_se.Jf_3);
        }

        // Top face y-hi
        if (msh.EtoF(3,ie) >= 0){
            int f1 = msh.EtoF(3,ie);
            msh.face[f1].p = p;
            msh.face[f1].x = vec(out_parent.x1.col(p));
            msh.face[f1].y = vec(out_parent.y1.col(p));
            msh.face[f1].nx = sd(vec(out_parent.Ja21.col(p)));
            msh.face[f1].ny = sd(vec(out_parent.Ja22.col(p)));
            msh.face[f1].nn = sd(vec((out_parent.Ja21.col(p).array().square() + out_parent.Ja22.col(p).array().square()).sqrt()));
            msh.face[f1].w1d = w;
            msh.face[f1].Jf = vec(out_parent.Jf_4);
        } else {
            int f1 = msh.EtoF(7,ie);
            int f2 = msh.EtoF(11,ie);
            // Face 8 - nw
            msh.face[f1].p = p;
            msh.face[f1].x = vec(out_nw.x1.col(p));
            msh.face[f1].y = vec(out_nw.y1.col(p));
            msh.face[f1].nx = sd(vec(out_nw.Ja21.col(p)));
            msh.face[f1].ny = sd(vec(out_nw.Ja22.col(p)));
            msh.face[f1].nn = sd(vec((out_nw.Ja21.col(p).array().square() + out_nw.Ja22.col(p).array().square()).sqrt()));
            msh.face[f1].w1d = w;
            msh.face[f1].Jf = vec(out_nw.Jf_4);
            // Face 12 - ne
            msh.face[f2].p = p;
            msh.face[f2].x = vec(out_ne.x1.col(p));
            msh.face[f2].y = vec(out_ne.y1.col(p));
            msh.face[f2].nx = sd(vec(out_ne.Ja21.col(p)));
            msh.face[f2].ny = sd(vec(out_ne.Ja22.col(p)));
            msh.face[f2].nn = sd(vec((out_ne.Ja21.col(p).array().square() + out_ne.Ja22.col(p).array().square()).sqrt()));
            msh.face[f2].w1d = w;
            msh.face[f2].Jf = vec(out_ne.Jf_4);
        }
    }

    return msh;
}