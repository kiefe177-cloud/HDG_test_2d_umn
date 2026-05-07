#pragma once

#include "hdg/Msh.h"
#include "hdg/amr_number.h"
#include "hdg/amr_connect.h"
#include "hdg/phys_mesh.h"

Msh msh_amr_curv(const int p, AmrNode& amr,const PhysMesh& phys_msh);