#pragma once

#include <string>
#include "hdg/Msh.h"
#include "hdg/phys_mesh.h"

Msh msh_test_grad(int p, const PhysMesh& phys_msh,
                  const std::string& amr_dump_path = "");