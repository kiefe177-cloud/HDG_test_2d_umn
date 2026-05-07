#include <fstream>
#include <iostream>
#include <stdexcept>
#include "hdg/read_cone_dat_file.h"

ConeData read_cone_dat_file(const std::string& filename){
    // Reads in a .dat file in a string format and outputs the grid struct
    // output [matrix x, matrix y, matrix z, int nx, int ny, matrix coords]

    // 1. Open the file
    std::ifstream infile(filename);

    if (!infile.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    // 2. Read the dimensions
    int nx, ny;
    if (!(infile >> nx >> ny)){
        throw std::runtime_error("Error reading dimensions from file.");
    }

    int numNodes = nx*ny;

    // 3. Prepare Outpus
    ConeData data;
    data.nx = nx;
    data.ny = ny;

    data.x.resize(ny,nx);
    data.y.resize(ny,nx);
    data.z.resize(ny,nx);

    data.coords.resize(numNodes,3);

    float val_x, val_y, val_z;

    for (int k = 0; k < numNodes; ++k){

        if (!(infile >> val_x >> val_y >> val_z)){
            throw std::runtime_error("File ended unexpectedly or bad format");
        }

        // Store raw coords (Row K)
        data.coords(k,0) = val_x;
        data.coords(k,1) = val_y;
        data.coords(k,2) = val_z;

    }
    // 4. Reshape columns into grids
    data.x = data.coords.col(0).reshaped(ny,nx);
    data.y = data.coords.col(1).reshaped(ny,nx);
    data.z = data.coords.col(2).reshaped(ny,nx);

    // 5. Close file
    infile.close();
    return data;
}
