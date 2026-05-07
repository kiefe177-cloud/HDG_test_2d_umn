Remove-Item -Recurse -Force build
mkdir build
cd build
cmake -G "MinGW Makefiles" .. 
mingw32-make
ctest

using namespace Eigen

Need to cd into output to plot data.

py ..\scripts\{filename}.py

-march=native or -march=x86-64-v3 for Eigenspeedup?


For sparse matrices:

Eigen::SparseMatrix<double> A(rows, cols);          // 1. empty shell
std::vector<Eigen::Triplet<double>> trips;          // 2. fill the recipe
trips.reserve(estimated_nnz);
//   ... emplace_back(row, col, value) for every non-zero ...
A.setFromTriplets(trips.begin(), trips.end());      // 3. bake the matrix