# CABANA toystar prototype



## To compile 
Install Cabana and Kokkos as described in https://github.com/ECP-copa/Cabana/wiki/Build-Instructions/#building-cabana 

After this, you need to change line 25 from "CabanaConfig.cmake" in the build directory inside Cabana directory, and replace "${CMAKE_CURRENT_LIST_DIR}" by "where_you_cloned_cabana/Cabana/build/install/lib/cmake/Cabana/"

then inside of this project directory 
make build
cd build 
cmake .. - -DCMAKE_PREFIX_PATH=where_you_cloned_cabana/Cabana/build/
make -j



