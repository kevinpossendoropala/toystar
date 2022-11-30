
#include <Cabana_Core.hpp>

#include <iostream>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <cmath>
#include <math.h>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <string>
#include <sstream>
#include <cstring>


//initial particle distribution lenght

// Spline Kernel
double kernel_spline(double x1, double y1,double z1,double x2,double y2, double z2, double h)
{

    double result = 0.0;
    double C = 1./(4.*M_PI*h*h*h);
    double q = sqrt(pow(x1-x2,2.)+pow((y1-y2),2.)+pow(z1-z2,2.))/h;
    if (q >= 0. && q < 1.)
    {
        result = C * (pow((2.0 - q), 3.) - (4.0 * pow((1.0 - q), 3.)));
    }

    if (q >= 1. && q < 2.)
    {
        result = C * (pow((2.0 - q), 3.));
    }

    if (q >= 2.)
    {
        result = 0.0;
    }

    return result;
}


double grad_kernel_x(double x1, double y1,double z1,double x2,double y2, double z2, double h)
{
    double result = 0.0;

    double dist = sqrt(pow(x1-x2,2.)+pow((y1-y2),2.)+pow(z1-z2,2.));
    double C = 1./(4.*M_PI*h*h*h);
    double q = dist/h;

    if (q >= 0 && q < 1)
    {
        result = ((C / pow(h, 3.)) * 3.0 * (x1 - x2) * (3.0 * dist - 4.0 * h));
    }

    if (q >= 1 && q < 2)
    {
        result = (((-3.0 * C) / (pow(h, 3.) * dist)) * (x1 - x2) * (pow((dist - 2.0 * h), 2.)));
    }

    if (q >= 2)
    {
        result = 0.0;
    }

    return result;
}
double grad_kernel_y(double x1, double y1,double z1,double x2,double y2, double z2, double h)
{
    double result = 0.0;

    double dist = sqrt(pow(x1-x2,2.)+pow((y1-y2),2.)+pow(z1-z2,2.));
    double C = 1./(4.*M_PI*h*h*h);
    double q = dist/h;

    if (q >= 0 && q < 1)
    {
        result = ((C / pow(h, 3.)) * 3.0 * (y1 - y2) * (3.0 * dist - 4.0 * h));
    }

    if (q >= 1 && q < 2)
    {
        result = (((-3.0 * C) / (pow(h, 3.) * dist)) * (y1 - y2) * (pow((dist - 2.0 * h), 2.)));
    }

    if (q >= 2)
    {
        result = 0.0;
    }

    return result;
}
double grad_kernel_z(double x1, double y1,double z1,double x2,double y2, double z2, double h)
{

    
    double dist = sqrt(pow(x1-x2,2.)+pow((y1-y2),2.)+pow(z1-z2,2.));
    double result = 0.0;
    double C = 1./(4.*M_PI*h*h*h);
    double q = dist/h;

    if (q >= 0 && q < 1)
    {
        result = ((C / pow(h, 3.)) * 3.0 * (z1 - z2) * (3.0 * dist - 4.0 * h));
    }

    if (q >= 1 && q < 2)
    {
        result = (((-3.0 * C) / (pow(h, 3.) * dist)) * (z1 - z2) * (pow((dist - 2.0 * h), 2.)));
    }

    if (q >= 2)
    {
        result = 0.0;
    }

    return result;
}

//---------------------------------------------------------------------------//
// Slice example.
//---------------------------------------------------------------------------//
void createParticles()
{

        double L = 1.;
    int npart = 10003;
    double h = 0.1;
    double R = 0.75;
    double dt = 0.01;
    double k = 0.1;
    double vis = 0.5;
    double M = 2.;
    double nu = 2./double(npart);
    double lambda = 15.*M_PI*4.*k*M/(8.*pow(R,5.)*pow(M_PI,3./2.));
    std::cout<< lambda <<std::endl;
    /*
       Start by declaring the types our tuples will store. Store a rank-2
       array of doubles, a rank-1 array of floats, and a single integer in
       each tuple.
    */
    using DataTypes = Cabana::MemberTypes<double[3] //position
                                          ,double[3]//velocity
                                          ,double[3]//acceleration
                                          ,double //pressure
                                          ,double /* density*/
                                          ,int /*id*/>;

    /*
      Next declare the vector length of our SoAs. This is how many tuples the
      SoAs will contain. A reasonable number for performance should be some
      multiple of the vector length on the machine you are using.
    */
    const int VectorLength = 4;

    /*
      Finally declare the memory space in which the AoSoA will be allocated
      and the execution space in which kernels will execute. In this example
      we are writing basic loops that will execute on the CPU. The HostSpace
      allocates memory in standard CPU RAM.

      Kokkos also supports execution on NVIDIA GPUs. To create an AoSoA
      allocated with CUDA Unified Virtual Memory (UVM) use
      `Kokkos::CudaUVMSpace` instead of `Kokkos::HostSpace`. The CudaUVMSpace
      allocates memory in managed GPU memory via `cudaMallocManaged`. This
      memory is automatically paged between host and device depending on the
      context in which the memory is accessed.
    */
    using MemorySpace = Kokkos::HostSpace;
    using ExecutionSpace = Kokkos::OpenMP;
    using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;

    /*
       Create the AoSoA. We define how many tuples the aosoa will
       contain. Note that if the number of tuples is not evenly divisible by
       the vector length then the last SoA in the AoSoA will not be entirely
       full (although its memory will still be allocated).
    */

   // it seens that it is best to use a even number, so that all particles can be accessed, if the number
   // divides evenly for VectorLenght, the arraySize( last_soa ) returns 0 : Kevin
    int num_particles = npart;
    Cabana::AoSoA<DataTypes, DeviceType, VectorLength> particles( "particles",
                                                              num_particles );

    /*
      Create a slice over each tuple member in the AoSoA. An integer template
      parameter is used to indicate which member to slice. A slice object
      simply wraps the data associated with an AoSoA member in a more
      conventient accessor structure. A slice therefore has the same memory
      space as the AoSoA from which it was derived. Slices may optionally be
      assigned a label. This label is not included in the memory tracker
      because slices are unmanaged memory but may still be used for diagnostic
      purposes.
    */

    auto position = Cabana::slice<0>( particles, "position" );
    auto velocity = Cabana::slice<1>( particles, "velocity" );
    auto acceleration = Cabana::slice<2>( particles, "acceleration" );
    auto pressure = Cabana::slice<3>( particles, "pressure" );
    auto density  = Cabana::slice<4>( particles, "density" );
    auto ids = Cabana::slice<5>( particles,"ids" );

/*
    auto po = Cabana::slice<0>( particles, "position" );
    auto v = Cabana::slice<1>( particles, "velocity" );
    auto pr = Cabana::slice<2>( particles, "density" );
    auto d  = Cabana::slice<3>( particles, "density" );
    auto i = Cabana::slice<4>( particles,"ids" );

    decltype( po )::atomic_access_slice position = po;
    decltype( v )::atomic_access_slice velocity = v;
    decltype( pr )::atomic_access_slice pressure = pr;
    decltype( d )::atomic_access_slice density = d;
    decltype( i )::atomic_access_slice ids = i;
*/
    /*
      Let's initialize the data using the 2D indexing scheme. Slice data can
      be accessed in 2D using the `access()` function. Note that both the SoA
      index and the array index are passed to this function.

      Also note that the slice object has a similar interface to the AoSoA for
      accessing the total number of tuples in the data structure and the array
      sizes.
    */
//#pragma omp parallel for
    for ( std::size_t s = 0; s < position.numSoA(); ++s ) //number of SoA from VectorLenght
    {
        
        //std::cout << s << std::endl;
        //std::cout << velocity.arraySize( s ) << std::endl;
        //initialize position for each particle
        //#pragma omp parallel for
        for ( std::size_t n = 0; n < position.arraySize( s ); ++n ){
     //number of particles loops
            for ( int i = 0; i < 3; ++i )
                    position.access( s, n, i ) = 2.*L*double((rand())%100)/100. -L ;
        }
        for ( int i = 0; i < 3; ++i )
            //#pragma omp parallel for
            for ( std::size_t a = 0; a < position.arraySize( s ); ++a ){
                velocity.access( s, a, i ) = 0.0;
                acceleration.access( s, a, i ) = 0.0;
            }
        //#pragma omp parallel for
        for ( std::size_t a = 0; a < position.arraySize(s); ++a )
            pressure.access( s, a ) = 10. ;
        //#pragma omp parallel for
        for ( std::size_t a = 0; a < position.arraySize( s ); ++a )
            density.access( s, a ) = 0 ;
    }

for ( std::size_t i = 0; i < particles.size(); ++i )
    ids( i ) = i;
    /*
    for ( std::size_t s = 0; s < position.numSoA(); ++s ) //number of SoA from VectorLenght
    {
        for ( std::size_t a = 0; a < position.arraySize( s ); ++a ){
        for ( int i = 0; i < 2; ++i )
                std::cout << "Particle " <<s*a+a << ", Position (" << i << "): " << position.access( s,a, i ) << std::endl;
        
        for ( int i = 0; i < 2; ++i )
            std::cout << "Particle " <<s+ s*a << ", velocity (" << i
                      << "): " << velocity.access( s,a, i ) << std::endl;

        std::cout << "Particle " <<s+ s*a << ", pressure: " << pressure.access(s,a)
                  << std::endl;
        }
    }/*

      Define the parameters of the Cartesian grid over which we will build the
      cell list. This is a simple 3x3x3 uniform grid on [0,3] in each
      direction. Each grid cell has a size of 1 in each dimension.
     */
    Cabana::deep_copy(acceleration,0);
    Cabana::deep_copy(density,0);
    double grid_min[3] = { -4.0, -4.0,-4.0 };
    double grid_max[3] = { 4.0, 4.0,4.0 };
    double grid_delta[3] = {2.*h, 2.*h ,2.*h};
    Cabana::LinkedCellList<DeviceType> cell_list( position, grid_delta,
                                                  grid_min, grid_max );

    /*
      Now permute the AoSoA (i.e. reorder the data) using the linked cell list.
    */
    Cabana::permute( cell_list, particles );

    
// define the kernel to add to density
auto vector_kernel =
    KOKKOS_LAMBDA( const int s, const int a ){
     Kokkos::atomic_add(&density.access(s,a),nu*kernel_spline(0.,0.,0. ,0.0,0.0,0.0,h));
 };

Cabana::SimdPolicy<VectorLength,ExecutionSpace> simd_policy( 0, num_particles );
Cabana::simd_parallel_for( simd_policy, vector_kernel, "vector_op" ); 
Kokkos::fence();


auto integrate_kernel =
    KOKKOS_LAMBDA( const int s, const int a )
    { Kokkos::atomic_add( &velocity.access(s,a,0) , acceleration.access(s,a,0)*dt);
    Kokkos::atomic_add( &velocity.access(s,a,1) , acceleration.access(s,a,1)*dt);
    Kokkos::atomic_add( &velocity.access(s,a,2), acceleration.access(s,a,2)*dt);
    Kokkos::atomic_add( &position.access(s,a,0), velocity.access(s,a,0)*dt);
    Kokkos::atomic_add( &position.access(s,a,1) , velocity.access(s,a,1)*dt);
    Kokkos::atomic_add( &position.access(s,a,2) , velocity.access(s,a,2)*dt);
 };

Cabana::SimdPolicy<VectorLength,ExecutionSpace> integrate_policy( 0, num_particles );


        double neighborhood_radius = 2.*h;
    double cell_ratio = 1.0;
    using ListAlgorithm = Cabana::FullNeighborTag;
    using ListType =
        Cabana::VerletList<MemorySpace, ListAlgorithm, Cabana::VerletLayout2D>;
    ListType verlet_list( position, 0, position.size(), neighborhood_radius,
                          cell_ratio, grid_min, grid_max );

    /*
      KERNEL 1 - First neighbors

      This kernel is used with the neighbor list created above and forwards
      it, along with indexing and threading tags to an underlying
      Kokkos::parallel_for. This first kernel thus indexes directly
      over both the central particle i and neighbors j.

      Note the atomic update to ensure multiple neighbors do not update the
      central particle simultaneously if threading over neighbors.
     */
    auto first_neighbor_kernel = KOKKOS_LAMBDA( const int i, const int j )
    {
        Kokkos::atomic_add( &density(i), nu*kernel_spline(position(i,0),position(i,1),position(i,2),
                                                        position(j,0),position(j,1),position(j,2),h ));
    };



    /*
      We define a standard Kokkos execution policy to use for our outer loop.
    */
    Kokkos::RangePolicy<ExecutionSpace> policy( 0, particles.size() );

    /*
      Finally, perform the parallel loop. This parallel_for concept
      in Cabana is complementary to existing parallel_for
      implementations in Kokkos. The serial tag indicates that the neighbor loop
      is serial, while the central particle loop uses threading.

      Notice that we do not have to directly interact with the neighbor list we
      created as in the previous VerletList example. This is instead done
      internally through the neighbor_parallel_for interface.

      Note: We fence after the kernel is completed for safety but this may not
      be needed depending on the memory/execution space being used. When the
      CUDA UVM memory space is used this fence is necessary to ensure
      completion of the kernel on the device before UVM data is accessed on
      the host. Not fencing in the case of using CUDA UVM will typically
      result in a bus error.
    */
    Cabana::neighbor_parallel_for( policy, first_neighbor_kernel, verlet_list,
                                   Cabana::FirstNeighborsTag(),
                                   Cabana::SerialOpTag(), "ex_1st_serial" );
    Kokkos::fence();
    auto eos_kernel =
    KOKKOS_LAMBDA( const int s, const int a )
    { Kokkos::atomic_add(&pressure.access(s,a),k*density.access(s,a)*density.access(s,a)) ;
    };
    


    Cabana::simd_parallel_for( simd_policy, eos_kernel, "eos" ); 
    Kokkos::fence();

    auto self_a_kernel =
    KOKKOS_LAMBDA( const int s, const int a )
    { Kokkos::atomic_add( &acceleration.access(s,a,0) ,-vis*velocity.access(s,a,0) -lambda*position.access(s,a,0));
    Kokkos::atomic_add( &acceleration.access(s,a,1) , -vis*velocity.access(s,a,1) -lambda*position.access(s,a,1));
    Kokkos::atomic_add( &acceleration.access(s,a,2), -vis*velocity.access(s,a,2) -lambda*position.access(s,a,2));
    };

    Cabana::simd_parallel_for( simd_policy, self_a_kernel, "self_a_kernel" ); 
    Kokkos::fence();

    auto acceleration_kernel = KOKKOS_LAMBDA( const int i, const int j )
    {
        Kokkos::atomic_add( &acceleration(i,0), -nu*(pressure(i)/(pow(density(i),2.)) +pressure(j)/(pow(density(j),2.) ))*grad_kernel_x(position(i,0),position(i,1),position(i,2),
                                                        position(j,0),position(j,1),position(j,2),h ));
        Kokkos::atomic_add( &acceleration(i,1), -nu*(pressure(i)/(pow(density(i),2.)) +pressure(j)/(pow(density(j),2.) ))*grad_kernel_y(position(i,0),position(i,1),position(i,2),
                                                        position(j,0),position(j,1),position(j,2),h ));
        Kokkos::atomic_add( &acceleration(i,2), -nu*(pressure(i)/(pow(density(i),2.)) +pressure(j)/(pow(density(j),2.) ))*grad_kernel_z(position(i,0),position(i,1),position(i,2),
                                                        position(j,0),position(j,1),position(j,2),h ));
    };

    Cabana::neighbor_parallel_for( policy, acceleration_kernel, verlet_list,
                                   Cabana::FirstNeighborsTag(),
                                   Cabana::SerialOpTag(), "acceleration" );
    Kokkos::fence();
    double Tmax = 15.;

    double t = 0.0;
    while(t < Tmax){
        std::ofstream myfile;
        std::string out_string;
        std::stringstream ss;
        std::stringstream sh;
        ss << t;
        out_string = "t-" + ss.str() + ".csv";
        std::cout << t << std:: endl;
        myfile.open(out_string.c_str());

        myfile <<"id"<<"," <<"x"
               << ","
               << "y"
               << ","
               << "z"
               << ","
               << "density"<< std::endl;
    for ( std::size_t i = 0; i < particles.size(); ++i )
        myfile << ids( i ) <<","
                  << position( i, 0 ) << "," << position( i, 1 ) << "," <<  position( i, 2 )<<", "<< density(i) <<std::endl;
    myfile.close();


    Cabana::simd_parallel_for( integrate_policy, integrate_kernel, "integrate" ); 
    Kokkos::fence();

    Cabana::deep_copy(acceleration,0);
    Cabana::deep_copy(density,0);

    Cabana::simd_parallel_for( simd_policy, vector_kernel, "vector_op" ); 
    Kokkos::fence();

    ListType verlet_list( position, 0, position.size(), neighborhood_radius,
                          cell_ratio, grid_min, grid_max );


    Cabana::neighbor_parallel_for( policy, first_neighbor_kernel, verlet_list,
                                   Cabana::FirstNeighborsTag(),
                                   Cabana::SerialOpTag(), "ex_1st_serial" );
    Kokkos::fence();
    auto eos_kernel =
    KOKKOS_LAMBDA( const int s, const int a )
    { pressure.access(s,a) = k*density.access(s,a)*density.access(s,a) ;
    };
    Cabana::simd_parallel_for( simd_policy, eos_kernel, "eos" ); 
    Kokkos::fence();

    Cabana::simd_parallel_for( simd_policy, self_a_kernel, "self_a_kernel" ); 
    Kokkos::fence();


    Cabana::neighbor_parallel_for( policy, acceleration_kernel, verlet_list,
                                   Cabana::FirstNeighborsTag(),
                                   Cabana::SerialOpTag(), "acceleration" );
    Kokkos::fence();
    std::cout << t <<std::endl;
    t += dt;
    }   






}

//---------------------------------------------------------------------------//
// Main.
//---------------------------------------------------------------------------//
int main( int argc, char* argv[] )
{   
    Kokkos::ScopeGuard scope_guard( argc, argv );
    srand (time(NULL));
    createParticles();

    return 0;
}

//---------------------------------------------------------------------------//
