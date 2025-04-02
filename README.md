# neuron_transport_3D_v3_archive
Repository for files to analyze mass transport around neurons from image stack data.

1) Data is downloaded from Microns Explorer using ImageSegmentation_14.0.py.  Note the folder structure in use: start one folder with the program files and another folder called 'Neuron_transport_data' in the same directory.  The settings are stored in the file 'Simulation_settings_4.0.csv. The simulation number is the first column and in the current program is set to '1'.  The base volume is specified in columns 2-5 and the number of adjacent volumes is specified in the next six columns.  The final column 'z_filled' is used to specify if zloc has been multiplied by 5 with 4 interpolated frames inserted.  At the start, z_filled is 0.
2) The downloaded data is mode filtered and interpolated.  This is in a separate repository.  Interpolation is necessary for the following steps, so follow the directions at the other repository to accomplish this.
3) After mode filtering and interpolation, run unique_neurons_4.1_smooth.jl.  This depends on the file 'Simulation_settings_4.0_smoothed.csv'. The sim_number is set in the program.  The program writes the file 'MarchingCubes_settings.csv' that is used by MarchingCubes*.c and Ply2JLD2*.jl.
4) Compile and run MarchingCubes3D_2.0a_smoothz_mpi.c.  This requires a functioning MPI implementation.  We used Intel's oneAPI/2023.2.1 or Intel(R) MPI Library 2018 Update 3 for Windows but other implementations of MPI should work.  Compile and run the program from the base folder as:
mpicc -g -o MarchingCubes3D_2.0a_mpi MarchingCubes3D_2.0a_smoothz_mpi.c -lm
mpiexec -n 40 ./MarchingCubes3D_2.0a_mpi 1
The argument to the program '1' refers to a row in 'MarchingCubes_settings*.csv'. This is a single analysis volume.  All analysis volumes may be run together using runmc.sh.
