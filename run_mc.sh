#!/bin/bash

# Loop through arguments 1-8
for i in {1..8}
do
    echo "Running MarchingCubes with argument $i"
    mpirun -n 40 ./MarchingCubes3D_2.0a_smoothz_mpi $i
    
    # Check if the program executed successfully
    if [ $? -eq 0 ]; then
        echo "Successfully completed run $i"
    else
        echo "Error in run $i"
    fi
done

echo "All runs completed"