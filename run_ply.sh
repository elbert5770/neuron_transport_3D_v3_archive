#!/bin/bash

# Loop through arguments 1-8
for i in {1..8}
do
    echo "Running PlyToJLD2 with argument $i"
    mpiexec -n 40 julia --depwarn=yes --project ./Neuron_transport_programs/PlyToJLD2_3D_1.0a.jl $i
    
    # Check if the program executed successfully
    if [ $? -eq 0 ]; then
        echo "Successfully completed run $i"
    else
        echo "Error in run $i"
    fi
done

echo "All runs completed"