# Written by: Donald L. Elbert & Dylan Esguerra, University of Washington
# Copyright (c) 2022-2024 University of Washington
# All rights reserved.

# Redistribution and use in source and binary forms, with or without modification, are permitted provided
#  that the following conditions are met:

# Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 
# Neither the name of the University of Washington nor the names of its contributors may be used to endorse 
# or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE UNIVERSITY OF WASHINGTON AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED 
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY OF WASHINGTON OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT 
# OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON 
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING 
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
using FileIO
using CSV
using Tables
using LinearAlgebra
using WriteVTK
using JLD2
using SparseArrays
using StaticArrays
using IterativeSolvers
using MPI
using Statistics: mean


struct BC
    ID::Int32
    type::Int32
    value::Float32
end

struct Simulation_parameters
    Δt::Float32
    BCarray::Vector{BC}
    ICvalue::Float32
    ℾ::Float32
    ρ::Float32
    ū::Vector{Float32}
    Sourceᵤ::Float32
    Sourceₚ::Float32
end

struct Structure_cell
    npf::Int32
    num_faces::Int32
    num_nodes::Int32
    num_edges::Int32
    num_boundary_edges::Int32
    faces::Array{Int32,2}
    nodes::Array{Float32,2}
    normals::Array{Float32,2}
    edges::Array{Int32,2}
    face_center::Array{Float32,2}
end

struct Solution_setup
    Sim_parameters::Simulation_parameters
    weight_node::Array{Float32,2}
    t̂_dot_lf_δ_edge::Array{Float32,2}
    volume_face::Vector{Float32}
    S::Array{Float32,2}
    Sedge::Array{Float32,3}
    E::Array{Float32,2}
    Ef::Vector{Float32}
    T::Array{Float32,2}
    dCF::Vector{Float32}
    rCF::Array{Float32,2}
    weight_edge::Vector{Float32}
end

struct Matrix_setup
    sourceᵤ::Vector{Float32}
    a_sparse::SparseMatrixCSC{Float32, Int32}
    boundary_node_values::Array{Float32,2}
end

struct Boundary_edges
    boundary_edges::Array{Int32,2}
    interior_boundary_faces::Array{Int32,1}
    sum_interior_faces::Int32
    split_boundary_edges::Int64
    edge_to_boundary_edge::Vector{Int64}
    interior_boundary_nodes::Array{Int32,2}
end

struct Match_face
    match_faces::Matrix{Int32}
    total_face_matches::Int64
    owner_temp::Matrix{Int32}
    number_matched_cells::Int32
end

struct Owner_neighbor
    num_cells::Int32
    num_matches::Int64
    cell_list::Vector{Int64}
    owner::Matrix{Int32}
    neighbor::Matrix{Int32}
    neighbor_sort_vec::Vector{Int32}
    owner_neighbor_ptr::Matrix{Int32}
    num_faces_array::Vector{Int32}
end

struct Cube_properties
    cells_on_faces::Array{Int32,2}
    cells_on_cube_side::Vector{Int32}
    edges_on_cube_side::Vector{Int32}
end

struct Ghost_cells_dynamic
    cube_face_conc::Vector{Vector{Float32}}
    cube_face_conc_ghost::Vector{Vector{Float32}}
end

struct Ghost_cells_static
    cube_face_S::Vector{Array{Float32,2}}
    cube_face_E::Vector{Array{Float32,2}}
    cube_face_Ef::Vector{Vector{Float32}} 
    cube_face_T::Vector{Array{Float32,2}}     
    cube_face_dCF::Vector{Vector{Float32}}
    cube_face_rCF::Vector{Array{Float32,2}}  
    cube_face_weight_edge::Vector{Vector{Float32}}
    cube_face_t̂_dot_lf_δ_edge::Vector{Vector{Float32}}
end


function neuron_solve(timestep::Int64, path::String, list_number::Int64, extension::String, plot_vtu::Bool, comm::MPI.Comm, rank_comm::Int64, size_comm::Int64, cell_list::Vector{Int64}, owner_neighbor_ptr::Matrix{Int32}, edges_on_cube_side::Vector{Int32}, win_owner_match::Vector{MPI.Win}, win_neighbor_match::Vector{MPI.Win}, win_phi::Vector{MPI.Win}, win_phi_owner::Vector{MPI.Win}, win_phi_neighbor::Vector{MPI.Win}, win_cells_on_faces::MPI.Win, win_cube_face_conc::Vector{MPI.Win}, win_cube_face_conc_ghost::Vector{MPI.Win},win_cube_face_area_delta::Vector{MPI.Win})
   
    cell_number = cell_list[list_number]
    fid_jld_st = jldopen(string(path,cell_number,extension,"_structure",".jld2"))
    # display(fid_jld_st)
    root = 0
    structure_setup::Structure_cell = fid_jld_st["structure_setup"] 

    npf::Int64 = structure_setup.npf
    num_faces::Int64 = structure_setup.num_faces
    num_nodes::Int64 = structure_setup.num_nodes
    num_edges::Int64 = structure_setup.num_edges
    faces::Array{Int32,2} = structure_setup.faces
    nodes::Array{Float32,2} = structure_setup.nodes
    edges::Array{Int32,2} = structure_setup.edges
    close(fid_jld_st)

    
    
    fid_jld_sol = jldopen(string(path,cell_number,extension,"_setup",".jld2"))
    # display(fid_jld_sol)

    solution_setup::Solution_setup = fid_jld_sol["solution_setup"]

    
    # S::Array{Float32,2} = solution_setup.S
    # Sedge::Array{Float32,3} = solution_setup.Sedge
    # E::Array{Float32,2} = solution_setup.E
    # Ef::Vector{Float32} = solution_setup.Ef
    # T::Array{Float32,2} = solution_setup.T
    # dCF::Vector{Float32} = solution_setup.dCF
    # rCF::Array{Float32,2} = solution_setup.rCF
    # weight_edge::Vector{Float32} = solution_setup.weight_edge
    volume_face::Vector{Float32} = solution_setup.volume_face
    weight_node::Array{Float32,2} = solution_setup.weight_node
    t̂_dot_lf_δ_edge::Array{Float32,2} = solution_setup.t̂_dot_lf_δ_edge


    Sim_parameters::Simulation_parameters = solution_setup.Sim_parameters
    Δt::Float32 = Sim_parameters.Δt
    BCarray::Vector{BC} = Sim_parameters.BCarray
    ICvalue::Float32 = Sim_parameters.ICvalue
    ℾ::Float32 = Sim_parameters.ℾ
    ū::Vector{Float32} = Sim_parameters.ū
   
    close(fid_jld_sol)
   
    fid_jld_mat = jldopen(string(path,cell_number,extension,"_matrix_setup",".jld2"))
    # display(fid_jld_sol)

    matrix_setup::Matrix_setup = fid_jld_mat["matrix_setup"]
    sourceᵤ::Vector{Float32} = matrix_setup.sourceᵤ
    a_sparse::SparseMatrixCSC{Float32, Int64} = matrix_setup.a_sparse
    boundary_node_values::Array{Float32,2} = matrix_setup.boundary_node_values
    close(fid_jld_mat)

    fid_jld_be = jldopen(string(path,cell_number,extension,"_boundary_edges",".jld2"))
    boundary_setup::Boundary_edges = fid_jld_be["boundary_setup"]
    boundary_edges::Array{Int64,2} = boundary_setup.boundary_edges 
    split_boundary_edges::Int64 = boundary_setup.split_boundary_edges
    sum_interior_faces::Int64 = boundary_setup.sum_interior_faces
    interior_boundary_faces = zeros(Int32,sum_interior_faces)
    interior_boundary_faces::Array{Int32,1} = boundary_setup.interior_boundary_faces
    
    close(fid_jld_be)
    
    
    phi = Vector{Float32}(undef, num_faces)
    phi° = Vector{Float32}(undef, num_faces)

    MPI.Win_lock(win_phi[list_number];rank=0,type=:exclusive)
    MPI.Get!(phi, win_phi[list_number];rank=0)
    MPI.Win_unlock(win_phi[list_number];rank=0)
    
    b = Vector{Float32}(undef, num_faces)
    sourceₛ = zeros(Float32,num_faces)
    sourceₜ = Vector{Float32}(undef,num_faces)
   

    if npf == 3
        circle_path = [1 2;2 3;3 1]
    else
        if npf == 4
            circle_path = [1 2;2 3;3 4;4 1]
        else
            println("npf (node per face) must be 3 or 4)")
            return
        end
    end
    
    eps = Float32(1e-14)
   
    cells_on_faces_local = Vector{Int32}(undef,18)
                   
    MPI.Win_lock(win_cells_on_faces;rank=root,type=:exclusive)
    MPI.Get!(cells_on_faces_local, win_cells_on_faces;rank=root,disp=(list_number-1)*18)
    MPI.Win_unlock(win_cells_on_faces;rank=root)
    
    for iter in 1:6
        if cells_on_faces_local[12+iter] > 0
        
            face_vector_start = cells_on_faces_local[12+iter]
            
            cell_vector_start = cells_on_faces_local[6+iter]
            num_boundary_faces = cells_on_faces_local[iter]
       
            ghost_conc = zeros(Float32,num_boundary_faces)
            
            MPI.Win_lock(win_cube_face_conc_ghost[iter];rank=root,type=:exclusive)
            MPI.Get!(ghost_conc, win_cube_face_conc_ghost[iter];rank=root,disp=face_vector_start-1)
            MPI.Win_unlock(win_cube_face_conc_ghost[iter];rank=root)
            

            size_area_delta = edges_on_cube_side[iter]
            
            ghost_Sx = zeros(Float32, num_boundary_faces)
            ghost_Sy = zeros(Float32, num_boundary_faces)
            ghost_Sz = zeros(Float32, num_boundary_faces)
            
            MPI.Win_lock(win_cube_face_area_delta[iter]; rank=root, type=:exclusive)
            MPI.Get!(ghost_Sx, win_cube_face_area_delta[iter]; rank=root, disp=face_vector_start-1)
            MPI.Win_unlock(win_cube_face_area_delta[iter]; rank=root)
            MPI.Win_lock(win_cube_face_area_delta[iter]; rank=root, type=:exclusive)
            MPI.Get!(ghost_Sy, win_cube_face_area_delta[iter]; rank=root, disp=size_area_delta+face_vector_start-1)
            MPI.Win_unlock(win_cube_face_area_delta[iter]; rank=root)
            MPI.Win_lock(win_cube_face_area_delta[iter]; rank=root, type=:exclusive)
            MPI.Get!(ghost_Sz, win_cube_face_area_delta[iter]; rank=root, disp=size_area_delta*2+face_vector_start-1)
            MPI.Win_unlock(win_cube_face_area_delta[iter]; rank=root)
            
            ghost_Ef = zeros(Float32, num_boundary_faces)
            
            MPI.Win_lock(win_cube_face_area_delta[iter]; rank=root, type=:exclusive)
            MPI.Get!(ghost_Ef, win_cube_face_area_delta[iter]; rank=root, disp=size_area_delta*12+face_vector_start-1)
            MPI.Win_unlock(win_cube_face_area_delta[iter]; rank=root)
            ghost_dCF = zeros(Float32, num_boundary_faces)
            MPI.Win_lock(win_cube_face_area_delta[iter]; rank=root, type=:exclusive)
            MPI.Get!(ghost_dCF, win_cube_face_area_delta[iter]; rank=root, disp=size_area_delta*13+face_vector_start-1)
            MPI.Win_unlock(win_cube_face_area_delta[iter]; rank=root)
            
            if mod(iter,2) == 0
                n̂_edge = 1.0
            else
                n̂_edge = -1.0
            end
            
            
            
            for edge_number in 1:num_boundary_faces
                cell_index_value = edge_number + cell_vector_start - 1
                face_value = interior_boundary_faces[cell_index_value]
                face_index_value = edge_number + face_vector_start - 1
                F = dot(ū, n̂_edge .*[ghost_Sx[edge_number] ghost_Sy[edge_number] ghost_Sz[edge_number]])
                ⎹F⎸ = abs(F)
                
                sourceᵤ[face_value] += ghost_conc[edge_number]* (ℾ * ghost_Ef[edge_number] / ghost_dCF[edge_number] - (F -⎹F⎸)/2)
               
            end

        end
        
    end


    max_cg_iter = 20
    

    # Vectorized operation to initialize phi°
    phi° .= ifelse.(phi .< eps, zero(Float32), phi)
    
    phi_nodes = zeros(Float32,num_nodes)
    
    grad_phi_edge = zeros(Float32,3)
    grad_phi_b = zeros(Float32,3)
    phi_bedge = Float32(0.0)
    if (timestep == 1 && plot_vtu == true  ) 
          
        vtk_faces = Vector{MeshCell}(undef,num_faces)
        for (i,face) in enumerate(eachcol(faces))
            vtk_faces[i] = MeshCell(VTKCellTypes.VTK_TRIANGLE,face[1:npf])
        end
        sim_write_vtk(path,cell_number,extension,timestep-1,Δt,nodes,vtk_faces,phi)
        
    end

    orientation = 0
    neighbor_face = 0
    edge = 0
    current_edge = 0 
    bedge_type = 0
    
    for cg_iter in range(1,max_cg_iter)

    
        fill!(phi_nodes, 0.0)

        for (ic,face) in enumerate(eachcol(faces))
            
            @inbounds for j in range(1,npf)
                phi_nodes[face[j]] += phi[ic]*weight_node[j,ic]               
            end
        end
        for (i,bedge) in enumerate(eachcol(boundary_edges))
            
            if bedge[5] < 7
                phi_nodes[bedge[1]] = boundary_node_values[1,i]
                phi_nodes[bedge[2]] = boundary_node_values[2,i]
       
                
            end
            
            
        end
        
        # Compute sources  
        for (ic,face) in enumerate(eachcol(faces))
           
            sourceₜ[ic] = phi°[ic]*volume_face[ic]/Δt
            sourceₛ[ic] = 0.0
            
            @inbounds for j in 1:npf         
                if face[npf*2+j] > 0
                    sourceₛ[ic] -= t̂_dot_lf_δ_edge[j,ic]*ℾ*(phi_nodes[face[circle_path[j,2]]] - phi_nodes[face[circle_path[j,1]]])
                end
                
            end
            
            b[ic] = sourceᵤ[ic]+sourceₜ[ic]+sourceₛ[ic]
        end
        

        cg!(phi, a_sparse, b; maxiter=10)

    end

    if (mod(timestep,25) == 1 && plot_vtu == true)
        vtk_faces = Vector{MeshCell}(undef,num_faces)
        for (i,face) in enumerate(eachcol(faces))
            vtk_faces[i] = MeshCell(VTKCellTypes.VTK_TRIANGLE,face[1:npf])
        end
        sim_write_vtk(path,cell_number,extension,timestep,Δt,nodes,vtk_faces,phi)
    
    end
    # end


    
    
    for iter in 1:6
        if cells_on_faces_local[12+iter] > 0
            face_vector_start = cells_on_faces_local[12+iter]
            
            cell_vector_start = cells_on_faces_local[6+iter]
            num_boundary_faces = cells_on_faces_local[iter]
            cube_face_conc = zeros(Float32,num_boundary_faces)
            
            for edge_number in 1:num_boundary_faces
                cell_index_value = edge_number + cell_vector_start - 1
                face_value = interior_boundary_faces[cell_index_value]
                
                cube_face_conc[edge_number] = phi[face_value]
              
            end
    
            
            MPI.Win_lock(win_cube_face_conc[iter];rank=root,type=:exclusive)
            MPI.Put!(cube_face_conc, win_cube_face_conc[iter];rank=root,disp=face_vector_start-1)
            MPI.Win_unlock(win_cube_face_conc[iter];rank=root)
            
            
        end
        
    end

    phi_mean = mean(phi)
    if isnan(phi_mean) == true 
        for ic in range(1,num_faces)
            phi[ic] = 0.0
        end
        @show "Setting NaN to zero, cell_number = ",cell_number,num_faces   
        
    end
    if phi_mean > 10.0
        for ic in range(1,num_faces)
            phi[ic] = 1.0
        end
        @show "Unstable solution, setting values to 1.0, cell_number = ",cell_number,num_faces,phi_mean   
        exit(1)
    end

    
    num_owner_matches = owner_neighbor_ptr[3,list_number]
    if num_owner_matches > 0
       
            owner_matches = zeros(Int32,num_owner_matches)#Vector{Int32}(undef,num_owner_matches)
        
        MPI.Win_lock(win_owner_match[list_number];rank=0,type=:exclusive)
        MPI.Get!(owner_matches, win_owner_match[list_number];rank=0)
        MPI.Win_unlock(win_owner_match[list_number];rank=0)
        
        phi_owner = Vector{Float32}(undef,owner_neighbor_ptr[3,list_number])
        for (i,own) in enumerate(owner_matches)
            phi_owner[i] = phi[own]
        end
        
        MPI.Win_lock(win_phi_owner[list_number];rank=0,type=:exclusive)
        MPI.Put!(phi_owner,0,0,win_phi_owner[list_number])
        MPI.Win_unlock(win_phi_owner[list_number];rank=0)
    end
    num_neighbor_matches = owner_neighbor_ptr[6,list_number]
    
    if num_neighbor_matches > 0
        neighbor_matches = zeros(Int32,num_neighbor_matches)
        MPI.Win_lock(win_neighbor_match[list_number];rank=0,type=:exclusive)
        MPI.Get!(neighbor_matches, win_neighbor_match[list_number];rank=0)
        MPI.Win_unlock(win_neighbor_match[list_number];rank=0)

        phi_neighbor = Vector{Float32}(undef,owner_neighbor_ptr[6,list_number])
        for (i,neigh) in enumerate(neighbor_matches)
            phi_neighbor[i] = phi[neigh]
        end
       
        MPI.Win_lock(win_phi_neighbor[list_number];rank=0,type=:exclusive)
        MPI.Put!(phi_neighbor,0,0,win_phi_neighbor[list_number])
        MPI.Win_unlock(win_phi_neighbor[list_number];rank=0)
    end
    
    

    MPI.Win_lock(win_phi[list_number];rank=0,type=:exclusive)
    MPI.Put!(phi,0,0,win_phi[list_number])
    MPI.Win_unlock(win_phi[list_number];rank=0)
   
end

        
function match_neighbors(timestep::Int64, path::String, list_number::Int64, extension::String, plot_vtu::Bool, comm::MPI.Comm, rank::Int64, root::Int64, world_size::Int64, cell_list::Vector{Int64}, num_faces_collection::Vector{Int32}, owner_neighbor_ptr::Matrix{Int32}, win_owner_match::Vector{MPI.Win}, win_neighbor_match::Vector{MPI.Win}, win_phi::Vector{MPI.Win}, win_phi_owner::Vector{MPI.Win}, win_phi_neighbor::Vector{MPI.Win}, win_cube_face_conc::Vector{MPI.Win}, win_cube_face_conc_ghost::Vector{MPI.Win},win_cells_on_faces::MPI.Win)
    cell_number = cell_list[list_number]
    # @show "start match_neighbors"
    num_faces = num_faces_collection[list_number]

    phi = Vector{Float32}(undef, num_faces)
    MPI.Win_lock(win_phi[list_number];rank=0,type=:exclusive)
    MPI.Get!(phi, win_phi[list_number];rank=0)
    MPI.Win_unlock(win_phi[list_number];rank=0)
    
    num_owner_matches = owner_neighbor_ptr[3,list_number]
    if num_owner_matches > 0
        owner_matches = zeros(Int32,num_owner_matches)#Vector{Int32}(undef,num_owner_matches)
        MPI.Win_lock(win_owner_match[list_number];rank=0,type=:exclusive)
        MPI.Get!(owner_matches, win_owner_match[list_number];rank=0)
        MPI.Win_unlock(win_owner_match[list_number];rank=0)
        
        phi_owner = Vector{Float32}(undef,owner_neighbor_ptr[3,list_number])
        MPI.Win_lock(win_phi_owner[list_number];rank=0,type=:exclusive)
        MPI.Get!(phi_owner,win_phi_owner[list_number];rank=0)
        MPI.Win_unlock(win_phi_owner[list_number];rank=0)
        for (i,own) in enumerate(owner_matches)
            phi[own] = phi_owner[i]
        end
        
    end

    num_neighbor_matches = owner_neighbor_ptr[6,list_number]
    if num_neighbor_matches > 0
        neighbor_matches = zeros(Int32,num_neighbor_matches)#Vector{Int32}(undef,num_neighbor_matches)
        MPI.Win_lock(win_neighbor_match[list_number];rank=0,type=:exclusive)
        MPI.Get!(neighbor_matches, win_neighbor_match[list_number];rank=0)
        MPI.Win_unlock(win_neighbor_match[list_number];rank=0)
        
        phi_neighbor = Vector{Float32}(undef,owner_neighbor_ptr[6,list_number])   
        MPI.Win_lock(win_phi_neighbor[list_number];rank=0,type=:exclusive)
        MPI.Get!(phi_neighbor,win_phi_neighbor[list_number];rank=0)
        MPI.Win_unlock(win_phi_neighbor[list_number];rank=0)
        
        for (i,neigh) in enumerate(neighbor_matches)
            phi[neigh] = phi_neighbor[i]
        end

    end  
    MPI.Win_lock(win_phi[list_number];rank=0,type=:exclusive)
    MPI.Put!(phi,0,0,win_phi[list_number])
    MPI.Win_unlock(win_phi[list_number];rank=0)

    # @show "end match_neighbors"
   
end


function Initial_conditions(num_faces,phi,ICvalue)
   
    for ic in 1:num_faces     
        phi[ic] = ICvalue
    end
    
    return
end




function job_queue(timestep,cell_list,num_faces_collection,path,extension,plot_vtu,comm,rank,world_size,nworkers,root,owner_neighbor_ptr,edges_on_cube_side,win_owner_match,win_neighbor_match,win_phi,win_phi_owner,win_phi_neighbor,win_cells_on_faces,win_cube_face_conc,win_cube_face_conc_ghost,win_cube_face_area_delta,MPI_function)

    T = eltype(cell_list)
    N = length(cell_list)
    send_mesg = Array{T}(undef, 1)
    recv_mesg = Array{T}(undef, 1)

    if rank == root 
        
        idx_recv = 0
        idx_sent = 1

        new_data = zeros(T,N)
        
        sreqs_workers = Array{MPI.Request}(undef,nworkers)
        # -1 = start, 0 = channel not available, 1 = channel available
        status_workers = ones(nworkers).*-1

        # Send message to workers
        for dst in 1:nworkers
            if idx_sent > N
                break
            end
            send_mesg[1] = idx_sent
            sreq = MPI.Isend(send_mesg, comm; dest=dst, tag=dst+32)
            # if mod(idx_sent,10) == 0
            #     println("Progress $idx_sent / $N")
            # end
            idx_sent += 1
            sreqs_workers[dst] = sreq
            status_workers[dst] = 0
            # print("Root: Sent number $(send_mesg[1]) to Worker $dst\n")
        end

        # Send and receive messages until all elements are added
        while idx_recv != N
            # Check to see if there is an available message to receive
            for dst in 1:nworkers
                if status_workers[dst] == 0
                    flag = MPI.Test(sreqs_workers[dst])
                    if flag
                        status_workers[dst] = 1
                    end
                end
            end
            for dst in 1:nworkers
                if status_workers[dst] == 1
                    ismessage = MPI.Iprobe(comm; source=dst, tag=dst+42)
                    if ismessage
                        # Receives message
                        MPI.Recv!(recv_mesg, comm; source=dst, tag=dst+42)
                        idx_recv += 1
                        new_data[idx_recv] = recv_mesg[1]
                        # print("Root: Received number $(recv_mesg[1]) from Worker $dst\n")
                        if idx_sent <= N
                            send_mesg[1] = idx_sent
                            # Sends new message
                            sreq = MPI.Isend(send_mesg, comm; dest=dst, tag=dst+32)
                            # if mod(idx_sent,10) == 0
                            #     println("Progress $idx_sent / $N")
                            # end
                            idx_sent += 1
                            sreqs_workers[dst] = sreq
                            status_workers[dst] = 0
                            # print("Root: Sent number $(send_mesg[1]) to Worker $dst\n")
                        end
                    end
                end
            end
        end

        for dst in 1:nworkers
            # Termination message to worker
            send_mesg[1] = -1
            sreq = MPI.Isend(send_mesg, comm; dest=dst, tag=dst+32)
            sreqs_workers[dst] = sreq
            status_workers[dst] = 0
            # print("Root: Finish Worker $dst\n")
        end
        
    else # If rank == worker
        # -1 = start, 0 = channel not available, 1 = channel available
        status_worker = -1
        sreqs_workers = Array{MPI.Request}(undef,1)
        while true
            if status_worker != 0
                ismessage = MPI.Iprobe(comm; source=root, tag=rank+32)

                if ismessage
                    # Receives message
                    MPI.Recv!(recv_mesg, comm; source=root, tag=rank+32)
                    # Termination message from root
                    if recv_mesg[1] == -1
                        # print("Worker $rank: Finish\n")
                        break
                    end
                    # print("Worker $rank: Received number $(recv_mesg[1]) from root\n")
                    
                    if MPI_function == 1

                        neuron_solve(timestep,path,recv_mesg[1],extension,plot_vtu,comm,rank,world_size,cell_list,owner_neighbor_ptr,edges_on_cube_side,win_owner_match,win_neighbor_match,win_phi,win_phi_owner,win_phi_neighbor,win_cells_on_faces,win_cube_face_conc,win_cube_face_conc_ghost,win_cube_face_area_delta)
                    else
                        if MPI_function == 2
                            match_neighbors(timestep,path,recv_mesg[1],extension,plot_vtu,comm,rank,root,world_size,cell_list,num_faces_collection,owner_neighbor_ptr,win_owner_match,win_neighbor_match,win_phi,win_phi_owner,win_phi_neighbor,win_cube_face_conc,win_cube_face_conc_ghost,win_cells_on_faces)
                        end
                    end
                    
                    send_mesg[1] = recv_mesg[1]
                    sreq = MPI.Isend(send_mesg[1], comm; dest=root, tag=rank+42)
                    sreqs_workers[1] = sreq
                    status_worker = 0
                end
            else
                # Check to see if there is an available message to receive
                flag = MPI.Test(sreqs_workers[1])
                if flag
                    status_worker = 1
                end
               
            end
        end
    end
    
    MPI.Barrier(comm)
    return
    
end



function sim_write_vtk(path,cell_number,extension,timestep,Δt,nodes,vtk_faces,phi)
    
    vtk_grid(string(path,cell_number,extension,"_",timestep),nodes,vtk_faces) do vtk
        vtk["Concentration", VTKCellData()] = phi
        vtk["time", VTKFieldData()] = timestep*Δt
    end
end



function setup_neighbor_matches(num_cells::Int64, win_neighbor_match::Vector{MPI.Win}, owner_neighbor_ptr::Matrix{Int32}, neighbor::Matrix{Int32}, match_collection::Vector{Array{Int32,2}}, comm::MPI.Comm, rank::Int64, root::Int64)
    if rank == 0
        # @show "Setting up neighbor matches"
        neighbor_match_collection = Vector{Vector{Int32}}(undef,num_cells)
        for i in 1:num_cells
            neighbor_size = owner_neighbor_ptr[6,i]
            if neighbor_size > 0
                neighbor_match_collection[i] = Vector{Int32}(undef,neighbor_size)
            else
                neighbor_match_collection[i] = zeros(Int32,1)
            end
        end
    
        for nbor in eachcol(neighbor)
            
            (neighbor_match_collection[nbor[2]])[nbor[5]:nbor[6]] .= view(match_collection[nbor[1]],nbor[3]:nbor[4],2)

        end
    end
    MPI.Barrier(comm)
    null_array = zeros(Int32,1)
    for i in 1:num_cells
        
        if rank == 0
            
            
            if owner_neighbor_ptr[4,i] > 0
                win_neighbor_match[i] = MPI.Win_create(neighbor_match_collection[i], comm)
                
            else
                win_neighbor_match[i] = MPI.Win_create(null_array, comm)
            end
                
        else
            
            win_neighbor_match[i] = MPI.Win_create(null_array, comm)
            
        end
        MPI.Barrier(comm)
    end
end

function setup_owner_matches(num_cells::Int64, win_owner_match::Vector{MPI.Win}, owner_neighbor_ptr::Matrix{Int32}, match_collection::Vector{Array{Int32,2}}, comm::MPI.Comm, rank::Int64, root::Int64, path::String, cell_list::Vector{Int64}, extension::String)
    if rank == root

        for i in 1:num_cells
             


            
            filename = string(path,cell_list[i],extension,"_match",".jld2")
            fid_jld_match = jldopen(filename)
            # display(fid_jld_match)
        
            match_setup::Match_face = fid_jld_match["match_setup"]
            total_face_matches::Int32 = match_setup.total_face_matches
            if total_face_matches > 0
                match_faces = Array{Int32,2}(undef,2,total_face_matches)
                match_faces::Array{Int32,2} = match_setup.match_faces
            else
                match_faces = zeros(Int32,2,1)
            end
            
            close(fid_jld_match)
            
            match_collection[i] = permutedims(match_faces, (2,1))

        end

    end

    MPI.Barrier(comm)
    
    null_array = zeros(Int32,1)
    for i in 1:num_cells
        if rank == 0
            
            if owner_neighbor_ptr[1,i] > 0
                win_owner_match[i] = MPI.Win_create(@view((match_collection[i])[:,1]), comm)
               
            else
                win_owner_match[i] = MPI.Win_create(null_array, comm)
            end
             
        else
            
            win_owner_match[i] = MPI.Win_create(null_array, comm)
            
        end
        MPI.Barrier(comm)
    end
end


function setup_phi(num_cells,owner_neighbor_ptr,phi_owner_collection,phi_neighbor_collection,win_phi_owner,win_phi_neighbor,comm,rank,root)
    if rank == root
        
        for i in 1:num_cells
            
            if owner_neighbor_ptr[3,i] > 0
                phi_owner_collection[i] = Vector{Float32}(undef,owner_neighbor_ptr[3,i])
            else
                phi_owner_collection[i] = zeros(Float32,1)
            end
            if owner_neighbor_ptr[6,i] > 0
                phi_neighbor_collection[i] = Vector{Float32}(undef,owner_neighbor_ptr[6,i])
            else
                phi_neighbor_collection[i] = zeros(Float32,1)
            end
        end
    end
    
    MPI.Barrier(comm)
    null_array = zeros(Float32,1)
    for i in 1:num_cells
        if rank == 0
            
            if owner_neighbor_ptr[3,i] > 0
                
                win_phi_owner[i] = MPI.Win_create(phi_owner_collection[i], comm)

            else
                win_phi_owner[i] = MPI.Win_create(null_array, comm)
            end
                
        else

            win_phi_owner[i] = MPI.Win_create(null_array, comm)
           
        end
        MPI.Barrier(comm)
    end
    for i in 1:num_cells
        if rank == 0
            
            if owner_neighbor_ptr[6,i] > 0
                
                win_phi_neighbor[i] = MPI.Win_create(phi_neighbor_collection[i], comm)
            
            else
                win_phi_neighbor[i] = MPI.Win_create(null_array, comm)
            end
                
        else
            
            win_phi_neighbor[i] = MPI.Win_create(null_array, comm)
            
        end
        MPI.Barrier(comm)
    end
    return nothing
end

function setup_ghost_cells(path::String, frame_name::String, location::Matrix{Int64}, cubes::Int64, xval::Int64, yval::Int64, startframe::Int64, slices::Int64, xy_elements::Int64, height::Int64, root::Int64, Sim_parameters::Simulation_parameters)
    
    
    cube_face_conc_array = Vector{Vector{Vector{Float32}}}(undef,cubes)
    cube_face_conc_ghost_array = Vector{Vector{Vector{Float32}}}(undef,cubes)
   
    cubes_x = (location[1,2]-location[1,1]+1)
    cubes_y = (location[2,2]-location[2,1]+1)
    cubes_z = (location[3,2]-location[3,1]+1)
    

    counter = 0
    counter_x = 0
    counter_y = 0
    counter_z = 0
    for xcounter in location[1,1]:location[1,2]
        counter_x += 1
        for ycounter in location[2,1]:location[2,2]
            counter_y += 1
            for zcounter in location[3,1]:location[3,2]
                counter_z += 1
                

                counter += 1
                current_startframe = startframe + zcounter*slices
                current_xval = xval + xcounter*xy_elements
                current_yval = yval + ycounter*xy_elements
                
                extension = string("_",current_startframe,"_",current_xval,"_",current_yval,"_",height)
                endframe = current_startframe+slices

                filename = string(path,frame_name,current_startframe,"_",endframe,"_",current_xval,"_",current_yval,"_",height,"_ghost_dynamic.jld2")
                fid_jld_ghost = jldopen(filename)
                # display(fid_jld_ghost)
                ghost_setup_dynamic::Ghost_cells_dynamic = fid_jld_ghost["ghost_setup_dynamic"]
               
                cube_face_conc_array[counter] = ghost_setup_dynamic.cube_face_conc
                cube_face_conc_ghost_array[counter] = ghost_setup_dynamic.cube_face_conc_ghost 
                
              
                close(fid_jld_ghost)


            end
        end
    end

   
    for (j,cube) in enumerate(cube_face_conc_array)
        
        for (i,cube_face) in enumerate(cube)
            
            if i == 2 || i == 4 || i == 6
                if size(cube_face)[1] > 1
                    for (k, conc_on_cube_face) in enumerate(cube_face)
                        # Determine indices based on i
                        area_index = i
                        neighbor_index = i == 2 ? j + cubes_z * cubes_y : (i == 4 ? j + cubes_z : j + 1)
                        neighbor_face = i - 1
            
                        
                        cube_face_conc_ghost_array[neighbor_index][neighbor_face][k] = conc_on_cube_face
                        cube_face_conc_ghost_array[j][i][k] = cube_face_conc_array[neighbor_index][neighbor_face][k]
                        
                    end
                end
            end
             
        end
    end
    counter = 0
    counter_x = 0
    counter_y = 0
    counter_z = 0
    for xcounter in location[1,1]:location[1,2]
        counter_x += 1
        for ycounter in location[2,1]:location[2,2]
            counter_y += 1
            for zcounter in location[3,1]:location[3,2]
                counter_z += 1
                
                # @show counter_x,counter_y,counter_z
                counter += 1
                current_startframe = startframe + zcounter*slices
                current_xval = xval + xcounter*xy_elements
                current_yval = yval + ycounter*xy_elements
                  
                extension = string("_",current_startframe,"_",current_xval,"_",current_yval,"_",height)
                endframe = current_startframe+slices

                filename = string(path,frame_name,current_startframe,"_",endframe,"_",current_xval,"_",current_yval,"_",height,"_ghost_dynamic.jld2")
                # @show filename
                jldsave(filename,true;ghost_setup_dynamic=Ghost_cells_dynamic(cube_face_conc_array[counter],cube_face_conc_ghost_array[counter]))
            end
        end
    end
end


function cube_solution(sim_number::Int64,
    counter::Int64,
    path::String,
    frame_name::String,
    comm::MPI.Comm,
    rank::Int64,
    world_size::Int64,
    nworkers::Int64,
    root::Int64,
    extension::String,
    slices::Int64,
    current_startframe::Int64,
    endframe::Int64,
    current_xval::Int64,
    current_yval::Int64,
    height::Int64,
    ICvalue::Float32,
    restart::Bool,
    plot_vtu::Bool,
    timestep_iter::Int64,
    timestep0::Int64)


    null_array = Vector{Int32}(undef,1)
    null_array_f32 = Vector{Float32}(undef,1)
    null_array_f64 = Vector{Float64}(undef,1)
    num_cells_buffer = Vector{Int64}(undef,1)
    
    if rank == root
        
        filename = string(path,"simulation",extension,".jld2")
        
        fid_jld_sim = jldopen(filename)
        # display(fid_jld_sim)

        match_setup::Owner_neighbor = fid_jld_sim["match_setup"]
        num_cells::Int64 = match_setup.num_cells
        
        num_cells_buffer[1] = num_cells
    end
    MPI.Barrier(comm)
    MPI.Bcast!(num_cells_buffer, comm; root=0)
    
    if num_cells_buffer[1] == 0
        @show "No cells, returning"
        return
    end
    num_cells = num_cells_buffer[1]
    if rank == root
        cell_list = Vector{Int64}(undef,num_cells)
        cell_list = match_setup.cell_list
        
        owner = match_setup.owner

        neighbor = match_setup.neighbor
        
        owner_neighbor_ptr = Array{Int32,2}(undef,6,num_cells)
        owner_neighbor_ptr = match_setup.owner_neighbor_ptr
        
        num_faces_collection = Vector{Int32}(undef,num_cells)
        num_faces_collection = match_setup.num_faces_array
        close(fid_jld_sim)
    
        
        
    end



    if rank != root
        num_faces_collection = Vector{Int32}(undef,num_cells)
        owner_neighbor_ptr = Array{Int32,2}(undef,6,num_cells)
    end
    MPI.Barrier(comm)

    
    
    MPI.Bcast!(num_faces_collection, comm; root=0)
    MPI.Bcast!(owner_neighbor_ptr, comm; root=0)

  
    
    
    
    if rank != root
        
        size_owner = sum(owner_neighbor_ptr[1,:])
        
        owner = Array{Int32,2}(undef,4,size_owner)
        neighbor = Array{Int32,2}(undef,6,size_owner)
        cell_list = Vector{Int64}(undef,num_cells)
    end
    MPI.Barrier(comm)


    MPI.Bcast!(owner, comm; root=0)
    
   
    MPI.Bcast!(neighbor, comm; root=0)
    

    MPI.Bcast!(cell_list, comm; root=0)


    
    if rank == root
        # @show "Creating match collection"
        match_collection = Vector{Array{Int32,2}}(undef,num_cells)
    else
        match_collection = Vector{Array{Int32,2}}(undef,1)
    end
    MPI.Barrier(comm)
    win_owner_match = Vector{MPI.Win}(undef,num_cells)
    setup_owner_matches(num_cells,win_owner_match,owner_neighbor_ptr,match_collection,comm,rank,root,path,cell_list,extension)
 
    MPI.Barrier(comm)
    win_neighbor_match = Vector{MPI.Win}(undef,num_cells)
    setup_neighbor_matches(num_cells,win_neighbor_match,owner_neighbor_ptr,neighbor,match_collection,comm,rank,root)
    MPI.Barrier(comm)
    # @show rank,"Neighbor matches setup complete"
  
    if rank == root
        phi_owner_collection = Vector{Vector{Float32}}(undef,num_cells)
        phi_neighbor_collection = Vector{Vector{Float32}}(undef,num_cells)
    else
        phi_owner_collection = Vector{Vector{Float32}}(undef,1)
        phi_neighbor_collection = Vector{Vector{Float32}}(undef,1)
    end
    MPI.Barrier(comm)
    win_phi_owner = Vector{MPI.Win}(undef,num_cells)
    win_phi_neighbor = Vector{MPI.Win}(undef,num_cells)
    # @show rank,"starting phi setup"
    setup_phi(num_cells,owner_neighbor_ptr,phi_owner_collection,phi_neighbor_collection,win_phi_owner,win_phi_neighbor,comm,rank,root)
    
    MPI.Barrier(comm)

    
    
    
    win_phi = Vector{MPI.Win}(undef,num_cells)
    
    if restart == false && timestep_iter == 1
        timestep = timestep_iter
        phi_collection = Vector{Vector{Float32}}(undef,num_cells)
        if rank == 0
            @show "starting at timestep = 1"
        end
     
        for i in 1:num_cells
            
            if rank == 0
                
                num_faces = num_faces_collection[i]
                

                phi_collection[i] = Vector{Float32}(undef,num_faces)
                Initial_conditions(num_faces,phi_collection[i],ICvalue)
                
                win_phi[i] = MPI.Win_create(phi_collection[i], comm)
                
            else
                
                win_phi[i] = MPI.Win_create(null_array_f32, comm)
                
            end
            
            MPI.Barrier(comm)
        end
    else
        timestep = timestep0+timestep_iter
        if rank == root
            
            
            if mod(timestep,2) == 0
                phi_collection = load(string(path,"restart_",extension,"_odd",".jld2"),"phi_collection")
            else
                phi_collection = load(string(path,"restart_",extension,"_even",".jld2"),"phi_collection")
            end
            @show "timestep = ",timestep
   
        end
        MPI.Barrier(comm)


        for i in 1:num_cells
            if rank == 0
                win_phi[i] = MPI.Win_create(phi_collection[i], comm)
            else
                win_phi[i] = MPI.Win_create(null_array_f32, comm)
            end
            MPI.Barrier(comm)
        end     
    end


    

    edges_on_cube_side = Vector{Int32}(undef,18)
    if rank == 0
         # @show "reading cubeface_cells"
         filename = string(path,frame_name,current_startframe,"_",endframe,"_",current_xval,"_",current_yval,"_",height,".jld2")
         fid_jld = jldopen(filename)
         cubeface_cells::Cube_properties = fid_jld["cubeface_cells"] 
         
         cells_on_faces::Array{Int32,2} = cubeface_cells.cells_on_faces
         
         edges_on_cube_side::Vector{Int32} = cubeface_cells.edges_on_cube_side 
         close(fid_jld)
        
         cells_on_faces_vec::Vector{Int32} = vec(cells_on_faces)
         
         win_cells_on_faces = MPI.Win_create(cells_on_faces_vec, comm)

        if restart == false && timestep_iter == 1
            cube_face_conc = Vector{Vector{Float32}}(undef,6)
            cube_face_conc_ghost = Vector{Vector{Float32}}(undef,6)
            
            for iter in 1:6
                num_edges_on_cube_side = edges_on_cube_side[iter]
                if num_edges_on_cube_side > 0
                    cube_face_conc[iter] = zeros(Float32,num_edges_on_cube_side) .+ ICvalue
                    cube_face_conc_ghost[iter] = zeros(Float32,num_edges_on_cube_side) .+ ICvalue
                else
                    cube_face_conc[iter] = zeros(Float32,1) 
                    cube_face_conc_ghost[iter] = zeros(Float32,1)
                end
            end
        else
            filename = string(path,frame_name,current_startframe,"_",endframe,"_",current_xval,"_",current_yval,"_",height,"_ghost_dynamic.jld2")
            fid_jld_ghost = jldopen(filename)
            ghost_setup_dynamic::Ghost_cells_dynamic = fid_jld_ghost["ghost_setup_dynamic"]
            cube_face_conc::Vector{Vector{Float32}} = ghost_setup_dynamic.cube_face_conc
            cube_face_conc_ghost::Vector{Vector{Float32}} = ghost_setup_dynamic.cube_face_conc_ghost 
           
            close(fid_jld_ghost)
        
        end

        
        
        filename = string(path,frame_name,current_startframe,"_",endframe,"_",current_xval,"_",current_yval,"_",height,"_ghost_static.jld2")
        fid_jld_ghost = jldopen(filename)
        ghost_setup::Ghost_cells_static = fid_jld_ghost["ghost_setup_static"]
        cube_face_S::Vector{Array{Float32,2}} = ghost_setup.cube_face_S
        cube_face_E::Vector{Array{Float32,2}} = ghost_setup.cube_face_E
        cube_face_Ef::Vector{Vector{Float32}} = ghost_setup.cube_face_Ef 
        cube_face_T::Vector{Array{Float32,2}} = ghost_setup.cube_face_T       
        cube_face_dCF::Vector{Vector{Float32}} = ghost_setup.cube_face_dCF
        cube_face_rCF::Vector{Array{Float32,2}}  = ghost_setup.cube_face_rCF  
        cube_face_weight_edge::Vector{Vector{Float32}} = ghost_setup.cube_face_weight_edge
        close(fid_jld_ghost)
        
       

    else
        
        win_cells_on_faces = MPI.Win_create(null_array, comm)
        
    end
    
    MPI.Barrier(comm)
   
    MPI.Bcast!(edges_on_cube_side, comm; root=0)
    
    MPI.Barrier(comm)
    
    win_cube_face_conc = Vector{MPI.Win}(undef,6)
    win_cube_face_conc_ghost = Vector{MPI.Win}(undef,6)
    
    for iter in 1:6
       
        if rank == 0 
            # @show "Setting up ghost cells"
            
            win_cube_face_conc[iter] = MPI.Win_create(cube_face_conc[iter], comm)
            win_cube_face_conc_ghost[iter] = MPI.Win_create(cube_face_conc_ghost[iter], comm)
           
        else
            # @show "Setting up neighbor ghost cells"
            win_cube_face_conc[iter] = MPI.Win_create(null_array_f32, comm)
            win_cube_face_conc_ghost[iter] = MPI.Win_create(null_array_f32, comm)

        end
        MPI.Barrier(comm)
    end
    

    win_cube_face_area_delta =  Vector{MPI.Win}(undef,6)
    for iter in 1:6
        if rank == 0
            # @show "Setting up ghost area/delta"
           
            size_area_delta = edges_on_cube_side[iter]
          
            if size_area_delta > 0
                

                area_delta = zeros(Float32,Int64(size_area_delta*15)) 
                for ad_iter in 1:3
                    lhs_iter = ad_iter
                
                    @views area_delta[size_area_delta*(lhs_iter-1)+1:size_area_delta*lhs_iter] .= cube_face_S[iter][ad_iter,:]
                end
                for ad_iter in 1:3
                    lhs_iter = ad_iter+3
                
                    @views area_delta[size_area_delta*(lhs_iter-1)+1:size_area_delta*lhs_iter] .= cube_face_E[iter][ad_iter,:]
                end
                for ad_iter in 1:3
                    lhs_iter = ad_iter+6
                
                    @views area_delta[size_area_delta*(lhs_iter-1)+1:size_area_delta*lhs_iter] .= cube_face_T[iter][ad_iter,:]
                end
                for ad_iter in 1:3
                    lhs_iter = ad_iter+9
                    
                    @views area_delta[size_area_delta*(lhs_iter-1)+1:size_area_delta*lhs_iter] .= cube_face_rCF[iter][ad_iter,:]
                end
                lhs_iter = 13
            
                area_delta[size_area_delta*(lhs_iter-1)+1:size_area_delta*lhs_iter] .= cube_face_Ef[iter][:]
                lhs_iter = 14
            
                area_delta[size_area_delta*(lhs_iter-1)+1:size_area_delta*lhs_iter] .= cube_face_dCF[iter][:]
                lhs_iter = 15
                
                area_delta[size_area_delta*(lhs_iter-1)+1:size_area_delta*lhs_iter] .= cube_face_weight_edge[iter][:]
            
                    

                win_cube_face_area_delta[iter] = MPI.Win_create(area_delta, comm)
            else
                win_cube_face_area_delta[iter] = MPI.Win_create(null_array_f32, comm) 
            end
            
        else
            win_cube_face_area_delta[iter] = MPI.Win_create(null_array_f32, comm)            
        end
    end
    MPI.Barrier(comm)



    MPI_function = 1
    job_queue(timestep,cell_list,num_faces_collection,path,extension,plot_vtu,comm,rank,world_size,nworkers,root,owner_neighbor_ptr,edges_on_cube_side,win_owner_match,win_neighbor_match,win_phi,win_phi_owner,win_phi_neighbor,win_cells_on_faces,win_cube_face_conc,win_cube_face_conc_ghost,win_cube_face_area_delta,MPI_function)
    
    # @show rank, "Finished job queue"
    MPI.Barrier(comm)
    if rank == root
        
        for nbor in eachcol(neighbor)
            owner_cell = nbor[1]
            neighbor_cell = nbor[2]
            owner_start_pos = nbor[3]
            owner_stop_pos = nbor[4]
            neighbor_start_pos = nbor[5]
            neighbor_stop_pos = nbor[6]
           
            @views phi_owner_collection[owner_cell][owner_start_pos:owner_stop_pos] .+= phi_neighbor_collection[neighbor_cell][neighbor_start_pos:neighbor_stop_pos]
            @views phi_owner_collection[owner_cell][owner_start_pos:owner_stop_pos] ./= 2
            @views phi_neighbor_collection[neighbor_cell][neighbor_start_pos:neighbor_stop_pos] .= phi_owner_collection[owner_cell][owner_start_pos:owner_stop_pos]
            
        end
    end
    MPI.Barrier(comm)
    MPI_function = 2
    job_queue(timestep,cell_list,num_faces_collection,path,extension,plot_vtu,comm,rank,world_size,nworkers,root,owner_neighbor_ptr,edges_on_cube_side,win_owner_match,win_neighbor_match,win_phi,win_phi_owner,win_phi_neighbor,win_cells_on_faces,win_cube_face_conc,win_cube_face_conc_ghost,win_cube_face_area_delta,MPI_function)

    MPI.Barrier(comm)
    if rank == root
        if mod(timestep,2) == 0
            
            jldsave(string(path,"restart_",extension,"_even",".jld2");phi_collection)
        else
            jldsave(string(path,"restart_",extension,"_odd",".jld2");phi_collection)
        end
    
        filename = string(path,frame_name,current_startframe,"_",endframe,"_",current_xval,"_",current_yval,"_",height,"_ghost_dynamic.jld2")
        jldsave(filename,true;ghost_setup_dynamic=Ghost_cells_dynamic(cube_face_conc,cube_face_conc_ghost))
    end
    MPI.Barrier(comm)
    
    MPI.free(win_cells_on_faces)
    MPI.Barrier(comm)
   
    
    for i in 1:6
        MPI.free(win_cube_face_conc[i])
    end
    MPI.Barrier(comm)
 
    for i in 1:6
        MPI.free(win_cube_face_conc_ghost[i])
    end
    MPI.Barrier(comm)
 
   
    for i in 1:6
        MPI.free(win_cube_face_area_delta[i])
    end
    MPI.Barrier(comm)
  
    for i in 1:num_cells
        MPI.free(win_owner_match[i])
    end
    MPI.Barrier(comm)
 
    for i in 1:num_cells
         MPI.free(win_neighbor_match[i])
    end
    MPI.Barrier(comm)
    for i in 1:num_cells
         MPI.free(win_phi_owner[i])
    end
    MPI.Barrier(comm)
    for i in 1:num_cells
         MPI.free(win_phi_neighbor[i])
    end
    MPI.Barrier(comm)
    for i in 1:num_cells
         MPI.free(win_phi[i])
    end
    MPI.Barrier(comm)
    # @show rank,"exiting"
    
end



function MPI_setup(sim_number,path,frame_name,comm,rank,world_size,restart,plot_vtu,num_timesteps)
  
    nworkers = world_size - 1
    root = 0
    if world_size <= 1
        println("World size must be greater than 1")
        return
    end
    path_variables = Vector{Int64}(undef,5)
    ICvalue_buffer = Vector{Float32}(undef,1)
    location = zeros(Int64,3,2)

    

    if rank == 0
        filename = string(@__DIR__,"/Simulation_settings_4.0_smoothed.csv")
        settings::Matrix{Int64} = CSV.File(filename) |> Tables.matrix

        
        
        xval::Int64 = settings[sim_number,2]
        yval::Int64 = settings[sim_number,3]
        startframe::Int64 = settings[sim_number,4]
        height::Int64 = settings[sim_number,5]
        slices::Int64 = height*25*5

        
        location[1,1:2] .= settings[sim_number,6:7]
        location[2,1:2] .= settings[sim_number,8:9]
        location[3,1:2] .= settings[sim_number,10:11]

        for i in 1:3
            if location[i,1] > 0
                location[i,1] = -location[i,1]
            end
            if location[i,2] < 0
                println("Error: plus location must be positive")
                return
            end 
        end

        path_variables .= slices,startframe,xval,yval,height

        fid_jld_sim = jldopen(string(path,"Simulation_parameters_",startframe,"_",xval,"_",yval,"_",height,".jld2"))
        # display(fid_jld_sim)
    
    
        Sim_parameters::Simulation_parameters = fid_jld_sim["Sim_parameters"] 
        close(fid_jld_sim)

        ICvalue_buffer[1] = Sim_parameters.ICvalue
        

    end
    MPI.Barrier(comm)

    MPI.Bcast!(location, comm; root=0)

    cubes = (location[1,2]-location[1,1]+1)*(location[2,2]-location[2,1]+1)*(location[3,2]-location[3,1]+1)
    


    MPI.Bcast!(path_variables, comm; root=0)
    slices,startframe,xval,yval,height = path_variables
    MPI.Bcast!(ICvalue_buffer, comm; root=0)
    ICvalue::Float32 = ICvalue_buffer[1]
    xy_elements = height*125

    timestep_buffer = Vector{Int64}(undef,1)
    if rank == 0
        if restart == false
           
            timestep0 = 0
        else
            timestep0::Int64 = load(string(path,"Last_completed_timestep_",startframe,"_",xval,"_",yval,"_",height,".jld2"),"timestep")
        end
        timestep_buffer[1] = timestep0
       
    end
    timestep_new = 0
    MPI.Bcast!(timestep_buffer, comm; root=0)
    timestep0 = timestep_buffer[1]
    for timestep_iter in 1:num_timesteps
        timestep_new = timestep0+timestep_iter
        counter = 0
        for xcounter in location[1,1]:location[1,2]
            for ycounter in location[2,1]:location[2,2]
                for zcounter in location[3,1]:location[3,2]
                    
                    counter += 1
                    current_startframe = startframe + zcounter*slices
                    current_xval = xval + xcounter*xy_elements
                    current_yval = yval + ycounter*xy_elements
                    
                    extension = string("_",current_startframe,"_",current_xval,"_",current_yval,"_",height)
                    endframe = current_startframe+slices
             

                    cube_solution(sim_number,counter,path,frame_name,comm,rank,world_size,nworkers,root,extension,slices,current_startframe,endframe,current_xval,current_yval,height,ICvalue,restart,plot_vtu,timestep_iter,timestep0)
                end
            end
        end
        MPI.Barrier(comm)
        if rank == 0
            
            setup_ghost_cells(path,frame_name,location,cubes,xval,yval,startframe,slices,xy_elements,height,root,Sim_parameters)
            
            jldsave(string(path,"Last_completed_timestep_",startframe,"_",xval,"_",yval,"_",height,".jld2");timestep=timestep_new)
            
        end
    end
    
end


##### User settings
restart = false
plot_vtu = true
num_timesteps = 40001
sim_number = parse(Int64,ARGS[1])
##### End user settings

path = string(@__DIR__, "/../Neuron_transport_data/")
frame_name = "frame_uniq_"
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
world_size = MPI.Comm_size(comm)



MPI_setup(sim_number,path,frame_name,comm,rank,world_size,restart,plot_vtu,num_timesteps)
MPI.Barrier(comm)


MPI.Finalize()