# Written by: Donald L. Elbert & Dylan Esguerra, University of Washington
# Copyright (c) 2022-2024 University of Washington
# All rights reserved.

# Redistribution and use in source and binary forms, with or without modification, are permitted provided
#  that the following conditions are met:

# -Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# -Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the 
# following disclaimer in the documentation and/or other materials provided with the distribution.
# -Neither the name of the University of Washington nor the names of its contributors may be used to endorse 
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
using JLD2
using SparseArrays
using StaticArrays
using MPI
using Statistics: mean
using TOML

include("neuron_transport_match_3D_3.0a.jl")
include("neuron_transport_interface_setup_3D_3.0a.jl")
include("neuron_transport_setup_job_queues_1.0.jl")

struct BC
    ID::Int32
    type::Int32
    value::Float32
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

function uniquenode(path,list_number,extension,Sim_parameters,comm,rank_comm,size_comm,cell_list,xcounter,ycounter,zcounter,cube_counter,location,startframe,endframe,xval,yval,height,win_accumulate_cells_on_cube_side,win_cube_face_match,win_accumulate_owner_ptr,win_num_faces)
   
    ## Read ply file for given cell (biological cell, not mesh cell)
    cell_number = cell_list[list_number]
    Δt::Float32 = Sim_parameters.Δt
    BCarray::Vector{BC} = Sim_parameters.BCarray
    ICvalue::Float32 = Sim_parameters.ICvalue
    ℾ::Float32 = Sim_parameters.ℾ
    ρ::Float32 = Sim_parameters.ρ
    ū::Vector{Float32} = Sim_parameters.ū
    Sourceᵤ::Float32 = Sim_parameters.Sourceᵤ
    Sourceₚ::Float32 = Sim_parameters.Sourceₚ
    
    filetype = "_structure.jld2"
    filename = string(path,cell_number,extension,filetype)
    fid_jld = jldopen(filename)
    # display(fid_jld)
    structure_setup = fid_jld["structure_setup"] 
    npf::Int32 = structure_setup.npf
    num_faces::Int32 = structure_setup.num_faces
    num_nodes::Int32 = structure_setup.num_nodes
    num_edges::Int32 = structure_setup.num_edges
    num_boundary_edges::Int32 = structure_setup.num_boundary_edges
    

    nodes = Array{Float32,2}(undef, 3,num_nodes)  
    normals = Array{Float32,2}(undef, 3,num_nodes)
    faces = Array{Int32,2}(undef, npf*4,num_faces)
    edges = Array{Int32,2}(undef, 11,num_edges)
    boundary_edges = Array{Int32,2}(undef, 11,num_boundary_edges)
    face_center = Array{Float32,2}(undef,3,num_faces)


    faces::Array{Int32,2} = structure_setup.faces
    nodes::Array{Float32,2} = structure_setup.nodes
    normals::Array{Float32,2} = structure_setup.normals
    edges::Array{Int32,2} = structure_setup.edges
    face_center::Array{Float32,2} = structure_setup.face_center
    close(fid_jld)

    count1 = 0
    for (ie,edge) in enumerate(eachcol(edges))
        if edge[6] == 1
            count1 += 1
            @views boundary_edges[1:11,count1] .= edge
            boundary_edges[9,count1] = ie
        end
    end

    setup_match_faces(list_number,cell_list,path,extension,filetype,win_accumulate_owner_ptr)
    
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

    

    eps = 1e-14

    num_faces_array = Vector{Int32}(undef,1)
    num_faces_array[1] = num_faces
    MPI.Win_lock(MPI.LOCK_EXCLUSIVE, 0, 0, win_num_faces)
    MPI.Put!(num_faces_array, win_num_faces;rank=0,disp=list_number-1) 
    MPI.Win_unlock(0, win_num_faces)

    t = Array{Float32,3}(undef,3,npf,num_faces)
    t̂ = Array{Float32,3}(undef,3,npf,num_faces)
    edge_mid = Array{Float32,3}(undef,3,npf,num_faces)
    n_face = Array{Float32,2}(undef,3,num_faces)
    n̂_face = Array{Float32,2}(undef,3,num_faces)
    n̂_edge = Array{Float32,3}(undef,3,npf,num_faces)
    volume_face = Vector{Float32}(undef,num_faces)
    d_face_edge = Array{Float32,2}(undef,npf,num_faces)
    
    if size(nodes,2) != num_nodes
        @warn "Nodes array has incorrect number of nodes"
        return
    end
  
    vertex_normals = zeros(Float32,3,num_nodes)
    vertex_index = Int32(0)
    count_face_per_vertex = zeros(Int64,num_nodes)
    running_count_face_per_vertex = zeros(Int64,num_nodes)
    # Each face has some associated data.  These are:
    # The center of the face (face_center)
    # The tangent to each edge (t), which are not unit vectors
    # The midpoints of each edge of the face (edge_mid)
    # The unit normal vector to each edge (n̂_edge), which are unit vectors
    # The normal vector of the face (n_face), a unit vector
    # The unitnormal vector of the face (n̂_face), a unit vector
    # The distance from face center to each edge midpoint (d_face_edge)
    # face area, also called face volume in 2D, even though it is an area, is face_volume
    # In contrast, the length of each 'edge' is called an area.
    
    for (ic,face) in enumerate(eachcol(faces))

        for j in range(1,npf)
            @views  t[1:3,j,ic] .= nodes[:,face[circle_path[j,2]]]-nodes[:,face[circle_path[j,1]]]
            @views  edge_mid[1:3,j,ic] .= (nodes[:,face[circle_path[j,2]]]+nodes[:,face[circle_path[j,1]]])/2
            
        end
       
        @views  n_face[1:3,ic] .= cross(t[1:3,1,ic],-t[1:3,3,ic])
        @views  n̂_face[1:3,ic] .= n_face[1:3,ic]./norm(n_face[1:3,ic])
        
        
        for j in range(1,npf)   
            vertex_index = face[j]
            if vertex_index > num_nodes
                @warn "Vertex index $vertex_index out of bounds for face $ic"
                continue
            end
            @views  vertex_normals[1:3, vertex_index] .+= n̂_face[1:3,ic]
            count_face_per_vertex[vertex_index] += 1        
            
            @views  t̂[1:3,j,ic] .= t[1:3,j,ic]./norm(t[1:3,j,ic])
            @views  n̂_edge[1:3,j,ic] .= cross(t̂[1:3,j,ic],n̂_face[1:3,ic])
           
            @views  d_face_edge[j,ic] = norm([face_center[1:3,ic]-edge_mid[1:3,j,ic]])
            
        end
       
        @views volume_face[ic] = norm(n_face[1:3,ic])/2 # face area, aka face volume in 2D
        
    end
    vertex_faces = Vector{Vector{Int32}}(undef,num_nodes)
    # @show "Allocating vertex faces"
    for i in 1:num_nodes
        @views vertex_normals[1:3, i] .= normalize(vertex_normals[1:3, i])
        vertex_faces[i] = zeros(Int32,count_face_per_vertex[i])
    end

    
    # @show "Filling vertex faces"
    for (ic, face) in enumerate(eachcol(faces))

        for j in 1:npf
            vertex_index = face[j]
            running_count_face_per_vertex[vertex_index] += 1
            
            vertex_faces[vertex_index][running_count_face_per_vertex[vertex_index]] = ic
        end
    end
    # @show "Calculating vertex normals"
    w = Vector{Float64}(undef,3)
    max_faces_per_vertex = maximum(count_face_per_vertex)
    vertex_normal_A = zeros(Float64, 3, max_faces_per_vertex)

    for (i, vertex) in enumerate(eachcol(nodes))
        
        faces_per_vertex = count_face_per_vertex[i]
        if faces_per_vertex > max_faces_per_vertex
            @warn "Vertex $i has more faces ($faces_per_vertex) than allocated space ($max_faces_per_vertex)"

        end
        
        for j in 1:faces_per_vertex
            face_index = vertex_faces[i][j]
            if face_index > size(n̂_face, 2)
                @warn "Face index $face_index out of bounds for vertex $i"
                continue
            end
            @views vertex_normal_A[1:3, j] .= n̂_face[1:3, face_index]
        end
        
       
        
        w = max_min_dot_product_mod(vertex_normal_A, vertex_normals[1:3, i], faces_per_vertex,0.1,i)
        @views vertex_normals[1:3, i] .= normalize(w)
    end

  
    arrow_scale = 0.5
    
    vertex_normal = Vector{Float64}(undef, 3)
    negative_volume_counter = 0
    for (ic, face) in enumerate(eachcol(faces))
        for j in 1:npf
            vertex_index = face[j]
           
            @views vertex_normal .= vertex_normals[1:3, vertex_index]
            
            if n̂_face[1,ic]*vertex_normal[1] + n̂_face[2,ic]*vertex_normal[2] + n̂_face[3,ic]*vertex_normal[3] < 0.2
               
                negative_volume_counter += 1
            end

        end
        
    end
   
    
    # The weight is used to calculate edge values at the midpoint of the edge.
    # The weight reflects the relative distance between each face's center
    # and the shared edge.  
   
    area_edge = Vector{Float32}(undef,num_edges)
    dCF = Vector{Float32}(undef,num_edges)
    rCF = Array{Float32,2}(undef,3,num_edges)
    S = Array{Float32,2}(undef,3,num_edges)
    Sedge = zeros(Float32,2,3,num_edges)
    E = Array{Float32,2}(undef,3,num_edges)
    Ef = Vector{Float32}(undef,num_edges)
    T = Array{Float32,2}(undef,3,num_edges)
    e = Array{Float32,2}(undef,3,num_edges)
    weight_edge = Vector{Float32}(undef,num_edges)
  
    n_bend = Vector{Float32}(undef,3)
    n_rCF = Vector{Float32}(undef,3)
    n̂_rCF = Array{Float32}(undef,3,num_edges)
    for (ie,edge) in enumerate(eachcol(edges))
        node1 = edge[1]
        node2 = edge[2]
        face1 = edge[3]
        face2 = edge[4]
        edge_face1 = edge[7]
        edge_face2 = edge[8]
        bedge_number = edge[9]
        @views area_edge[ie] = norm(nodes[1:3,node2]-nodes[1:3,node1])
        if edge[6] == 0
            # weight_edge depends on the face order in edges.  In edges, the first face is the one
            # with lower number.
            weight_edge[ie] = d_face_edge[edge_face2,face2]/(d_face_edge[edge_face1,face1] + d_face_edge[edge_face2,face2])
            
            
            @views  rCF[1:3,ie] .=  face_center[1:3,face2] .- face_center[1:3,face1]
            
        else
            # Only one face per edge at boundary
            # area_bedge[bedge_number] = area_edge[ie]
            weight_edge[ie] = 1.0
           
            
            @views  rCF[1:3,ie] .= edge_mid[1:3,edge_face1,face1] .- face_center[1:3,face1]
        end
        dCF[ie] = norm(rCF[1:3,ie])
        e[1:3,ie] .= normalize(rCF[1:3,ie])
        n_bend .= cross(rCF[1:3,ie],t̂[1:3,edge_face1,face1])
        
        n_rCF .= cross(t̂[1:3,edge_face1,face1],n_bend)
        n̂_rCF[1:3,ie] .= normalize(n_rCF)
        S[1:3,ie] .= n̂_rCF[1:3,ie] .* area_edge[ie]
        @views Sedge[1,1:3,ie] .= n̂_edge[1:3,edge_face1,face1] .* area_edge[ie]
        if face2 > 0
            @views Sedge[2,1:3,ie] .= n̂_edge[1:3,edge_face2,face2] .* area_edge[ie]

        end
        E[1:3,ie] .= dot(S[1:3,ie],S[1:3,ie])/dot(e[1:3,ie],S[1:3,ie]).*e[1:3,ie]
        Ef[ie] = norm(E[1:3,ie])
        T[1:3,ie] .= S[1:3,ie] .- E[1:3,ie]
        
    end
   
    weight_node = Array{Float32,2}(undef,npf,num_faces)
    weight_node_sum = zeros(num_nodes)
    t̂_dot_lf_δ_edge = Array{Float32,2}(undef,npf,num_faces)
    counter = 0
    δ_edge = Float32(0.0)
    for (ic,face) in enumerate(eachcol(faces))
        for j in range(1,npf)
            if norm(nodes[1:3,face[j]]-face_center[1:3,ic]) == 0.0
                counter += 1
                
            end
            @views weight_node[j,ic] = 1/norm(nodes[1:3,face[j]]-face_center[1:3,ic])
            weight_node_sum[face[j]] += weight_node[j,ic]
            edge = face[npf+j]
            if edges[6,edge] == 0
                δ_edge = dot(rCF[1:3,edge],n̂_rCF[1:3,edge])
                @views t̂_dot_lf_δ_edge[j,ic] = dot(t̂[1:3,j,ic],rCF[1:3,edge].*face[npf*3+j])/δ_edge
            else
                δ_edge = dot(rCF[1:3,edge],n̂_rCF[1:3,edge])
                @views t̂_dot_lf_δ_edge[j,ic] = dot(t̂[1:3,j,ic],rCF[1:3,edge].*face[npf*3+j])/δ_edge
            end
        end
    end

    for (ic,face) in enumerate(eachcol(faces))
        for j in range(1,npf)
            weight_node[j,ic] = weight_node[j,ic]/weight_node_sum[face[j]]
        end
    end
    
    if counter > 0
        @show "Defective face element,exiting",counter,mean(weight_node),mean(nodes),mean(face_center),mean(weight_node_sum)
        throw(DomainError(counter, "# defective face elements"))
        exit()
    end



    eps = 1e-4
    span = height*62.5
    bx = [-span span] .+ xval
    by = [-span span] .+ yval
    bz = ([0.0 500.0] .+ startframe) #* height*1.25
    zone_bx = [1 2]
    zone_by = [3 4]
    zone_bz = [5 6]
    cell_edges_on_boundary = zeros(Int32,18)

    for (i,bedge) in enumerate(eachcol(boundary_edges))
        missing = true

        for j in 1:2
            if (abs(nodes[1,bedge[1]] - bx[j]) < eps  && abs(nodes[1,bedge[2]] - bx[j]) < eps  )
                if xcounter == location[1,j]
                    boundary_edges[5,i] = zone_bx[j]
                    boundary_edges[6,i] = -1
                    cell_edges_on_boundary[j] -= 1
                else
                    boundary_edges[5,i] = 7
                    boundary_edges[6,i] = zone_bx[j]
                    cell_edges_on_boundary[j] += 1
                end
                
                missing = false
                
            end
            if (abs(nodes[2,bedge[1]] - by[j]) < eps  && abs(nodes[2,bedge[2]] - by[j]) < eps  )
                if ycounter == location[2,j]
                    boundary_edges[5,i] = zone_by[j]
                    boundary_edges[6,i] = -1
                    cell_edges_on_boundary[2+j] -= 1
                else
                    boundary_edges[5,i] = 7
                    boundary_edges[6,i] = zone_by[j]
                    cell_edges_on_boundary[2+j] += 1
                end
                
                missing = false
                
            end
            if (abs(nodes[3,bedge[1]] - bz[j]) < eps  && abs(nodes[3,bedge[2]] - bz[j]) < eps  )
                if zcounter == location[3,j]
                    boundary_edges[5,i] = zone_bz[j]
                    boundary_edges[6,i] = -1
                    cell_edges_on_boundary[4+j] -= 1
                else
                    boundary_edges[5,i] = 7
                    boundary_edges[6,i] = zone_bz[j]
                    cell_edges_on_boundary[4+j] += 1
                end
                
                missing = false
                
            end
        end

        if missing == true
            println("Missing boundary edge \n")
            
            @show i,boundary_edges[5,i],nodes[:,bedge[1]],nodes[:,bedge[2]],bx,by,bz
            
        end
       
    end


    
    p1 = sortperm(boundary_edges[6,:],rev=false)
    boundary_edges = boundary_edges[:,p1]
    edge_to_boundary_edge = collect(1:num_boundary_edges)
    edge_to_boundary_edge[p1] = edge_to_boundary_edge
  

    bedge_owner_ptr_tmp = Vector{UnitRange}(undef,6)
    bedge_owner_ptr_length = zeros(Int32,6)
    bedge_owner_ptr_present = zeros(Int32,6)
    split_boundary_edges::Int64 = 0
    sum_interior_faces::Int32 = 0

    for i in 1:6
        bedge_owner_ptr_tmp[i] = searchsorted(boundary_edges[6,:],i)
        bedge_owner_ptr_length[i] = length(bedge_owner_ptr_tmp[i])
        if bedge_owner_ptr_length[i] > 0
            bedge_owner_ptr_present[i] = 1
            cell_edges_on_boundary[6+i] =  sum_interior_faces + 1
            if split_boundary_edges == 0
                split_boundary_edges = bedge_owner_ptr_tmp[i].start
            end
            sum_interior_faces += bedge_owner_ptr_length[i]
        end

    end
    interior_boundary_faces = zeros(Int32,sum_interior_faces)

    interior_boundary_faces .= boundary_edges[3,split_boundary_edges:split_boundary_edges+sum_interior_faces-1]
    
    interior_boundary_nodes = zeros(Int32,2,sum_interior_faces)
    
    interior_boundary_nodes[1,1:sum_interior_faces] .= boundary_edges[1,split_boundary_edges:split_boundary_edges+sum_interior_faces-1]
    interior_boundary_nodes[2,1:sum_interior_faces] .= boundary_edges[2,split_boundary_edges:split_boundary_edges+sum_interior_faces-1]
  
    if sum(bedge_owner_ptr_length) > 0
        MPI.Win_lock(MPI.LOCK_EXCLUSIVE, 0, 0, win_accumulate_cells_on_cube_side)
        MPI.Accumulate!(bedge_owner_ptr_present, MPI.SUM, win_accumulate_cells_on_cube_side;rank=0) 
        MPI.Win_unlock(0, win_accumulate_cells_on_cube_side)

        MPI.Win_lock(MPI.LOCK_EXCLUSIVE, 0, 0, win_cube_face_match)
        MPI.Put!(cell_edges_on_boundary, win_cube_face_match;rank=0,disp=18*(list_number-1)) 
        MPI.Win_unlock(0, win_cube_face_match)
    end
    
                                                                                              
    jldsave(string(path,cell_number,extension,"_setup",".jld2"),true;solution_setup=Solution_setup(Sim_parameters,weight_node,t̂_dot_lf_δ_edge,volume_face,S,Sedge,E,Ef,T,dCF,rCF,weight_edge))
    jldsave(string(path,cell_number,extension,"_boundary_edges",".jld2"),true;boundary_setup=Boundary_edges(boundary_edges,interior_boundary_faces,sum_interior_faces,split_boundary_edges,edge_to_boundary_edge,interior_boundary_nodes))

    return
end
   

function max_min_dot_product_mod(A::Matrix{Float64}, direction::Vector{Float32},face_per_vertex::Int64,alpha::Float64,ic::Int64)
    # Ensure A is 3xn and direction is a 3-element vector
    @assert size(A, 1) == 3 && length(direction) == 3

    # Normalize the direction vector
    direction_normalized = normalize(direction)

    # Calculate dot products with each column of A
    dot_products = [dot(A[:, i], direction_normalized) for i in 1:face_per_vertex]
    
    # Find the minimum dot product
    min_dot, min_index = findmin(dot_products)
    min_indices = findall(x -> isapprox(x, min_dot, atol=1e-5), dot_products)
    

    
    # We'll use an iterative approach to find the best direction
    best_direction = direction_normalized
    best_min_dot = min_dot
    
    for _ in 1:100  # Limit iterations to prevent infinite loops
        gradient = mean(A[:, i] for i in min_indices)
        new_direction = normalize(best_direction + alpha * gradient)
        
        new_dot_products = [dot(A[:, i], new_direction) for i in 1:face_per_vertex]
        new_min_dot, new_min_index = findmin(new_dot_products)
        
        if new_min_dot > best_min_dot
            best_direction = new_direction
            best_min_dot = new_min_dot
            min_indices = findall(x -> isapprox(x, new_min_dot, atol=1e-5), new_dot_products)
        else
            break
        end
    end
    if best_min_dot < 0.1
        @show ic,best_min_dot,best_direction
    end
    return best_direction
end

function cube_matrix_setup(path,extension,comm,rank,world_size,xcounter,ycounter,zcounter,counter,location,current_startframe,endframe,current_xval,current_yval,height)
    num_cells_buffer = Vector{Int64}(undef,1)
    null_array = Vector{Int32}(undef,1)
    if rank == 0            
        filename = string(path,frame_name,current_startframe,"_",endframe,"_",current_xval,"_",current_yval,"_",height,".csv")
        cell_list_temp = CSV.File(filename) |> Tables.matrix

        num_cells_buffer[1] = length(cell_list_temp)

    end
    MPI.Barrier(comm)
    MPI.Bcast!(num_cells_buffer, comm; root=0)
    MPI.Barrier(comm)
    if num_cells_buffer[1] == 0
        @show "No cells, continuing",rank
        return
    end
     
    cell_list = Vector{Int64}(undef,num_cells_buffer[1])
    
    if rank == 0
        cell_list = cell_list_temp[:,1]
    end
    MPI.Bcast!(cell_list, comm; root=0)

    edges_on_cube_side = Vector{Int32}(undef,18)
    if rank == 0


        filename = string(path,frame_name,current_startframe,"_",endframe,"_",current_xval,"_",current_yval,"_",height,"_ghost_static.jld2")
        fid_jld_ghost = jldopen(filename)
        ghost_setup_static::Ghost_cells_static = fid_jld_ghost["ghost_setup_static"]
        cube_face_S::Vector{Array{Float32,2}} = ghost_setup_static.cube_face_S
        cube_face_E::Vector{Array{Float32,2}} = ghost_setup_static.cube_face_E
        cube_face_Ef::Vector{Vector{Float32}} = ghost_setup_static.cube_face_Ef 
        cube_face_T::Vector{Array{Float32,2}} = ghost_setup_static.cube_face_T       
        cube_face_dCF::Vector{Vector{Float32}} = ghost_setup_static.cube_face_dCF
        cube_face_rCF::Vector{Array{Float32,2}}  = ghost_setup_static.cube_face_rCF  
        cube_face_weight_edge::Vector{Vector{Float32}} = ghost_setup_static.cube_face_weight_edge
        cube_face_t̂_dot_lf_δ_edge::Vector{Vector{Float32}} = ghost_setup_static.cube_face_t̂_dot_lf_δ_edge
        close(fid_jld_ghost)

        filename = string(path,frame_name,current_startframe,"_",endframe,"_",current_xval,"_",current_yval,"_",height,".jld2")
        fid_jld = jldopen(filename)
        cubeface_cells::Cube_properties = fid_jld["cubeface_cells"] 
        
        cells_on_faces::Array{Int32,2} = cubeface_cells.cells_on_faces
        
        edges_on_cube_side::Vector{Int32} = cubeface_cells.edges_on_cube_side 
        close(fid_jld)
        cells_on_faces_vec::Vector{Int32} = vec(cells_on_faces)

        win_cells_on_faces = MPI.Win_create(cells_on_faces_vec, comm)
    else
        win_cells_on_faces = MPI.Win_create(null_array, comm)
    end

    MPI.Barrier(comm)
    MPI.Bcast!(edges_on_cube_side, comm; root=0)
   
    MPI.Barrier(comm)
    win_cube_face_area_delta =  Vector{MPI.Win}(undef,6)
    null_array_f32 = Vector{Float32}(undef,1)
    for iter in 1:6
        if rank == 0
            
            size_area_delta = length(cube_face_Ef[iter])
            
            if size_area_delta > 1
                
                
            
                
                area_delta = zeros(Float32,Int64(size_area_delta*16)) 
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
                
                lhs_iter = 16
                
                @views area_delta[size_area_delta*(lhs_iter-1)+1:size_area_delta*lhs_iter] .= cube_face_t̂_dot_lf_δ_edge[iter][:]
            
                
 
                win_cube_face_area_delta[iter] = MPI.Win_create(area_delta, comm)
            else
                win_cube_face_area_delta[iter] = MPI.Win_create(null_array_f32, comm) 
            end
            
        else
            win_cube_face_area_delta[iter] = MPI.Win_create(null_array_f32, comm)            
        end
    end
    MPI.Barrier(comm)

    job_queue_matrix(cell_list,path,extension,comm,rank,world_size,xcounter,ycounter,zcounter,counter,location,current_startframe,endframe,current_xval,current_yval,height,edges_on_cube_side,win_cells_on_faces,win_cube_face_area_delta)
    MPI.Barrier(comm)
    MPI.free(win_cells_on_faces)
    MPI.Barrier(comm)
    for i in 1:6
        MPI.free(win_cube_face_area_delta[i])
    end
    MPI.Barrier(comm)
end

function matrix_setup(cell_list,list_number,path,extension,comm,rank,root,world_size,xcounter,ycounter,zcounter,counter,location,current_startframe,endframe,current_xval,current_yval,height,edges_on_cube_side,win_cells_on_faces,win_cube_face_area_delta)
    

    cell_number = cell_list[list_number]
 

    fid_jld_sol = jldopen(string(path,cell_number,extension,"_setup",".jld2"))
    # display(fid_jld_sol)

    solution_setup::Solution_setup = fid_jld_sol["solution_setup"]
    Sim_parameters::Simulation_parameters = solution_setup.Sim_parameters
    Δt::Float32 = Sim_parameters.Δt
    BCarray::Vector{BC} = Sim_parameters.BCarray
    ICvalue::Float32 = Sim_parameters.ICvalue
    ℾ::Float32 = Sim_parameters.ℾ
    ρ::Float32 = Sim_parameters.ρ
    ū::Vector{Float32} = Sim_parameters.ū
    Sourceᵤ::Float32 = Sim_parameters.Sourceᵤ
    Sourceₚ::Float32 = Sim_parameters.Sourceₚ
    S::Array{Float32,2} = solution_setup.S
    Sedge::Array{Float32,3} = solution_setup.Sedge
    E::Array{Float32,2} = solution_setup.E
    Ef::Vector{Float32} = solution_setup.Ef
    T::Array{Float32,2} = solution_setup.T
    dCF::Vector{Float32} = solution_setup.dCF
    rCF::Array{Float32,2} = solution_setup.rCF
    weight_edge::Vector{Float32} = solution_setup.weight_edge
    volume_face::Vector{Float32} = solution_setup.volume_face
    weight_node::Array{Float32,2} = solution_setup.weight_node
    t̂_dot_lf_δ_edge::Array{Float32,2} = solution_setup.t̂_dot_lf_δ_edge
    close(fid_jld_sol)


    
    filetype = "_structure.jld2"
    filename = string(path,cell_number,extension,filetype)
    fid_jld = jldopen(filename)
    # display(fid_jld)
    structure_setup = fid_jld["structure_setup"] 
    npf::Int32 = structure_setup.npf
    num_faces::Int32 = structure_setup.num_faces
    num_nodes::Int32 = structure_setup.num_nodes
    num_edges::Int32 = structure_setup.num_edges
    num_boundary_edges::Int32 = structure_setup.num_boundary_edges
    

    nodes = Array{Float32,2}(undef, 3,num_nodes)  
    faces = Array{Int32,2}(undef, npf*4,num_faces)
    edges = Array{Int32,2}(undef, 11,num_edges)
    boundary_edges = Array{Int32,2}(undef, 11,num_boundary_edges)
    boundary_node_values = zeros(Float32,2,num_boundary_edges)
    face_center = Array{Float32,2}(undef,3,num_faces)
    

    faces::Array{Int32,2} = structure_setup.faces
    nodes::Array{Float32,2} = structure_setup.nodes
    edges::Array{Int32,2} = structure_setup.edges
    face_center::Array{Float32,2} = structure_setup.face_center
    close(fid_jld)

    fid_jld_be = jldopen(string(path,cell_number,extension,"_boundary_edges",".jld2"))
    boundary_setup::Boundary_edges = fid_jld_be["boundary_setup"]
    boundary_edges::Array{Int64,2} = boundary_setup.boundary_edges 
    split_boundary_edges::Int64 = boundary_setup.split_boundary_edges
    sum_interior_faces::Int64 = boundary_setup.sum_interior_faces
    interior_boundary_faces = zeros(Int32,sum_interior_faces)
    interior_boundary_faces::Array{Int32,1} = boundary_setup.interior_boundary_faces
    edge_to_boundary_edge::Vector{Int64} = boundary_setup.edge_to_boundary_edge
    close(fid_jld_be)

    cells_on_faces_local = Vector{Int32}(undef,18)
    MPI.Win_lock(win_cells_on_faces;rank=root,type=:exclusive)
    MPI.Get!(cells_on_faces_local, win_cells_on_faces;rank=root,disp=(list_number-1)*18)
    MPI.Win_unlock(win_cells_on_faces;rank=root)



 
    
    aₚ = Vector{Float32}(undef,num_faces)

    aₙ = Array{Float32}(undef,npf,num_faces)

    sourceᵤ = zeros(Float32,num_faces)
    sourceₚ = zeros(Float32,num_faces)
    count_interior = 0
    bedge = Int32(0)
    for (ic,face) in enumerate(eachcol(faces))
        aₚ[ic] = 0.0
        sourceᵤ[ic] = Sourceᵤ*volume_face[ic]
        sourceₚ[ic] = Sourceₚ*volume_face[ic]
        for j in range(1,npf)   
            edge = face[npf+j] 
            
            F = ū[1]*S[1,edge]*face[npf*3+j] + ū[2]*S[2,edge]*face[npf*3+j] + ū[3]*S[3,edge]*face[npf*3+j]
            
            ⎹F⎸ = abs(F)
            if edges[6,edge] == 0  
                aₙ[j,ic] = ℾ*Ef[edge]/dCF[edge]
                aₚ[ic] += aₙ[j,ic]
                aₙ[j,ic] += -(F -⎹F⎸)/2 #-max(-F,0)
                aₚ[ic] += (F +⎹F⎸)/2 #max(F,0)
              
            else

                aₙ[j,ic] = 0.0
                bedge = Int32(edge_to_boundary_edge[edges[9,edge]])
                for bvalue in BCarray
                    
                    # bvalue.ID will be 1-7, 7 being an interior cube face
                    
                    if boundary_edges[5,bedge] == bvalue.ID
                        if bvalue.type == 3 # Interior 
                            
                            count_interior += 1
                           
                        else
                            if bvalue.type == 2  # Neumann, 
                                sourceᵤ[ic] += bvalue.value
                                
                            else 
                                if bvalue.type == 1 # Dirichlet
                                    sourceᵤ[ic] += bvalue.value*(ℾ*Ef[edge]/dCF[edge] - (F -⎹F⎸)/2)
                                   
                                    sourceₚ[ic] += ℾ*Ef[edge]/dCF[edge] + (F +⎹F⎸)/2
                                    
                                    boundary_node_values[1,bedge] = bvalue.value
                                    boundary_node_values[2,bedge] = bvalue.value
                                else
                                    @show "Error",bvalue.type
                                end
                            end
                        end
                        break
                    end
                end
            end
            
        end
        aₚ[ic] += sourceₚ[ic]
        aₚ[ic] += volume_face[ic]/Δt
    end

    for iter in 1:6
        
        if cells_on_faces_local[12+iter] > 0


            face_vector_start = cells_on_faces_local[12+iter]
            
            cell_vector_start = cells_on_faces_local[6+iter]
            num_boundary_faces = cells_on_faces_local[iter]
            size_area_delta = edges_on_cube_side[iter]
            ghost_S = zeros(Float32,num_boundary_faces*3)
            ghost_E = zeros(Float32,num_boundary_faces*3)
            ghost_Ef = zeros(Float32,num_boundary_faces)
            ghost_T = zeros(Float32,num_boundary_faces*3)   
            ghost_dCF = zeros(Float32,num_boundary_faces)
            ghost_rCF = zeros(Float32,num_boundary_faces*3) 
            ghost_weight_edge = zeros(Float32,num_boundary_faces)
            ghost_t̂_dot_lf_δ_edge = zeros(Float32,num_boundary_faces)
             
            MPI.Win_lock(win_cube_face_area_delta[iter];rank=root,type=:exclusive)
            MPI.Get!(ghost_S, win_cube_face_area_delta[iter];rank=root,disp=face_vector_start-1)
            MPI.Get!(ghost_E, win_cube_face_area_delta[iter];rank=root,disp=face_vector_start-1+size_area_delta*3)
            MPI.Get!(ghost_T, win_cube_face_area_delta[iter];rank=root,disp=face_vector_start-1+size_area_delta*6)
            MPI.Get!(ghost_rCF, win_cube_face_area_delta[iter];rank=root,disp=face_vector_start-1+size_area_delta*9)
            MPI.Get!(ghost_Ef, win_cube_face_area_delta[iter];rank=root,disp=face_vector_start-1+size_area_delta*12)
            MPI.Get!(ghost_dCF, win_cube_face_area_delta[iter];rank=root,disp=face_vector_start-1+size_area_delta*13)
            MPI.Get!(ghost_weight_edge, win_cube_face_area_delta[iter];rank=root,disp=face_vector_start-1+size_area_delta*14)
            MPI.Get!(ghost_t̂_dot_lf_δ_edge, win_cube_face_area_delta[iter];rank=root,disp=face_vector_start-1+size_area_delta*15)
            
            MPI.Win_unlock(win_cube_face_area_delta[iter];rank=root)
        
           
            if mod(iter,2) == 0
              
                n̂_edge = 1.0
            else
                
                n̂_edge = -1.0
            end
           
            for edge_number in 1:num_boundary_faces
                cell_index_value = edge_number + cell_vector_start - 1
                face_value = interior_boundary_faces[cell_index_value]
                face_index_value = edge_number + face_vector_start - 1
                edge_index = boundary_edges[9,cell_index_value + split_boundary_edges -1]
                @views S[1:3,edge_index] = [ghost_S[edge_number] ghost_S[edge_number+num_boundary_faces] ghost_S[edge_number+num_boundary_faces*2]]
                
                @views E[1:3,edge_index] = [ghost_E[edge_number] ghost_E[edge_number+num_boundary_faces] ghost_E[edge_number+num_boundary_faces*2]]
                @views T[1:3,edge_index] = [ghost_T[edge_number] ghost_T[edge_number+num_boundary_faces] ghost_T[edge_number+num_boundary_faces*2]]
                @views rCF[1:3,edge_index] = [ghost_rCF[edge_number] ghost_rCF[edge_number+num_boundary_faces] ghost_rCF[edge_number+num_boundary_faces*2]]
                Ef[edge_index] = ghost_Ef[edge_number]
                dCF[edge_index] = ghost_dCF[edge_number]
                weight_edge[edge_index] = ghost_weight_edge[edge_number]
                t̂_dot_lf_δ_edge[edge_index] = ghost_t̂_dot_lf_δ_edge[edge_number]

                F = ū[1]*n̂_edge*ghost_S[edge_number] + ū[2]*n̂_edge*ghost_S[edge_number+num_boundary_faces] + ū[3]*n̂_edge*ghost_S[edge_number+num_boundary_faces*2]
                ⎹F⎸ = abs(F)
               
                                
            
                    

               
                aₚ[face_value] += ℾ*ghost_Ef[edge_number]/ghost_dCF[edge_number] + (F +⎹F⎸)/2 #+ max(F,0)    
                
            end
        end

    end
    jldsave(string(path,cell_number,extension,"_setup",".jld2"),true;solution_setup=Solution_setup(Sim_parameters,weight_node,t̂_dot_lf_δ_edge,volume_face,S,Sedge,E,Ef,T,dCF,rCF,weight_edge))
    
    I = zeros(Int32,num_faces*4)
    J = zeros(Int32,num_faces*4)
    Va = zeros(Float32,num_faces*4)
    edge_count = 0
    count = 0
    for (ic,face) in enumerate(eachcol(faces))
        count += 1
        I[count] = ic
        J[count] = ic
        Va[count] = aₚ[ic]
        for j in range(1,3)
            if face[npf*2+j] == 0
                edge_count += 1
            else
                count += 1
                I[count] = ic
                J[count] = face[npf*2+j]
                Va[count] = -aₙ[j,ic]
            end
        end
    end
    I = I[1:count]
    J = J[1:count]
    Va = Va[1:count] 

    a_sparse = sparse(I,J,Va)

 

    
    jldsave(string(path,cell_number,extension,"_matrix_setup",".jld2"),true;matrix_setup=Matrix_setup(sourceᵤ,a_sparse,boundary_node_values))
end





# This function initializes the global setup for a neuron transport simulation.
# It sets up MPI communication, defines boundary conditions, and initializes
# simulation parameters.
# After the basic setup applicable to all analysis cubes, the function 'job_queue'
# parallelizes the function 'uniquenode', which sets up the geometric parameters of the mesh
# and the boundary edge properties.
# 'job_queue_match' parallelizes 'match_owners', which identifies face matches between
# biological cells.  This is followed by 'match_neighbors' that finalizes the match information.
#
# 'interface_setup' identifies edges at boundaries between analysis volumes
# 
# 'cube_matrix_setup' reads in geometric parameters for all analysis volumes and calls
# 'job_queue_matrix' that parallelizes 'matrix_setup', which produces the a_sparse matrix
function global_setup(sim_number::Int64, path::String, frame_name::String)
    MPI.Init()
    root = 0
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    world_size = MPI.Comm_size(comm)
    if world_size <= 1
        println("World size must be greater than 1")
        return
    end

    #######User settings############
    

    # Read the TOML file
    settings = TOML.parsefile(string(@__DIR__,"/simulation_settings.toml"))
 
    # Access the settings
    
    BCarray = [BC(Int32(bc["ID"]), Int32(bc["type"]), Float32(bc["value"])) for (_, bc) in settings["boundary_conditions"]]
    Sourceᵤ = settings["sources"]["Source_u"]
    Sourceₚ = settings["sources"]["Source_p"]
    ICvalue = settings["parameters"]["ICvalue"]
    Δt = settings["parameters"]["deltat"]
    ℾ = settings["parameters"]["Diffusion coefficient"]
    ρ = settings["parameters"]["Density"]
    ū = settings["parameters"]["velocity"]

 
    #########End user settings###########################

    Sim_parameters = Simulation_parameters(Δt,BCarray,ICvalue,ℾ,ρ,ū,Sourceᵤ,Sourceₚ)

        
    path_variables = Vector{Int64}(undef,5)
    location = zeros(Int64,3,2)

    cells_on_cube_side = zeros(Int32,6)
    
    win_accumulate_cells_on_cube_side = MPI.Win_create(cells_on_cube_side, MPI.COMM_WORLD)
    
    if rank == 0
        filename = string(@__DIR__,"/Simulation_settings_4.0_smoothed.csv")
        settings = CSV.File(filename) |> Tables.matrix
        
             
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
        
        jldsave(string(path,"Simulation_parameters_",startframe,"_",xval,"_",yval,"_",height,".jld2");Sim_parameters=Sim_parameters)
       
    end
    MPI.Barrier(comm)

    MPI.Bcast!(location, comm; root=0)

    cubes = (location[1,2]-location[1,1]+1)*(location[2,2]-location[2,1]+1)*(location[3,2]-location[3,1]+1)
    
    cube_face_match_collection = Vector{Vector{Int32}}(undef,cubes)
    MPI.Barrier(comm)

    MPI.Bcast!(path_variables, comm; root=0)
    slices,startframe,xval,yval,height = path_variables
    
    xy_elements = height*125
    MPI.Barrier(comm)


    num_cells_buffer = Vector{Int64}(undef,1)

    counter = 0
    for xcounter in location[1,1]:location[1,2]
        for ycounter in location[2,1]:location[2,2]
            for zcounter in location[3,1]:location[3,2]
                MPI.Barrier(comm)
                counter += 1
                current_startframe = startframe + zcounter*slices
                current_xval = xval + xcounter*xy_elements
                current_yval = yval + ycounter*xy_elements
                if rank == 0
                    @show xcounter,ycounter,zcounter
                    @show current_startframe,current_xval,current_yval  
                end
                extension = string("_",current_startframe,"_",current_xval,"_",current_yval,"_",height)
    
                endframe = current_startframe+slices
                if rank == 0            
                    filename = string(path,frame_name,current_startframe,"_",endframe,"_",current_xval,"_",current_yval,"_",height,".csv")
                    cell_list_temp = CSV.File(filename) |> Tables.matrix
    
                    num_cells_buffer[1] = length(cell_list_temp)

                end
                MPI.Barrier(comm)
                MPI.Bcast!(num_cells_buffer, comm; root=0)
                MPI.Barrier(comm)
                if num_cells_buffer[1] == 0
                    @show "No cells, continuing",rank
                    continue
                end
                 
                cell_list = Vector{Int64}(undef,num_cells_buffer[1])
                null_array = zeros(Int32,1)
                if rank == 0
                    owner_ptr = zeros(Int32,3*num_cells_buffer[1]) 
                    cell_list = cell_list_temp[:,1]
                    num_faces_array = zeros(Int32,num_cells_buffer[1])

                    cube_face_match_collection[counter] = zeros(Int32,18*num_cells_buffer[1])
                    win_cube_face_match = MPI.Win_create(cube_face_match_collection[counter], comm)
                    win_accumulate_owner_ptr = MPI.Win_create(owner_ptr, comm)
                    win_num_faces = MPI.Win_create(num_faces_array, comm)
                else
                    win_cube_face_match = MPI.Win_create(null_array, comm)
                    win_accumulate_owner_ptr = MPI.Win_create(null_array, comm)
                    win_num_faces = MPI.Win_create(null_array, comm)
                end
                MPI.Barrier(comm)
                MPI.Bcast!(cell_list, comm; root=0)

                
                job_queue(cell_list,path,extension,Sim_parameters,comm,rank,world_size,xcounter,ycounter,zcounter,counter,location,current_startframe,endframe,current_xval,current_yval,height,win_accumulate_cells_on_cube_side,win_cube_face_match,win_accumulate_owner_ptr,win_num_faces)

                MPI.Barrier(comm)
                
                if rank == 0

                    
                    owner_ptr = reshape(owner_ptr,3,num_cells_buffer[1])
                    counter_owner = 1
                    for (i,owner) in enumerate(eachcol(owner_ptr))
                        if owner[1] > 0
                            owner_ptr[2,i] = counter_owner
                            counter_owner += owner_ptr[1,i]
                        end
                    end

                    num_matches = counter_owner-1
                    owner = zeros(Int32,4*(num_matches))
                else
                    owner = zeros(Int32,1)
                end   

                owner_location = Vector{Int32}(undef,num_cells_buffer[1])
                if rank == 0
                    owner_location .= owner_ptr[2,:]
                end
                MPI.Barrier(comm)
                MPI.Bcast!(owner_location, comm; root=0)
                MPI.Barrier(comm)


                win_owner = MPI.Win_create(owner, comm)
                MPI.Barrier(comm)
                
                job_queue_match(cell_list,path,extension,comm,rank,world_size,win_owner,owner_location)
                MPI.Barrier(comm)
                
                if rank == 0
                    owner = reshape(owner,4,num_matches)
                    match_neighbors(owner,owner_ptr,num_matches,num_cells_buffer[1],path,extension,cell_list,num_faces_array)
                                      
                    cells_on_faces = reshape(cube_face_match_collection[counter],18,num_cells_buffer[1])
                    
                    edges_on_cube_side = vec(sum(cells_on_faces,dims=2))
                    
                    for owner_face in 2:2:6
                        running_total = 1
                        if cells_on_cube_side[owner_face] > 0
                            for (column,interface_edges) in enumerate(eachcol(cells_on_faces))
                                if interface_edges[6+owner_face] > 0
                                    cells_on_faces[12+owner_face,column] = running_total
                                    running_total += interface_edges[owner_face]
                                    
                                end
                            end

                        end
                    end
                    
                   
    
     
                    jldsave(string(path,frame_name,current_startframe,"_",endframe,"_",current_xval,"_",current_yval,"_",height,".jld2"),true;cubeface_cells=Cube_properties(cells_on_faces,cells_on_cube_side,edges_on_cube_side))
                    cells_on_cube_side .= 0,0,0,0,0,0

                end
                
                MPI.Barrier(comm)
                MPI.free(win_cube_face_match)
                MPI.Barrier(comm)

                MPI.free(win_accumulate_owner_ptr)
                MPI.free(win_num_faces)
                MPI.Barrier(comm)
                MPI.free(win_owner)
                MPI.Barrier(comm)
            end
        end
    end
    
    MPI.free(win_accumulate_cells_on_cube_side)
    MPI.Barrier(comm)

    if rank == 0
        interface_setup(location,xy_elements,ICvalue,xval,yval,startframe,height,slices)
    end
    MPI.Barrier(comm)

    counter = 0
    for xcounter in location[1,1]:location[1,2]
        for ycounter in location[2,1]:location[2,2]
            for zcounter in location[3,1]:location[3,2]
                MPI.Barrier(comm)
                counter += 1
                current_startframe = startframe + zcounter*slices
                current_xval = xval + xcounter*xy_elements
                current_yval = yval + ycounter*xy_elements
                if rank == 0
                    @show xcounter,ycounter,zcounter
                    @show current_startframe,current_xval,current_yval  
                end
                extension = string("_",current_startframe,"_",current_xval,"_",current_yval,"_",height)
    
                endframe = current_startframe+slices
                


                cube_matrix_setup(path,extension,comm,rank,world_size,xcounter,ycounter,zcounter,counter,location,current_startframe,endframe,current_xval,current_yval,height)
                

                MPI.Barrier(comm)
                
                
            end
        end
    end

    MPI.Barrier(comm)
    
    MPI.Finalize()
end




sim_number = parse(Int64,ARGS[1])
path = string(@__DIR__, "/../Neuron_transport_data/")
frame_name = "frame_uniq_"

global_setup(sim_number,path,frame_name)
