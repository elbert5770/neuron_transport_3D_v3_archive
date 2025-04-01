# Copyright Donald L. Elbert & Dylan Esguerra (c) 2022-2024 University of Washington
# All rights reserved.

# Redistribution and use in source and binary forms, with or without modification, are permitted provided
#  that the following conditions are met:

# -Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# -Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the 
# following disclaimer in the documentation and/or other materials provided with the distribution.
# -Neither the name of the University of Washington nor the names of its contributors may be used to endorse 
# or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE UNIVERSITY OF WASHINGTON AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED 
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY OF WASHINGTON OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT 
# OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON 
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING 
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

using FileIO
using CSV
using Tables
using JLD2
using SparseArrays
using SymRCM
using MPI
using PlyIO

include("build_edges_from_mesh_3D_3.0a.jl")

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

function MeshToJLD2(filename,cell_number,list_number,file_type,win_nodes)
    # In theory either .msh (Fluent) or .ply may be read in plain text or binary format

    # npf is nodes per face.  Triangles are npf = 3, quadrilaterals are npf = 4, other shapes are not supported
    npf = 3

    filetypes = [".msh",".ply"]
    
    file_type_num = 0
    for (file_num,file) in enumerate(filetypes)
        if cmp(file,file_type) == 0
            # println(string(file))
            file_type_num = file_num
        end
    end
    
    if file_type_num == 0
        println("Filetype must be .msh (Fluent) or .ply")
        return 1
    end

    # mshparser and plyparser are in build_edges_from_mesh_X.jl
    if file_type_num == 1
        num_dim,num_nodes,num_faces,num_zones,num_edges,count = mshheader(filename)
        nodes = Array{Float32,2}(undef, 3,num_nodes)   
        faces = Array{Int32,2}(undef, npf*4,num_faces)*0
        faceedges = Array{Int32,1}(undef,3)
        mshparser(filename,nodes,faces,faceedges,count,num_dim) 
    end

    if file_type_num == 2
        num_dim,num_nodes,num_faces,num_zones,num_edges,count,binary = plyheader(filename)
        if binary == 0
            println("Error: File format not supported. File must be binary PLY.")
            return
        end
        nodes = Array{Float32,2}(undef, 3,num_nodes)  
        normals = Array{Float32,2}(undef, 3,num_nodes)
        faces = Array{Int32,2}(undef, npf*4,num_faces)*0
        faceedges = Array{Int32,1}(undef,3)
        plyparser(filename,nodes,normals,faces,faceedges,count,binary,num_nodes,num_faces)     
    end

    # The manager core records the number of nodes using one sided communication
    num_nodes_array = zeros(Int32,1)
    num_nodes_array[1] = num_nodes

    MPI.Win_lock(MPI.LOCK_EXCLUSIVE, 0, 0, win_nodes)
    MPI.Put!(num_nodes_array, win_nodes;rank=0,disp=list_number-1)
    MPI.Win_unlock(0, win_nodes)

    # This is a convenience to simplify proceeding around the edges of a face
    if npf == 3
        circle_path = [1 2;2 3;3 1]
    else
        if npf == 4
            circle_path = [1 2;2 3;3 4;4 1]
        else
            println("npf (number of faces per node) must be 3 or 4)")
            return
        end
    end

    # Edges are not recorded in .msh or .ply files.  Information about edges will be stored in
    # 'edges' but for now are stored in edge_to_face
    face = Vector{Int32}(undef,npf*4)
    edge_to_face = Array{Int32,2}(undef,5,num_faces*npf)
    edge_to_face_sort = Array{Int32,2}(undef,5,num_faces*npf)
    
    for (ic,face) in enumerate(eachcol(faces))
        for j in range(1,npf)      
            if face[circle_path[j,1]] < face[circle_path[j,2]]
                @views  edge_to_face[1:5,(ic-1)*npf+j] .=  [face[circle_path[j,1]],face[circle_path[j,2]],ic,j,0]
            else
                @views  edge_to_face[1:5,(ic-1)*npf+j] .=  [face[circle_path[j,2]],face[circle_path[j,1]],ic,j,1]
            end   
        end    
    end

    # Sort 'edge_to_face' by nodes so that colums of edge_to_face that represent the same a edge are adjacent
    # An edge is duplicated in edge_to_face if it is an interior edge, and appears once if it is a boundary edge
    # sort is not necessarily type stable so this is done in two steps
    p1 = sortperm(eachslice(edge_to_face; dims=2), by=x->(x[1],x[2]))
    edge_to_face_sort = edge_to_face[:,p1]

    # "edge_comparison()" is in build_edges_from_mesh_X.jl
    # This function fills in all of the useful information to the arrays edges and faces
    # Julia is column major so the arrays record one face or edge per column
    # This allows more rapid access
    node_list = Array{Int32,2}(undef,5,2)
    count1 = 0
    count2 = 0
    skip = 1
    # Edges starts at the maximum possible size, i.e. all edges are boundary edges, which is actually not possible
    edges = Array{Int32,2}(undef,11,num_faces*npf)
    num_edges,num_boundary_edges = edge_comparison(edge_to_face_sort,edges,faces,node_list,npf,count1,count2,skip)
#    


    # The size of edges was previously unknown because the number of boundary edges was unknown
    # The number of edges is now known
    edges = edges[:,1:num_edges]
  

    # To apply the reverse Cuthill-Mckee algorithm requires that the faces are in a sparse matrix
    # reverse Cuthill-Mckee reorders the faces to minimize bandwidth of the face
    # connectivity matrix
    I = zeros(Int32,num_faces*4)
    J = zeros(Int32,num_faces*4)
    V = zeros(Int32,num_faces*4)
    edge_count = 0
    counter = 0
    for (ic,face) in enumerate(eachcol(faces))

        counter += 1
        I[counter] = ic
        J[counter] = ic
        V[counter] = 1
        for j in range(1,3)
            if face[npf*2+j] == 0
                edge_count += 1
            else
                counter += 1
                I[counter] = ic
                J[counter] = face[npf*2+j]
                V[counter] = 1
            end
        end
    end
    I = I[1:counter]
    J = J[1:counter]
    V = V[1:counter]     
    
    face_sparse = sparse(I,J,V)

    # display(face_sparse)

    # This is the RCM, p is a vector the renumbered faces
    p = symrcm(face_sparse)

  

    # The inverse of p allows a lookup of new face numbers from old face numbers 
    ip = similar(p)
    ip[p] = 1:length(p)
    # @show typeof(p),typeof(ip)
    # @show issymmetric(new_face)
    # @show isposdef(new_face)
    # @show ishermitian(new_face)
    edge_temp = Int32(0)
    for (ie,edge) in enumerate(eachcol(edges))
        # Faces connected to an edge are stored in rows 3 and 4
        for j in 3:4
            if edge[j] > 0
                edges[j,ie] = ip[edge[j]]
            end
        end
        edge_temp = edges[4,ie]
        if edges[3,ie] > edge_temp && edge_temp > 0
            
            edges[4,ie] = edges[3,ie]
            edges[3,ie] = edge_temp
            edge_temp = edges[8,ie]
            edges[8,ie] = edges[7,ie]
            edges[7,ie] = edge_temp
            edge_temp = edges[11,ie]
            edges[11,ie] = edges[10,ie]
            edges[10,ie] = edge_temp
        end
    end
    new_face = Int32(0)
    for (ic,face) in enumerate(eachcol(faces))
        # Faces adjacent to a face are stored in rows npf*2.+[1:npf]
        for j in 1:npf
            if face[npf*2+j] > 0
                new_face = ip[face[npf*2+j]]
                faces[npf*2+j,ic] = new_face
                if new_face > ip[ic]
                    faces[npf*3+j,ic] = 1
                else
                    faces[npf*3+j,ic] = -1
                end
            else
                faces[npf*3+j,ic] = 1
            end
        end
    end
    faces = faces[1:12,p]

    # Calculate the location of the centroid of each face
    vertex_temp = Array{Float32,2}(undef,3,npf)
    face_center = Array{Float32,2}(undef,3,num_faces)
    for (ic,face) in enumerate(eachcol(faces))
        for j in range(1,npf)
            @views  vertex_temp[1:3,j] .= nodes[1:3,face[j]]
        end
        face_center[1:3,ic] = sum(vertex_temp,dims=2)./npf
    end

    # All of the needed structural information is stored in files 'X_structure.jld2', which obviates further need for .msh or .ply files
    jldsave(string(filename,"_structure",".jld2"),true;structure_setup=Structure_cell(npf,num_faces,num_nodes,num_edges,num_boundary_edges,faces,nodes,normals,edges,face_center))
 

end


function job_queue(sim_number,path,frame_name,file_type)
    # job_queue parallelizes based on biological cells listed the frame_uniq file
    # The code uses a manager/worker model to use one-sided communication for information
    # needed by the workers, as well as allowing the workers to record the number of nodes
    # to the manager.  This allows the manager to sort the list of cells by descending size
    # so that larger cells are analyzed first. This helps maximize worker utilization in later
    # steps. Larger cells that require longer analysis times are started first.
    MPI.Init()

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    world_size = MPI.Comm_size(comm)
    nworkers = world_size - 1

    root = 0
    path_variables = Vector{Int64}(undef,5)
    if rank == 0
        filename = string(path,"MarchingCubes_settings_smooth.csv")
        settings = CSV.File(filename) |> Tables.matrix
        # @show settings
        slices = settings[sim_number,2]
        startframe = settings[sim_number,3]
        xval = settings[sim_number,4]
        yval = settings[sim_number,5]
        height = settings[sim_number,6]
        
        path_variables .= slices,startframe,xval,yval,height
    end
    MPI.Barrier(comm)
    MPI.Bcast!(path_variables, comm; root=0)
    # path variables are needed by all worker nodes to create proper filenames
    # The MPI paradigm can be confusing, but path_variables is created on all ranks
    # Then shared from the root rank to all other ranks
    # Then unpacked into individual variables on all ranks
    # MPI one-sided communication only passes vectors, so it is more efficient to pack
    # the variables into a vector and then unpack
    slices,startframe,xval,yval,height = path_variables

    # extension for part of the filenames
    extension = string("_",startframe,"_",xval,"_",yval,"_",height)
    
    MPI.Barrier(comm)
    
   
    num_cells_buffer = Vector{Int64}(undef,1)
    if rank == root
        endframe = startframe+slices
        filename = string(path,frame_name,startframe,"_",endframe,"_",xval,"_",yval,"_",height,".csv")
        # @show filename
        cell_list_temp = CSV.File(filename) |> Tables.matrix  
        num_cells_buffer[1] = length(cell_list_temp)
    end

    MPI.Barrier(comm)
    # Here's an example of a single int being broadcast as a 1 element vector
    MPI.Bcast!(num_cells_buffer, comm; root=root)
    if num_cells_buffer[1] == 0
        @show "No cells, exiting"
        return
    end
    
    cell_list = Vector{Int64}(undef,num_cells_buffer[1])

    if rank == root
        cell_list = cell_list_temp[:,1]

    end
    MPI.Barrier(comm)
    MPI.Bcast!(cell_list, comm; root=root)

        
    # Variables for the manager/worker model    
    T = eltype(cell_list)
    N = length(cell_list)

    # Creating a window for RMA communication. The vector of node number is
    # only full on the root rank. The 'vector' needs to be present on all other
    # ranks but does not need to be full sized
    null_array = zeros(Int32,1)
    if rank == root
        num_nodes_array = Vector{Int32}(undef,N)
        win_nodes = MPI.Win_create(num_nodes_array, comm)
    else
        win_nodes = MPI.Win_create(null_array, comm)
    end
    MPI.Barrier(comm)
   
    
    send_mesg = Array{T}(undef, 2)
    recv_mesg = Array{T}(undef, 2)

    if rank == root 
        # root setting up the job queue
        idx_recv = 0
        idx_sent = 1

        new_data = Array{T}(undef, N)
        # Array of workers requests
        sreqs_workers = Array{MPI.Request}(undef,nworkers)
        # -1 = start, 0 = channel not available, 1 = channel available
        status_workers = ones(nworkers).*-1

        # Send message to workers
        for dst in 1:nworkers
            if idx_sent > N
                break
            end
            send_mesg .= cell_list[idx_sent],idx_sent
            sreq = MPI.Isend(send_mesg, comm; dest=dst, tag=dst+32)
            idx_sent += 1
            sreqs_workers[dst] = sreq
            status_workers[dst] = 0
            print("Root: Sent number $(send_mesg[1]) to Worker $dst\n")
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
                            send_mesg .= cell_list[idx_sent],idx_sent
                            # Sends new message
                            sreq = MPI.Isend(send_mesg, comm; dest=dst, tag=dst+32)
                            idx_sent += 1
                            sreqs_workers[dst] = sreq
                            status_workers[dst] = 0
                            print("Root: Sent number $(send_mesg[1]) to Worker $dst\n")
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

        MPI.Waitall(sreqs_workers)
        # print("Root: New data = $new_data\n")
    else # If rank == worker
        # -1 = start, 0 = channel not available, 1 = channel available for new job
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
                    
                    filename = string(path,recv_mesg[1],extension)
                    MeshToJLD2(filename,recv_mesg[1],recv_mesg[2],file_type,win_nodes)
                    send_mesg[1] = recv_mesg[1]
                    sreq = MPI.Isend(send_mesg, comm; dest=root, tag=rank+42)
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

    if rank == root
        # sort cell list so that largest cells are first
        # largest cells have the most nodes
    
        p1 = sortperm(num_nodes_array,rev=true)
     
        cell_list = cell_list[p1]
     
                  
        CSV.write(string(path,frame_name,startframe,"_",endframe,"_",xval,"_",yval,"_",height,".csv"), Tables.table(cell_list), header=true)
 
    end
    MPI.Barrier(comm)
    
    MPI.free(win_nodes)
    MPI.Barrier(comm)
    MPI.Finalize()

end

# sim_number is read from file "MarchingCubes_settings.csv".
# This file has one line per analysis cube.
# It has columnsw:
# SimNumber,Slices,Startframe,Xval,Yval,Height
# sim_number is an input argument to this program
# If 8 analysis cubes are analyzed, the program is run eight times, 
# with arguments 1,2,3...8

sim_number = parse(Int32,ARGS[1])

#######User settings################
path = string(@__DIR__, "/../Neuron_transport_data/")
frame_name = "frame_uniq_"
file_type = ".ply"
###################

job_queue(sim_number,path,frame_name,file_type)