# Written by: Donald L. Elbert, University of Washington
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



function interface_setup(location,xy_elements,ICvalue,xval,yval,startframe,height,slices)


    eps = 1e-4
    
   
    counter = 0
    n̂_edge = zeros(Float32,3)
    num_edges_on_cube_side::Int64 = 0
    # Iterating over the analysis volumes (cubes)
    for xcounter in location[1,1]:location[1,2]
        for ycounter in location[2,1]:location[2,2]
            for zcounter in location[3,1]:location[3,2]
                
                counter += 1
                current_startframe = startframe + zcounter*slices
                current_xval = xval + xcounter*xy_elements
                current_yval = yval + ycounter*xy_elements
  
                endframe = current_startframe+slices
                # Read in list of (biological) cells for the current cube
                filename = string(path,frame_name,current_startframe,"_",endframe,"_",current_xval,"_",current_yval,"_",height,".csv")
                cell_list0 = vec(CSV.File(filename) |> Tables.matrix)
              
                if length(cell_list0) == 0
                    @show "No cells, continuing",rank
                    continue
                end
                # Read in lists of cells and edges on cube side
                filename = string(path,frame_name,current_startframe,"_",endframe,"_",current_xval,"_",current_yval,"_",height,".jld2")
                fid_jld = jldopen(filename)
                cubeface_cells = fid_jld["cubeface_cells"] 
                
                cells_on_faces0::Array{Int32,2} = cubeface_cells.cells_on_faces
                cells_on_cube_side0::Vector{Int32} = cubeface_cells.cells_on_cube_side
                edges_on_cube_side0::Vector{Int32} = cubeface_cells.edges_on_cube_side 
                close(fid_jld)
    

                # Allocate space for storing conc, conc neighbor and structural info
                # The '6' refers to the number of faces on a cube
                
                cube_face_conc = Vector{Vector{Float32}}(undef,6)
                cube_face_conc_ghost = Vector{Vector{Float32}}(undef,6)

                cube_face_S = Vector{Array{Float32,2}}(undef,6)
                cube_face_E = Vector{Array{Float32,2}}(undef,6)
                cube_face_Ef = Vector{Vector{Float32}}(undef,6)   
                cube_face_T = Vector{Array{Float32,2}}(undef,6)       
                cube_face_dCF = Vector{Vector{Float32}}(undef,6)
                cube_face_rCF = Vector{Array{Float32,2}}(undef,6)  
                cube_face_weight_edge = Vector{Vector{Float32}}(undef,6)
                cube_face_t̂_dot_lf_δ_edge = Vector{Vector{Float32}}(undef,6)
                
                # Below, the even number cube_faces will be treated as the owners of the
                # relationships, and the odd cube_faces will be neighbors. The cells span
                # the cubes and the cube_faces share edges at the boundary. The structural parameters
                # are identical across the boundary. Analysis of structural variables 
                # will be done on the even cube_faces and later copied to the odd cube_faces
                # after all cubes have been analyzed, which cuts down the disk writes by a factor of 3
                for cube_faces in 1:6  
                    num_edges_on_cube_side = edges_on_cube_side0[cube_faces]
                    if num_edges_on_cube_side > 0
                        cube_face_conc[cube_faces] = zeros(Float32,num_edges_on_cube_side) 
                        cube_face_conc_ghost[cube_faces] = zeros(Float32,num_edges_on_cube_side) 
    
                        cube_face_S[cube_faces] = zeros(Float32,3,num_edges_on_cube_side)
                        cube_face_E[cube_faces] = zeros(Float32,3,num_edges_on_cube_side)
                        cube_face_Ef[cube_faces] = zeros(Float32,num_edges_on_cube_side)
                        cube_face_T[cube_faces] = zeros(Float32,3,num_edges_on_cube_side)
                        cube_face_dCF[cube_faces] = zeros(Float32,num_edges_on_cube_side)
                        cube_face_rCF[cube_faces] = zeros(Float32,3,num_edges_on_cube_side)
                        cube_face_weight_edge[cube_faces] = zeros(Float32,num_edges_on_cube_side)
                        cube_face_t̂_dot_lf_δ_edge[cube_faces] = zeros(Float32,num_edges_on_cube_side)
                    else
                        cube_face_conc[cube_faces] = zeros(Float32,1)
                        cube_face_conc_ghost[cube_faces] = zeros(Float32,1)

                        cube_face_S[cube_faces] = zeros(Float32,1,1)
                        cube_face_E[cube_faces] = zeros(Float32,1,1)
                        cube_face_Ef[cube_faces] = zeros(Float32,1)
                        cube_face_T[cube_faces] = zeros(Float32,1,1)
                        cube_face_dCF[cube_faces] = zeros(Float32,1)
                        cube_face_rCF[cube_faces] = zeros(Float32,1,1)
                        cube_face_weight_edge[cube_faces] = zeros(Float32,1)
                        cube_face_t̂_dot_lf_δ_edge[cube_faces] = zeros(Float32,1)
                    end

                end


                # The neighbor cubes need to share information with the 
                # current cube. extension0 is the current cube
                extension0 = string("_",current_startframe,"_",current_xval,"_",current_yval,"_",height)
               

                # extensions[1] is in the +x direction
                # extensions[2] is in the +y direction
                # extensions[3] is in the +z direction
                neighbor_vals = [current_xval + xy_elements, current_yval + xy_elements, current_startframe + slices]
                extensions = [
                    string("_", current_startframe, "_", neighbor_vals[1], "_", current_yval, "_", height),
                    string("_", current_startframe, "_", current_xval, "_", neighbor_vals[2], "_", height),
                    string("_", neighbor_vals[3], "_", current_xval, "_", current_yval, "_", height)
                ]

                # Some filenames use the startframe(z direction) and 
                # endframe to avoid filename clashes
                # Only the neighbor in the +z direction has a different endframe
                endframes = [endframe, endframe, neighbor_vals[3] + slices]

                # This information was read in for the current cube,
                # these vectors of vectors/array store the same information 
                # for the neighbor cubes
                cell_lists = Vector{Vector{Int64}}(undef, 3)
                cells_on_faces = Vector{Array{Int32,2}}(undef, 3)
                cells_on_cube_side = Vector{Vector{Int32}}(undef, 3)
                edges_on_cube_side = Vector{Vector{Int32}}(undef, 3)

                for i in 1:3
                    face_index = 2 * i
                    if cells_on_cube_side0[face_index] > 0
                        # Read in all cells in the neighbor cube
                        filename_csv = string(path, frame_name, i == 3 ? neighbor_vals[i] : current_startframe, "_", endframes[i], "_", 
                                              i == 1 ? neighbor_vals[i] : current_xval, "_", 
                                              i == 2 ? neighbor_vals[i] : current_yval, "_", height, ".csv")
                        cell_lists[i] = CSV.File(filename_csv) |> Tables.matrix |> vec
                        if isempty(cell_lists[i]) || eltype(cell_lists[i]) != Int64
                            error("cell_lists[$i] is empty or not a vector of Int64")
                        end

                        # Read in info on cells and edges on neighbor cube faces
                        filename_jld2 = string(path, frame_name, i == 3 ? neighbor_vals[i] : current_startframe, "_", endframes[i], "_", 
                                               i == 1 ? neighbor_vals[i] : current_xval, "_", 
                                               i == 2 ? neighbor_vals[i] : current_yval, "_", height, ".jld2")
                        fid_jld = jldopen(filename_jld2)
                        cubeface_cells = fid_jld["cubeface_cells"]
                        cells_on_faces[i] = cubeface_cells.cells_on_faces
                        cells_on_cube_side[i] = cubeface_cells.cells_on_cube_side
                        edges_on_cube_side[i] = cubeface_cells.edges_on_cube_side
                        close(fid_jld)
                    end
                end

                for (ic0,cell_on_face0) in enumerate(eachcol(cells_on_faces0))
                    # if the current cube owns any faces
                    if cell_on_face0[2] > 0 || cell_on_face0[4] > 0 || cell_on_face0[6] > 0 
                        filename_edge = string(path,cell_list0[ic0],extension0,"_boundary_edges",".jld2")
                        fid_jld_cube1 = jldopen(filename_edge)
                        boundary_setup = fid_jld_cube1["boundary_setup"]
                        # display(fid_jld_cube1)

                        # Read in the boundary edges for the cell
                        boundary_edges0 = boundary_setup.boundary_edges
                        split_boundary_edges0 = boundary_setup.split_boundary_edges
                        sum_interior_faces0 = boundary_setup.sum_interior_faces
                        # area_interior_boundary_edges0 = boundary_setup.area_interior_boundary_edges
                        close(fid_jld_cube1)
                        
                        filename_edge = string(path,cell_list0[ic0],extension0,"_structure",".jld2")
                        fid_jld_cube1 = jldopen(filename_edge)
                        structure_setup = fid_jld_cube1["structure_setup"]
                        nodes0 = structure_setup.nodes
                        face_center0 = structure_setup.face_center
                        close(fid_jld_cube1)
                        
                        nodes0_1 = nodes0[:,boundary_edges0[1,split_boundary_edges0:split_boundary_edges0+sum_interior_faces0-1]] 
                        nodes0_2 = nodes0[:,boundary_edges0[2,split_boundary_edges0:split_boundary_edges0+sum_interior_faces0-1]]
                        face_center0_interface = face_center0[:,boundary_edges0[3,split_boundary_edges0:split_boundary_edges0+sum_interior_faces0-1]]
                        area_edge0 = Vector{Float32}(undef,sum_interior_faces0)
                        t0 = Vector{Float32}(undef,3)
                        t̂0 = Array{Float32,2}(undef,3,sum_interior_faces0)
                        rCF = Vector{Float32}(undef,3)
                        n_bend = Vector{Float32}(undef,3)
                        n_rCF = Vector{Float32}(undef,3)
                        n̂_rCF = Vector{Float32}(undef,3)
                        S = Vector{Float32}(undef,3)
                        E = Vector{Float32}(undef,3)
                        e = Vector{Float32}(undef,3)
                        T = Vector{Float32}(undef,3)
                        Ef = Float32(0.0)
                        δ_edge = Float32(0.0)
                        t̂_dot_lf_δ_edge = Float32(0.0)
                        d_face_edge_neighbor = Float32(0.0)
                        edge_mid0 = Array{Float32,2}(undef,3,sum_interior_faces0)
                        d_face_edge0 = Vector{Float32}(undef,sum_interior_faces0)
                        for (countnodes,nodes1) in enumerate(eachcol(nodes0_1))
                            t0 .= nodes0_2[:,countnodes] .- nodes1
                            area_edge0[countnodes] = norm(t0)
                            t̂0[1:3,countnodes] .= t0 ./area_edge0[countnodes]
                            edge_mid0[1:3,countnodes] .= (nodes0_2[:,countnodes] .+ nodes1)./2
                            d_face_edge0[countnodes] = norm(face_center0_interface[1:3, countnodes] .- edge_mid0[1:3,countnodes])
                        end
                        
                        # # Consistency check
                        # if sum(area_edge0 .- area_interior_boundary_edges0) > 0
                        #     @show "Error, areas inconsistent",area_edge0 .- area_interior_boundary_edges0
                        #     exit()
                        # end

                        

                        for face_index in 2:2:6
                            if cell_on_face0[face_index] > 0
                                # If there are cells on the owned face, then read in 
                                # neighbor boundary info plus enough structural info
                                # to calculate vectors S,E,T,rCF,as well as Ef,dCF,
                                # weight_edge and invRQ
                                neighbor_val = face_index == 2 ? current_xval + xy_elements :
                                               face_index == 4 ? current_yval + xy_elements :
                                               current_startframe + slices

                                extension_neighbor = face_index == 6 ? string("_", neighbor_val, "_", current_xval, "_", current_yval, "_", height) :
                                                    face_index == 4 ? string("_", current_startframe, "_", current_xval, "_", neighbor_val, "_", height) :
                                                    string("_", current_startframe, "_", neighbor_val, "_", current_yval, "_", height)

                                endframe_neighbor = face_index == 6 ? neighbor_val + slices : endframe

                                filename_edge = string(path, cell_list0[ic0], extension_neighbor, "_boundary_edges.jld2")
                                fid_jld_cube = jldopen(filename_edge)
                                boundary_setup = fid_jld_cube["boundary_setup"]
                                boundary_edges_neighbor = boundary_setup.boundary_edges
                                split_boundary_edges_neighbor = boundary_setup.split_boundary_edges
                                close(fid_jld_cube)

                                filename_edge = string(path, cell_list0[ic0], extension_neighbor, "_structure.jld2")
                                fid_jld_cube = jldopen(filename_edge)
                                structure_setup = fid_jld_cube["structure_setup"]
                                nodes_neighbor = structure_setup.nodes
                                face_center_neighbor = structure_setup.face_center
                                close(fid_jld_cube)

                                # cells_on_faces has 18 rows
                                # The first 6 store the total number of edges on the face
                                # with negative values if it is an exterior edge
                                # The next 6 store a cell-centric start position for the 
                                # list of edges for each cube face, which is used to store
                                # e.g. the start position in the list of interior boundary edges
                                # that is stored on a per cell basis
                                # The next 6 store a cube face-centric start position for the 
                                # list of edges for each cube face, which is used to store
                                # e.g. the phi values associated with each boundary edge
                                # that is stored on a per cube face basis
                                range_face_start = cells_on_faces0[12 + face_index, ic0]
                                range_face_stop = range_face_start + cells_on_faces0[face_index, ic0] - 1
                                range_cell_start = cells_on_faces0[6 + face_index, ic0]
                                range_cell_stop = range_cell_start + cells_on_faces0[face_index, ic0] - 1

                                cube_face_index = Int64(face_index/2)
                                

                                cell_list_neighbor = face_index == 2 ? cell_lists[1] :
                                                     face_index == 4 ? cell_lists[2] : cell_lists[3]
                                cells_on_faces_neighbor = face_index == 2 ? cells_on_faces[1] :
                                                          face_index == 4 ? cells_on_faces[2] : cells_on_faces[3]

                                for (ic_neighbor, cell_neighbor) in enumerate(cell_list_neighbor)
                                    # It isn't known yet the position of cell_list0[ic0] in 
                                    # cell_list_neighbor. cell_list0 is an 18 digit cell identifier.
                                    # When this matches cell_neighbor, then record ic_neighbor
                                    # and look up where the edges are stored for cell_neighbor
                                    if cell_neighbor == cell_list0[ic0]
                                        # cells_on_faces[13:18,ic] only has entries for even numbered cube faces
                                        # This is because the even numbered cube faces own the relationship
                                        # The odd numbered cube faces don't even know which ic_neigbor corresponds
                                        # to ic0. To capture this, though, the only info that is needed is to store
                                        # the start value in the odd cube face centric rows (i.e. 13-18) of cells_on_faces
                                        # This is later saved back into the file that contains the Cube_properties structure
                                        cells_on_faces[cube_face_index][11 + face_index, ic_neighbor] = cells_on_faces0[12 + face_index, ic0]

                                        # Defining ranges for face based lookups of boundary edges
                                        start_interior_edge0 = cells_on_faces0[6 + face_index, ic0] + split_boundary_edges0 - 1
                                        start_interior_edge_neighbor = cells_on_faces_neighbor[5 + face_index, ic_neighbor] + split_boundary_edges_neighbor - 1
                                        range_interior_edge0 = start_interior_edge0:start_interior_edge0 + cells_on_faces0[face_index, ic0] - 1
                                        range_interior_edge_neighbor = start_interior_edge_neighbor:start_interior_edge_neighbor + cells_on_faces_neighbor[face_index - 1, ic_neighbor] - 1
                                        # Defining ranges for cube face based storage of calculated quantities
                                        start_delta = cells_on_faces0[12 + face_index, ic0]
                                        # Debug prints
                                        

                                        faces0 = boundary_edges0[3, range_interior_edge0]
                                        faces_neighbor = boundary_edges_neighbor[3, range_interior_edge_neighbor]

                                        for iter in 1:cell_on_face0[face_index]
                                            
                                            t̂0_iter = range_cell_start + iter - 1
                                            delta_iter = start_delta + iter - 1
                                            rCF = face_center_neighbor[1:3, faces_neighbor[iter]] .- face_center0[1:3, faces0[iter]]
                                            n_bend = cross(rCF, t̂0[1:3, t̂0_iter])
                                            n_rCF = cross(t̂0[1:3, t̂0_iter], n_bend)
                                            n̂_rCF = normalize(n_rCF)
                                            dCF = norm(rCF)
                                            e .= normalize(rCF)
                                            S .= n̂_rCF .* area_edge0[t̂0_iter]
                                            E .= dot(S,S)/dot(e,S).*e
                                            Ef = norm(E)
                                            T .= S .- E
                                            δ_edge = dot(rCF,n̂_rCF)
                                            t̂_dot_lf_δ_edge = dot(t̂0[1:3, t̂0_iter],rCF)/δ_edge
                                            
                                            d_face_edge_neighbor = norm(face_center_neighbor[1:3, faces_neighbor[iter]] .- edge_mid0[1:3,t̂0_iter])
                                            cube_face_S[face_index][1:3, delta_iter] .= S
                                            cube_face_E[face_index][1:3, delta_iter] .= E
                                            cube_face_Ef[face_index][delta_iter] = Ef
                                            cube_face_T[face_index][1:3, delta_iter] .= T
                                            cube_face_dCF[face_index][delta_iter] = dCF
                                            cube_face_rCF[face_index][1:3, delta_iter] .= rCF
                                            cube_face_weight_edge[face_index][delta_iter] = d_face_edge_neighbor/(d_face_edge_neighbor+d_face_edge0[t̂0_iter])
                                            cube_face_t̂_dot_lf_δ_edge[face_index][delta_iter] = t̂_dot_lf_δ_edge
                                          
                                            if cell_list0[ic0] == 864691135162494509 #864691136723381757
                                                println("Debug information:")
                                                println("xcounter: ", xcounter)
                                                println("ycounter: ", ycounter)
                                                println("zcounter: ", zcounter)
                                                println("ic0: ", ic0)
                                                println("ic_neighbor: ", ic_neighbor)
                                                println("cell_list0[ic0]: ", cell_list0[ic0])
                                                println("cell_list_neighbor[ic_neighbor]: ", cell_list_neighbor[ic_neighbor])
                                                println("face_index: ", face_index)
                                                println("range_interior_edge0: ", range_interior_edge0)
                                                println("range_interior_edge_neighbor: ", range_interior_edge_neighbor)
                                                println("cells_on_faces0[12 + face_index, ic0]: ", cells_on_faces0[face_index, ic0])
                                                println("cells_on_faces_neighbor[5 + face_index, ic_neighbor]: ", cells_on_faces_neighbor[face_index-1, ic_neighbor])
                                                @show cells_on_faces0[:, ic0],cells_on_faces_neighbor[:, ic_neighbor]

                                                @show Ef,face_center_neighbor[1:3, faces_neighbor[iter]],face_center0[1:3, faces0[iter]]
                                                @show rCF,n̂_rCF,cell_neighbor,cell_list0[ic0]
                                            end
          
                                        end

                                        p_neighbor = sum((nodes_neighbor[:, boundary_edges_neighbor[1, range_interior_edge_neighbor]] .+ nodes_neighbor[:, boundary_edges_neighbor[2, range_interior_edge_neighbor]]) ./ 2 .- 
                                                         (nodes0[:, boundary_edges0[1, range_interior_edge0]] .+ nodes0[:, boundary_edges0[2, range_interior_edge0]]) ./ 2)

                                        if p_neighbor > eps
                                            @show p_neighbor
                                            println("List of boundary_edges in owner and neighbor out of order")
                                            exit()
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
                            
                       

                if cells_on_cube_side0[2] > 0
                    jldsave(string(path,frame_name,current_startframe,"_",endframe,"_",neighbor_vals[1],"_",current_yval,"_",height,".jld2"),true;cubeface_cells=Cube_properties(cells_on_faces[1],cells_on_cube_side[1],edges_on_cube_side[1]))
                end
                
                if cells_on_cube_side0[4] > 0
                    jldsave(string(path,frame_name,current_startframe,"_",endframe,"_",current_xval,"_",neighbor_vals[2],"_",height,".jld2"),true;cubeface_cells=Cube_properties(cells_on_faces[2],cells_on_cube_side[2],edges_on_cube_side[2]))
                end
                
                if cells_on_cube_side0[6] > 0
                    jldsave(string(path,frame_name,neighbor_vals[3],"_",endframes[3],"_",current_xval,"_",current_yval,"_",height,".jld2"),true;cubeface_cells=Cube_properties(cells_on_faces[3],cells_on_cube_side[3],edges_on_cube_side[3]))
                end

                filename_dynamic = string(path,frame_name,current_startframe,"_",endframe,"_",current_xval,"_",current_yval,"_",height,"_ghost_dynamic.jld2")
                filename_static = string(path,frame_name,current_startframe,"_",endframe,"_",current_xval,"_",current_yval,"_",height,"_ghost_static.jld2")
                
                jldsave(filename_dynamic, true; ghost_setup_dynamic=Ghost_cells_dynamic(cube_face_conc,cube_face_conc_ghost))
                jldsave(filename_static, true; ghost_setup_static=Ghost_cells_static(cube_face_S,cube_face_E,cube_face_Ef,cube_face_T,cube_face_dCF,cube_face_rCF,cube_face_weight_edge,cube_face_t̂_dot_lf_δ_edge))
            end
        end
    end
    counter = 0
    for xcounter in location[1,1]:location[1,2]
        for ycounter in location[2,1]:location[2,2]
            for zcounter in location[3,1]:location[3,2]
               
                cube_face_S = Vector{Array{Float32,2}}(undef,6)
                cube_face_E = Vector{Array{Float32,2}}(undef,6)
                cube_face_Ef = Vector{Vector{Float32}}(undef,6)   
                cube_face_T = Vector{Array{Float32,2}}(undef,6)       
                cube_face_dCF = Vector{Vector{Float32}}(undef,6)
                cube_face_rCF = Vector{Array{Float32,2}}(undef,6)  
                cube_face_weight_edge = Vector{Vector{Float32}}(undef,6)
                cube_face_t̂_dot_lf_δ_edge = Vector{Vector{Float32}}(undef,6)  
                cube_face_S_owner = Vector{Array{Float32,2}}(undef,6)
                cube_face_E_owner = Vector{Array{Float32,2}}(undef,6)
                cube_face_Ef_owner = Vector{Vector{Float32}}(undef,6)   
                cube_face_T_owner = Vector{Array{Float32,2}}(undef,6)       
                cube_face_dCF_owner = Vector{Vector{Float32}}(undef,6)
                cube_face_rCF_owner = Vector{Array{Float32,2}}(undef,6)  
                cube_face_weight_edge_owner = Vector{Vector{Float32}}(undef,6)
                cube_face_t̂_dot_lf_δ_edge_owner = Vector{Vector{Float32}}(undef,6) 

                counter += 1
                current_startframe = startframe + zcounter*slices
                current_xval = xval + xcounter*xy_elements
                current_yval = yval + ycounter*xy_elements
 
                endframe = current_startframe+slices
           
                filename_static = string(path,frame_name,current_startframe,"_",endframe,"_",current_xval,"_",current_yval,"_",height,"_ghost_static.jld2")
            
            
                fid_jld_ghost_static = jldopen(filename_static)
                
                
                ghost_setup_static = fid_jld_ghost_static["ghost_setup_static"]
                
                cube_face_S::Vector{Array{Float32,2}} = ghost_setup_static.cube_face_S
                cube_face_E::Vector{Array{Float32,2}} = ghost_setup_static.cube_face_E
                cube_face_Ef::Vector{Vector{Float32}} = ghost_setup_static.cube_face_Ef 
                cube_face_T::Vector{Array{Float32,2}} = ghost_setup_static.cube_face_T       
                cube_face_dCF::Vector{Vector{Float32}} = ghost_setup_static.cube_face_dCF
                cube_face_rCF::Vector{Array{Float32,2}}  = ghost_setup_static.cube_face_rCF  
                cube_face_weight_edge::Vector{Vector{Float32}} = ghost_setup_static.cube_face_weight_edge
                cube_face_t̂_dot_lf_δ_edge::Vector{Vector{Float32}} = ghost_setup_static.cube_face_t̂_dot_lf_δ_edge
           
                close(fid_jld_ghost_static)
                
                for i in 1:3
                    if length(cube_face_S[2*i-1]) > 1
                        neighbor_val = i == 1 ? current_xval - xy_elements :
                                       i == 2 ? current_yval - xy_elements :
                                       current_startframe - slices

                       
                        endframe_neighbor = i == 3 ? neighbor_val + slices : endframe

                        filename_static = string(path, frame_name, 
                                          i == 3 ? neighbor_val : current_startframe, "_",
                                          endframe_neighbor, "_",
                                          i == 1 ? neighbor_val : current_xval, "_",
                                          i == 2 ? neighbor_val : current_yval, "_",
                                          height, "_ghost_static.jld2")

                        fid_jld_ghost = jldopen(filename_static)
                        # display(fid_jld_ghost)
                        ghost_setup = fid_jld_ghost["ghost_setup_static"]

                        cube_face_S_owner = ghost_setup.cube_face_S
                        cube_face_E_owner = ghost_setup.cube_face_E
                        cube_face_Ef_owner = ghost_setup.cube_face_Ef
                        cube_face_T_owner = ghost_setup.cube_face_T      
                        cube_face_dCF_owner = ghost_setup.cube_face_dCF
                        cube_face_rCF_owner = ghost_setup.cube_face_rCF  
                        cube_face_weight_edge_owner = ghost_setup.cube_face_weight_edge
                        cube_face_t̂_dot_lf_δ_edge_owner = ghost_setup.cube_face_t̂_dot_lf_δ_edge
                        close(fid_jld_ghost)

                        

                        cube_face_S[2*i-1] .= cube_face_S_owner[2*i]
                        cube_face_E[2*i-1] .= cube_face_E_owner[2*i]
                        cube_face_Ef[2*i-1] .= cube_face_Ef_owner[2*i]
                        cube_face_T[2*i-1] .= cube_face_T_owner[2*i]
                        cube_face_dCF[2*i-1] .= cube_face_dCF_owner[2*i]
                        cube_face_rCF[2*i-1] .= cube_face_rCF_owner[2*i]
                        cube_face_weight_edge[2*i-1] .= cube_face_weight_edge_owner[2*i]
                        cube_face_t̂_dot_lf_δ_edge[2*i-1] .= cube_face_t̂_dot_lf_δ_edge_owner[2*i]
                    end
                end
                filename = string(path,frame_name,current_startframe,"_",endframe,"_",current_xval,"_",current_yval,"_",height,"_ghost_static.jld2")
                jldsave(filename,true;ghost_setup_static=Ghost_cells_static(cube_face_S,cube_face_E,cube_face_Ef,cube_face_T,cube_face_dCF,cube_face_rCF,cube_face_weight_edge,cube_face_t̂_dot_lf_δ_edge))
            end
        end
    end
  

end




