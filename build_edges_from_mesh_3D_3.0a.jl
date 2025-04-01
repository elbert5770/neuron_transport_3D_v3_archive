# Written by: Donald L. Elbert, University of Washington
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
# THIS SOFTWARE IS PROVIDED BY THE UNIVERSITY OF WASHINGTON AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED 
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY OF WASHINGTON OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT 
# OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON 
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING 
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

function edge_comparison(edge_to_face_sort,edges,faces,node_list,npf,count1,count2,skip)
    # edge_to_face_sort has one column per edge, but interior faces share faces
    # and edges are duplicated in this case
    # The sorted array has duplicates that are adjacent, however, boundary faces have
    # one or two edges not shared by any other face, thus they are not duplicated.
    # Looking for duplicates, with the possibility that the first column is not
    # duplicated and the last column is not duplicated
    # If a column is not duplicated, then it is a boundary edge and this must be recorded
    for ftc in eachcol(edge_to_face_sort)
        if skip == 1
            node_list[1:5,1] .= ftc
            skip = 0
            
            continue
        end
       
        node_list[1:5,2] .= ftc
        if @views node_list[1:2,1] == node_list[1:2,2]
            count1 += 1
            
            # edges = [node 1 of both faces, node 2 of both faces, face 1,face 2, zone,boundary edge?,edge# of edge in face 1,edge# of edge in face 2,boundary edge #,node order face 1 reversed?,node order face 2 reversed?]
            # node_list = [node 1 of an edge in face 1, node 2 of this edge, number of face 1, edge# of edge in face 1 (1 to npf),was order of nodes reversed?
            #              node 1 of another edge in face 2, node 2 of this edge, number of face 2, edge# of edge in face 1 (1 to npf),was order of nodes reversed?]
            # order of nodes reveresed? 0 = false, 1 = true; order needs to be reversed for sorting and matching
            # faces = [node1,node2,node3,edge1,edge2,edge3,adjacent face1,adjacent face2,adjacent face3,lf_direction_adj_face1,lf_direction_adj_face2,lf_direction_adj_face3]
            @views edges[1:11,count1] .= node_list[1,1],node_list[2,1],node_list[3,1],node_list[3,2],1,0,node_list[4,1],node_list[4,2],0,node_list[5,1],node_list[5,2]
            faces[npf+node_list[4,1],node_list[3,1]] = count1
            faces[npf+node_list[4,2],node_list[3,2]] = count1
            faces[npf*2+node_list[4,1],node_list[3,1]] = node_list[3,2]
            faces[npf*2+node_list[4,2],node_list[3,2]] = node_list[3,1]

            skip = 1
        else
            # Boundary edge
            count1 += 1
            count2 += 1
 
            @views edges[1:11,count1] .= node_list[1,1],node_list[2,1],node_list[3,1],0,-1,1,node_list[4,1],0,count2,node_list[5,1],0
            
            faces[npf+node_list[4,1],node_list[3,1]] = count1
            faces[npf*2+node_list[4,1],node_list[3,1]] = 0
   
            @views node_list[1:5,1] = node_list[1:5,2]
            skip = 0
        end
    end
    if skip == 0
        # The final column, if not yet processed, must be a boundary edge
        count1 += 1
        count2 += 1
        @views edges[1:11,count1] .= node_list[1,1],node_list[2,1],node_list[3,1],0,-1,1,node_list[4,1],0,count2,node_list[5,1],0
        
        faces[npf+node_list[4,1],node_list[3,1]] = count1
        faces[npf*2+node_list[4,1],node_list[3,1]] = 0
        faces[npf*3+node_list[4,1],node_list[3,1]] = 1
    end
    return count1,count2
end

# Functions to extract nodes and face-node connectivity from .msh (Fluent) or .ply files
function mshheader(filename)
    num_nodes = 0
    num_faces = 0
    num_zones = 0
    num_edges = 0
    num_dim = 0
    count = 0
    for i in eachline(filename)

        if startswith(i,"(2 ")
            num_dim = parse(Int32,chop(i, head = 3,tail = 1))
            count += 1
            continue
       end

        if startswith(i,"(10 (0 1 ")
             num_nodes = parse(Int,chop(i, head = 9,tail = 3),base=16)
             count += 1
             continue
        end

        if startswith(i,"(13 (0 1 ")
            num_faces = parse(Int,chop(i, head = 9,tail = 4),base=16)
            count += 1
            continue
       end
        

         
        if startswith(i,"(10 (")
            break
        end
        count += 1
    end
 
    return num_dim,num_nodes,num_faces,num_zones,num_edges,count

end

function plyheader(filename)
    num_nodes = 0
    num_faces = 0
    num_zones = 0
    num_edges = 0
    num_dim = 0
    count = 0
    binary = 0
    filename = string(filename,".ply")
    for i in eachline(filename)
        
        if startswith(i,"format ")
            format = chop(i, head = 7,tail = 0)
            if startswith(format,"binary")
                binary = 1
            end
        end
        if startswith(i,"element vertex ")
            num_nodes = parse(Int32,chop(i, head = 15,tail = 0))
        end
        if startswith(i,"element face ")
             num_faces = parse(Int32,chop(i, head = 13,tail = 0))
        end
        count += 1
        if startswith(i,"end_header")
            break
        end
    end
    if binary == 1
        count2 = 0
        position = 0
        open(filename, "r") do stream
            while true
                read_char = readvalue_c(stream, position)
                
                if Char(read_char) == '\n'
                    
                    count2 += 1
                   
                    if count2 == count
                        count = position+1
                        break
                    end
                end
                position += 1
            end
        end
    else
        println("Error: File format not supported. File must be binary PLY.")
        return 0,0,0,0,0,0,0
        
        
    end
    

    return num_dim,num_nodes,num_faces,num_zones,num_edges,count,binary

end

function mshparser(filename,nodes,faces,faceedges,count,num_dim)
    # Read nodes (vertex) and faces (faces)
    
    count2 = 0
    count3 = 1
    count4 = 1
    add_zone = false
    zone_props = Vector{Int32}(undef,5)
    zone_num = 0
    zone_start = 0
    zone_end = 0
    add_node = false
    for i in eachline(filename)
        count2 += 1
        if count2 <= count
            continue
        end
        # Nodes are stored first in the vertex section of file
        if startswith(i,"(10 (")
            zone_props = parse.(Int,split(chop(i, head = 5,tail = 2)),base=16)
            zone_num = zone_props[1]                
            zone_start = zone_props[2]
            zone_end = zone_props[3]
            
            add_node = true
            
            continue
            
        end
        
        if startswith(i,"(13 (")
            zone_props = parse.(Int,split(chop(i, head = 5,tail = 2)),base=16)
            zone_num = zone_props[1]                
            zone_start = zone_props[2]
            zone_end = zone_props[3]
           
            
            add_zone = true
            
            continue
            
        end

           
        if add_node == true
            if count4 >= zone_start && count4 <= zone_end
                nodes[1:num_dim,count4] = parse.(Float32,split(i))

                count4 += 1         
            else
                
                add_node = false
            end
        end
            
        if add_zone == true 
            if count3 >= zone_start && count3 <= zone_end
                facetemp = parse.(Int,split(i),base=16)
                
                faceedges = facetemp[1:4]
                
                faces[1:4,count3] = faceedges
                
                count3 += 1
            
            else
                add_zone = false
            end
        end

    end
   
end

function plyparser(filename,nodes,normals,faces,faceedges,count,binary,num_nodes,num_faces)
    ply = load_ply(string(filename,".ply"))
    ply["vertex"]
    
    nodes[1,1:num_nodes] .= ply["vertex"]["x"]
    nodes[2,1:num_nodes] .= ply["vertex"]["y"]
    nodes[3,1:num_nodes] .= ply["vertex"]["z"]
    normals[1,1:num_nodes] .= ply["vertex"]["nx"]
    normals[2,1:num_nodes] .= ply["vertex"]["ny"]
    normals[3,1:num_nodes] .= ply["vertex"]["nz"]
   
    for (i,face) in enumerate(eachrow(ply["face"]["vertex_indices"]))
        faces[1:3,i] = Int32.(face[1][1:3].+1)
    end
      
end

# Helper functions to read characters, floats or ints from files
# only readvalue_c is currently used, others may be useful if load_ply
# in the PlyIO package ever breaks
function readvalue_f(stream, position)
    seek(stream, position)
    return read(stream, Float32)
end

function readvalue_i(stream, position)
    seek(stream, position)
    return read(stream, Int32)
end
function readvalue_c(stream, position)
    seek(stream, position)
    return read(stream, UInt8)
end

