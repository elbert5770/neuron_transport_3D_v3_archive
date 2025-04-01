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
# THIS SOFTWARE IS PROVIDED BY THE UNIVERSITY OF WASHINGTON AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED 
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY OF WASHINGTON OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT 
# OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON 
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING 
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

function setup_match_faces(cell_list_pos,cell_list,path,extension,filetype,win_accumulate_owner_ptr)
    
    eps = 1e-6
    
    num_cells = length(cell_list)
    cell1 = cell_list[cell_list_pos]
    filename1 = string(path,cell1,extension,filetype)
    fid_jld1 = jldopen(filename1)

    # display(fid_jld)
    structure_setup1 = fid_jld1["structure_setup"] 
    
    num_faces1::Int32 = structure_setup1.num_faces
    
    face_center1 = Array{Float32,2}(undef,3,num_faces1)
    face_center1::Array{Float32,2} = structure_setup1.face_center
    close(fid_jld1)
    
    rface_center1 = Array{Float32,2}(undef,3,num_faces1)
    p1 = Vector{Int32}(undef,num_faces1)
    rface_center1 = reverse(face_center1,dims=1)
    p1 = sortperm(eachslice(rface_center1; dims=2))
    
    # Initialize variables for face matching
    face_match = zeros(Int32,3,num_faces1)
    face_match_count = zeros(Int32,num_cells)
    num_matches = 0
    total_face_matches = 0
    total_cell_matches = 0
    for j = cell_list_pos+1:num_cells
        cell2 = cell_list[j]
        filename2 = string(path,cell_list[j],extension,filetype)   
        fid_jld2 = jldopen(filename2)
        structure_setup2 = fid_jld2["structure_setup"] 
        num_faces2::Int32 = structure_setup2.num_faces
        face_center2 = Array{Float32,2}(undef,3,num_faces2)
        face_center2::Array{Float32,2} = structure_setup2.face_center
        close(fid_jld2)
        
        rface_center2 = Array{Float32,2}(undef,3,num_faces2)
        p2 = Vector{Int32}(undef,num_faces2)
        rface_center2 = reverse(face_center2,dims=1)
        p2 = sortperm(eachslice(rface_center2; dims=2))
        num_matches = match(face_center1[:,p1],face_center2[:,p2],num_faces1,num_faces2,face_match,p1,p2,cell2,eps,j)
        face_match_count[j] = num_matches
        total_face_matches += num_matches
        if num_matches > 0
            total_cell_matches += 1
        end
    end
    
    # face_match_sort = [number of neighbor in cell_list,cell number of neighbor,face # of neighbor,face # of owner]
    if total_face_matches > 0
        # Sort face_match by first row, which contains the cell_list_number.
        # No match is indicated by a cell_list_number of 0.  These appear at the front of the array.
        # The cell_list_number is the index of the cell in the cell_list.
        face_match_sort = zeros(Int32,2,total_face_matches)
        
        pm = sortperm(eachslice(face_match; dims=2))
        @views face_match_sort .= face_match[2:3,pm[num_faces1-total_face_matches+1:num_faces1]]      
    else
        face_match_sort = zeros(Int32,1,1)
    end
    # @show face_match_count,cell_list_pos,cell1
    
    owner_temp = zeros(Int32,4,total_cell_matches)
    
    counter1 = 0

    for (i,face_start) in enumerate(face_match_count)
        
        if face_start > 0
            counter1 += 1
            start1 = sum(face_match_count[1:i-1]) + 1
            stop1 = start1 - 1 + face_match_count[i]
            owner_temp[1:4,counter1] .= cell_list_pos,i,start1,stop1
            
        end
        
       
    end
    
    number_matched_cells::Int32 = counter1
    fid_jld3 = jldsave(string(path,cell1,extension,"_match",".jld2"),true;match_setup=Match_face(face_match_sort,total_face_matches,owner_temp,number_matched_cells))
    
    owner_ptr = Vector{Int32}(undef,3)
    owner_ptr .= number_matched_cells,Int32(0),Int32(total_face_matches)
    if number_matched_cells > 0
        MPI.Win_lock(MPI.LOCK_EXCLUSIVE, 0, 0, win_accumulate_owner_ptr)
        MPI.Put!(owner_ptr, win_accumulate_owner_ptr;rank=0,disp=3*(cell_list_pos-1)) 
        MPI.Win_unlock(0, win_accumulate_owner_ptr)
    end
    cell_number = cell_list[cell_list_pos]

    return nothing

end

function match_owners(cell_list,cell_list_pos,num_cells,path,extension,win_owner,owner_location)
    cell1 = cell_list[cell_list_pos]
    
    if owner_location == 0
        return
    end

    filetype = "_match.jld2"
    
    
    owner = zeros(Int32,4,owner_location)
    
    filename1 = string(path,cell1,extension,filetype)   
   
    fid_jld1 = jldopen(filename1;parallel_read=true)
    # display(fid_jld1)
    match_setup1 = fid_jld1["match_setup"] 
    num_matched_cells::Int32 = match_setup1.number_matched_cells
    owner_temp = zeros(Int32,4,num_matched_cells)
    owner_temp::Matrix{Int32} = match_setup1.owner_temp
    close(fid_jld1)
    
    owner_temp_vec = vec(owner_temp) 
    
    
    MPI.Win_lock(MPI.LOCK_EXCLUSIVE, 0, 0, win_owner)
    MPI.Put!(owner_temp_vec, win_owner;rank=0,disp=4*(owner_location-1)) 
    MPI.Win_unlock(0, win_owner)

    return
end

function match_neighbors(owner,owner_ptr,num_matches,num_cells,path,extension,cell_list,num_faces_array)

    neighbor = zeros(Int32,6,num_matches)
    p1 = sortperm(eachslice(owner; dims=2), by=x->x[2])
    
    neighbor[1:4,1:num_matches] .= owner[1:4,p1]
 

    neighbor_ptr_temp = Vector{UnitRange}(undef,num_cells)
    
    neighbor_ptr = Array{Int32,2}(undef,3,num_cells)
    
    for i in 1:num_cells    
        neighbor_ptr_temp[i] = searchsorted(neighbor[2,:],i)
        neighbor_ptr[1:2,i] .= length(neighbor_ptr_temp[i]),neighbor_ptr_temp[i].start
    end
    
    num_neighbor_matches = zeros(Int32,num_cells)
    for i in 2:num_cells
        start_neighbor = neighbor_ptr[2,i]
        stop_neighbor = start_neighbor + neighbor_ptr[1,i] - 1
        
        for j in start_neighbor:stop_neighbor        
            start_position = neighbor[3,j]
            stop_position = neighbor[4,j]
            neighbor[5,j] = num_neighbor_matches[i] + 1
            num_neighbor_matches[i] += stop_position-start_position+1            
            neighbor[6,j] = num_neighbor_matches[i]
        end
    end

    for i in 1:num_cells
        neighbor_ptr[3,i] = num_neighbor_matches[i]
    end
    owner_neighbor_ptr = vcat(owner_ptr,neighbor_ptr)
    
    
    fid_jld3 = jldsave(string(path,"simulation",extension,".jld2"),true;match_setup=Owner_neighbor(num_cells,num_matches,cell_list,owner,neighbor,p1,owner_neighbor_ptr,num_faces_array))
    
    
end
 
    






function match(s1,s2,m,n,face_match1,p1,p2,cell2,eps,cell_list_number)
    # # Sort arrays by values in col3, col2, then col1
    # that correspond to z, y, and x coordinates, respectively
    # there are fewer unique points in the z direction so this 
    # direction is sorted first



    i = 1
    j = 1

    count = 0
    counter = 0
    stemp = Vector{Float32}(undef,3)
    stemp_abs = Vector{Int32}(undef,3)
    
    while i <= m && j <= n
        
        if abs(s1[3,i] - s2[3,j])  < eps && abs(s1[2,i] - s2[2,j])  < eps && abs(s1[1,i] - s2[1,j])  < eps
            # One of the two matches will be the owner for distributed processing later
     
            counter += 1
           
            face_match1[1:3,p1[i]] .= cell_list_number,p1[i],p2[j]
     

            i += 1
            j += 1
        else
            # Determine if i or j should be incremented
            # Increment i if s1[i,:] < s2[j,:], otherwise increment j
            # 'less than' is judged from the first non-zero element in the difference vector
            # Thus, if the faces are at the same z, then the y is compared, etc.
            # The order of coordinates is reversed due to the lower resolution in the z direction
            @views stemp .= (s1[1:3,i] .- s2[1:3,j])
            stemp_abs .= abs.(stemp) .> eps # 0 if false, so it is within eps of zero
            count = stemp_abs[3] > 0 ? 3 : (stemp_abs[2] > 0 ? 2 : 1)            
            stemp[count] < 0 ? i += 1 : j += 1

            # The four lines above replace the following 20 lines
            # if s1[3,i] < s2[3,j]
            #     i += 1
            # else
            #     if s1[3,i] > s2[3,j]
            #         j += 1
            #     else
            #         if s1[2,i] < s2[2,j]
            #             i += 1
            #         else
            #             if s1[2,i] > s2[2,j]
            #                 j += 1
            #             else
            #                 if s1[1,i] < s2[1,j]
            #                     i += 1
            #                 else
            #                     if s1[1,i] > s2[1,j]
            #                         j += 1
            #                     end
            #                 end
            #             end
            #         end

            #     end
            # end
        end
    
    end
    return counter
end



    

