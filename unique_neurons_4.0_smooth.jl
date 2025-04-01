using CSV
# using DataFrames
# using DataFrames
# using MarchingCubes
# using GLMakie
#using MeshViz
# using Meshes
# using SparseArrays
# using ColorSchemes
# import ColorSchemes.rainbow
using FileIO
# using StaticArrays
using Tables

function main(sim_number,dir_name,frame_name)

    filename = string(dir_name,"MarchingCubes_settings_smooth.csv")
    settings = CSV.File(filename) |> Tables.matrix
    # @show settings
    slices = settings[sim_number,2]
    startframe = settings[sim_number,3]
    xval = settings[sim_number,4]
    yval = settings[sim_number,5]
    height = settings[sim_number,6]
    @show sim_number,xval,yval,startframe,height
    filename = string(dir_name, "szmframe_",startframe, "_",xval,"_",yval,"_",height,".csv")
    frame_image = CSV.File(filename, header=true) |> Tables.matrix
    @show size(frame_image)
    cell_list = unique(frame_image)
    for i in startframe:startframe+slices
        filename = string(dir_name, "szmframe_",i, "_",xval,"_",yval,"_",height,".csv")
        frame_image = CSV.File(filename, header=true) |> Tables.matrix
        cell_list2 = unique(frame_image)
        # for cell in cell_list2
        #     if cell == 864691136698463485
        #         @show i
        #     end
        # end
        cell_list = vcat(cell_list,cell_list2)
        # @show size(cell_list)
        # @show cell_list
        cell_list = unique(cell_list)
        # @show cell_list
        # @show size(cell_list)
    end
    cell_list = sort(cell_list,rev=true)
    num_cells = length(cell_list)
    if num_cells > 1 
        for i in 1:num_cells
            if cell_list[i] == 0
                num_cells = i -1
                break
            end
        end
        cell_list = cell_list[1:num_cells]
    end
    @show size(cell_list),cell_list
    endframe = startframe+slices
    output_file = string(dir_name,frame_name,startframe,"_",endframe,"_",xval,"_",yval,"_",height,".csv")
    CSV.write(output_file, Tables.table(cell_list), header=true)
end


for i in 1:8
    sim_number = i
    dir_name = string(@__DIR__, "/../Neuron_transport_data/")
    frame_name = "frame_uniq_"

    main(sim_number,dir_name,frame_name)
end