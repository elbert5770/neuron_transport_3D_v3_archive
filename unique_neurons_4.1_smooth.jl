using CSV
using FileIO
using Tables

function main(sim_number,dir_name,frame_name)
    # Load the simulation settings from a CSV file
    filename = string(@__DIR__, "/Simulation_settings_4.0_smoothed.csv")
    settings = CSV.File(filename) |> Tables.matrix
    @show settings
    # Extract relevant parameters from the settings
    xval::Int64 = settings[sim_number, 2]
    yval::Int64 = settings[sim_number, 3]
    startframe::Int64 = settings[sim_number, 4]
    height::Int64 = settings[sim_number, 5]
    slices::Int64 = height * 25 *5
    location = zeros(Int64, 3, 2)
    location[1, 1:2] .= settings[sim_number, 6:7]
    location[2, 1:2] .= settings[sim_number, 8:9]
    location[3, 1:2] .= settings[sim_number, 10:11]
    xy_elements = height * 125
    number_volumes = (location[1, 2] + location[1, 1] + 1) * (location[2, 2] + location[2, 1] + 1) * (location[3, 2] + location[3, 1] + 1)
    @show number_volumes
    # Initialize a buffer to store the settings for the marching cubes algorithm
    settings_file_buffer = Array{Int64, 2}(undef, number_volumes, 6)

    # Ensure that the location values are within the expected range
    for i in 1:3
        if location[i, 1] > 0
            location[i, 1] = -location[i, 1]
        end
        if location[i, 2] < 0
            println("Error: plus location must be positive")
            return
        end
    end
    
    # Iterate through the volumes and process the data
    counter = 0
    for xcounter in location[1, 1]:location[1, 2]
        for ycounter in location[2, 1]:location[2, 2]
            for zcounter in location[3, 1]:location[3, 2]
                counter += 1
                cell_list = Vector{Int64}()
                @show xcounter, ycounter, zcounter

                # Calculate the current start frame, x-value, and y-value
                current_startframe = startframe + zcounter * slices
                current_xval = xval + xcounter * xy_elements
                current_yval = yval + ycounter * xy_elements
                endframe = current_startframe + slices

                # Store the settings in the buffer
                settings_file_buffer[counter, 1:6] .= counter, slices, current_startframe, current_xval, current_yval, height

                # Iterate through the remaining frames and concatenate the cell data
                for i in startframe:endframe
                    filename = string(dir_name, "szmframe_",i, "_",xval,"_",yval,"_",height,".csv")
                    frame_image = CSV.File(filename, header=true) |> Tables.matrix
                    cell_list2 = unique(frame_image)
                
                    cell_list = vcat(cell_list,cell_list2)
                    
                    cell_list = unique(cell_list)
                end
                # Sort the cell list and remove any zeros 
                cell_list = sort(cell_list,rev=true)
                num_cells = length(cell_list)
                if num_cells > 1 
                    for i in 1:num_cells
                        if cell_list[i] == 0
                            num_cells = i - 1
                            break
                        end
                    end
                    cell_list = cell_list[1:num_cells]
                end
                @show size(cell_list),cell_list
                
                output_file = string(dir_name,frame_name,startframe,"_",endframe,"_",xval,"_",yval,"_",height,".csv")
                CSV.write(output_file, Tables.table(cell_list), header=true)
            end
        end
    end

    # Write the settings file for the marching cubes algorithm
    CSV.write(string(dir_name, "MarchingCubes_settings_smooth.csv"), Tables.table(settings_file_buffer), writeheader=true, header=["SimNumber", "Slices", "Startframe", "Xval", "Yval", "Height"])
end



sim_number = 4
dir_name = string(@__DIR__, "/../Neuron_transport_data/")
frame_name = "frame_uniq_"

main(sim_number,dir_name,frame_name)
