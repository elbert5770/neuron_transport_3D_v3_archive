# julia --threads=auto --depwarn=yes --project ./Neuron_transport_programs/plot_effective_diffusion_optimize_3D_mt_tensor_pos.jl

using ReadVTK
# using GLMakie
# using WriteVTK
# using ColorSchemes
# import ColorSchemes.rainbow
using Optim

# Function to compute the analytical solution
function anisotropic_diffusion(x, y, z, t, L, Dxx, Dxy, Dxz, Dyy, Dyz, Dzz, nterms=20)
    a = b = c = L/2
    ϕ = 0.0
    
    for l in 0:nterms
        for m in 0:nterms
            for n in 0:nterms
                # Coefficient for each term
                Almn = (-1)^(l+m+n) / ((2*l+1) * (2*m+1) * (2*n+1))
                
                # Spatial components remain the same
                spatial = cos((2l+1)*π*x/(2*a)) * cos((2m+1)*π*y/(2*b)) * cos((2n+1)*π*z/(2*c))
                
                # Time decay now uses different D values for each direction
                decay = exp(-π^2/4 * (
                    Dxx*(2l+1)^2/(a^2) + 
                    Dxy*(2l+1)*(2m+1)/(a*b) + 
                    Dxz*(2l+1)*(2n+1)/(a*c) + 
                    Dxy*(2m+1)*(2l+1)/(a*b) + 
                    Dyy*(2m+1)^2/(b^2) + 
                    Dyz*(2m+1)*(2n+1)/(b*c) + 
                    Dxz*(2n+1)*(2l+1)/(a*c) + 
                    Dyz*(2n+1)*(2m+1)/(b*c) + 
                    Dzz*(2n+1)^2/(c^2)
                ) * t)
                
                ϕ += Almn * spatial * decay
            end
        end
    end
    
    A = 64 / (π^3)
    return ϕ * A
end

function main(D,final_plot=false)
    path = string(@__DIR__, "/../Neuron_transport_data/")
    @show D
    ssq_resid = Threads.Atomic{Float64}(0.0)
    # Parameters
    L = 1000.0  # cube side length
    # D = 0.5 # diffusion coefficient
    center_x = 170398+250
    center_y = 61241-250
    center_z = 87455

    time = 2501
    real_time = Float64(time*10)
    # locations = [1,0,0]
    # if locations[1] == 1
    #     xloc = 170398
    # else
    #     xloc = 170898
    # end
    # if locations[2] == 1
    #     yloc = 60741
    # else
    #     yloc = 61241
    # end
    # if locations[3] == 1
    #     zloc = 86955
    # else
    #     zloc = 87455
    # end
    # cell = 1
    # if cell == 1
    #     cell = 864691136812081779
    # elseif cell == 2
    #     cell = 864691136099789941
    # elseif cell == 3
    #     cell = 864691136222809022
    # end

    cell_list = [864691136812081779,864691136099789941,864691136222809022,864691136334080435,864691135570814445,864691135539037938,864691136024224825,864691135544672808,864691135865680005,864691136792370798,864691136697855485,864691136194020822,864691135988581578,864691135737609108,864691135639263035,864691135494427536,864691135382491610,864691135256146095]
    # cells = [864691136697855485,864691136194020822,864691135988581578,864691135737609108,864691135639263035,864691135494427536,864691135382491610,864691135256146095]
    # cells = [864691135737609108]
    # cells = [864691136697855485]
    # cell_list = [864691135666842850]
    # cells = [864691136240768958]
    # fig = Figure(size = (1600, 1600),
    #     fontsize = 20)
    # ax = Axis(fig[1, 1], 
    #     xlabel = "Modeled Concentration", 
    #     ylabel = "Analytical Concentration",
    #     limits = (0, 1, 0, 1),
    #     backgroundcolor = :black)

    
    # Define a vector of colors for the different cells
    plot_colors = [:red, :yellow, :cyan, :lime, :magenta, :orange, :purple, :green, :pink, :brown, :gray, :teal, :gold, :violet, :white, :indigo, :turquoise, :maroon]
        # @show size(plot_colors)
    counter = Threads.Atomic{Int}(0)
    Threads.@threads for (i,cell) in collect(enumerate(cell_list))
        @show cell,plot_colors[mod1(i,18)]
    # for (i,cell) in enumerate(cells)
        # @show cell,plot_colors[mod1(i,18)]
        for xloc in [170398,170898]
            for yloc in [60741,61241]
                for zloc in [86955,87455]
                    local_counter = Threads.atomic_add!(counter, 1)
                    # @show cell,plot_colors[mod1(counter,18)],xloc,yloc,zloc
                    # if xloc != 170898 && yloc != 60741 && zloc != 87455
                    #     continue
                    # end
                    file = string(cell,"_",zloc,"_",xloc,"_",yloc,"_4_",time)

                    # filename = "D:\\Neuron_transport_data\\HYAK8001final\\Final3\\Final5\\Final10\\864691136222809022_87455_170398_61241_4_5001.vtu"
                    # filename = "D:\\Neuron_transport_data\\final_eight\\$file.vtu"
                    filename = string(path,"$file.vtu")
                    if !isfile(filename)
                        continue
                    end
                    vtk = VTKFile(filename)



                    cell_data = get_cell_data(vtk)
                    element_ids = cell_data["Concentration"]
                    # data_vtk = get_point_data(vtk)
                    # element_ids = data_vtk["Concentration"]
                    data1 = get_data(element_ids)
                    # size(data_vtk)

                    points1 = get_points(vtk)
                    # @show length_array = size(points1)[2]
                    # @show points1[1:3,1:10]
                    # @show size(data1)
                    # @show data1[1:100]
                    # @show size(cell_data)
                    # @show size(element_ids)
                    # @show element_ids[1:10]

                    cells = get_cells(vtk)
                        # @show length(cells)
                        # @show cells.connectivity[1:3]
                    connect = reshape(cells.connectivity,(3,length(cells)))
                        # @show size(connect),typeof(connect)
                        # @show connect[:,1:10]
                        # @show size(cells),typeof(cells)
                        
                    # Calculate center coordinates for each triangle
                    centers = zeros(3, size(connect,2))
                    for i in 1:size(connect,2)
                        # Get the three vertex indices for this triangle
                        vertex_indices = connect[:,i]
                        
                        # Calculate center as average of three vertices
                        for j in 1:3
                            centers[j,i] = sum(points1[j,vertex_indices]) / 3
                        end
                    end

                    # @show size(centers)
                    # @show centers[:,1:5] .- [center_x, center_y, center_z]  # Show first 5 triangle centers
                    adj_centers = zeros(3,size(centers,2))
                    adj_centers .= centers .- [center_x, center_y, center_z]
                    data2 = zeros(size(data1))
                    data3 = zeros(size(data1))
                    threshold = 20.0
                    nterms = 10
                    for (i,data) in enumerate(data1)
                        
                        pred_conc = anisotropic_diffusion(adj_centers[1,i], adj_centers[2,i], adj_centers[3,i], real_time, L, D[1], D[2], D[3], D[4], D[5], D[6],nterms)
                        # pred_conc = diffusion_solution(adj_centers[1,i], adj_centers[2,i], adj_centers[3,i], real_time, L, D, nterms)
                        data2[i] = pred_conc #(data-pred_conc)#/pred_conc
                        data3[i] = data-pred_conc
                        # if adj_centers[1,i] > -threshold && adj_centers[1,i] < threshold && adj_centers[2,i] > -threshold && adj_centers[2,i] < threshold && adj_centers[3,i] > -threshold && adj_centers[3,i] < threshold
                        #     @show data,pred_conc


                        # end
                    end

                    # @show minimum(adj_centers, dims=2)  # Show min values for each dimension
                    # @show maximum(adj_centers, dims=2)  # Show max values for each dimension

                    # Create a new VTK file
                    # if final_plot
                    #     vtk_filename = string("D:\\Neuron_transport_data\\final_eight\\",file,"_excess.vtu")
                    #     # Convert points to the expected format (3×N array of points)
                    #     points_array = Array(points1)  # Convert to regular Array if it's not already

                    #     # Convert cells to MeshCell format
                    #     mesh_cells = [MeshCell(VTKCellTypes.VTK_TRIANGLE, connect[:,i]) for i in 1:size(connect,2)]

                    #     vtkfile = vtk_grid(vtk_filename, points_array, mesh_cells)

                    #     # Add the original concentration data
                    #     vtk_cell_data(vtkfile, data1, "Concentration")

                    #     # Add the new excess concentration data
                    #     vtk_cell_data(vtkfile, data3, "Excess Conc")

                    #     # Save the file
                    #     vtk_save(vtkfile)
                    # end

                    Threads.atomic_add!(ssq_resid, sum((data3).^2))
                    # diffusion_solution(0, 0, 0, real_time, L, D, nterms)

                    # Create a scatter plot comparing data1 vs data2
                    

                    # Get the data range for the line of unity
                    # x_range = range(minimum(sqrt.(-log10.(data1))), maximum(sqrt.(-log10.(data1))), length=100)
                    # x_range = range(minimum(data1), maximum(data2), length=100)
                    # Add the line of unity
                
                    #plot_colors[mod1(i, length(plot_colors))]  
                    # if xloc == 170398 && yloc == 61241 && zloc == 87455
                    # if final_plot
                    #     scatter!(ax, data1, data2, 
                    #         markersize = 2,
                    #         color = plot_colors[mod1(counter+1,18)],  # Use mod1 to cycle through colors
                    #         alpha = 0.25,label = string(cell) => (; markersize = 15))
                    #         # scatter!(ax, sqrt.(-log10.(data1)), sqrt.(-log10.(data2)), 
                    #         # markersize = 2,
                    #         # color = :blue,
                    #         # alpha = 0.5)
                    # end
                    # else
                    #     scatter!(ax, data1, data2, 
                    #         markersize = 2,
                    #         color = plot_colors[mod1(i,15)],  # Use mod1 to cycle through colors
                    #         alpha = 0.25)
                    # end
                end
            end
        end
    end
    # if final_plot
    #     lines!(ax, [0,1], [0,1], color = :yellow, linestyle = :dash)
    #     axislegend(ax, merge = true, unique = true, position = :lt,backgroundcolor = :black,labelcolor = :yellow)

    #     # Display the figure
    #     display(fig)
    # end
    return ssq_resid[]
    # Optional: save the figure
    # save("concentration_comparison_850_$(time).png", fig)

end

# Add optimization code at the bottom instead of direct main() call
# Define the objective function
function objective(D)
    final_plot = false
    D = exp.(D)
    return main(D,final_plot)  # Optim requires vector input, so we take first element
end

# Initial guess for D
initial_D = [0.5 0.001 0.001 0.5 0.001 0.5]
initial_D = [1.0 1.0 1.0 1.0 1.0 1.0]
initial_D = [0.47072337597582675 0.08860122957284434 0.00438270120279417 0.44634128567648185 0.04520994394537368 0.4238346684699281]
initial_D = [0.5066926564342022 0.005040738455530669 0.012233065656919869 0.5422145853318132 0.0041280964154034525 0.4727260531743884]
initial_D = log.(initial_D)
println("Number of threads: ", Threads.nthreads())
# Set up optimization
result = optimize(objective, initial_D, NelderMead())

# Get the optimal D value
optimal_D = Optim.minimizer(result)[1]
minimum_ssq = Optim.minimum(result)

# Print results
println("Optimal diffusion coefficient D: ", exp.(optimal_D))
println("Minimum sum of squared residuals: ", minimum_ssq)
# final_plot = true
# # Optional: Run main one final time with optimal D to generate plot
# main(optimal_D,final_plot)
