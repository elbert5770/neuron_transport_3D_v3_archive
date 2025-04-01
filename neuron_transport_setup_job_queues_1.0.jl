function job_queue(cell_list,path,extension,Sim_parameters,comm,rank,world_size,xcounter,ycounter,zcounter,counter,location,current_startframe,endframe,xval,yval,height,win_accumulate_cells_on_cube_side,win_cube_face_match,win_accumulate_owner_ptr,win_num_faces)

    nworkers = world_size - 1

    root = 0

    MPI.Barrier(comm)
    T = eltype(cell_list)
    N = length(cell_list)
    send_mesg = Array{T}(undef, 1)
    recv_mesg = Array{T}(undef, 1)

    if rank == root # I am root

        idx_recv = 0
        idx_sent = 1

        new_data = Array{T}(undef, N)*0
        # Array of workers requests
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
            idx_sent += 1
            # if mod(idx_sent,10) == 0
            #     println("Progress $idx_sent / $N")
            # end
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
                            idx_sent += 1
                            # if mod(idx_sent,10) == 0
                            #     println("Progress $idx_sent / $N")
                            # end
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

        MPI.Waitall(sreqs_workers)
        # print("Root: New data = $new_data\n")
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

                    uniquenode(path,recv_mesg[1],extension,Sim_parameters,comm,rank,world_size,cell_list,xcounter,ycounter,zcounter,counter,location,current_startframe,endframe,xval,yval,height,win_accumulate_cells_on_cube_side,win_cube_face_match,win_accumulate_owner_ptr,win_num_faces)

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
    return

end

function job_queue_matrix(cell_list,path,extension,comm,rank,world_size,xcounter,ycounter,zcounter,counter,location,current_startframe,endframe,current_xval,current_yval,height,edges_on_cube_side,win_cells_on_faces,win_cube_face_area_delta)

    nworkers = world_size - 1

    root = 0

    MPI.Barrier(comm)
    T = eltype(cell_list)
    N = length(cell_list)
    send_mesg = Array{T}(undef, 1)
    recv_mesg = Array{T}(undef, 1)

    if rank == root # I am root

        idx_recv = 0
        idx_sent = 1

        new_data = Array{T}(undef, N)*0
        # Array of workers requests
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
            idx_sent += 1
            # if mod(idx_sent,10) == 0
            #     println("Progress $idx_sent / $N")
            # end
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
                            idx_sent += 1
                            # if mod(idx_sent,10) == 0
                            #     println("Progress $idx_sent / $N")
                            # end
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

        MPI.Waitall(sreqs_workers)
        # print("Root: New data = $new_data\n")
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

                    matrix_setup(cell_list,recv_mesg[1],path,extension,comm,rank,root,world_size,xcounter,ycounter,zcounter,counter,location,current_startframe,endframe,current_xval,current_yval,height,edges_on_cube_side,win_cells_on_faces,win_cube_face_area_delta)

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
    return

end

function job_queue_match(cell_list,path,extension,comm,rank,world_size,win_owner,owner_location)

    nworkers = world_size - 1

    root = 0

    MPI.Barrier(comm)
    T = eltype(cell_list)
    N = length(cell_list)
    send_mesg = Array{T}(undef, 2)
    recv_mesg = Array{T}(undef, 2)

    if rank == root # I am root

        idx_recv = 0
        idx_sent = 1

        new_data = Array{T}(undef, N)*0
        # Array of workers requests
        sreqs_workers = Array{MPI.Request}(undef,nworkers)
        # -1 = start, 0 = channel not available, 1 = channel available
        status_workers = ones(nworkers).*-1

        # Send message to workers
        for dst in 1:nworkers
            if idx_sent > N
                break
            end
   
            send_mesg .= @views idx_sent,owner_location[idx_sent]
            sreq = MPI.Isend(send_mesg, comm; dest=dst, tag=dst+32)
            idx_sent += 1
            # if mod(idx_sent,10) == 0
            #     println("Progress $idx_sent / $N")
            # end
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
                            
                            send_mesg .= @views idx_sent,owner_location[idx_sent]
                            # Sends new message
                            sreq = MPI.Isend(send_mesg, comm; dest=dst, tag=dst+32)
                            idx_sent += 1
                            # if mod(idx_sent,10) == 0
                            #     println("Progress $idx_sent / $N")
                            # end
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

        MPI.Waitall(sreqs_workers)
        # print("Root: New data = $new_data\n")
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

                    match_owners(cell_list,recv_mesg[1],N,path,extension,win_owner,recv_mesg[2])

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
    return

end