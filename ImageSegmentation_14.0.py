
# local user: use python environment "microns", conda init bash, conda activate /gscratch/elbert/python/microns,, python ./ImageSegmentation_14.0.py from
# /gscratch/elbert/julia/Neuron_transport_programs
# Note: cloudvolume seems to require python 3.9 or higher

# user must supply 'simulation_number'
# The simulation number is the number of the simulation to run from Simulation_settings_4.0.csv
# Simulation_settings_4.0.csv must be in the same directory as this script
# The format of Simulation_settings_4.0.csv is as follows:
# Simulation,xloc,yloc,zloc,minus_x,plus_x,minus_y,plus_y,minus_z,plus_z,height
# 1,170398,61241,17391,4,0,1,1,0,0,1,0
# 2,170398,61241,17291,16,0,0,0,0,0,0,0
# 3,170398,61241,86955,4,0,0,0,0,0,0,1
# The minus_x and plus_x are the desired number of 4 um cubes to the left and right of the center cube
# The minus_y and plus_y are the desired number of 4 um cubes to the left and right of the center cube
# The minus_z and plus_z are the desired number of 4 um cubes to the left and right of the center cube
# The height is the height of the cube in microns
# The xloc, yloc are the coordinates of the center of the cube's x and y faces
# The zloc is the z coordinate of the center of the cube's bottom face


import cloudvolume
import pandas as pd
from imageryclient import ImageryClient
# import matplotlib.pyplot as plt
import numpy as np
import fastremap
import os


os.makedirs("../Neuron_transport_data/", exist_ok=True)  
df_settings = pd.read_csv("Simulation_settings_4.0.csv")
os.chdir("../Neuron_transport_data/")

seg_source = "precomputed://https://storage.googleapis.com/iarpa_microns/minnie/minnie65/seg_m343/"
# seg_source = "precomputed://https://minnie.microns-daf.com/segmentation/table/minnie65_public"
seg_cv = cloudvolume.CloudVolume(seg_source, use_https=False)

###User input#####

simulation_number = 1
# z_start_slice = 17391
# xval = 170398 # x value of center of cube
# yval = 60741#61241 # y value of center of cube
####End user input#####




df_set = df_settings.loc[df_settings['Simulation'] == simulation_number]
print(df_set)
z_start_slice = df_set['zloc'][simulation_number-1]
del_microns = df_set['height'][simulation_number-1]
xval = df_set['xloc'][simulation_number-1]
yval = df_set['yloc'][simulation_number-1]
minus_x = df_set['minus_x'][simulation_number-1]
plus_x = df_set['plus_x'][simulation_number-1]
minus_y = df_set['minus_y'][simulation_number-1]
plus_y = df_set['plus_y'][simulation_number-1]
minus_z = df_set['minus_z'][simulation_number-1]
plus_z = df_set['plus_z'][simulation_number-1]
print(z_start_slice,del_microns,xval,yval,minus_x,plus_x,minus_y,plus_y,minus_z,plus_z)

# # z range should be 100 for 4 um cube, 200 for 8 um, 400 for 16 um

for iterx in range(-minus_x,plus_x+1): 
    for itery in range(-minus_y,plus_y+1): 
        for iterz in range(-minus_z,plus_z+1): 
            print(iterx,itery,iterz)
            z_range = 25*del_microns
            current_start_slice = z_start_slice + iterz*z_range
            z_end_slice = current_start_slice + z_range
            for zval in range(current_start_slice,z_end_slice+1): 
                
                #print(xval,yval,zval)
                
                # img = img_cv[xval-250:xval+250, yval-250:yval+250, zval:zval+1]

                # seg_cv = cloudvolume.CloudVolume(seg_source)
                seg_cv.bounds
                del_xy = int(del_microns*125/2) # 250 for 4 micron, 500 for 8 micron or 1000 for 16 micron case
                current_xval = xval + iterx*2*del_xy
                current_yval = yval + itery*2*del_xy
                print(current_xval,current_yval,zval)
                size_xy = int(del_xy*2 + 1)
                seg = seg_cv[int(current_xval-del_xy):int(current_xval+del_xy+1), int(current_yval-del_xy):int(current_yval+del_xy+1), int(zval)]

                seg_flat = np.reshape(seg,(size_xy,size_xy))
                df = pd.DataFrame(seg_flat)
                #df.to_csv("frame_" + str(zval) + "_" + str(xval) + "_" + str(yval) + "_" + str(del_microns) + ".csv",index=False)
                df.to_csv(f"frame_{zval}_{current_xval}_{current_yval}_{del_microns}.csv", index=False)
                uniq, cts = fastremap.unique(seg, return_counts=True) 
                # print(uniq)
                # print(cts)
                df = pd.DataFrame(uniq)
                #df.to_csv("frame_uniq_" + str(zval) + "_" + str(xval) + "_" + str(yval) + "_" + str(del_microns) + ".csv",index=False)
                df.to_csv(f"frame_uniq_{zval}_{current_xval}_{current_yval}_{del_microns}.csv", index=False) 