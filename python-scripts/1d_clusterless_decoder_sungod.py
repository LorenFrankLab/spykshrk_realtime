#cell 1
# Setup and import packages
import os 
import glob
import trodes2SS
from trodes2SS import AttrDict, TrodesImport
import sungod_linearization
from sungod_linearization import createTrackGraph, hack_determinearmorder, turn_array_into_ranges, \
chunk_data, change_to_directory_make_if_nonexistent
import numpy as np
import scipy.io
import scipy as sp
import scipy.stats as ss
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import holoviews as hv
import json
import functools
import dask
import dask.dataframe as dd
import dask.array as da
import networkx as nx
import loren_frank_data_processing as lfdp
import scipy.io as sio # for saving .mat files 
import inspect # for inspecting files (e.g. finding file source)
import multiprocessing 
import sys 
import pickle
from tempfile import TemporaryFile
from multiprocessing import Pool
import math 

# set path to folders where spykshrk core scripts live
path_main = '/home/mcoulter/spykshrk_hpc'
os.chdir(path_main)
from spykshrk.franklab.data_containers import FlatLinearPosition, SpikeFeatures, Posteriors, \
        EncodeSettings, pos_col_format, SpikeObservation, RippleTimes, DayEpochEvent, DayEpochTimeSeries
from spykshrk.franklab.pp_decoder.util import normal_pdf_int_lookup, gaussian, apply_no_anim_boundary, normal2D
from spykshrk.franklab.pp_decoder.pp_clusterless import OfflinePPEncoder, OfflinePPDecoder
from spykshrk.franklab.pp_decoder.visualization import DecodeVisualizer
from spykshrk.util import Groupby


#cell 2
# Import data

# Define path bases 
path_base_rawdata = '/home/mcoulter/spykshrk_hpc/'

# Define parameters
rat_name = 'remy'
directory_temp = path_base_rawdata + rat_name + '/'
day_dictionary = {'remy':[20], 'gus':[24]}
epoch_dictionary = {'remy':[2], 'gus':[7]} 
tetrodes_dictionary = {'remy': [4,6,9,10,11,12,13,14,15,17,19,20,21,22,23,24,25,26,28,29,30], # 4,6,9,10,11,12,13,14,15,17,19,20,21,22,23,24,25,26,28,29,30
                       'gus': list(range(6,13)) + list(range(17,22)) + list(range(24,28)) + [30]} # list(range(6,13)) + list(range(17,22)) + list(range(24,28)) + [30]

# Maze information
os.chdir('/home/mcoulter/spykshrk_hpc/')
#maze_coordinates = scipy.io.loadmat('set_arm_nodes.mat',variable_names = 'linearcoord_NEW')
# new maze coordinates with only one segment for box
maze_coordinates = scipy.io.loadmat('set_arm_nodes.mat',variable_names = 'linearcoord_one_box')

datasrc = TrodesImport(directory_temp, rat_name, day_dictionary[rat_name], 
                       epoch_dictionary[rat_name], tetrodes_dictionary[rat_name])

# Import marks
marks = datasrc.import_marks()
# # os.chdir('/data2/jguidera/data/')
# # np.load('marks.npy')

# Import position #? concerned about use of sampling rate in the definition for position
# Temporary small definition of encoding settings-- need 'arm_coordinates' to use datasrc.import_pos 
encode_settings = AttrDict({'arm_coordinates': [[0,0]]})
# Import position (#? concerned about use of sampling rate in the definition for position)
pos = datasrc.import_pos(encode_settings, xy='x')
posY = datasrc.import_pos(encode_settings, xy='y')

# Import ripples
rips = datasrc.import_rips(pos, velthresh=4)

# Define path bases
path_base_dayepoch = 'day' + str(day_dictionary[rat_name][0]) + '_epoch' + str(epoch_dictionary[rat_name][0])
path_base_analysis = '/home/mcoulter/spykshrk_hpc/'

#cell 3
#filter ripples for velocity < 4
#re-shape ripples input table into format for get_irregular_resample
rips['timestamp'] = rips['starttime']
rips['time'] = rips['starttime']
rips.timestamp = rips.timestamp*30000
rips['timestamp'] = rips['timestamp'].astype(int)
rips.reset_index(level=['event'], inplace=True)
rips.columns = ['event','starttime','endtime','maxthresh','timestamp','time']
rips.set_index(['timestamp', 'time'], drop=True, append=True, inplace=True)

#filter for velocity < 4 with get_irregular_resample
linflat_obj = pos.get_mapped_single_axis()
linflat_ripindex = linflat_obj.get_irregular_resampled(rips)
linflat_ripindex_encode_velthresh = linflat_ripindex.query('linvel_flat < 4')

#re-shape to RippleTimes format for plotting
rips_vel_filt = rips.loc[linflat_ripindex_encode_velthresh.index]
rips_vel_filt.reset_index(level=['timestamp','time'], inplace=True)
rips_vel_filt.set_index(['event'], drop=True, append=True, inplace=True)
rips_vel_filtered = RippleTimes.create_default(rips_vel_filt, 1)

print('rips when animal velocity <= 4: '+str(linflat_ripindex_encode_velthresh.shape[0]))

#cell 4
# Encoding input data, position and spikes
# **** time is 30x sec
subset_start = 0
subset_end = 10000
chunkstart = pos.index.get_level_values('time')[subset_start]
chunkend = pos.index.get_level_values('time')[subset_end]
speed_threshold_save = 0; 

pos_subset = pos.loc[(pos.index.get_level_values('time') <= chunkend) & (pos.index.get_level_values('time') >= chunkstart)]
posY_subset = posY.loc[(posY.index.get_level_values('time') <= chunkend) & (posY.index.get_level_values('time') >= chunkstart)] 
pos_start= pos_subset.index.get_level_values('time')[0]
pos_end = pos_subset.index.get_level_values('time')[-1]
spk_subset = marks.loc[(marks.index.get_level_values('time') <  pos_end) & (marks.index.get_level_values('time') >  pos_start)]
rip_subset = rips.loc[(rips['starttime'].values >  pos_start) & (rips['endtime'].values <  pos_end)]
#rip_subset = rips_vel_filtered.loc[(rips_vel_filtered['starttime'].values >  pos_start) & (rips_vel_filtered['endtime'].values <  pos_end)]

spk_subset_sparse = trodes2SS.threshold_marks(spk_subset, maxthresh=2000,minthresh=100)
print('original length: '+str(spk_subset.shape[0]))
print('after filtering: '+str(spk_subset_sparse.shape[0]))

spk_subset_sparse.groupby('elec_grp_id')

# Filter encoding marks for times when rat velocity > 2 cm/s
# The purpose of this is to remove most of the stationary time from the encoding, to focus on times of movement

linflat_obj = pos_subset.get_mapped_single_axis()
linflat_spkindex = linflat_obj.get_irregular_resampled(spk_subset_sparse)
linflat_spkindex_encode_velthresh = linflat_spkindex.query('linvel_flat > 2')

spk_subset_sparse_encode = spk_subset_sparse.loc[linflat_spkindex_encode_velthresh.index]

print('encoding spikes after filtering: '+str(spk_subset_sparse_encode.shape[0]))

#cell 5
# Decoding input data, position and spikes
# **** time is 30x sec
chunkstart_decode = pos.index.get_level_values('time')[subset_start]
chunkend_decode = pos.index.get_level_values('time')[subset_end]
speed_threshold_save = 0; 

pos_subset_decode = pos.loc[(pos.index.get_level_values('time') <= chunkend_decode) & (pos.index.get_level_values('time') >= chunkstart_decode)]
posY_subset_decode = posY.loc[(posY.index.get_level_values('time') <= chunkend_decode) & (posY.index.get_level_values('time') >= chunkstart_decode)] 
pos_start_decode = pos_subset_decode.index.get_level_values('time')[0]
pos_end_decode = pos_subset_decode.index.get_level_values('time')[-1]
spk_subset_decode = marks.loc[(marks.index.get_level_values('time') <  pos_end_decode) & (marks.index.get_level_values('time') >  pos_start_decode)]
rip_subset_decode = rips.loc[(rips['starttime'].values >  pos_start) & (rips['endtime'].values <  pos_end)]
#rip_subset_decode = rips_vel_filtered.loc[(rips_vel_filtered['starttime'].values > pos_start_decode) & (rips_vel_filtered['endtime'].values <  pos_end_decode)]

spk_subset_sparse_decode = trodes2SS.threshold_marks(spk_subset_decode, maxthresh=2000,minthresh=100)
print('original length: '+str(spk_subset_decode.shape[0]))
print('after filtering: '+str(spk_subset_sparse_decode.shape[0]))

spk_subset_sparse_decode.groupby('elec_grp_id')

#cell 6
# linearize the whole epoch - should only have to do this once.
linear_start = pos.index.get_level_values('time')[subset_start]
linear_end = pos.index.get_level_values('time')[subset_end]
# Define path base
path_base_timewindow = str(int(round(linear_start))) + 'to' + str(int(round(linear_end))) + 'sec'
path_base_foranalysisofonesessionepoch = path_base_analysis + rat_name + '/' + path_base_dayepoch + '/' + path_base_timewindow

# Change to directory with saved linearization result
# Define folder for saved linearization result 
linearization_output_save_path = path_base_foranalysisofonesessionepoch + '/linearization_output/'
linearization_output_save_path
# Check if it exists, make if it doesn't
directory_path = linearization_output_save_path
change_to_directory_make_if_nonexistent(directory_path)

# Define name of linearization result
linearization_output1_save_filename = 'linearization_' + path_base_timewindow + '_speed' + str(speed_threshold_save) + '_linear_distance_arm_shift' + '.npy'
linearization_output2_save_filename = 'linearization_' + path_base_timewindow + '_speed' + str(speed_threshold_save) + '_track_segment_id_use' + '.npy'
# If linearization result doesn't exist, do linearization calculation
if os.path.exists(linearization_output1_save_filename) == False:
    print('Linearization result doesnt exist. Doing linearization calculation')

    # Prepare for linearization 
    
    # Create graph elements
    track_graph, track_segments, center_well_id = createTrackGraph(maze_coordinates)
    #track_segments = lfdp.track_segment_classification.get_track_segments_from_graph(track_graph)

    # Define shift amounts 
    # 1-13-19 trying 10cm bins with flat transition matrix, set hardcode_shiftamount to 20
    # **** 
    hardcode_armorder = hack_determinearmorder(track_segments) # add progressive stagger in this order
    hardcode_shiftamount = 20 # add this stagger to sum of previous shifts
    # ****
    linearization_arm_lengths = []
    # Caculate length of outer arms, plot
    for track_segment in enumerate(track_segments): # for each track segment
        #plt.plot(track_segment[1][:,0],track_segment[1][:,1]) # plot track segment
        #plt.text(track_segment[1][0,0],track_segment[1][0,1],str(track_segment[0])) # label with segment number
        # Calculate and plot length of outer arms 
        if track_segment[0] < 8: # if an outer arm, calculate length 
            linearization_arm_lengths.append(np.linalg.norm(track_segment[1][0,:] - track_segment[1][1,:])) # calculate length
            #plt.text(track_segment[1][0,0],track_segment[1][0,1] - 4,str(linearization_arm_lengths[track_segment[0]])) # text to show length 
    # Define dictionary for shifts for each arm segment
    shift_linear_distance_by_arm_dictionary = dict() # initialize empty dictionary 
    for arm in enumerate(hardcode_armorder): # for each outer arm
        if arm[0] == 0: # if first arm, just shift hardcode_shiftamount
            temporary_variable_shift = hardcode_shiftamount 
        else: # if not first arm, add to hardcode_shiftamount length of previous arm 
            temporary_variable_shift = hardcode_shiftamount + linearization_arm_lengths[arm[0]] + shift_linear_distance_by_arm_dictionary[hardcode_armorder[arm[0] - 1]]
        shift_linear_distance_by_arm_dictionary[arm[1]] = temporary_variable_shift
        
    
    # Pull node coordinates (store as node_coords)
    temp2 = [] # list for node coordinates
    for eachnode in track_graph.nodes: # for each node
        temp = list(track_graph.nodes[eachnode]['pos'])
        temp2.append(temp)
    node_coords = np.asarray(temp2)
    # Assign nodes to track segments
    track_segment_id_nodes = lfdp.track_segment_classification.find_nearest_segment(track_segments, node_coords)

    # Calculate linear distance of nodes to back well 
    linear_distance_nodes = lfdp.track_segment_classification.calculate_linear_distance(
            track_graph, track_segment_id_nodes, center_well_id, node_coords)

    # Linearize position
    pos_subset_linear = pos.loc[(pos.index.get_level_values('time') <= linear_end) & (pos.index.get_level_values('time') >= linear_start)]
    posY_subset_linear = posY.loc[(posY.index.get_level_values('time') <= linear_end) & (posY.index.get_level_values('time') >= linear_start)] 

    # Vector with position
    simplepos = np.vstack([pos_subset_linear['linpos_flat'],posY_subset_linear['linpos_flat']]) # x pos, y pos
    # Store under different name to plot for debugging 
    simplepos_original = simplepos 

    # Assign each position to a track segment
    # ****
    sensor_std_dev = 1 # 10
    assign_track_segments_one_is_Markov_two_is_naive = 2 # 1 for hidden markov model, 2 for naive
    # ****
    # Define back well
    #center_well_id = 17
    center_well_id = 16
    # HIDDEN MARKOV MODEL:
    # Assign position to track segment
    track_segment_id = lfdp.track_segment_classification.classify_track_segments(track_graph,
                                simplepos.T, sensor_std_dev=sensor_std_dev, route_euclidean_distance_scaling=1)
    # SIMPLER WAY: 
    #track_segments = lfdp.track_segment_classification.get_track_segments_from_graph(track_graph)
    track_segment_id_naive = lfdp.track_segment_classification.find_nearest_segment(track_segments, simplepos.T)
    # Choose track segment assignment 
    if assign_track_segments_one_is_Markov_two_is_naive == 1:
        track_segment_id_use = track_segment_id
    elif assign_track_segments_one_is_Markov_two_is_naive == 2:   
        track_segment_id_use = track_segment_id_naive
    # Find linear distance of position from back well 
    linear_distance = lfdp.track_segment_classification.calculate_linear_distance(track_graph, 
                                 track_segment_id_use, center_well_id, simplepos.T)

    # Modify: 1) collapse non-arm locations, 2) shift linear distance for the 8 arms
    newseg = np.copy(track_segment_id_use)
    # 1) Collapse non-arm locations
    # newseg[(newseg < 16) & (newseg > 7)] = 8
    # newseg[(newseg == 16)] = 9
    #try making one segment for box
    newseg[(newseg < 17) & (newseg > 7)] = 8
    
    # 2) Shift linear distance for each arm 
    linear_distance_arm_shift = np.copy(linear_distance)
    for seg in shift_linear_distance_by_arm_dictionary:
        linear_distance_arm_shift[(newseg==seg)]+=shift_linear_distance_by_arm_dictionary[seg]  
    # Incorporate modifications 

    pos_subset['linpos_flat']=linear_distance_arm_shift[(subset_start-subset_start):(subset_end-subset_start+1)]

    # Store some linearization results in python format for quick loading (pos_subset) 
    np.save(linearization_output1_save_filename, linear_distance_arm_shift)
    np.save(linearization_output2_save_filename, track_segment_id_use)
    
    # Save some linearization results in .mat file
    # Convert dictionary with shift for each arm to array since matlab can't read the dictionary 
    linearization_shift_segments_list = []
    for key in shift_linear_distance_by_arm_dictionary:
        temp = [key,shift_linear_distance_by_arm_dictionary[key]]
        linearization_shift_segments_list.append(temp)    
    # Change directory
    change_to_directory_make_if_nonexistent(linearization_output_save_path)
    # Define file name 
    file_name_temp = [rat_name + '_day' + str(day_dictionary[rat_name][0]) + '_epoch' + str(epoch_dictionary[rat_name][0]) + 
                      '_' + path_base_timewindow +
                      '_speed' + str(speed_threshold_save) + 
                      '_linearization_variables.mat']    

    # Store variables 
    export_this = AttrDict({'linearization_segments': track_segments,
                            'linearization_nodes_coordinates': node_coords,
                            'linearization_nodes_distance_to_back_well':linear_distance_nodes,
                            'linearization_shift_segments_list': linearization_shift_segments_list,
                            'linearization_position_segments':track_segment_id_use,
                            'linearization_position_distance_from_back_well':linear_distance,
                            'linearization_position_distance_from_back_well_arm_shift':linear_distance_arm_shift
                           })
    # Warn before overwriting file 
    if os.path.exists(file_name_temp[0]) == True:
        input("Press Enter to overwrite file")
        print('overwriting')
    # Save 
    print('saving file:',file_name_temp)
    sio.savemat(file_name_temp[0],export_this)
    
# If linearization result exists, load it 
else:
    print('Linearization result exists. Loading it.')
    linear_distance_arm_shift = np.load(linearization_output1_save_filename)
    #test = np.load(linearization_output3_save_filename)
    track_segment_id_use = np.load(linearization_output2_save_filename)
    pos_subset['linpos_flat'] = linear_distance_arm_shift[(subset_start-subset_start):(subset_end-subset_start+1)]

#cell 7
# Define position bins #!!! HARD CODE: ASSUMES POSITION BIN OF WIDTH 1 !!!
# need to use the indices of the encoding time subset in this cell

# Initialize variables 
tracksegment_positionvalues_min_and_max = []
tracksegment_positionvalues_for_bin_edges = [] 

# Find min and max position for each track segment 
tracksegments_temp = np.unique(track_segment_id_use[subset_start:(subset_end+1)])
for t_loop in tracksegments_temp: # for each track segment
    indiceswewant_temp = track_segment_id_use[subset_start:(subset_end+1)] == t_loop
    tracksegment_positionvalues_temp = pos_subset.values[indiceswewant_temp,0] # second dimension of pos_subset: zero for position, 1 for velocity
    tracksegment_positionvalues_min_and_max.append([tracksegment_positionvalues_temp.min(), tracksegment_positionvalues_temp.max()])
    # To define edges, floor mins and ceil maxes
    tracksegment_positionvalues_for_bin_edges.append([np.floor(tracksegment_positionvalues_temp.min()), np.ceil(tracksegment_positionvalues_temp.max())])

# Floor to get bins #? Is this right? Does 0 mean the bin spanning [0, 1]?
tracksegment_positionvalues_min_and_max_floor = np.floor(tracksegment_positionvalues_min_and_max)

# Find only bins in range of segments
binswewant_temp = []
for t_loop in tracksegment_positionvalues_min_and_max_floor: # for each track segment
    binswewant_temp.append(np.ndarray.tolist(np.arange(t_loop[0],t_loop[1] + 1))) # + 1 to account for np.arange not including last index
# Do same for edges
edgeswewant_temp = []
for t_loop in tracksegment_positionvalues_for_bin_edges: # for each track segment
    edgeswewant_temp.append(np.ndarray.tolist(np.arange(t_loop[0],t_loop[1] + 1))) # + 1 to account for np.arange not including last index

# Flatten (combine bins from segments)
binswewant_temp_flat = [y for x in binswewant_temp for y in x]
edgeswewant_temp_flat = [y for x in edgeswewant_temp for y in x]

# Find unique elements
arm_coords_wewant = (np.unique(binswewant_temp_flat))
edges_wewant = (np.unique(edgeswewant_temp_flat))

# Turn list of edges into ranges 
start_temp, end_temp = turn_array_into_ranges(edges_wewant)
arm_coordinates_WEWANT = np.column_stack((start_temp, end_temp))
print(arm_coordinates_WEWANT)

#cell 7.1
# this cell speeds up encoding with larger position bins
# try 5cm bins - do this by dividing position subset by 5 and arm coords by 5

#pos_subset['linpos_flat'] = (pos_subset['linpos_flat'])/5

#arm_coordinates_WEWANT = arm_coordinates_WEWANT/5
#arm_coordinates_WEWANT = np.around(arm_coordinates_WEWANT)
#print(arm_coordinates_WEWANT)

#cell 8
# DEFINE encoding settings #? Ideally would only define once
max_pos = int(round(linear_distance_arm_shift.max()) + 20)

# if you are using 5cm position bins, use this max_pos instead
#max_pos = int(round(linear_distance_arm_shift.max()/5)+5)

encode_settings = AttrDict({'sampling_rate': 3e4,
                            'pos_bins': np.arange(0,max_pos,1), # arm_coords_wewant
                            'pos_bin_edges': np.arange(0,max_pos + .1,1), # edges_wewant, 
                            'pos_bin_delta': 1, 
                            # 'pos_kernel': sp.stats.norm.pdf(arm_coords_wewant, arm_coords_wewant[-1]/2, 1),
                            'pos_kernel': sp.stats.norm.pdf(np.arange(0,max_pos,1), max_pos/2, 1), #note that the pos_kernel mean should be half of the range of positions (ie 180/90) # sp.stats.norm.pdf(np.arange(0,560,1), 280, 1),    
                            'pos_kernel_std': 1, 
                            'mark_kernel_std': int(20), 
                            'pos_num_bins': max_pos, # len(arm_coords_wewant)
                            'pos_col_names': [pos_col_format(ii, max_pos) for ii in range(max_pos)], # or range(0,max_pos,10)
                            'arm_coordinates': arm_coordinates_WEWANT}) # includes box, removes bins in the gaps 'arm_coordinates': [[0,max_pos]]})

#cell 9
#define decode settings
decode_settings = AttrDict({'trans_smooth_std': 2,
                            'trans_uniform_gain': 0.001,
                            'time_bin_size':60})

#cell 10
# Run encoder
# these time-table lines are so that we can record the time it takes for encoder to run even if notebook disconnects
# look at the time stamps for the two files in /data2/mcoulter called time_stamp1 and time_stamp2
print('Starting encoder')
time_table_data = {'age': [1, 2, 3, 4, 5]}
time_table = pd.DataFrame(time_table_data)
time_table.to_csv('/home/mcoulter/spykshrk_hpc/time_stamp1.csv')

encoder = OfflinePPEncoder(linflat=pos_subset, dec_spk_amp=spk_subset_sparse_decode, encode_settings=encode_settings, 
                           decode_settings=decode_settings, enc_spk_amp=spk_subset_sparse, dask_worker_memory=1e9,
                           dask_chunksize = None)

results = encoder.run_encoder()
time_table.to_csv('/home/mcoulter/spykshrk_hpc/time_stamp2.csv')
print('Enocder finished!')

#cell 11
#make observations table from results

#tet_ids = np.unique(spk_subset_sparse.index.get_level_values('elec_grp_id'))
tet_ids = np.unique(spk_subset_sparse_decode.index.get_level_values('elec_grp_id'))
observ_tet_list = []
#grp = spk_subset_sparse.groupby('elec_grp_id')
grp = spk_subset_sparse_decode.groupby('elec_grp_id')
for tet_ii, (tet_id, grp_spk) in enumerate(grp):
    tet_result = results[tet_ii]
    tet_result.set_index(grp_spk.index, inplace=True)
    observ_tet_list.append(tet_result)

observ = pd.concat(observ_tet_list)
observ_obj = SpikeObservation.create_default(observ.sort_index(level=['day', 'epoch', 
                                                                      'timestamp', 'elec_grp_id']), 
                                             encode_settings)

observ_obj['elec_grp_id'] = observ_obj.index.get_level_values('elec_grp_id')
observ_obj.index = observ_obj.index.droplevel('elec_grp_id')

#cell 13
# save observations
#observ_obj._to_hdf_store('/data2/mcoulter/remy_20_4_observ_obj_0_2000.h5','/analysis', 
#                         'decode/clusterless/offline/observ_obj', 'observ_obj')
#print('Saved observations to /data2/mcoulter/remy_20_4_observ_obj_0_2000.h5')

#cell 14
# load previously generated observations
# hacky but reliable way to load a dataframe stored as hdf
# Posteriors is imported from data_containers
#observ_obj = Posteriors._from_hdf_store('/data2/mcoulter/remy_20_4_observ_obj_0_20000.h5','/analysis', 
#                         'decode/clusterless/offline/observ_obj', 'observ_obj')

#cell 15
# Run PP decoding algorithm
# NOTE 1-11-19 had to add spk_amp and vel to encode settings in order for decoding to run
# what should these be set to? and why are they here now?
time_bin_size = 60
decode_settings = AttrDict({'trans_smooth_std': 2,
                            'trans_uniform_gain': 0.001,
                            'time_bin_size':60})

encode_settings = AttrDict({'sampling_rate': 3e4,
                            'pos_bins': np.arange(0,max_pos,1), # arm_coords_wewant
                            'pos_bin_edges': np.arange(0,max_pos + .1,1), # edges_wewant, 
                            'pos_bin_delta': 1, 
                            # 'pos_kernel': sp.stats.norm.pdf(arm_coords_wewant, arm_coords_wewant[-1]/2, 1),
                            'pos_kernel': sp.stats.norm.pdf(np.arange(0,max_pos,1), max_pos/2, 1), #note that the pos_kernel mean should be half of the range of positions (ie 180/90) # sp.stats.norm.pdf(np.arange(0,560,1), 280, 1),    
                            'pos_kernel_std': 1, 
                            'mark_kernel_std': int(20), 
                            'pos_num_bins': max_pos, # len(arm_coords_wewant)
                            'pos_col_names': [pos_col_format(ii, max_pos) for ii in range(max_pos)], # [pos_col_format(int(ii), len(arm_coords_wewant)) for ii in arm_coords_wewant],
                            'arm_coordinates': arm_coordinates_WEWANT, # 'arm_coordinates': [[0,max_pos]]})
                            'spk_amp': 60,
                            'vel': 0})

print('Starting decoder')
decoder = OfflinePPDecoder(observ_obj=observ_obj, trans_mat=encoder.trans_mat['flat'], 
                           prob_no_spike=encoder.prob_no_spike,
                           encode_settings=encode_settings, decode_settings=decode_settings, 
                           time_bin_size=time_bin_size)

posteriors = decoder.run_decoder()
print('Decoder finished!')

#cell 16
#save posteriors
#posteriors._to_hdf_store('/data2/mcoulter/posteriors/remy_20_4_linearized_0_2000.h5','/analysis', 
#                         'decode/clusterless/offline/posterior', 'learned_trans_mat')
#print('Saved posteriors to /data2/mcoulter/posteriors/remy_20_4_linearized_0_2000.h5')

#cell 17
#load previously generated posteriors
#posteriors = Posteriors._from_hdf_store('/data2/mcoulter/posteriors/remy_20_4_linearized_alltime_decode.h5','/analysis',
#                                        'decode/clusterless/offline/posterior', 'learned_trans_mat')
