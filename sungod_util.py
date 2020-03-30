# based on sungod_linearization
# based on Loren Frank Data Processing by ED
# 
# Jun 2018 JAG, Jan 2019, MEC, AKG

import os
import numpy as np
import scipy as sp
import scipy.stats as ss
import scipy.io
import networkx as nx
import loren_frank_data_processing as lfdp
from loren_frank_data_processing import Animal
import scipy.io as sio # for saving .mat files 
import inspect # for inspecting files (e.g. finding file source)
import pandas as pd
import json
import functools
from replay_trajectory_classification.state_transition import strong_diagonal_discrete
from replay_trajectory_classification.core import _causal_classify, _acausal_classify
from spykshrk.franklab.pp_decoder.util import apply_no_anim_boundary
from trodes2SS import convert_dan_posterior_to_xarray, AttrDict


def createTrackGraph(maze_coordinates):
    #linearcoord = maze_coordinates['linearcoord_NEW'].squeeze()
    linearcoord = maze_coordinates['linearcoord_one_box'].squeeze()
    track_segments = [np.stack((arm[:-1], arm[1:]), axis=1) for arm in linearcoord]
    center_well_position = track_segments[0][0][0]
    track_segments, center_well_position = (np.unique(np.concatenate(track_segments), axis=0),
                center_well_position) # ## what does redefining center_well_position here do?
    nodes = np.unique(track_segments.reshape((-1, 2)), axis=0)
    edges = np.zeros(track_segments.shape[:2], dtype=int)
    for node_id, node in enumerate(nodes):
        edge_ind = np.nonzero(np.isin(track_segments, node).sum(axis=2) > 1)
        edges[edge_ind] = node_id
    edge_distances = np.linalg.norm(
        np.diff(track_segments, axis=-2).squeeze(), axis=1)
    track_graph = nx.Graph()
    for node_id, node_position in enumerate(nodes):
        track_graph.add_node(node_id, pos=tuple(node_position))
    for edge, distance in zip(edges, edge_distances):
        track_graph.add_edge(edge[0], edge[1], distance=distance)
    center_well_id = np.unique(
        np.nonzero(np.isin(nodes, center_well_position).sum(axis=1) > 1)[0])[0]
    return track_graph, track_segments, center_well_id

def hack_determinearmorder(track_segments): 
    # order arm segments based on y position. ASSUMES CERTAIN LAYOUT OF TRACK_SEGMENTS. 
    d_temp = [] 
    for track_segment in enumerate(track_segments):
        if track_segment[0] < 8:
            d_temp.append(track_segment[1][1,1])

    rank = ss.rankdata(d_temp) - 1 # - 1 to account for python indexing

    order1 = [None]*len(rank)
    for r in enumerate(rank):
        order1[int(r[1])] = int(r[0])
    return(order1)

def turn_array_into_ranges(array1):
    array1_diff = np.ediff1d(array1)

    start_temp = [] 
    end_temp = []
    start_temp.append(array1[0]) 

    some_end_indices = np.where(array1_diff > 1)

    for i in range(len(some_end_indices[0])):
        # This is always an end index
        end_temp.append(array1[some_end_indices[0][i]])
        if array1[some_end_indices[0][i]] == start_temp[i]: # if this is the same as the last start index, it was already added as a start index-- don't need to add it again
            start_temp.append(array1[some_end_indices[0][i] + 1]) # define next start index   
        elif array1_diff[some_end_indices[0][i] - 1] > 1: # if last value was more than 1 away, this is also a start index
            start_temp.append(array1[some_end_indices[0][i]])    
        else: # if last value was NOT more than 1 away, this is JUST an end index, and next start index is next index
            start_temp.append(array1[some_end_indices[0][i] + 1])   
    # The last entry in array is always the last end index
    end_temp.append(array1[-1])  

    return start_temp, end_temp

# Function to define chunked data 
def chunk_data(data,number_of_chunks):
    print('chunking data of length',len(data),'samples into',str(number_of_chunks),'chunk(s)')
    # Takes 1D data and splits into number of chunks (as equally as possible)
    
    # Calculate number of data points per chunk
    datapoints_per_chunk = math.ceil(len(data)/number_of_chunks)
    print('datapoints_per_chunk:',datapoints_per_chunk)
    
    # Initialize empty list for chunked data
    chunked_data = [] 
    
    # Define chunks
    for chunk_number in range(number_of_chunks): # for each chunk
        chunked_data.append(data[chunk_number*datapoints_per_chunk:(chunk_number + 1)*datapoints_per_chunk]) 
    
    return chunked_data

    # Toy example
    #hi = np.concatenate((np.ones(5),np.ones(5)*2),axis=0)
    #print(hi)
    #chunked_data = []
    #chunk_number = 2
    #for i in range(10): 
    #    x = hi[chunk_number*i:chunk_number*(i + 1)]
    #    print(x)
    #    chunked_data.append(x)
    #print(chunked_data)

def change_to_directory_make_if_nonexistent(directory_path):
    # Make directory if it doesn't exist
    if os.path.exists(directory_path) == False:
        print('making path ' + directory_path)
        os.chdir('/')        
        os.makedirs(directory_path)
    # Change to directory 
    os.chdir(directory_path)
    # Print working directory
    os.getcwd()
    

def run_linearization_routine(animal, day, epoch, linearization_path, raw_path, gap_size=20, optional_alternate_nodes=None, optional_output_suffix=None,lfdp_v9=False):

    if optional_alternate_nodes:
        node_ref = optional_alternate_nodes
    else:
        node_ref = linearization_path + 'set_arm_nodes.mat'

    #node_ref = linearization_path + 'fievel_new_arm_nodes.mat'
    #node_ref = linearization_path + 'remy_20_2_new_arm_nodes.mat'
    linearcoord = sio.loadmat(node_ref)['linearcoord_one_box'][0]

    animalinfo  = {animal: Animal(directory=raw_path, short_name=animal)}
    epoch_key = (animal, day, epoch)
    position_info = lfdp.position._get_pos_dataframe(epoch_key, animalinfo)

    center_well_position = linearcoord[0][0]
    nodes = [center_well_position[np.newaxis, :]]

    for arm in linearcoord: 
        for point in arm[1:]:
            nodes.append(point[np.newaxis, :])
    nodes = np.concatenate(nodes)

    dist = []
    for arm in linearcoord:
        dist.append(np.linalg.norm(np.diff(arm, axis=0), axis=1))
    np.stack([*dist])

    edges = [(0, 1),(1, 2),(0, 3),(3, 4),(0, 5),(5, 6),(0, 7),(7, 8),
            (0, 9),(9, 10),(0, 11),(11, 12),(0, 13),(13, 14),(0, 15),(15, 16)]
    edge_distances = np.concatenate([*dist])

    track_graph = nx.Graph()
    for node_id, node_position in enumerate(nodes):
        track_graph.add_node(node_id, pos=tuple(node_position))
    for edge, distance in zip(edges, edge_distances):
        track_graph.add_edge(edge[0], edge[1], distance=distance)
    #lfdp.track_segment_classification.plot_track(track_graph)

    position = position_info.loc[:, ['x_position', 'y_position']].values
    center_well_id = 0

    if lfdp_v9:  # slight change in syntax and params for the new version
        position_info['track_segment_id'] = lfdp.track_segment_classification.classify_track_segments(
                track_graph, position,
                route_euclidean_distance_scaling=1,
                sensor_std_dev=3, diagonal_bias=0)
        
        (position_info['linear_distance'], position_info['projected_x'], position_info['projected_y']) = lfdp.track_segment_classification.calculate_linear_distance(
                    track_graph, position_info['track_segment_id'], center_well_id, position)

        #backfill nans, and report how many
        print('Nans backfilled in linearization routine:' +str(len(np.where(np.isnan(position_info['track_segment_id'])))))
        position_info['track_segment_id'].fillna(method='backfill',inplace=True)
        position_info['linear_distance'].fillna(method='backfill',inplace=True)

        track_segment_id = np.copy(position_info['track_segment_id'])
        linear_distance_arm_shift = np.copy(position_info['linear_distance'])

    else:
        track_segment_id = lfdp.track_segment_classification.classify_track_segments(
                track_graph, position,
                route_euclidean_distance_scaling=1,
                sensor_std_dev=1)

        linear_distance = lfdp.track_segment_classification.calculate_linear_distance(
                    track_graph, track_segment_id, center_well_id, position)

        linear_distance_arm_shift = np.copy(linear_distance)

        # this section calculates the shift amounts for each arm
    arm_distances = (edge_distances[1],edge_distances[3],edge_distances[5],edge_distances[7],
                        edge_distances[9],edge_distances[11],edge_distances[13],edge_distances[15])

    shift_linear_distance_by_arm_dictionary = dict() # initialize empty dictionary 

    hardcode_armorder = [0,1,2,3,4,5,6,7]

    for arm in enumerate(hardcode_armorder): # for each outer arm
        if arm[0] == 0: # if first arm, just shift hardcode_shiftamount
            temporary_variable_shift = gap_size
        else: # if not first arm, add to hardcode_shiftamount length of previous arm 
            temporary_variable_shift = gap_size + arm_distances[arm[0]] + shift_linear_distance_by_arm_dictionary[hardcode_armorder[arm[0] - 1]]
        shift_linear_distance_by_arm_dictionary[arm[1]] = temporary_variable_shift

        # Modify: 1) collapse non-arm locations (segments 0-7), 
        # 2) shift linear distance for the 8 arms (segments 8-15)
    newseg = np.copy(track_segment_id)
    newseg[(newseg < 8)] = 0

        # 2) Shift linear distance for each arm 
    for seg in shift_linear_distance_by_arm_dictionary:
        linear_distance_arm_shift[(newseg==(seg+8))]+=shift_linear_distance_by_arm_dictionary[(seg)]  

    #save outputs
    output_base = linearization_path + animal + '_' + str(day) + '_' +str(epoch) + '_'
    if optional_output_suffix:
        lin_output1 = output_base + optional_output_suffix + '_distance.npy'
        lin_output2 = output_base + optional_output_suffix + '_track_segments.npy'
    else:
        lin_output1 = output_base + 'linearized_distance.npy'
        lin_output2 = output_base + 'linearized_track_segments.npy'
    #lin_output3 = output_base + 'linearization_variables.mat'
    np.save(lin_output1, linear_distance_arm_shift)
    np.save(lin_output2, track_segment_id)
    os.chmod(lin_output1,0o774)
    os.chmod(lin_output2,0o774)

    # linearization_shift_segments_list = []
    # for key in shift_linear_distance_by_arm_dictionary:
    #     temp = [key,shift_linear_distance_by_arm_dictionary[key]]
    #     linearization_shift_segments_list.append(temp)    

    #     # Store variables 
    # export_this = AttrDict({'linearization_segments': edges,
    #                             'linearization_nodes_coordinates': nodes,
    #                             'linearization_nodes_distance_to_back_well':arm_distances,
    #                             'linearization_shift_segments_list': linearization_shift_segments_list,
    #                             'linearization_position_segments':track_segment_id,
    #                             'linearization_position_distance_from_back_well':linear_distance,
    #                             'linearization_position_distance_from_back_well_arm_shift':linear_distance_arm_shift
    #                            })

        
    # sio.savemat(lin_output3,export_this)

def run_linearization_routine_4_arm(animal, day, epoch, linearization_path, raw_path, gap_size=20):
    animalinfo  = {animal: Animal(directory=raw_path, short_name=animal)}
    epoch_key = (animal, day, epoch)
    position_info = lfdp.position._get_pos_dataframe(epoch_key, animalinfo)

    nodes = np.array([
        (169.1, 105), # home
        (140, 106.8), # past center
        (119.8, 95.8), # start arm 1
        (118.6, 106.2), # start arm 2
        (119.8, 115.4),  # start arm 3
        (123, 125), # start arm 4 
        (51.4, 88.2), # end arm 1
        (49.5, 109.2),  # end arm 2 
        (52.8, 130),  # end arm 3
        (61.1, 152.5), # end arm 4
    ])

    edges = np.array([
        (0, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (2, 6),
        (3, 7),
        (4, 8),
        (5, 9),
    ])

    track_segments = np.array([(nodes[e1], nodes[e2]) for e1, e2 in edges])
    edge_distances = np.linalg.norm(
        np.diff(track_segments, axis=-2).squeeze(), axis=1)

    track_graph = nx.Graph()
    for node_id, node_position in enumerate(nodes):
        track_graph.add_node(node_id, pos=tuple(node_position))
    #old:
    #for edge, distance in zip(edges, edge_distances):
    #    track_graph.add_edge(edge[0], edge[1], distance=distance)

    # new
    for edge, distance in zip(edges, edge_distances):
        nx.add_path(track_graph, edge, distance=distance)

    #lfdp.track_segment_classification.plot_track(track_graph)

    position = position_info.loc[:, ['x_position', 'y_position']].values

    track_segment_id = lfdp.track_segment_classification.classify_track_segments(
            track_graph, position,
            route_euclidean_distance_scaling=1,
            sensor_std_dev=1)

    center_well_id = 0
    linear_distance = lfdp.track_segment_classification.calculate_linear_distance(
                track_graph, track_segment_id, center_well_id, position)

    # this section calculates the shift amounts for each arm
    arm_distances = (edge_distances[5:9])

    shift_linear_distance_by_arm_dictionary = dict()
    # this order is used to shift the linear distances
    hardcode_armorder = [0,1,2,3,4,5,6,7,8]
    # add this gap to sum of previous shifts (was 20 for 1cm)
    hardcode_shiftamount = 20

    for arm in hardcode_armorder:
        # home to wait
        if arm == 0:
            temporary_variable_shift = 0

        # outer half of box - also no shift
        elif arm > 0 and arm < 5:
            temporary_variable_shift = 0

        # first outer arm - only hardcode shift because arms dont start at 0
        elif arm == 5:
            temporary_variable_shift = hardcode_shiftamount
                
        # outer arms - add length of previous arm
        else:
            temporary_variable_shift = hardcode_shiftamount + arm_distances[arm-6] + shift_linear_distance_by_arm_dictionary[hardcode_armorder[arm - 1]]
                
        shift_linear_distance_by_arm_dictionary[arm] = temporary_variable_shift

    # 2) Shift linear distance for each arm 
    linear_distance_arm_shift = np.copy(linear_distance)
    for seg in shift_linear_distance_by_arm_dictionary:
        linear_distance_arm_shift[(track_segment_id==(seg))]+=shift_linear_distance_by_arm_dictionary[(seg)]

    #save outputs
    output_base = linearization_path + animal + '_' + str(day) + '_' +str(epoch) + '_'
    lin_output1 = output_base + 'linearized_distance.npy'
    lin_output2 = output_base + 'linearized_track_segments.npy'
    #lin_output3 = output_base + 'linearization_variables.mat'
    np.save(lin_output1, linear_distance_arm_shift)
    np.save(lin_output2, track_segment_id)
    os.chmod(lin_output1,0o774)
    os.chmod(lin_output2,0o774)

def define_segment_coordinates(pos_obj, track_segment_ids):
    
    ''' 
    Use segment assignments of position data to determine the ends of each segment.
     
    Important: do this per epoch in case maze coverage is incomplete - this will exclude areas
     where the animal never goes)

    Note: this implicity puts data into 1cm bins, since we are rounding position to integers
    
    Output: start and end bins for each segment (0-based, INCLUSIVE)
    '''

    track_segments = np.unique(track_segment_ids)
    seg_pos_range = []
    seg_pos_range_edges = [] 
    for seg in track_segments:
        seg_inds = track_segment_ids==seg      #get index of positions on that segment
        seg_pos = pos_obj['linpos_flat'].values[seg_inds]    # store those positions (pos only, not vel)
        seg_pos_range.append([seg_pos.min() ,seg_pos.max()])    # get min and max vals per segment
        seg_pos_range_edges.append([np.floor(seg_pos.min()), np.ceil(seg_pos.max())])

    binswewant_tmp = []
    for seg_range in np.floor(seg_pos_range_edges): # for each track segment
        binswewant_tmp.append(np.arange(seg_range[0],seg_range[1]+1)) # + 1 to account for np.arange not including last index

    occupied_bins = np.unique(np.concatenate(binswewant_tmp))   # concatenate and get rid of duplicate bins from box segs
    arm_coordinates = np.column_stack(turn_array_into_ranges(occupied_bins))   #  get start & end of each region

    return arm_coordinates, occupied_bins

def bin_position_data(pos_obj, arm_coordinates, pos_bin_size = 1):
    '''
    Bin linearized position 
    Bin size defaults to 1 

    In order to make arms all evenly divisible by bin_size, add extra values to arm bounds
    This could be problematic for extremely large bins, but should be accounted for by pos_bin_delta later

    Note that gap bins may differ in number and size but this shouldnt matter; dependent on arm bounds 

    Returns a new pos object with [linpos_flat] pos values replaced with indices of position bins
    '''

    armcoordinates_corrected = []
    pos_bins = []
    for arm in np.arange(0,arm_coordinates.shape[0],1):
        arm_bounds = arm_coordinates[arm]
        # add to arm edge value to make it divisible by posbin. each arm starts at the node, and now will end some 5cm bins later
        remainder = np.mod((arm_bounds[1]-arm_bounds[0]),pos_bin_size)   
        if remainder: 
            extra_needed = pos_bin_size-remainder     
        else:
            extra_needed = 0
        new_bounds = [arm_bounds[0], arm_bounds[1]+extra_needed]
        armcoordinates_corrected.append(new_bounds)
        # construct pos bin edges. this is not just a range, because the gaps are not strictly 5cm (thanks to the adjustments above)
        pos_bins.append(np.arange(new_bounds[0],new_bounds[1]+1,pos_bin_size))   
        if arm_bounds[1] < arm_coordinates[-1,-1]:  
            pos_bins.append(np.arange(new_bounds[1],arm_coordinates[arm+1][0]+1,pos_bin_size))  # add gap bins
        
    # there are lots of repeat bins in segments in the box - get rid of them         
    pos_bins = np.unique(np.concatenate(pos_bins))  # analog to mike's position_bins
    print(pos_bins)

    armcoordinates_binned = []     # find the indexes of the arm ends (mike's new_arm_coords)
    for end in armcoordinates_corrected:
        startind=np.where(pos_bins == end[0])
        endind = np.where(pos_bins == end[1])
        armcoordinates_binned.append([startind[0][0],endind[0][0]-1])
    armcoordinates_binned = np.array(armcoordinates_binned)

    print(armcoordinates_binned)

    # bin linearized position data with these bins
    # digitize returns the bin index for each value, converting position measure into binindex measure (1-based)
    digitized = np.digitize(pos_obj['linpos_flat'], pos_bins)
    binned_linear_pos = pos_obj.copy()    # make complete separate copy (deep)
    binned_linear_pos['linpos_flat'] = digitized-1  # -1  for python 0-based

    return binned_linear_pos, armcoordinates_binned, pos_bins

def define_pos_bin_delta(armcoordinates_binned, pos_bins, pos_raw, pos_bin_size):
    ''' 
    Determine what proportion of the bin is covered by original (unbinned) position data
    This will scale the decode likelihood down for end bins with only partial coverage 
    only define this for arm end bins, because everything else should necessarily be complete
    '''

    pos_delta = np.ones(armcoordinates_binned[-1][-1]+1)
    for i in np.arange(0,len(pos_bins),1):  #iterate through pos bins (excludes box end)
        if i in armcoordinates_binned[1:,1]:     # calc coverage if this is an arm end bin
            pos_in_bin = pos_raw['linpos_flat'][(pos_raw['linpos_flat']>pos_bins[i]) & (pos_raw['linpos_flat']<pos_bins[i+1])]        
            #if pos_in_bin.size > 0:
            pos_delta[i] = (max(pos_in_bin)-pos_bins[i])/pos_bin_size

    pos_delta = 1/pos_delta   # take inverse since we end up dividing by this instead of multiplying

    return pos_delta

def reorder_data_by_random_trial_order(trials, pos_obj, marks_obj):
    '''
    Randomize trial order so that chunks (for cross validation) don't contain clusters of certain arms
    '''

    trials.reset_index(level=['trial'], inplace=True)   
    trialsindex = trials['trial'].values
    print('Number of trials: ',len(trialsindex))
    np.random.shuffle(trialsindex)

    starttimes_shuffled = trials['starttime'].iloc[trialsindex]
    endtimes_shuffled = trials['endtime'].iloc[trialsindex]

    pos_reordered = pos_obj.head(0)   #initialize
    for i in range(len(starttimes_shuffled)):
        random_trial_pos = pos_obj.loc[(pos_obj.index.get_level_values('time') <= endtimes_shuffled.iloc[i]) 
                                    & (pos_obj.index.get_level_values('time') >= starttimes_shuffled.iloc[i])]
        pos_reordered = pos_reordered.append(random_trial_pos)
             
        #marks
    marks_reordered = marks_obj.head(0)
    for i in range(len(starttimes_shuffled)):
        random_trial_marks = marks_obj.loc[(marks_obj.index.get_level_values('time') <= endtimes_shuffled.iloc[i]) & (marks_obj.index.get_level_values('time') >= starttimes_shuffled.iloc[i])]
        marks_reordered = marks_reordered.append(random_trial_marks)

    trials.set_index(['trial'], drop=True, append=True, inplace=True)

    return pos_reordered, marks_reordered, trialsindex

def assign_enc_dec_set_by_velocity(pos_obj, marks_obj, velthresh, buffer = 0):
    '''
    Add column to marks object and pos object which designates whether data is part of encoding set (1) or decoding set (0)
    Based on velocity threshold
    Optional buffer, which will include n seconds after the rat slows down in the encoding set
    '''

    posinfo_at_mark_times = pos_obj.get_irregular_resampled(marks_obj).reset_index()

    if buffer:
        # calculate buffer times for marks
        above = posinfo_at_mark_times['linvel_flat']>velthresh  # generate boolean of above/below velocity threshold
        abovediff = np.insert(np.diff(above.astype(int)),0,0)   # convert boolean to int, calc diff, add a leading 0 to restore array length
        not_transition_inds = np.where(abovediff!=-1)  # find indexes of everything OTHER THAN transitions from above threshold to below 
        transition_times = pd.DataFrame({'time':posinfo_at_mark_times['time']})   # initialize a df to track the transision times 
        transition_times['time'].iloc[not_transition_inds]=np.nan    # turn everything in that df to nans except for the transition times
        transition_times = transition_times.ffill().fillna(0)     # forward fill the nans with the most recent transistion times. turn remaining nans (at beginning) to 0s 
        inBuffer = (posinfo_at_mark_times['time']-transition_times['time'])<=buffer   # any thing within 2s after a transition time will be in buffer
        mark_assignments = (above|inBuffer).values              # assign anything above velthresh or in buffer to enc set
        # calculate buffer times for pos
        pos_obj.reset_index(level=['time'], inplace=True)   
        above = pos_obj['linvel_flat']>velthresh    # generate boolean of above/below velocity threshold
        abovediff = np.insert(np.diff(above.astype(int)),0,0)   # convert boolean to int, calc diff, add a leading 0 to restore array length
        not_transition_inds = np.where(abovediff!=-1)  # find indexes of everything OTHER THAN transitions from above threshold to below 
        transition_times = pd.DataFrame({'time':pos_obj['time']})   # initialize a df to track the transision times 
        transition_times['time'].iloc[not_transition_inds]=np.nan    # turn everything in that df to nans except for the transition times
        transition_times = transition_times.ffill().fillna(0)     # forward fill the nans with the most recent transistion times. turn remaining nans (at beginning) to 0s 
        inBuffer = (pos_obj['time']-transition_times['time'])<=buffer   # any thing within 2s after a transition time will be in buffer
        pos_assignments = (above|inBuffer)             # assign anything above velthresh or in buffer to enc set\
        pos_obj.set_index(['time'], drop=True, append=True, inplace=True)

    else:
        mark_assignments = (posinfo_at_mark_times['linvel_flat'].values>velthresh)   # get boolean of above thresh, convert to int
        pos_assignments = (pos_obj['linvel_flat'].values>velthresh)

    marks_obj['encoding_set'] = mark_assignments   # then add values
    pos_obj['encoding_set'] = pos_assignments

    return marks_obj, pos_obj

def shift_enc_marks_for_shuffle(marks_obj, shift):

    '''
    shift encoding marks by a fraction of total epoch time in order to break relationship between position and spikes
    returns a new marks object with marks shifted by a certain number of indices (shift_amount)

    Note that the shift amount is approximate; it is based on the # of spikes that happen in the original order (and this
    may be slightly different after the trials are reordered) but this should not be a problem

    '''
    encoding_marks_shifted = marks_obj.copy()   #deepcopy to create a new obj to fill with shifted values
    encoding_marks_shifted.reset_index(level=['time'],inplace=True)
    encoding_marks_shifted.sort_index(level='timestamp',inplace=True)
    min_time =encoding_marks_shifted['time'].min()
    max_time = encoding_marks_shifted['time'].max()
    epoch_length = max_time - min_time
    print('Total epoch time (sec): ',epoch_length)
    shift_target_time = min_time + shift * epoch_length
    print(shift_target_time)
    timediffs = np.abs(encoding_marks_shifted['time'].values-shift_target_time)
    shift_amount = np.argmin(timediffs)   # calculate rowindex of closest value of minimum time difference (sorted)
    print(shift_amount)
    encoding_marks_shifted['c00'] = np.roll(encoding_marks_shifted['c00'],shift_amount)
    encoding_marks_shifted['c01'] = np.roll(encoding_marks_shifted['c01'],shift_amount)
    encoding_marks_shifted['c02'] = np.roll(encoding_marks_shifted['c02'],shift_amount)
    encoding_marks_shifted['c03'] = np.roll(encoding_marks_shifted['c03'],shift_amount)

    encoding_marks_shifted.set_index(['time'], drop=True, append=True, inplace=True)

    return encoding_marks_shifted, shift_amount

def decode_with_classifier(likelihoods_obj, sungod_transmat, occupancy, discrete_tm_val=.99,velmask = None):
        
    '''
    '''
    # set up ingredients for classifier
    num_bins = sungod_transmat.shape[0]
    gapmask = np.ceil(np.nan_to_num(occupancy))
    ic_tmp = gapmask * np.ones((3,num_bins)) # in all bins, 0s in all gaps, then repeat over 3 rows
    initial_conditions = ic_tmp[:,:,np.newaxis]   # add on the extra 3rd dimension 
    trans_mat_dict = {'continuous':sungod_transmat, 
                      'identity':gapmask*np.eye(num_bins),
                      'uniform':gapmask*gapmask[:,np.newaxis]*np.ones((num_bins,num_bins))/sum(gapmask)}

    all_tms = [[trans_mat_dict['continuous'],trans_mat_dict['uniform'], trans_mat_dict['identity']],
         [trans_mat_dict['uniform'],trans_mat_dict['uniform'],trans_mat_dict['uniform']],
         [trans_mat_dict['continuous'],trans_mat_dict['uniform'], trans_mat_dict['identity']]]

    continuous_state_transition = np.stack(all_tms,axis=0)
    discrete_state_transition = strong_diagonal_discrete(3,discrete_tm_val)  # controls how sticky the discrete state trans is

    likelihoods = likelihoods_obj.drop(['num_spikes','dec_bin'],axis=1)

    # initialize output structures
    causal_state1 = likelihoods.copy()   # continuous
    causal_state2 = likelihoods.copy()   # fragmented
    causal_state3 = likelihoods.copy()  # hover 
    acausal_state1 = likelihoods.copy()
    acausal_state2 = likelihoods.copy()
    acausal_state3 = likelihoods.copy() 
    # nan out all the values
    causal_state1.loc[:] = 0   # continuous
    causal_state2.loc[:] = 0   # fragmented
    causal_state3.loc[:] = 0  # hover 
    acausal_state1.loc[:] = 0
    acausal_state2.loc[:] = 0
    acausal_state3.loc[:] = 0

    if velmask is None: 
        print('using first row nans')
        nanbins = np.isnan(likelihoods['x000'])  # use the first posbin to find nan time chunks
        blocks = (nanbins != nanbins.shift()).cumsum()   # define each group of non-nan values as a block
        immoblocks = np.unique(blocks.loc[~nanbins])

    else:
        print('using velmask')  #if a mask has been provided, use it to decide when to run. mask = 1 when vel>thresh; 0 otherwise
        blocks = (velmask != velmask.shift()).cumsum()
        immoblocks = np.unique(blocks.loc[~velmask])

    # run classifier
    for b in immoblocks:    #  iterate through non-nan blocks
        print('running valid block '+ str(b) +' of ' + str(max(blocks)))
        chunk = likelihoods.loc[blocks==b].fillna(0).values   # get rid of the gap nans
        chunk_expanded = chunk[:,np.newaxis,:,np.newaxis] * np.ones((1,3,num_bins,1))   # put things in the right shape
        causal_posterior = _causal_classify(initial_conditions, continuous_state_transition,discrete_state_transition,chunk_expanded)
        causal_state1.loc[blocks==b]=causal_posterior[:,0,:,0]   # store posteriors for each state separately bc df is 2d only
        causal_state2.loc[blocks==b]=causal_posterior[:,1,:,0]
        causal_state3.loc[blocks==b]=causal_posterior[:,2,:,0]

        acausal_posterior = _acausal_classify(causal_posterior,continuous_state_transition,discrete_state_transition)
        acausal_state1.loc[blocks==b]=acausal_posterior[:,0,:,0]   # store posteriors for each state separately bc df is 2d only
        acausal_state2.loc[blocks==b]=acausal_posterior[:,1,:,0]
        acausal_state3.loc[blocks==b]=acausal_posterior[:,2,:,0]

    return causal_state1, causal_state2, causal_state3, acausal_state1, acausal_state2, acausal_state3, trans_mat_dict
            
def calc_sungod_trans_mat(encode_settings, decode_settings, uniform_gain=-1, to_power=1):
    ''' this is basically a duplication of the one in pp_clusterless; 
    allows you to change parameters (such as uniform gain, AKA offset, directly)

    if uniform gain is explicitly provided, it will override the one set in dec_settings
    '''

    if uniform_gain==-1:   
        uniform_gain = decode_settings.trans_uniform_gain
    else:
        print('overriding uniform gain value to ' + str(uniform_gain))

    n = len(encode_settings.pos_bins)
    transition_mat = np.zeros([n,n])
    k = np.array([(1/3)*np.ones(n-1),(1/3)*np.ones(n),(1/3)*np.ones(n-1)])
    offset = [-1,0,1]
    transition_mat = sp.sparse.diags(k,offset).toarray()
    box_end_bin = encode_settings.arm_coordinates[0,1]
    for x in encode_settings.arm_coordinates[:,0]:
        transition_mat[int(x),int(x)] = (5/9)
        transition_mat[box_end_bin,int(x)] = (1/9)
        transition_mat[int(x),box_end_bin] = (1/9)
    for y in encode_settings.arm_coordinates[:,1]:
        transition_mat[int(y),int(y)] = (2/3)
    transition_mat[box_end_bin,0] = 0
    transition_mat[0,box_end_bin] = 0
    transition_mat[box_end_bin,box_end_bin] = 0
    transition_mat[0,0] = (2/3)
    transition_mat[box_end_bin-1, box_end_bin-1] = (5/9)
    transition_mat[box_end_bin-1,box_end_bin] = (1/9)
    transition_mat[box_end_bin, box_end_bin-1] = (1/9)

            # uniform offset (gain, currently 0.0001)
            # needs to be set before running the encoder cell
            # normally: decode_settings.trans_uniform_gain
    uniform_dist = np.ones(transition_mat.shape)*uniform_gain

            # apply uniform offset
    transition_mat = transition_mat + uniform_dist
    
            # apply no animal boundary - make gaps between arms (zeros)
    transition_mat = apply_no_anim_boundary(encode_settings.pos_bins, encode_settings.arm_coordinates, transition_mat)
    
            # to smooth: take the transition matrix to a power
    transition_mat = np.linalg.matrix_power(transition_mat,to_power)
    
            # normalize transition matrix - this turns the gaps into nans, so then turn them back to 0
    transition_mat = transition_mat/(transition_mat.sum(axis=0)[None, :])
    transition_mat[np.isnan(transition_mat)] = 0

    return transition_mat
