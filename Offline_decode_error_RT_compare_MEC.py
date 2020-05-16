# python script to run 1d clusterless decoder on sungod data using spykshrk
# also calculates offline and real-time decoding error
# also compares offline and real-time ripple detection
# also compares offline and real-time replay classification
# written by MEC from notebooks written by AKG and JG
# 4-1-20
# on LLNL, this script runs from the folder '/usr/workspace/wsb/coulter5/spykshrk_realtime/LLNL_run_scripts'

# SECTION 1: run offline decoder

#cell 1
# Setup and import packages
import sys
import os
#import pdb
from datetime import datetime, date

import numpy as np
import scipy as sp
import pandas as pd

import loren_frank_data_processing as lfdp
from loren_frank_data_processing import Animal

import trodes2SS
import sungod_util
import realtime_analysis_util

from spykshrk.franklab.data_containers import RippleTimes, pos_col_format, Posteriors

from spykshrk.franklab.pp_decoder.pp_clusterless import OfflinePPEncoder, OfflinePPDecoder

def main(path_base, rat_name, day, epoch, shift_amt, path_out, realtime_rec, 
         velthresh=4, use_enc_as_dec_flag=0, dec_all_flag=0,suffix = ''):
    print(datetime.now())
    today = str(date.today())

    #cell 2
    # Define parameters

    print(rat_name)
    print('Shift amount is: ',shift_amt)

    #day = 19
    #epoch = 2
    print('Day: ',day,'Epoch: ',epoch)

    # define data source filepaths
    path_base = path_base
    raw_directory = path_base + rat_name + '/filterframework/'
    #linearization_path = raw_directory + 'decoding/'   # need to update paths now that sungod_util doesn't add rat folder - add it here instead! 
    linearization_path = path_out
    day_ep = str(day) + '_' + str(epoch)

    #tetlist = None
    tetlist = [4]

    if tetlist is None:
        animalinfo  = {rat_name: Animal(directory=raw_directory, short_name=rat_name)}
        tetinfo = lfdp.tetrodes.make_tetrode_dataframe(animalinfo)
        tetrodes = tetinfo.query('area=="ca1" & day==@day & epoch==@epoch').index.get_level_values('tetrode_number').unique().tolist() 
    else:
        tetrodes= tetlist

    print('Tetrodes: ',tetrodes)

    pos_bin_size = 5
    velocity_thresh_for_enc_dec = velthresh
    velocity_buffer = 0

    print('Velocity thresh: ',velocity_thresh_for_enc_dec)

    shift_amt_for_shuffle = shift_amt

    use_enc_as_dec = use_enc_as_dec_flag
    decode_all = dec_all_flag

    discrete_tm_val=.98   # for classifier

    # IMPORT and process data

    #initialize data importer
    datasrc = trodes2SS.TrodesImport(raw_directory, rat_name, [day], [epoch], tetrodes)
    # Import marks
    marks = datasrc.import_marks()
    print('original length: '+str(marks.shape[0]))

    # fill in any deadchans with zeros
    #specific_tetinfo = tetinfo.query('tetrode_number==@tetrodes')  # pull the tetinfo for tets in list 
    #marks = datasrc.fill_dead_chans(marks, specific_tetinfo)

    # OPTIONAL: to reduce mark number, can filter by size. Current detection threshold is 100  
    marks = trodes2SS.threshold_marks(marks, maxthresh=2000,minthresh=300)
    # remove any big negative events (artifacts?)
    marks = trodes2SS.threshold_marks_negative(marks, negthresh=-999)
    print('after filtering: '+str(marks.shape[0]))

    # Import trials
    # there won't be any trials... or should we make an easy way to identify differnt trials???
    # we could write a specific phase out the statescript and then look for that in statescript log
    trials = datasrc.import_trials()

    # Import raw position 
    linear_pos_raw = datasrc.import_pos(xy='x')   # pull in xpos and speed, x will be replaced by linear
    posY = datasrc.import_pos(xy='y')          #  OPTIONAL; useful for 2d visualization

    # Import ripples
    rips_tmp = datasrc.import_rips(linear_pos_raw, velthresh=4) 
    rips = RippleTimes.create_default(rips_tmp,1)  # cast to rippletimes obj
    print('Rips less than velocity thresh: '+str(len(rips)))

    # Position linearization
    # if linearization exists, load it. if not, run the linearization.
    lin_output1 = os.path.join(linearization_path + rat_name + '_' + day_ep + '_' + 'linearized_distance.npy')
    lin_output2 = os.path.join(linearization_path + rat_name + '_' + day_ep + '_' + 'linearized_track_segments.npy')
    print('linearization file 1: ',lin_output1)
    if os.path.exists(lin_output1) == False:
        print('Linearization result doesnt exist. Doing linearization calculation!')
        nodepath = linearization_path+'remy_20_2_new_arm_nodes.mat'
        sungod_util.run_linearization_routine(rat_name, day, epoch, linearization_path, raw_directory, 
            gap_size=20, optional_alternate_nodes=nodepath) 
        linear_pos_raw['linpos_flat'] = np.load(lin_output1)   #replace x pos with linerized 
        track_segment_ids = np.load(lin_output2)
       
    else: 
        print('Linearization found. Loading it!')
        linear_pos_raw['linpos_flat'] = np.load(lin_output1)   #replace x pos with linerized 
        track_segment_ids = np.load(lin_output2)

    # generate boundary definitions of each segment
    arm_coords, _ = sungod_util.define_segment_coordinates(linear_pos_raw, track_segment_ids)  # optional addition output of all occupied positions (not just bounds)

    #bin linear position 
    binned_linear_pos, binned_arm_coords, pos_bins = sungod_util.bin_position_data(linear_pos_raw, arm_coords, pos_bin_size)

    # important for new arm nodes:
    binned_arm_coords[:,1] = 1+binned_arm_coords[:,1]

    # calculate bin coverage based on determined binned arm bounds   TO DO: prevent the annnoying "copy of a slice" error [prob need .values rather than a whole column]
    #pos_bin_delta = sungod_util.define_pos_bin_delta(binned_arm_coords, pos_bins, linear_pos_raw, pos_bin_size)
    pos_bin_delta = 1

    max_pos = binned_arm_coords[-1][-1]+1

    # cell 8
    # decide what to use as encoding and decoding data
    marks, binned_linear_pos = sungod_util.assign_enc_dec_set_by_velocity(binned_linear_pos, marks, velocity_thresh_for_enc_dec, velocity_buffer)

    # rearrange data by trials 
    pos_reordered, marks_reordered, order = sungod_util.reorder_data_by_random_trial_order(trials, binned_linear_pos, marks)

    encoding_marks = marks_reordered.loc[marks_reordered['encoding_set']==1]

    if decode_all==0:
        print('decoding marks set by use_enc_as_dec')
        decoding_marks = marks_reordered.loc[marks_reordered['encoding_set']==use_enc_as_dec]
    else: 
        print('decoding marks set to all marks')
        decoding_marks = marks_reordered  #use all of them

    #drop column of encoding/decoding mask - to speed up encoder
    encoding_marks.drop(columns='encoding_set',inplace=True)
    decoding_marks.drop(columns='encoding_set',inplace=True)

    print('Encoding spikes: '+str(len(encoding_marks)))
    print('Decoding spikes: '+str(len(decoding_marks)))

    encoding_pos = pos_reordered.loc[pos_reordered['encoding_set']==1]

    #explicity define decoding set - for nan mask
    if use_enc_as_dec:
        binned_linear_pos['decoding_set'] = binned_linear_pos['encoding_set']
    else:
        binned_linear_pos['decoding_set'] = ~binned_linear_pos['encoding_set']
    if decode_all:
        binned_linear_pos['decoding_set'] = True

    # apply shift for shuffling 
    encoding_marks_shifted, shift_amount = sungod_util.shift_enc_marks_for_shuffle(encoding_marks, shift_amt_for_shuffle)
    # put marks back in chronological order for some reason
    encoding_marks_shifted.sort_index(level='time',inplace=True)
    print('Marks index shift: ',shift_amount)
    print('Shifted marks shape: ', encoding_marks_shifted.shape)

    # cell 9
    # populate enc/dec settings. any parameter settable should be defined in parameter cell above and used here as a variable

    encode_settings = trodes2SS.AttrDict({'sampling_rate': 3e4,
                                    'pos_bins': np.arange(0,max_pos,1), # actually indices of valid bins. different from pos_bins above 
                                    'pos_bin_edges': np.arange(0,max_pos + .1,1), # indices of valid bin edges
                                    'pos_bin_delta': pos_bin_delta, 
                                    # 'pos_kernel': sp.stats.norm.pdf(arm_coords_wewant, arm_coords_wewant[-1]/2, 1),
                                    'pos_kernel': sp.stats.norm.pdf(np.arange(0,max_pos,1), max_pos/2, 1), #note that the pos_kernel mean should be half of the range of positions (ie 180/90)     
                                    'pos_kernel_std': 0, # 0 for histogram encoding model, 1+ for smoothing
                                    'mark_kernel_std': int(20), 
                                    'pos_num_bins': max_pos, 
                                    'pos_col_names': [pos_col_format(ii, max_pos) for ii in range(max_pos)], # or range(0,max_pos,10)
                                    'arm_coordinates': binned_arm_coords,   
                                    'spk_amp': 60,
                                    'vel': 0}) 

    decode_settings = trodes2SS.AttrDict({'trans_smooth_std': 2,
                                    'trans_uniform_gain': 0.0001,
                                    'time_bin_size':150})

    sungod_trans_mat = sungod_util.calc_sungod_trans_mat(encode_settings, decode_settings)

    print('Encode settings: ',encode_settings)
    print('Decode settings: ',decode_settings)

    #cell 10
    # Run encoder
    print('Starting encoder')
    time_started = datetime.now()
    print(len(encoding_marks_shifted))
    print(np.sum([np.dtype(dtype).itemsize for dtype in encoding_marks_shifted.dtypes]))
    print(np.dtype(dtype).itemsize for dtype in encoding_marks_shifted.dtypes)

    encoder = OfflinePPEncoder(linflat=encoding_pos, dec_spk_amp=decoding_marks, encode_settings=encode_settings, 
                               decode_settings=decode_settings, enc_spk_amp=encoding_marks_shifted, dask_worker_memory=1e9,
                               dask_chunksize = None)

    observ_obj = encoder.run_encoder()

    time_finished = datetime.now()

    print('Enocder finished!')
    print('Encoder started at: %s'%str(time_started))
    print('Encoder finished at: %s'%str(time_finished))

    #cell 15
    # Run PP decoding algorithm

    time_started = datetime.now()
    print('Starting decoder')
    decoder = OfflinePPDecoder(observ_obj=observ_obj, trans_mat=sungod_trans_mat, 
                               prob_no_spike=encoder.prob_no_spike,
                               encode_settings=encode_settings, decode_settings=decode_settings, 
                               time_bin_size=decode_settings.time_bin_size, all_linear_position=binned_linear_pos)

    posteriors = decoder.run_decoder()
    time_finished = datetime.now()
    print('Decoder finished!')
    print('Posteriors shape: '+ str(posteriors.shape))
    print('Decoder started at %s'%str(time_started))
    print('Decoder finished at %s'%str(time_finished))

    #cell 15.1
    # save posterior and linear position - netcdf
    posterior_file_name = os.path.join(path_out, rat_name + '_' + str(day) + '_' + str(epoch) + 
                                       '_shuffle_' + str(shift_amount) + '_posteriors'+ suffix + '.nc')
    posterior_file_name_h5 = os.path.join(path_out, rat_name + '_' + str(day) + '_' + str(epoch) + 
                                          '_shuffle_' + str(shift_amount) + '_posteriors' + suffix + '.h5')

    post1 = posteriors.copy()
    post2 = post1.apply_time_event(rips, event_mask_name='ripple_grp')
    post3 = post2.reset_index()
    post4 = trodes2SS.convert_dan_posterior_to_xarray(post3, tetrodes, velocity_thresh_for_enc_dec, 
                                                      encode_settings, decode_settings, sungod_trans_mat, 
                                                      order, shift_amount)
    post4.to_netcdf(posterior_file_name)
    print('Saved netcdf posteriors to '+posterior_file_name)

    # to export linearized position to MatLab: again convert to xarray and then save as netcdf

    position_file_name = os.path.join(path_out, rat_name + '_' + str(day) + '_' + 
                                      str(epoch) + '_shuffle_' + str(shift_amount) + 
                                      '_linearposition_v2'+suffix+'.nc')

    linearized_pos0 = binned_linear_pos.copy()
    linearized_pos1 = linearized_pos0.apply_time_event(rips, event_mask_name='ripple_grp')
    linearized_pos2 = linearized_pos1.reset_index()
    linearized_pos3 = linearized_pos2.to_xarray()
    linearized_pos3.to_netcdf(position_file_name)
    print('Saved netcdf linearized position to '+position_file_name)

    # save offline decoder result as hdf5 file
    post_to_save = posteriors.copy()
    post_to_save1 = post_to_save.apply_time_event(rips, event_mask_name='ripple_grp')
    post_to_save1._to_hdf_store(posterior_file_name_h5,'/analysis', 
                               'decode/clusterless/offline/posterior', 
                               'sungod_trans_mat', overwrite=True)
    print('Saved posteriors to '+posterior_file_name_h5)

    #cell 16
    ## run replay classifier
    # removed this cell

    ## to calculate histogram of posterior max position in each time bin
    #hist_bins = []
    #post_hist1 = posteriors.drop(['num_spikes','dec_bin','ripple_grp'], axis=1)
    #post_hist1.fillna(0,inplace=True)
    #post_hist3 = post_hist1.idxmax(axis=1)
    #post_hist3 = post_hist3.str.replace('x','')
    #post_hist3 = post_hist3.astype(int)
    ##print(post_hist3.shape)
    #hist_bins = np.histogram(post_hist3,bins=np.arange(0,147))
    ##print(hist_bins)
    #unique, counts = np.unique(post_hist3, return_counts=True)
    #print(dict(zip(unique,counts)))

    print("End of offline decoding!")

    # SECTION 2: calculate offline decoding error
    
    #calculate posterior max
    post_to_merge = posteriors.copy()
    post_to_merge1 = post_to_merge.drop(['num_spikes','dec_bin'], axis=1)
    post_to_merge1.fillna(0,inplace=True)
    post_to_merge1['posterior_max'] = post_to_merge1.idxmax(axis=1)
    post_to_merge1['posterior_max'] = post_to_merge1['posterior_max'].str.replace('x','')
    post_to_merge1['posterior_max'] = post_to_merge1['posterior_max'].astype(int)

    post_to_merge1.reset_index(inplace=True)
    post_to_merge1['timestamp1']=post_to_merge1['timestamp']

    # offline pos
    pos_vel = binned_linear_pos.copy()
    pos_vel1 = pos_vel.reset_index()
    pos_vel1['timestamp1'] = pos_vel1['timestamp']
    pos_vel2 = pos_vel1.drop(['day','epoch','time','timestamp'], axis=1)

    posterior_with_pos_vel = pd.merge_asof(post_to_merge1,pos_vel2,on='timestamp1',direction='nearest')
    posterior_with_pos_vel['error_cm'] = abs(posterior_with_pos_vel['posterior_max']-posterior_with_pos_vel['linpos_flat'])*5
    posterior_with_pos_vel

    post_error_plot_off = posterior_with_pos_vel.copy()

    # remote error
    print('calulating offline decoding error')
    # only calculate error during movement
    error_off = post_error_plot_off.copy()
    error_off = error_off[error_off['linvel_flat']>velthresh]
    print(error_off.shape)

    realtime_analysis_util.non_local_error(error_off,binned_arm_coords)
    
    # local error
    realtime_analysis_util.local_error(error_off,binned_arm_coords)

    # SECTION 3: calculate real-time decoding error

    # load real-time merged rec file
    realtime_rec_file = path_out+realtime_rec
    store_rt = pd.HDFStore(realtime_rec_file, mode='r')

    encoder_data = store_rt['rec_3']
    decoder_data = store_rt['rec_4']
    decoder_missed_spikes = store_rt['rec_5']
    likelihood_data = store_rt['rec_6']
    occupancy_data = store_rt['rec_7']
    #ripple_data1 = store_rt['rec_1']
    #stim_state = store_rt['rec_10']
    stim_lockout = store_rt['rec_11']
    stim_message = store_rt['rec_12']
    #timing1 = store_rt['rec_100']

    print('loaded realtime rec file:',realtime_rec)

    # calculate max for real-time posteriors
    post_error = decoder_data.copy()

    post_error.drop(columns=['rec_ind','bin_timestamp','wall_time','velocity','real_pos',
                             'raw_x','raw_y','smooth_x','smooth_y','next_bin',
                             'spike_count','ripple','ripple_number','ripple_length',
                             'shortcut_message','box','arm1','arm2','arm3','arm4','arm5',
                             'arm6','arm7','arm8'], inplace=True)
    post_error.fillna(0,inplace=True)
    post_error['posterior_max'] = post_error.idxmax(axis=1)
    post_error['posterior_max'] = post_error['posterior_max'].str.replace('x','')
    post_error['posterior_max'] = post_error['posterior_max'].astype(int)

    #now need to add back columns 'timestamp','real_pos_time','real_pos'.'spike_count'
    post_error['timestamp'] = decoder_data['bin_timestamp']
    post_error['linvel_flat'] = decoder_data['velocity']
    post_error['linpos_flat'] = decoder_data['real_pos']
    post_error['spike_count'] = decoder_data['spike_count']
    #this is the error column in centimeters
    post_error['error_cm'] = abs(post_error['posterior_max']-decoder_data['real_pos'])*5

    # arm_coords real-time
    binned_arm_coords_realtime = [[0,8],[13,24],[29,40],[45,56],[61,72],[77,88],[93,104],[109,120],[125,136]]

    # real-time remote error
    print('calulating real-time decoding error')
    error_realtime = post_error.copy()
    error_realtime = error_realtime[error_realtime['linvel_flat']>5]
    print(error_realtime.shape)

    realtime_analysis_util.non_local_error(error_realtime,binned_arm_coords_realtime)
  
    # real-time local error
    realtime_analysis_util.non_local_error(error_realtime,binned_arm_coords_realtime)

    # SECTION 4: calculate offline and real-time ripple matching
    # specific to remy day 20, epoch 2: to match offline, add 1958987 to realtime timestamps

    # choose files
    stim_lockout_file = stim_lockout

    realtime_rips = stim_lockout_file[stim_lockout_file['lockout_state']==1]
    print('realtime rips:',realtime_rips.shape)
    offline_rips = rips.reset_index()
    print('offline rips:',offline_rips.shape)
    offline_rips['adj_timestamp'] = offline_rips['timestamp']
    realtime_rips['adj_timestamp'] = realtime_rips['timestamp']+1958987

    realtime_rips
    # merge real-time and offline ripples 
    # offline ripple start is 50-100 msec before real-time
    # try a tolerance of 300 msec = 9000 timestamps
    # one problem is that we are only matching to the start of the offline ripple - but long ripples will get split in 2
    # so we really want to match over the whole interval of the offline ripple
    merged_ripple_times = pd.merge_asof(realtime_rips,offline_rips,on='adj_timestamp',tolerance=9000,direction='backward')
    matching_offline_rips = merged_ripple_times[merged_ripple_times['lockout_num']>0]

    # offline ripple duration
    matching_offline_rips['off_dur'] = matching_offline_rips['endtime']-matching_offline_rips['starttime']

    # note: still a few duplicates - this is coming from the original list of realtime ripples (stim_lockout)

    # save mismatched ripples separately
    nonmatching_offline_rips = matching_offline_rips[matching_offline_rips['day'].isnull()]
    # now need to remove ripples that dont match
    matching_offline_rips = matching_offline_rips[matching_offline_rips['day']>0]

    # 2 real-time rips match 1 offline rip
    matching_offline_rips['two_RT_rips'] = matching_offline_rips['event'].diff()
    # only keep rows where diff > 0 - this will remove double matches, specifically the second real-time rip
    # matching_offline_rips = matching_offline_rips[matching_offline_rips['two_RT_rips']>0]

    print('matching rips:',matching_offline_rips.shape)

    # ripple detection delay for matching rips
    ripple_delay_rt, ripple_delay_rt_bins = np.histogram((matching_offline_rips['adj_timestamp'].values-
                                                          matching_offline_rips['timestamp_y'].values)/30,
                                                          bins=np.arange(0,300,10))
    print('ripple delay:',ripple_delay_rt)
    print('ripple delay bins:',ripple_delay_rt_bins)
       
    # SECTION 5: calculate offline and real-time replay arm matching

    # merge real-time decoder output and stim messages
    # for stim_message this merge ('nearest') put the arm number and ripple number at all timebins after the end of the rippple
    # for stim_lockout this merge ('backward') will highlight the ripple time

    # which files to use
    stim_message_file = stim_message
    stim_lockout_file = stim_lockout
    decoder_data_file = decoder_data

    stim_message_1 = stim_message_file.copy()
    stim_message_2 = stim_message_1.drop(['rec_ind','spike_timestamp','time','lfp_timestamp',
                                       'ripple_time_bin','spike_count',
                                       'content_threshold','max_arm_repeats','box','arm1','arm2',
                                       'arm3','arm4','arm5','arm6','arm7','arm8'], axis=1)
    
    decode_to_merge = decoder_data_file.copy()
    merged_decoder_stim = pd.merge_asof(decode_to_merge,stim_message_2,on='bin_timestamp',direction='nearest')

    stim_lockout_1 = stim_lockout_file.copy()
    stim_lockout_1['timestamp_shift'] = stim_lockout_file['timestamp']
    stim_lockout_1['bin_timestamp'] = stim_lockout_1['timestamp_shift']

    stim_lockout_2 = stim_lockout_1.drop(['rec_ind','timestamp','time','tets_above_thresh',
                                          'big_rip_message_sent'], axis=1)
    stim_lockout_2
    merged_decoder_lockout = pd.merge_asof(merged_decoder_stim,stim_lockout_2,on='bin_timestamp',direction='backward')
    print('merged decoder/lockout shape:',merged_decoder_lockout.shape)

    # record replay arm classification result for each real-time ripple

    # for looping, use stim_lockout
    realtime_posterior_sum_all = np.zeros((len(stim_lockout_file[stim_lockout_file['lockout_state']==1]),4))

    counter = -1
    summarize_all_rips = True

    for timestamp in stim_lockout_file[stim_lockout_file['lockout_state']==1]['timestamp'].values:
        counter += 1
        #print(timestamp)
        if summarize_all_rips:
        #print(timestamp-30*300,timestamp+30*300)
        #posterior from decode/stim message merged table
            merged_to_plot = merged_decoder_lockout[(merged_decoder_lockout['bin_timestamp'] > timestamp-30*300) & 
                                                (merged_decoder_lockout['bin_timestamp'] < timestamp+30*300)]
            merged_to_plot.set_index('bin_timestamp',inplace=True)

            posterior_only_merged = merged_to_plot.drop(['rec_ind','wall_time','velocity','real_pos',
                                                    'spike_count','ripple','ripple_length','timestamp_shift',
                                                    'shortcut_message','box','arm1','arm2','arm3','arm4','arm5',
                                                    'arm6','arm7','arm8','posterior_max_arm','shortcut_message_sent',
                                                    'raw_x','raw_y','smooth_x','smooth_y','ripple_number_x',
                                                    'ripple_number_y','ripple_end','lockout_state','lockout_num',
                                                    'delay','next_bin'], axis=1)
            # ripple time - generated from lockout_state
            # note: this does not inlcude the 70 msec before ripple detection
            # lockout_state changes at posterior_lock end, so this should post_lock (now 50 msec)
            ripples_to_plot = merged_to_plot.reset_index()
            ripple_times_rt = ripples_to_plot.index[ripples_to_plot['lockout_state'] > 0].tolist()

            # get timestamp when shortcut message was sent - try to just isolate single ripple
            # bin_timestamp will show the delay - it will appear before the start of the ripple
            # in contrast, lfp_timestamp would line up exactly with ripple start
            shortcut_message_to_plot = stim_message_file[(stim_message_file['lfp_timestamp'] > timestamp-30*30) & 
                                                 (stim_message_file['lfp_timestamp'] < timestamp+30*100)]
            merged_to_plot_index = merged_to_plot.reset_index()
            # loop through multiple entries in shortcut_message
            shortcut_message_times = np.zeros(shortcut_message_to_plot.shape[0])
            for i in np.arange(0,shortcut_message_to_plot.shape[0]):
                shortcut_message_times[i] = merged_to_plot_index.index[merged_to_plot_index['bin_timestamp'].values == shortcut_message_to_plot['bin_timestamp'][i:i+1].values].tolist()[0]

            #shortcut_message_times = merged_to_plot_index.index[merged_to_plot_index['bin_timestamp'].values == shortcut_message_to_plot['bin_timestamp'].values].tolist()
            shortcut_messages = shortcut_message_to_plot['shortcut_message_sent'].values*123
        
            #plot title: include ripple number, max arm, and delay
            title_index = int(len(merged_to_plot)*0.6)
            max_arm = merged_to_plot[title_index:title_index+1]['posterior_max_arm'].values
            ripple_num = merged_to_plot[title_index:title_index+1]['lockout_num'].values
            message_delay = np.around(merged_to_plot[title_index:title_index+1]['delay'].values,decimals=0)
            #offline_max = non_matching[(non_matching['realtime_rip']>=counter) & 
            #                           (non_matching['realtime_rip']<counter+1)]['off_max_arm'].values

            # fill in current row of posterior sum array - seems to work
            realtime_posterior_sum_all[counter,0] = ripple_num
            realtime_posterior_sum_all[counter,1] = max_arm
            realtime_posterior_sum_all[counter,2] = shortcut_message_to_plot.shape[0]-1
            realtime_posterior_sum_all[counter,3] = message_delay
        
    # convert realtime_post_sum_summary array to pandas
    realtime_post_sum_summary = pd.DataFrame(data=realtime_posterior_sum_all,columns=('realtime_rip','rt_max_arm',
                                                                                      'rt_two_messages','rt_delay'))
    print('realtime replay summary shape:',realtime_post_sum_summary.shape)

    # calculate replay arm classification result for each offline ripple
    # running sum version: 50 msec (11 bins)
    # offline posteriors, 150 uV threshold

    # offline arm coords: new linearization
    arm_coords = [[0,8],[13,25],[29,41],[45,57],[62,74],[78,90],[94,106],[111,123],[127,139]]

    summarize_all_rips_offline = True

    # add ripple to posteriors dataframe
    posteriors1 = posteriors.copy()
    posteriors2 = posteriors1.apply_time_event(rips, event_mask_name='ripple_grp')    
    posterior_offline = posteriors2.reset_index()
    offline_pos = binned_linear_pos.reset_index()
    merged_off_post_pos = pd.merge_asof(posterior_offline,offline_pos,on='timestamp',direction='nearest')
    offline_posterior_sum_all = np.zeros((len(matching_offline_rips),14))

    # number of bins for sliding window sum
    sliding_window = 11

    # updated for new matching_offline_rips merge timestamp_x -> timestamp_y
    # for some reason, final offline rip timestamp is after the decoder has ended - odd
    for index, rip_timestamp in enumerate(matching_offline_rips['timestamp_y'][0:940]):
        posterior_sum_array = np.zeros((sliding_window,9))
        short_ripple = False
        # to plot all ripples
        if summarize_all_rips_offline:
            #print(rip_timestamp-30*300,rip_timestamp+30*300,index)
            
            posterior_to_plot = merged_off_post_pos[(merged_off_post_pos['timestamp'] > rip_timestamp-30*300) & 
                                                (merged_off_post_pos['timestamp'] < rip_timestamp+30*300)]

            realtime_ripple_num = matching_offline_rips['lockout_num'][index:index+1].values
            ripple_num_index = int(len(posterior_to_plot)*0.55)
            ripple_num = posterior_to_plot[ripple_num_index:ripple_num_index+1]['ripple_grp'].values
            if index > 920:
                print(index, rip_timestamp)
        
            # calculate posterior sum during ripple
            # we need to only take out the time when ripple_grp matches ripple_grp at the middle of the plotting bin
        
            #post_sum_times = posterior_to_plot[posterior_to_plot['ripple_grp'] > 0]
            post_sum_times = posterior_to_plot[posterior_to_plot['ripple_grp'] == ripple_num[0]]
            ripple_length = post_sum_times.shape[0]
            
            # if ripple is less than 12 bins: sum whole ripple
            if ripple_length < 12:
                # sum each arm over whole ripple
                post_sum_ripple = np.zeros((ripple_length,9))
                for i in np.arange(0,ripple_length):
                    if i == 0:
                        for j in np.arange(0,len(arm_coords),1):
                            post_sum_ripple[i,j] = post_sum_times.iloc[i,4:150].values[arm_coords[j][0]:arm_coords[j][1]].sum()
                    else:
                        for j in np.arange(0,len(arm_coords),1):
                            post_sum_ripple[i,j] = post_sum_ripple[i-1,j] + post_sum_times.iloc[i,4:150].values[arm_coords[j][0]:arm_coords[j][1]].sum()
                
                # normalize sum of whole ripple - this is the final row
                short_ripple = True
                post_sum_ripple[i] = post_sum_ripple[i]/post_sum_ripple[i].sum()
                posterior_sum = np.around(post_sum_ripple[-1:],decimals=1)
                posterior_sum = posterior_sum[0]
        
                if len(np.argwhere(posterior_sum>=0.5)):
                    arm_max = np.argwhere(posterior_sum>=0.5)[0][0]
                else:
                    arm_max = 99 
                    
            # if ripple 12 or more bins: use sliding window sum
            else:
                #print('long_ripple',ripple_length)
                stop_post_sum = False
                post_sum_ripple = np.zeros((ripple_length,9))
                for i in np.arange(0,ripple_length):
                    # first sum across arms
                    for j in np.arange(0,len(arm_coords),1):
                        post_sum_ripple[i,j] = post_sum_times.iloc[i,4:150].values[arm_coords[j][0]:arm_coords[j][1]].sum()
                    # then fill in sliding window array
                    posterior_sum_array[np.mod(i,sliding_window),:] = post_sum_ripple[i]
                    sum_array_sum = np.sum(posterior_sum_array,axis=0)
                    norm_posterior_arm_sum = sum_array_sum/sliding_window
                    if i > 10 and not stop_post_sum:
                        posterior_sum = np.around(norm_posterior_arm_sum,decimals=1)
                        #print(posterior_sum)
                        #posterior_sum = posterior_sum[0]
                        if len(np.argwhere(posterior_sum>=0.5)):
                            arm_max = np.argwhere(posterior_sum>=0.5)[0][0]
                        else:
                            arm_max = 99
                        if arm_max > 0 and arm_max < 9:
                            stop_post_sum = True
                            #print('post sum meets criteria. bin',i,'arm',arm_max)
            
            ripple_times = posterior_to_plot.index[posterior_to_plot['ripple_grp'] > 0].tolist()

            posterior_offline1 = posterior_to_plot.drop(['day_x','epoch_x','timestamp','time_x','num_spikes','dec_bin',
                                                     'ripple_grp','day_y','epoch_y','time_y','linpos_flat',
                                                     'linvel_flat'], axis=1)

            posterior_offline2 = posterior_offline1.fillna(0)

            # fill in current row of posterior sum array - seems to work
            offline_posterior_sum_all[index,0] = ripple_num
            offline_posterior_sum_all[index,1] = realtime_ripple_num
            offline_posterior_sum_all[index,2] = arm_max
            offline_posterior_sum_all[index,3] = posterior_sum[0]
            offline_posterior_sum_all[index,4] = posterior_sum[1]
            offline_posterior_sum_all[index,5] = posterior_sum[2]
            offline_posterior_sum_all[index,6] = posterior_sum[3]
            offline_posterior_sum_all[index,7] = posterior_sum[4]
            offline_posterior_sum_all[index,8] = posterior_sum[5]
            offline_posterior_sum_all[index,9] = posterior_sum[6]
            offline_posterior_sum_all[index,10] = posterior_sum[7]
            offline_posterior_sum_all[index,11] = posterior_sum[8]
            offline_posterior_sum_all[index,12] = ripple_length
            offline_posterior_sum_all[index,13] = short_ripple
                                         
    # convert offline_posterior_sum_all array to pandas
    off_post_sum_summary = pd.DataFrame(data=offline_posterior_sum_all,columns=('offline_rip','realtime_rip',
                                                                                'off_max_arm','box','arm1','arm2',
                                                                                'arm3','arm4','arm5','arm6',
                                                                                'arm7','arm8',
                                                                                'rip_length','short_rip'))

    print('offline replay summary shape:',off_post_sum_summary.shape)

    # match replay arm classification between offline and real-time
    replay_summary_combined = pd.DataFrame.join(off_post_sum_summary,realtime_post_sum_summary,on='realtime_rip',
                                                how='outer',lsuffix='off',rsuffix='rt')
    replay_combined_matching = replay_summary_combined[replay_summary_combined['offline_rip']>0]
    #replay_combined_matching = replay_combined_matching[replay_combined_matching['rt_two_messages']>0]
    #replay_summary_combined['realtime_ripoff'].values
    print('combined replays shape:',replay_combined_matching.shape)

    #summarize matching between offline and realtime
    print('total matching:',replay_combined_matching.shape[0])

    #exact match
    replay_exact_match = (replay_combined_matching[replay_combined_matching['off_max_arm'].values == replay_combined_matching['rt_max_arm'].values]).shape[0]
    print('exact match:',replay_exact_match)

    # non-matching
    non_matching = replay_combined_matching[replay_combined_matching['off_max_arm'].values != replay_combined_matching['rt_max_arm'].values]
    print('non-matching:',non_matching.shape[0])
    # count for each arm in realtime replays
    print('realtime below 0.5:',non_matching[non_matching['rt_max_arm'] == 99].shape[0])
    print('realtime box:',non_matching[non_matching['rt_max_arm'] == 0].shape[0])
    print('realtime arm 1:',non_matching[non_matching['rt_max_arm'] == 1].shape[0])
    print('realtime arm 2:',non_matching[non_matching['rt_max_arm'] == 2].shape[0])
    print('realtime arm 3:',non_matching[non_matching['rt_max_arm'] == 3].shape[0])
    print('realtime arm 4:',non_matching[non_matching['rt_max_arm'] == 4].shape[0])
    print('realtime arm 5:',non_matching[non_matching['rt_max_arm'] == 5].shape[0])
    print('realtime arm 6:',non_matching[non_matching['rt_max_arm'] == 6].shape[0])
    print('realtime arm 7:',non_matching[non_matching['rt_max_arm'] == 7].shape[0])
    print('realtime arm 8:',non_matching[non_matching['rt_max_arm'] == 8].shape[0])

    # offline: no arm above 0.5
    print('offline < 0.5, mismatch total:',non_matching[non_matching['off_max_arm'] == 99].shape[0])
    print('offline < 0.5, realtime box:',non_matching[(non_matching['off_max_arm'] == 99) & (non_matching['rt_max_arm'] == 0)].shape[0])
    print('offline < 0.5, realtime other arm:',non_matching[(non_matching['off_max_arm'] == 99) & (non_matching['rt_max_arm'] != 0)].shape[0])

    # offline: box
    print('offline box, mismatch total:',non_matching[non_matching['off_max_arm'] == 0].shape[0])
    print('offline box, realtime < 0.5:',non_matching[(non_matching['off_max_arm'] == 0) & (non_matching['rt_max_arm'] == 99)].shape[0])
    print('offline box, realtime other arm:',non_matching[(non_matching['off_max_arm'] == 0) & (non_matching['rt_max_arm'] != 99)].shape[0])

    # offline: outer arm
    print('offline arm, mismatch total:',non_matching[(non_matching['off_max_arm'] > 0)&(non_matching['off_max_arm'] < 10)].shape[0])
    print('offline arm, realtime < 0.5:',non_matching[(non_matching['off_max_arm'] > 0)&(non_matching['off_max_arm'] < 10) & (non_matching['rt_max_arm'] == 99)].shape[0])
    print('offline arm, realtime box:',non_matching[(non_matching['off_max_arm'] > 0)&(non_matching['off_max_arm'] < 10) & (non_matching['rt_max_arm'] == 0)].shape[0])
    print('offline arm, realtime other arm:',non_matching[(non_matching['off_max_arm'] > 0)&(non_matching['off_max_arm'] < 10) & (non_matching['rt_max_arm'] > 0)&(non_matching['rt_max_arm'] < 10)].shape[0])

    # calculate summary statistics based on these values
    ### TO DO: just use the same summary stats you have in the lab notebook
    print('matching fraction:',np.around(replay_exact_match/replay_combined_matching.shape[0],decimals = 3))
    print('box or less 0.5:',np.around((non_matching[(non_matching['off_max_arm'] == 0) & (non_matching['rt_max_arm'] == 99)].shape[0] +
                                        non_matching[(non_matching['off_max_arm'] == 99) & (non_matching['rt_max_arm'] == 0)].shape[0])/
                                        replay_combined_matching.shape[0],decimals = 3))
    print('false negative:',np.around((non_matching[(non_matching['off_max_arm'] > 0)&(non_matching['off_max_arm'] < 10) & (non_matching['rt_max_arm'] == 99)].shape[0] + 
                                       non_matching[(non_matching['off_max_arm'] > 0)&(non_matching['off_max_arm'] < 10) & (non_matching['rt_max_arm'] == 0)].shape[0])/
                                       replay_combined_matching.shape[0],decimals = 3))
    print('false positive:',np.around((non_matching[(non_matching['off_max_arm'] == 99) & (non_matching['rt_max_arm'] != 0)].shape[0] + 
                                       non_matching[(non_matching['off_max_arm'] == 0) & (non_matching['rt_max_arm'] != 99)].shape[0] + 
                                       non_matching[(non_matching['off_max_arm'] > 0)&(non_matching['off_max_arm'] < 10) & (non_matching['rt_max_arm'] > 0)&(non_matching['rt_max_arm'] < 10)].shape[0])/
                                       replay_combined_matching.shape[0],decimals = 3))

    print("End of script!")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', action='store', dest='path_base', help='Base path')
    parser.add_argument('-n', action='store', dest='rat_name', help='Rat Name')
    parser.add_argument('-d', action='store', dest='day', type=int, help='Day')
    parser.add_argument('-e', action='store', dest='epoch', type=int, help='Epoch')
    parser.add_argument('-s', action='store', dest='shift_amt', type=float, help='Shift amount')
    parser.add_argument('-o', action='store', dest='path_out', help='Path to output')
    parser.add_argument('-r', action='store', dest='realtime_rec', help='Real-time rec filename')    
    results = parser.parse_args()
    main(results.path_base,results.rat_name,results.day,results.epoch,results.shift_amt,results.path_out,results.realtime_rec)

