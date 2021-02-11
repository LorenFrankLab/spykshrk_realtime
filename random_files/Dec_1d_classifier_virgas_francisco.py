# python script to run 1d clusterless decoder on sungod data using spykshrk
# written by MEC from notebooks written by AKG and JG
# 7-21-19
# on LLNL, this script runs from the folder '/usr/workspace/wsb/coulter5/spykshrk_realtime/LLNL_run_scripts'

#cell 1
# Setup and import packages
import sys
import os
#import pdb
from datetime import datetime, date

import numpy as np
import scipy as sp

import loren_frank_data_processing as lfdp
from loren_frank_data_processing import Animal

import trodes2SS
import sungod_util

from spykshrk.franklab.data_containers import RippleTimes, pos_col_format, Posteriors

from spykshrk.franklab.pp_decoder.pp_clusterless import OfflinePPEncoder, OfflinePPDecoder

def main(path_base, rat_name, day, epoch, shift_amt, path_out):
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
    linearization_path = raw_directory + 'decoding/'   # need to update paths now that sungod_util doesn't add rat folder - add it here instead! 
    day_ep = str(day) + '_' + str(epoch)

    tetlist = None
    #tetlist = [4]

    if tetlist is None:
        animalinfo  = {rat_name: Animal(directory=raw_directory, short_name=rat_name)}
        tetinfo = lfdp.tetrodes.make_tetrode_dataframe(animalinfo)
        tetinfo['ndtype'] = tetinfo['deadchans'].apply(lambda d: isinstance(d,np.ndarray)) # add column with datatype of deadchans entry
        tmp = tetinfo['deadchans'][tetinfo['ndtype'].values].apply(lambda d: len(d))   # add length of deadchans list
        tetinfo['ndlength'] = tmp   # store lengths as an additional column. no dead chans = length 0 
        tetrodes = tetinfo.query('area=="ca1" & ndlength==0 & day==@day & epoch==@epoch').index.get_level_values('tetrode_number').unique().tolist()   
    else:
        tetrodes= tetlist

    print('Tetrodes: ',tetrodes)

    pos_bin_size = 5
    velocity_thresh_for_enc_dec = 4
    velocity_buffer = 0

    shift_amt_for_shuffle = shift_amt

    use_enc_as_dec = 0

    discrete_tm_val=.99   # for classifier

    # IMPORT and process data

    #initialize data importer
    datasrc = trodes2SS.TrodesImport(raw_directory, rat_name, [day], [epoch], tetrodes)
    # Import marks
    marks = datasrc.import_marks()
    print('original length: '+str(marks.shape[0]))
    # OPTIONAL: to reduce mark number, can filter by size. Current detection threshold is 100  
    marks = trodes2SS.threshold_marks(marks, maxthresh=2000,minthresh=100)
    # remove any big negative events (artifacts?)
    marks = trodes2SS.threshold_marks_negative(marks, negthresh=-999)
    print('after filtering: '+str(marks.shape[0]))

    # Import trials
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
        sungod_util.run_linearization_routine(rat_name, day, epoch, linearization_path, raw_directory, gap_size=20)
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

    # calculate bin coverage based on determined binned arm bounds   TO DO: prevent the annnoying "copy of a slice" error [prob need .values rather than a whole column]
    pos_bin_delta = sungod_util.define_pos_bin_delta(binned_arm_coords, pos_bins, linear_pos_raw, pos_bin_size)

    max_pos = binned_arm_coords[-1][-1]+1

    # cell 8
    # decide what to use as encoding and decoding data
    marks, binned_linear_pos = sungod_util.assign_enc_dec_set_by_velocity(binned_linear_pos, marks, velocity_thresh_for_enc_dec, velocity_buffer)

    # rearrange data by trials 
    pos_reordered, marks_reordered, order = sungod_util.reorder_data_by_random_trial_order(trials, binned_linear_pos, marks)

    encoding_marks = marks_reordered.loc[marks_reordered['encoding_set']==1]
    decoding_marks = marks_reordered.loc[marks_reordered['encoding_set']==use_enc_as_dec]

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
                                    'time_bin_size':60})

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

    time_finished =datetime.now()

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
    time_finished =datetime.now()
    print('Decoder finished!')
    print('Posteriors shape: '+ str(posteriors.shape))
    print('Decoder started at %s'%str(time_started))
    print('Decoder finished at %s'%str(time_finished))

    #cell 15.1
    # save posterior and linear position - netcdf
    posterior_file_name = os.path.join(path_out,  rat_name + '_' + str(day) + '_' + str(epoch) + '_shuffle_' + str(shift_amount) + '_posteriors_functionalized.nc')

    post1 = posteriors.apply_time_event(rips, event_mask_name='ripple_grp')
    post2 = post1.reset_index()
    post3 = trodes2SS.convert_dan_posterior_to_xarray(post2, tetrodes, 
                                            velocity_thresh_for_enc_dec, encode_settings, decode_settings, sungod_trans_mat, order, shift_amount)
    post3.to_netcdf(posterior_file_name)
    print('Saved netcdf posteriors to '+posterior_file_name)

    # to export linearized position to MatLab: again convert to xarray and then save as netcdf

    position_file_name = os.path.join(path_out, rat_name + '_' + str(day) + '_' + str(epoch) + '_shuffle_' + str(shift_amount) + '_linearposition_functionalized.nc')

    linearized_pos1 = binned_linear_pos.apply_time_event(rips, event_mask_name='ripple_grp')
    linearized_pos2 = linearized_pos1.reset_index()
    linearized_pos3 = linearized_pos2.to_xarray()
    linearized_pos3.to_netcdf(position_file_name)
    print('Saved netcdf linearized position to '+position_file_name)

    #cell 15.2
    # save posterior as hdf5 
    posterior_file_name_hdf5 = os.path.join(path_out,  rat_name + '_' + str(day) + '_' + str(epoch) + '_shuffle_' + str(shift_amount) + '_posteriors_functionalized.h5')
    posteriors._to_hdf_store(posterior_file_name_hdf5,'/analysis', 'decode/clusterless/offline/posterior', 'sungod_trans_mat',overwrite=True)
    print('Saved hdf5 posteriors to '+posterior_file_name_hdf5)

    #cell 16
    ## run replay classifier
    time_started = datetime.now()
    ## continuous trans_mat has small offset = 0 (no jumping)
    sungod_no_offset = sungod_util.calc_sungod_trans_mat(encode_settings, decode_settings, uniform_gain=0)
    causal_state1, causal_state2, causal_state3, acausal_state1, acausal_state2, acausal_state3, trans_mat_dict = sungod_util.decode_with_classifier(decoder.likelihoods, sungod_no_offset, encoder.occupancy, discrete_tm_val)
    time_finished = datetime.now()

    print('Classifier started at %s'%str(time_started))
    print('Classifier finished at %s'%str(time_finished))
    
    #cell 17
    ## save classifier output
    base_name = os.path.join(path_out, rat_name + '_' + day_ep + '_shuffle_' + str(shift_amount) + '_posterior_')
    fname = 'causal'
    trodes2SS.convert_save_classifier(base_name, fname, causal_state1, causal_state2, causal_state3, tetrodes, decoder.likelihoods,
                                      encode_settings, decode_settings, rips, velocity_thresh_for_enc_dec, velocity_buffer, sungod_no_offset, order, shift_amount)

    fname = 'acausal'
    trodes2SS.convert_save_classifier(base_name, fname, acausal_state1, acausal_state2, acausal_state3, tetrodes, decoder.likelihoods,
                                      encode_settings, decode_settings, rips, velocity_thresh_for_enc_dec, velocity_buffer, sungod_no_offset, order, shift_amount)

    print('Saved classifier results to: ',base_name+fname)

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
    results = parser.parse_args()

