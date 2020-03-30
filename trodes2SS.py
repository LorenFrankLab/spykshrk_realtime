"""
Parse trodes data into SS data containers
# Written by AKG
# Edited 3-8-19 by MEC to accomodate tetrodes and tritrodes (line 94)
# Edited 3-22-19 by MEC to make a filter for marks with large negative channels because
#                       this crashes the decoder by going outside the bounds of the
#                       normal_pdf_int_lookup function

"""

import numpy as np
import scipy as sp
import scipy.stats
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import holoviews as hv
import xarray as xr

import json
import functools

import dask
import dask.dataframe as dd
import dask.array as da

from spykshrk.franklab.data_containers import FlatLinearPosition, SpikeFeatures, \
        EncodeSettings, pos_col_format, SpikeObservation, RippleTimes, Posteriors

def get_all_below_threshold(self, threshold):
    ind = np.nonzero(np.all(self.values < threshold, axis=1))
    return self.iloc[ind]

def get_any_above_threshold(self, threshold):
    ind = np.nonzero(np.any(self.values >= threshold, axis=1))
    return self.iloc[ind]

def get_all_above_threshold(self, threshold):
    ind = np.nonzero(np.all(self.values > threshold, axis=1))
    return self.iloc[ind]

class AttrDict(dict):
	def __init__(self, *args, **kwargs):
		super(AttrDict, self).__init__(*args, **kwargs)
		self.__dict__ = self

def threshold_marks_negative(marks, negthresh=-999):
		pre_length = marks.shape
		marks = get_all_above_threshold(marks, negthresh)
		print(str(pre_length[0]-marks.shape[0])+' below '+str(negthresh)+'uV events removed')

		return marks

def threshold_marks(marks, maxthresh=2000, minthresh=0):
		pre_length = marks.shape
		marks = get_all_below_threshold(marks, maxthresh)
		print(str(pre_length[0]-marks.shape[0])+' above '+str(maxthresh)+'uV events removed')
		pre_length = marks.shape
		marks = get_any_above_threshold(marks, minthresh)
		print(str(pre_length[0]-marks.shape[0])+' below '+str(minthresh)+'uV events removed')

		return marks

class TrodesImport:
	""" animalinfo - takes in animal d/e/t information; parses FFmat files into dataframes of each datatype 
		default 30k sampling rate
	"""
	def __init__(self, ff_dir, name, days, epochs, tetrodes, Fs=3e4):
	    """ init function
	    
	    Args:
	        base_dir: root directory of animal data
	        name: name of animal
	        days: array of which days of data to process
	        tetrodes: array of which tetrodes to process
	        epochs: list of epochs for encoding
	    
	    """
	    self.ff_dir = ff_dir
	    self.name = name
	    self.days = days
	    self.epochs = epochs
	    self.tetrodes = tetrodes
	    self.Fs = Fs


	def import_marks(self):

		spk_amps = pd.DataFrame()
		

		for day in self.days:
			markname = self.ff_dir+self.name+'marks'+f'{day:02}'+'.mat'  #pads day with leading 0 if needed
			markmat = scipy.io.loadmat(markname,squeeze_me=True,struct_as_record=False)

			for ep in self.epochs:
				de_amps = pd.DataFrame()

				for tet in self.tetrodes:
					marktimes = markmat['marks'][day-1][ep-1][tet-1].times*self.Fs
					marktimes = marktimes.astype(np.int64,copy=False)
					marks = markmat['marks'][day-1][ep-1][tet-1].marks
					marks = marks.astype(np.int16,copy=False)
					tet_marks = SpikeFeatures.from_numpy_single_epoch_elec(day ,ep, tet, marktimes,marks,sampling_rate=self.Fs)
					if len(tet_marks.columns) == 4:
						tet_marks.columns=['c00','c01','c02','c03']
					if len(tet_marks.columns) == 3:
						tet_marks.columns=['c00','c01','c02']
					if len(tet_marks.columns) == 2:
						tet_marks.columns=['c00','c01']
					if len(tet_marks.columns) == 1:
						tet_marks.columns=['c00']
					de_amps = de_amps.append(tet_marks)

				de_amps.sort_index(level='timestamp', inplace=True)
				print('Duplicate marks found (and removed): '+str(de_amps[de_amps.index.duplicated(keep='first')].size))
				de_amps = de_amps[~de_amps.index.duplicated(keep='first')]
				spk_amps = spk_amps.append(de_amps)

		spk_amps.sampling_rate = self.Fs
		return spk_amps

	def fill_dead_chans(self, marks, tetinfo):

		# add number of dead chan col to tetinfo
		tmp = np.zeros((len(tetinfo),)) #initialize to get size right
		#if deadchans entry is an array (multiple deadchans) or empty, we can calculate length
		isarray = tetinfo['deadchans'].apply(lambda d: isinstance(d,np.ndarray)) 
		tmp[isarray] = tetinfo['deadchans'][isarray].apply(lambda d: len(d))  # add length of deadchans list
		# if it's an int (1 deadchan) fill in length with 1 (can't calc using len)
		isint = tetinfo['deadchans'].apply(lambda d: isinstance(d,int)) 
		tmp[isint] = 1
		tetinfo['ndlength'] = tmp   # store lengths as an additional column. no dead chans = length 0 

		marks.reset_index(inplace=True)

		day= self.days
		epoch = self.epochs
		subtets = tetinfo.query('area=="ca1" & ndlength>0 & day==@day & epoch==@epoch').index.get_level_values('tetrode_number').unique().tolist() 
		col_names = np.array(['c00','c01','c02','c03'])
		for tet in subtets:
			# if any chans contain all nans, then ignore which deadchan it is (data has already been removed; deadchans will always be the last channels)
			# otherwise, deadchans contain real data and deadchan tells you which one should be zero'd out 
			nanchans = np.array([np.all(np.isnan(marks['c00'].loc[marks['elec_grp_id']==tet])), np.all(np.isnan(marks['c01'].loc[marks['elec_grp_id']==tet])),
				np.all(np.isnan(marks['c02'].loc[marks['elec_grp_id']==tet])), np.all(np.isnan(marks['c03'].loc[marks['elec_grp_id']==tet]))])
			if np.any(nanchans):
				deadlist= col_names[nanchans]
				print('deadchans pre-stripped; replacing nans with zeros')
			else:
				deadchans = tetinfo.query("day==@day & epoch==@epoch & tetrode_number==@tet")['deadchans'].tolist()
				deadlen = tetinfo.query("day==@day & epoch==@epoch & tetrode_number==@tet")['ndlength'].tolist()
				print('deadchan data present; filling with nans according to deadchans list')
				if deadlen[0] >1:   # multiple deadchans, process as array
					deadlist = [col_names[x-1] for x in deadchans]
					deadlist = deadlist[0]
				else: # single deadchan, process as int
					deadlist = [col_names[deadchans[0]-1]]
			for channame in deadlist:
				print('subbing zeros for '+ channame + ' on tet ' + str(tet))
				marks[channame].mask(marks['elec_grp_id']==tet,0,inplace=True)

		marks.set_index(['day','epoch','elec_grp_id','timestamp','time'],inplace=True,drop=True)
		marks = marks.astype(np.int16, copy=False) 
		return marks



	def import_pos(self, xy = 'x'):

		allpos = pd.DataFrame()

		for day in self.days:
			for ep in self.epochs:
				posname = self.ff_dir+self.name+'pos'+f'{day:02}'+'.mat'
				posmat = scipy.io.loadmat(posname,squeeze_me=True,struct_as_record=False)
				pos_time = self.Fs*posmat['pos'][day-1][ep-1].data[:,0]
				pos_time = pos_time.astype(np.int64,copy=False)
				pos_runx = posmat['pos'][day-1][ep-1].data[:,5]
				pos_runy = posmat['pos'][day-1][ep-1].data[:,6]
				pos_vel = posmat['pos'][day-1][ep-1].data[:,8]

				if 'x' in xy:
					pos_obj = FlatLinearPosition.from_numpy_single_epoch(day, ep, pos_time, pos_runx, pos_vel, self.Fs,
                                                               [[0,0]])
				if 'y' in xy:
					pos_obj = FlatLinearPosition.from_numpy_single_epoch(day, ep, pos_time, pos_runy, pos_vel, self.Fs,
                                                               [[0,0]])
				allpos = allpos.append(pos_obj)

		allpos.sampling_rate = self.Fs
		return allpos

	def import_rips(self,pos_obj, velthresh=0):

		''' Converts ca1rippleskons mat file into RipplesTimes dataframe

		Parameters
		---------
		Requires position and velocity threshold (default 0, ie no filtering) to get rid of rips detected during movement

		'''
		allrips = pd.DataFrame()

		for day in self.days:
			for ep in self.epochs:
				ripname = self.ff_dir+self.name+'ca1rippleskons'+f'{day:02}'+'.mat'
				ripmat = scipy.io.loadmat(ripname,squeeze_me=True,struct_as_record=False)

				#generate a pandas table with starttime, endtime, and maxthresh columns, then instantiate RippleTimes 
				ripdata = {'time':ripmat['ca1rippleskons'][day-1][ep-1].starttime,
    			        'endtime':ripmat['ca1rippleskons'][day-1][ep-1].endtime,
       				    'maxthresh':ripmat['ca1rippleskons'][day-1][ep-1].maxthresh}
				rips = pd.DataFrame(ripdata,pd.MultiIndex.from_product([[day],[ep],
                        range(len(ripmat['ca1rippleskons'][day-1][ep-1].maxthresh))],
                        names=['day','epoch','event']))

				#specify field order
				rips = rips[['time','endtime','maxthresh']]

				# calculate timestamp based on time and Fs, convert from float to int
				rips['timestamp'] = (rips['time']*self.Fs).astype(int)

				#in order to use get_irregular_resampled, need to have a specific multiindex [day ep timestamp time]. reformat accordingly
				rips.reset_index(level=['event'], inplace=True)  # remove event number as multiindex
				rips.set_index(['timestamp', 'time'], drop=True, append=True, inplace=True)

				#use get_irregular_resampled to identify velocity at the time of ripple start, identify those that occur below the speed threshold
				posinfo_at_rip_times = pos_obj.get_irregular_resampled(rips)
				ripidx_below_vel = posinfo_at_rip_times['linvel_flat'] < velthresh
				rips_below_thresh = rips.loc[ripidx_below_vel].copy()

				#reformat multiindex and cast to RippleTimes object 
				rips_below_thresh.reset_index(level=['timestamp','time'], inplace=True)
				rips_below_thresh.set_index(['event'], drop=True, append=True, inplace=True)
				rips_below_thresh.rename({'time':'starttime'},axis='columns',inplace=True)
				#final_rips = RippleTimes.create_default(rips_below_thresh, 1)  # unclear why this won't work inside the function - just cast to rippletimes obj outside for now

				allrips = allrips.append(rips_below_thresh)

		return allrips

	def import_trials(self):

		alltrials = pd.DataFrame()

		for day in self.days:
			for ep in self.epochs:
				trialsname = self.ff_dir+self.name+'trials'+f'{day:02}'+'.mat'
				trialsmat = scipy.io.loadmat(trialsname,squeeze_me=True,struct_as_record=False)
				#generate a pandas table with starttime, endtime, and maxthresh columns, then instantiate RippleTimes 
				trialdata = {'starttime':trialsmat['trials'][day-1][ep-1].starttime,
	    			        'endtime':trialsmat['trials'][day-1][ep-1].endtime }
				
				trials = pd.DataFrame(trialdata,pd.MultiIndex.from_product([[day],[ep],
	                     range(len(trialsmat['trials'][day-1][ep-1].starttime))],
	                     names=['day','epoch','trial']))
				
				alltrials = alltrials.append(trials)

		return alltrials


def convert_dan_posterior_to_xarray(posterior_df, tetrode_dictionary, velocity_filter, encode_settings, decode_settings, transition_matrix, trial_order, marks_time_shift_amount, position_bin_centers=None):
    '''Converts pandas dataframe from Dan's 1D decoder to xarray Dataset
    
    Parameters
    ----------
    posterior_df : pandas.DataFrame, shape (n_time, n_columns)
    position_bin_centers : None or ndarray, shape (n_position_bins,), optional
    
    Returns
    -------
    results : xarray.Dataset
    
    '''
    is_position_bin = posterior_df.columns.str.startswith('x')
    
    if position_bin_centers is None:
        n_position_bins = is_position_bin.sum()
        position_bin_centers = np.arange(n_position_bins)
        
    coords = dict(
        day=posterior_df.loc[:, 'day'].values,
        epoch=posterior_df.loc[:, 'epoch'].values,
        timestamp=posterior_df.loc[:, 'timestamp'].values,
        time=posterior_df.loc[:, 'time'].values,
        position=position_bin_centers,
        num_spikes=posterior_df.loc[:, 'num_spikes'].values,
        dec_bin=posterior_df.loc[:, 'dec_bin'].values,
        ripple_grp=posterior_df.loc[:, 'ripple_grp'].values,
    )

    return xr.Dataset(
	    {'posterior': (('time','position'), posterior_df.loc[:, is_position_bin].values),
	     'velocity filter encode': velocity_filter,
	     'velocity filter decode': velocity_filter,
	     'tetrodes': tetrode_dictionary,
	     'marks_time_shift_amount': marks_time_shift_amount,
	     'trial_order': trial_order,
	     'sampling_rate': encode_settings['sampling_rate'],
	     'pos_bins': encode_settings['pos_bins'],
	     'pos_bin_edges': encode_settings['pos_bin_edges'],
	     'pos_bin_delta': encode_settings['pos_bin_delta'],
	     'pos_kernel': encode_settings['pos_kernel'],
	     'pos_kernel_std': encode_settings['pos_kernel_std'],
	     'mark_kernel_std': encode_settings['mark_kernel_std'],
	     'pos_num_bins': encode_settings['pos_num_bins'],
	     'pos_col_names': encode_settings['pos_col_names'],
	     'arm_coordinates': (encode_settings['arm_coordinates'][0]),
	     'trans_smooth_std': decode_settings['trans_smooth_std'],
	     'trans_uniform_gain': decode_settings['trans_uniform_gain'],
	     'time_bin_size': decode_settings['time_bin_size'],
	     'transition_matrix_name': 'flat powered',
	     'multiindex': ['day','epoch','timestamp','time'],
	     'transition_matrix': (('position','position'), transition_matrix)},
	    coords=coords)

def convert_save_classifier(base_name, fname, state1, state2, state3, tetrode_dictionary, likelihoods, encode_settings, decode_settings, rips, velthresh, vel_buffer, transition_matrix, trial_order, shift_amount, position_bin_centers=None):
    
    '''Converts pandas dataframe from Dan's 1D decoder to xarray Dataset
    
    Parameters
    ----------
    posterior_df : pandas.DataFrame, shape (n_time, n_columns)
    position_bin_centers : None or ndarray, shape (n_position_bins,), optional
    
    Returns
    -------
    results : xarray.Dataset
    
    '''
	#classifier_out['num_spikes'] = likelihoods['num_spikes']
    #classifier_out['dec_bin'] = likelihoods['dec_bin']
    state1_obj = Posteriors.from_dataframe(state1, enc_settings=encode_settings,
                                                            		dec_settings=decode_settings,
                                                            		user_key={'encode_settings': encode_settings,
                                                                      'decode_settings': decode_settings,
                                                                      'multi_index_keys': state1.index.names})

    state2_obj = Posteriors.from_dataframe(state2, enc_settings=encode_settings,
                                                            		dec_settings=decode_settings,
                                                            		user_key={'encode_settings': encode_settings,
                                                                      'decode_settings': decode_settings,
                                                                      'multi_index_keys': state2.index.names})
    state3_obj = Posteriors.from_dataframe(state3, enc_settings=encode_settings,
                                                            		dec_settings=decode_settings,
                                                            		user_key={'encode_settings': encode_settings,
                                                                      'decode_settings': decode_settings,
                                                                      'multi_index_keys': state3.index.names})
    
    #use state1 to populate these details (will apply to all)
    ripmask = state1_obj.apply_time_event(rips, event_mask_name='ripple_grp')
    s1_nomulti= state1_obj.reset_index()
    s2_nomulti = state2_obj.reset_index()
    s3_nomulti = state3_obj.reset_index()

    #likelihoods.reset_index(inplace=True)

    is_position_bin = s1_nomulti.columns.str.startswith('x')
    
    if position_bin_centers is None:
        n_position_bins = is_position_bin.sum()
        position_bin_centers = np.arange(n_position_bins)
    
    coords = dict(
        day=s1_nomulti['day'].values,
        epoch=s1_nomulti['epoch'].values,
        timestamp=s1_nomulti['timestamp'].values,
        time=s1_nomulti['time'].values,
        position=position_bin_centers,
        num_spikes=likelihoods['num_spikes'].values,
        dec_bin=likelihoods['dec_bin'].values,
        ripple_grp=s1_nomulti['ripple_grp'].values,
    )

    xr_obj =  xr.Dataset(
	    {'state1_posterior': (('time','position'), s1_nomulti.loc[:, is_position_bin].values),
	    'state2_posterior': (('time','position'), s2_nomulti.loc[:, is_position_bin].values),
	    'state3_posterior': (('time','position'), s3_nomulti.loc[:, is_position_bin].values),
	     'velocity_filter': velthresh,
	     'velocity_buffer': vel_buffer,
	     'tetrodes': tetrode_dictionary,
	     'shift_amount':shift_amount,
	     'trial_order': trial_order,
	     'sampling_rate': encode_settings['sampling_rate'],
	     'pos_bins': encode_settings['pos_bins'],
	     'pos_bin_edges': encode_settings['pos_bin_edges'],
	     'pos_bin_delta': encode_settings['pos_bin_delta'],
	     'pos_kernel': encode_settings['pos_kernel'],
	     'pos_kernel_std': encode_settings['pos_kernel_std'],
	     'mark_kernel_std': encode_settings['mark_kernel_std'],
	     'pos_num_bins': encode_settings['pos_num_bins'],
	     'arm_coordinates': (encode_settings['arm_coordinates'][0]),
	     'trans_smooth_std': decode_settings['trans_smooth_std'],
	     'trans_uniform_gain': decode_settings['trans_uniform_gain'],
	     'time_bin_size': decode_settings['time_bin_size'],
	     'transition_matrix_name': 'flat powered',
	     'multiindex': ['day','epoch','timestamp','time'],
	     'transition_matrix': (('position','position'), transition_matrix)},
	    coords=coords)

    xr_obj.to_netcdf(base_name+fname+'.nc')