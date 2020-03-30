import functools
import logging

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
import scipy as sp
import scipy.signal
import warnings

from spykshrk.franklab.data_containers import LinearPosition, FlatLinearPosition, SpikeObservation, EncodeSettings, DecodeSettings, Posteriors, pos_col_format
from spykshrk.franklab.pp_decoder.util import gaussian, normal2D, apply_no_anim_boundary, normal_pdf_int_lookup
from spykshrk.util import Groupby

logger = logging.getLogger(__name__)

class OfflinePPEncoder(object):

    def __init__(self, linflat, enc_spk_amp, dec_spk_amp, encode_settings: EncodeSettings, decode_settings: DecodeSettings,
                 dask_worker_memory=None, dask_memory_utilization=0.5, dask_chunksize=None):
        """
        Constructor for OfflinePPEncoder.
        
        Args:
            linflat (FlatLinearPosition): Observered 1D unique animal position
            enc_spk_amp (SpikeObservation): Observered spikes in encoding model
            dec_spk_amp (SpikeObservation): Observered spikes in decoding model
            encode_settings (EncodeSettings): Realtime encoder settings.
            decode_settings (DecodeSettings): Realtime decoder settings.

        """
        if dask_worker_memory is None and dask_chunksize is None:
            raise TypeError('OfflinePPEncoder requires either dask_memory or dask_chunksize to be set.')
        if dask_worker_memory is not None and dask_chunksize is not None:
            raise TypeError('OfflinePPEncoder only allows one to be set, dask_memory or dask_chunksize.')
        if dask_chunksize is not None:
            memory_per_dec = (len(enc_spk_amp) * np.sum([np.dtype(dtype).itemsize for dtype in enc_spk_amp.dtypes]))
            self.dask_chunksize = dask_chunksize
            logger.info('Manual Dask chunksize: {}'.format(self.dask_chunksize))
            logger.info('Expected worker peak memory usage: {:0.2f} MB'.format(self.dask_chunksize * memory_per_dec / 2**20))
            logger.info('Worker total memory: UNKNOWN')

        if dask_worker_memory is not None:
            memory_per_dec = (len(enc_spk_amp) * np.sum([np.dtype(dtype).itemsize for dtype in enc_spk_amp.dtypes]))
            self.dask_chunksize = np.int(dask_memory_utilization * dask_worker_memory / memory_per_dec)
            logger.info('Dask chunksize: {}'.format(self.dask_chunksize))
            logger.info('Memory utilization at: {:0.1f}%'.format(dask_memory_utilization * 100))
            logger.info('Expected worker peak memory usage: {:0.2f} MB'.
                        format(self.dask_chunksize * memory_per_dec / 2**20))

        self.linflat = linflat
        self.enc_spk_amp = enc_spk_amp
        self.dec_spk_amp = dec_spk_amp
        self.encode_settings = encode_settings
        self.decode_settings = decode_settings

        self.calc_occupancy()
        self.calc_firing_rate()
        self.calc_prob_no_spike()

        self.trans_mat = dict.fromkeys(['learned', 'simple', 'uniform','sungod'])
        self.trans_mat['learned'] = self.calc_learned_state_trans_mat(self.linflat,self.encode_settings, self.decode_settings)
        self.trans_mat['simple'] = self.calc_simple_trans_mat(self.encode_settings)
        self.trans_mat['uniform'] = self.calc_uniform_trans_mat(self.encode_settings)


    def run_encoder(self):
        logging.info("Setting up encoder dask task.")
        task = self.setup_encoder_dask()
        logging.info("Running compute tasks on dask workers.")
        self.results = dask.compute(*task)

        tet_ids = np.unique(self.enc_spk_amp.index.get_level_values('elec_grp_id'))
        observ_tet_list = []
        grp = self.dec_spk_amp.groupby('elec_grp_id')
        for tet_ii, (tet_id, grp_spk) in enumerate(grp):
            tet_result = self.results[tet_ii]
            tet_result.set_index(grp_spk.index, inplace=True)
            observ_tet_list.append(tet_result)

        observ = pd.concat(observ_tet_list)
        self.observ_obj = SpikeObservation.create_default(observ.sort_index(level=['day', 'epoch',
                                                                              'timestamp', 'elec_grp_id']),
                                                     self.encode_settings)

        self.observ_obj['elec_grp_id'] = self.observ_obj.index.get_level_values('elec_grp_id')
        self.observ_obj.index = self.observ_obj.index.droplevel('elec_grp_id')

        self.observ_obj['position'] = (self.linflat.get_irregular_resampled(self.observ_obj).
                                  get_mapped_single_axis()['linpos_flat'])

        self.observ_obj.set_distribution(self.observ_obj.get_distribution_as_np() + np.finfo(float).eps)

        return self.observ_obj

    def setup_encoder_dask(self):
        # grp = self.spk_amp.groupby('elec_grp_id')
        dec_grp = self.dec_spk_amp.groupby('elec_grp_id')
        observations = {}
        task = []

        for dec_tet_id, dec_spk_tet in dec_grp:
            enc_spk_tet = self.enc_spk_amp.query('elec_grp_id==@dec_tet_id')
            if len(enc_spk_tet) == 0 | len(dec_spk_tet) == 0:
                continue
            enc_tet_lin_pos = self.linflat.get_irregular_resampled(enc_spk_tet)
            if len(enc_tet_lin_pos) == 0:
                continue
            # maintain elec_grp_id info. get mark and index column names to reindex dask arrays
            mark_columns = dec_spk_tet.columns
            index_columns = dec_spk_tet.index.names
            dask_dec_spk_tet = dd.from_pandas(dec_spk_tet.reset_index(), chunksize=self.dask_chunksize)

            df_meta = pd.DataFrame([], columns=[pos_col_format(ii, self.encode_settings.pos_num_bins)
                                                for ii in range(self.encode_settings.pos_num_bins)])

            # setup decode of decode spikes from encoding of encoding spikes
            task.append(dask_dec_spk_tet.map_partitions(functools.partial(self.compute_observ_tet, enc_spk=enc_spk_tet,
                                                                          tet_lin_pos=enc_tet_lin_pos,
                                                                          occupancy=self.occupancy,
                                                                          encode_settings=self.encode_settings),
                                                        meta=df_meta,
                                                        mark_columns=mark_columns,
                                                        index_columns=index_columns))
        return task

    def compute_observ_tet(self, dec_spk, enc_spk, tet_lin_pos, occupancy, encode_settings, mark_columns, index_columns):
        import sys
        # MEC added 6-22-19
        # make an optional switch for using a histogram or gaussian smoothing for pos_distrib_tet
        # the histogram option is activated if encoder_settings.pos_kernel_std = 0
        # the (original) gaussian smoothing option is activated in all other cases

        # this is the new version of pos_distrib_tet with a histogram
        if encode_settings.pos_kernel_std == 0:
            #print('histogram')
            enc_spikes_range = np.arange(0,len(tet_lin_pos['linpos_flat']),1)
            pos_distrib_tet = np.zeros((len(tet_lin_pos['linpos_flat']),len(encode_settings.pos_bins)))
            pos_distrib_tet[np.expand_dims(enc_spikes_range, 1),np.expand_dims(tet_lin_pos['linpos_flat'], 1)] = 1

        # this is the original gaussian smoothing version
        else:
            #print('gaussian')
            pos_distrib_tet = sp.stats.norm.pdf(np.expand_dims(encode_settings.pos_bins, 0),
                                                np.expand_dims(tet_lin_pos['linpos_flat'], 1),
                                                encode_settings.pos_kernel_std)

        mark_contrib = normal_pdf_int_lookup(np.expand_dims(dec_spk[mark_columns], 1),
                                             np.expand_dims(enc_spk, 0),
                                             encode_settings.mark_kernel_std)
        all_contrib = np.prod(mark_contrib, axis=2)

        # set weight any matching spike to zero (if enc and dec set containing any overlapping events)  
        # max weight is peak of gaussian kernel ^4 (b/c each mark channel gets multiplied together)
        max_weight = normal_pdf_int_lookup(1,1,encode_settings.mark_kernel_std)**4 
        all_contrib[all_contrib>round(max_weight,20)]=0

        del mark_contrib
        observ = np.matmul(all_contrib, pos_distrib_tet)
        del all_contrib
        #print('observation before occ',[0])

        # occupancy normalize 
        # AKG added enc_settings.pos_bin_delta to take into account different bin delta values
        observ = observ / (occupancy * encode_settings.pos_bin_delta)
        #print('observation',observ[0])
        #print('occupancy',occupancy)

        # normalize factor for each row (#dec spks x #pos_bins)
        observ_sum = np.nansum(observ, axis=1)

        # replace all rows that are all zeros with uniform distribution
        observ_sum_zero = observ_sum == 0
        observ[observ_sum_zero, :] = 1/(self.encode_settings.pos_bins[-1] - self.encode_settings.pos_bins[0])
        observ_sum[observ_sum_zero] = 1

        # apply normalization factor
        observ = observ / observ_sum[:, np.newaxis]
        ret_df = pd.DataFrame(observ, index=dec_spk.set_index(index_columns).index,
                              columns=[pos_col_format(pos_ii, observ.shape[1])
                                       for pos_ii in range(observ.shape[1])])
        return ret_df

    def calc_occupancy(self):
        self.occupancy = self._calc_occupancy(self.linflat, self.encode_settings)

    def calc_firing_rate(self):
        self.firing_rate = self._calc_firing_rate_tet(self.enc_spk_amp, self.linflat, self.encode_settings)

    def calc_prob_no_spike(self):
        self.prob_no_spike = self._calc_prob_no_spike(self.firing_rate, self.occupancy, self.encode_settings, self.decode_settings)

    @staticmethod
    def _calc_occupancy(lin_obj: FlatLinearPosition, enc_settings: EncodeSettings):
        """
        Args:
            lin_obj (LinearPositionContainer): Linear position of the animal.
            enc_settings (EncodeSettings): Realtime encoding settings.
        Returns (np.array): The occupancy of the animal
        """
        occupancy, occ_bin_edges = np.histogram(lin_obj['linpos_flat'], bins=enc_settings.pos_bin_edges,
                                                normed=True)
        # dont use convovle becuase of boundary problems
        #occupancy = np.convolve(occupancy, enc_settings.pos_kernel, mode='same')

        # occupancy
        occupancy = apply_no_anim_boundary(enc_settings.pos_bins, enc_settings.arm_coordinates, occupancy, np.nan)
        occupancy += 1e-10
        return occupancy

    @staticmethod
    def _calc_firing_rate_tet(observ: SpikeObservation, lin_obj: FlatLinearPosition, enc_settings: EncodeSettings):
        # initialize conditional intensity function
        firing_rate = {}
        enc_tet_lin_pos = (lin_obj.get_irregular_resampled(observ))
        #enc_tet_lin_pos['elec_grp_id'] = observ.index.get_level_values(level='elec_grp_id')
        tet_pos_groups = enc_tet_lin_pos.loc[:, 'linpos_flat'].groupby('elec_grp_id')
        for tet_id, tet_spikes in tet_pos_groups:
            tet_pos_hist, _ = np.histogram(tet_spikes, bins=enc_settings.pos_bin_edges)
            firing_rate[tet_id] = tet_pos_hist
        for fr_key in firing_rate.keys():
            # dont use convovle because of boundary problems at 0
            #firing_rate[fr_key] = np.convolve(firing_rate[fr_key], enc_settings.pos_kernel, mode='same')
            firing_rate[fr_key] = apply_no_anim_boundary(enc_settings.pos_bins, enc_settings.arm_coordinates,
                                                         firing_rate[fr_key])
            firing_rate[fr_key] = firing_rate[fr_key] / (firing_rate[fr_key].sum() * enc_settings.pos_bin_delta)
        return firing_rate

    @staticmethod
    def _calc_prob_no_spike(firing_rate: dict, occupancy, enc_settings: EncodeSettings, dec_settings: DecodeSettings):
        """
        
        Args:
            firing_rate (pd.DataFrame): Occupancy firing rate, from calc_observation_intensity(...).
            occupancy (np.array): The occupancy of the animal.
            enc_settings (EncodeSettings): Realtime encode settings.
            dec_settings (DecodeSettings): Realtime decoding settings.

        Returns (dict[int, np.array]): Dictionary of probability that no spike occured per tetrode.

        """
        prob_no_spike = {}
        for tet_id, tet_fr in firing_rate.items():
            # original
            prob_no_spike[tet_id] = np.exp(-dec_settings.time_bin_size/enc_settings.sampling_rate * tet_fr / occupancy)
            # prob_no_spike[tet_id] = np.nan_to_num(prob_no_spike[tet_id], copy=False)
            # MEC: to turn off prob_no_spike
            #prob_no_spike[tet_id] = np.ones(enc_settings.pos_bins)
            #prob_no_spike[tet_id] = apply_no_anim_boundary(enc_settings.pos_bins, enc_settings.arm_coordinates, 
            #                                               prob_no_spike[tet_id], np.nan)
            #print('prob no spike',prob_no_spike[tet_id])
        return prob_no_spike

    
    @staticmethod
    def calc_learned_state_trans_mat(linpos_simple, enc_settings, dec_settings):
        """
        Calculate the point process transition matrix using the real behavior of the animal.
        This is the 2D matrix that defines the possible range of transitions in the position
        estimate.
        
        The learned values are smoothed with a gaussian kernel and a uniform offset is added,
        specified by the encoding config.  The matrix is column normalized.
        
        Args:
            linpos_simple (pd.DataFrame): Linear position pandas table with no MultiIndex.
            enc_settings (EncodeSettings): Encoder settings from a realtime config.
            dec_settings (DecodeSettings): Decoder settings from a realtime config.

        Returns: Learned transition matrix.

        """
        pos_num_bins = len(enc_settings.pos_bins)

        # Smoothing kernel for learned pos transition matrix
        xv, yv = np.meshgrid(np.arange(-20, 21), np.arange(-20, 21))
        kernel = normal2D(xv, yv, dec_settings.trans_smooth_std)
        kernel /= kernel.sum()

        linpos_state = linpos_simple
        linpos_ind = np.searchsorted(enc_settings.pos_bins, linpos_state, side='right') - 1

        # Create learned pos transition matrix
        learned_trans_mat = np.zeros([pos_num_bins, pos_num_bins])
        for first_pos_ind, second_pos_ind in zip(linpos_ind[:-1], linpos_ind[1:]):
            learned_trans_mat[first_pos_ind, second_pos_ind] += 1

        # normalize
        learned_trans_mat = learned_trans_mat / (learned_trans_mat.sum(axis=0)[None, :])
        learned_trans_mat[np.isnan(learned_trans_mat)] = 0

        # smooth
        learned_trans_mat = sp.signal.convolve2d(learned_trans_mat, kernel, mode='same')
        learned_trans_mat = apply_no_anim_boundary(enc_settings.pos_bins, enc_settings.arm_coordinates,
                                                   learned_trans_mat)

        # uniform offset
        uniform_gain = dec_settings.trans_uniform_gain
        uniform_dist = np.ones(learned_trans_mat.shape)

        # no-animal boundary
        uniform_dist = apply_no_anim_boundary(enc_settings.pos_bins, enc_settings.arm_coordinates, uniform_dist)

        # normalize uniform offset
        uniform_dist = uniform_dist / (uniform_dist.sum(axis=0)[None, :])
        uniform_dist[np.isnan(uniform_dist)] = 0

        # apply uniform offset
        learned_trans_mat = learned_trans_mat * (1 - uniform_gain) + uniform_dist * uniform_gain

        # renormalize
        learned_trans_mat = learned_trans_mat / (learned_trans_mat.sum(axis=0)[None, :])
        learned_trans_mat[np.isnan(learned_trans_mat)] = 0

        return learned_trans_mat

    @staticmethod
    def calc_simple_trans_mat(enc_settings):
        """
        Calculate a simple point process transition matrix using a gaussian kernel.
        
        Args:
            enc_settings (EncodeSettings): Encoder setting from realtime config.

        Returns (np.array): Simple gaussian transition matrix.

        """
        pos_num_bins = len(enc_settings.pos_bins)

        # Setup transition matrix
        transition_mat = np.ones([pos_num_bins, pos_num_bins])
        for bin_ii in range(pos_num_bins):
            transition_mat[bin_ii, :] = gaussian(enc_settings.pos_bins, enc_settings.pos_bins[bin_ii], 3)

        transition_mat = apply_no_anim_boundary(enc_settings.pos_bins, enc_settings.arm_coordinates,
                                                transition_mat)

        # uniform offset
        uniform_gain = 0.01
        uniform_dist = np.ones(transition_mat.shape)
        uniform_dist = apply_no_anim_boundary(enc_settings.pos_bins, enc_settings.arm_coordinates,
                                              uniform_dist)

        # normalize transition matrix
        transition_mat = transition_mat/(transition_mat.sum(axis=0)[None, :])
        transition_mat[np.isnan(transition_mat)] = 0

        # normalize uniform offset
        uniform_dist = uniform_dist/(uniform_dist.sum(axis=0)[None, :])
        uniform_dist[np.isnan(uniform_dist)] = 0

        # apply uniform offset
        transition_mat = transition_mat * (1 - uniform_gain) + uniform_dist * uniform_gain

        return transition_mat

    @staticmethod
    def calc_uniform_trans_mat(enc_settings):
        """
        Calculate a simple uniform point process transition matrix.
        
        Args:
            enc_settings (EncodeSettings): Encoder setting from realtime config.

        Returns (np.array): Simple uniform transition matrix.

        """

        pos_num_bins = len(enc_settings.pos_bins)

        # Setup transition matrix
        transition_mat = np.ones([pos_num_bins, pos_num_bins])
        transition_mat = apply_no_anim_boundary(enc_settings.pos_bins, enc_settings.arm_coordinates, transition_mat)

        # normalize transition matrix
        transition_mat = transition_mat/(transition_mat.sum(axis=0)[None, :])

        transition_mat = np.nan_to_num(transition_mat)

        return transition_mat



class OfflinePPDecoder(object):
    """
    Implementation of Adaptive Marked Point Process Decoder [Deng, et. al. 2015].
    
    Requires spike observation containers (spykshrk.franklab.pp_decoder.SpikeObservation).
    along with encoding (spykshrk.franklab.pp_decoder.EncodeSettings) 
    and decoding settings (spykshrk.franklab.pp_decoder.DecodeSettings).
    
    """
    def __init__(self, observ_obj: SpikeObservation, encode_settings: EncodeSettings,
                 decode_settings: DecodeSettings, time_bin_size=30, 
                 all_linear_position=None, 
                 parallel=True, trans_mat=None,
                 prob_no_spike=None):
        """
        Constructor for OfflinePPDecoder.
        
        Args:
            observ_obj (SpikeObservation): Observered position distribution for each spike.
            encode_settings (EncodeSettings): Realtime encoder settings.
            decode_settings (DecodeSettings): Realtime decoder settings.
            
            time_bin_size (float, optional): Delta time per bin to run decode, defaults to decoder_settings value.
        """
        if prob_no_spike:
            self.prob_no_spike = prob_no_spike
        else:
            self.prob_no_spike = {tet_id: np.ones(encode_settings.pos_num_bins) for tet_id in encode_settings.tetrodes}
        self.trans_mat = trans_mat
        self.observ_obj = observ_obj
        self.encode_settings = encode_settings
        self.decode_settings = decode_settings
        # self.which_trans_mat = which_trans_mat
        self.time_bin_size = time_bin_size

        self.parallel = parallel

        self.likelihoods = None
        self.posteriors = None
        self.posteriors_obj = None

        self.all_linear_position = all_linear_position

    def __del__(self):
        print('decoder deleting')

    def run_decoder(self):
        """
        Run the decoder at a given time bin size.  Intermediate results are saved as
        attributes to the class.

        Returns (pd.DataFrame): Final decoded posteriors that estimate position.

        """

        print("Beginning likelihood calculation")
        self.recalc_likelihood()

        # MEC mask for encoding times - now AKG mask based on enc set indicator in pos_obj
        # use get_irregular_resample to find all timebins during the encoding state, then set those rows to NaN
        # the real point of this is to nan out anything that isn't being decoded, so do that 
        posinfo_per_bin = self.all_linear_position.get_irregular_resampled(self.likelihoods)
        self.likelihoods.loc[posinfo_per_bin['decoding_set']==0] = np.nan

        print("Beginning posterior calculation")
        self.recalc_posterior()

        return self.posteriors_obj

    def recalc_likelihood(self):
        self.likelihoods = self.calc_observation_intensity(self.observ_obj,
                                                           self.prob_no_spike,
                                                           self.encode_settings,
                                                           self.decode_settings,
                                                           time_bin_size=self.time_bin_size)

    def recalc_posterior(self):
        self.posteriors = self.calc_posterior(self.likelihoods, self.trans_mat, self.encode_settings)
        self.posteriors_obj = Posteriors.from_dataframe(self.posteriors, enc_settings=self.encode_settings,
                                                        dec_settings=self.decode_settings,
                                                        user_key={'encode_settings': self.encode_settings,
                                                                  'decode_settings': self.decode_settings,
                                                                  'multi_index_keys': self.posteriors.index.names})



    @staticmethod
    def calc_observation_intensity(observ: SpikeObservation,
                                   prob_no_spike,
                                   enc_settings: EncodeSettings,
                                   dec_settings: DecodeSettings,
                                   time_bin_size=None):
        """
        
        Args:
            observ (SpikeObservation): Object containing observation data frame, one row per spike observed.
            enc_settings (EncodeSettings): Encoder settings from realtime config.
            dec_settings (DecodeSettings): Decoder settings from realtime config.
            time_bin_size (float): Delta time per bin.

        Returns: (pd.DataFrame, dict[int, np.array]) DataFrame of observation per time bin in each row.
            Dictionary of numpy arrays, one per tetrode, containing occupancy firing rate.

        """

        day = observ.index.get_level_values('day')[0]
        epoch = observ.index.get_level_values('epoch')[0]

        if time_bin_size is not None:
            observ.update_observations_bins(time_bin_size=time_bin_size, inplace=True)
        else:
            time_bin_size = dec_settings.time_bin_size
            observ.update_observations_bins(time_bin_size=time_bin_size, inplace=True)

        observ.update_parallel_bins(60000, inplace=True)

        observ.update_num_missing_future_bins(inplace=True)

        observ_dask = dd.from_pandas(observ.get_no_multi_index(), chunksize=30000)
        observ_grp = observ_dask.groupby('parallel_bin')

        observ_meta = [(key, 'f8') for key in [pos_col_format(ii, enc_settings.pos_num_bins)
                                               for ii in range(enc_settings.pos_num_bins)]]
        observ_meta.append(('timestamp', 'f8'))
        observ_meta.append(('num_spikes', 'f8'))
        observ_meta.append(('dec_bin', 'f8'))

        elec_grp_list = observ['elec_grp_id'].unique()
        observ_task = observ_grp.apply(functools.partial(OfflinePPDecoder._calc_observation_single_bin,
                                                         elec_grp_list=elec_grp_list,
                                                         prob_no_spike=prob_no_spike,
                                                         time_bin_size=time_bin_size,
                                                         enc_settings=enc_settings),
                                       meta=observ_meta)

        dec_agg_results = observ_task.compute()
        dec_agg_results.sort_values('timestamp', inplace=True)

        #encoding mask: convert timestamps to int in order to run get_irregular_resample
        dec_agg_results = dec_agg_results.astype({'timestamp': int})

        dec_new_ind = pd.MultiIndex.from_product([[day], [epoch], dec_agg_results['timestamp']])
        lev = list(dec_new_ind.levels)
        lab = list(dec_new_ind.labels)

        lev.append(dec_agg_results['timestamp']/float(enc_settings.sampling_rate))
        dec_agg_results.drop(columns='timestamp', inplace=True)

        lab.append(range(len(dec_agg_results)))

        dec_new_ind = pd.MultiIndex(levels=lev, labels=lab, names=['day', 'epoch', 'timestamp', 'time'])

        dec_agg_results.set_index(dec_new_ind, inplace=True)

        binned_observ = dec_agg_results

        #dec_agg_results['day'] = day
        #dec_agg_results['epoch'] = epoch
        #dec_agg_results['time'] = dec_agg_results['timestamp']/float(enc_settings.sampling_rate)
        #binned_observ = dec_agg_results.set_index(['day', 'epoch', 'timestamp', 'time'])

        # Smooth and normalize firing rate (conditional intensity function)

        return binned_observ

    @staticmethod
    def _calc_observation_single_bin(spikes_in_parallel, elec_grp_list,
                                     prob_no_spike, time_bin_size, enc_settings):

        global_prob_no_spike = np.prod(list(prob_no_spike.values()), axis=0)
        global_prob_no_spike = global_prob_no_spike/np.nansum(global_prob_no_spike)  # AKG added; normalize

        results = []
        #parallel_id = spikes_in_parallel['parallel_bin'].iloc[0]
        #dec_grp = spikes_in_parallel.groupby('dec_bin')

        dec_grp = Groupby(spikes_in_parallel.values, spikes_in_parallel['dec_bin'].values)
        pos_col_ind = spikes_in_parallel.columns.slice_locs(enc_settings.pos_col_names[0],
                                                            enc_settings.pos_col_names[-1])
        elec_grp_ind = spikes_in_parallel.columns.get_loc('elec_grp_id')
        num_missing_ind = spikes_in_parallel.columns.get_loc('num_missing_bins')
        dec_bin_start_ind = spikes_in_parallel.columns.get_loc('dec_bin_start')

        for dec_bin_ii, spikes_in_bin in dec_grp:
            #print('parallel #{} dec #{}'.format(parallel_id, dec_bin_ii))
            obv_in_bin = np.ones(enc_settings.pos_num_bins)

            num_spikes = len(spikes_in_bin)

            elec_set = set()

            spike_bin_raw = spikes_in_bin

            """obv_in_bin = spike_bin_raw[:, slice(*pos_col_ind)].prod(axis=0)

            for elec_grp_id in spike_bin_raw[:, elec_grp_ind]:
                elec_set.add(elec_grp_id)
                obv_in_bin *= prob_no_spike[elec_grp_id]

            obv_in_bin = obv_in_bin / (np.sum(obv_in_bin) * enc_settings.pos_bin_delta)

            num_missing_bins = spike_bin_raw[-1,num_missing_ind]
            """
            # Contribution of each spike
            missing_bins_list = []
            dec_bin_timestamp = spike_bin_raw[0, dec_bin_start_ind]
            for obv, elec_grp_id, num_missing_bins in zip(spike_bin_raw[:, slice(*pos_col_ind)],
                                                          spike_bin_raw[:, elec_grp_ind],
                                                          spike_bin_raw[:, num_missing_ind]):

                elec_set.add(elec_grp_id)
                missing_bins_list.append(num_missing_bins)

                obv_in_bin = obv_in_bin * obv
                obv_in_bin = obv_in_bin * prob_no_spike[elec_grp_id]
                obv_in_bin = obv_in_bin / (np.nansum(obv_in_bin) * enc_settings.pos_bin_delta)

            # Contribution for electrodes that no spikes in this bin
            # MEC: NOTE 2-17-20 Loren says we should multiply by global_prob_no spike here
            #for elec_grp_id in elec_set.symmetric_difference(elec_grp_list):
            #    obv_in_bin = obv_in_bin * prob_no_spike[elec_grp_id]
            #    obv_in_bin = obv_in_bin / (np.nansum(obv_in_bin) * enc_settings.pos_bin_delta)   # AKG added, normalize

            # NEW 3/1/20: multiply by global prob_no_spike
            obv_in_bin * global_prob_no_spike
            obv_in_bin = obv_in_bin / (np.nansum(obv_in_bin) * enc_settings.pos_bin_delta)   #  normalize

            # Checking if missing bin has more than 1 whole number (and the rest are zeros)
            missing_bins_list = np.array(missing_bins_list)
            if np.count_nonzero(missing_bins_list) > 1:
                warnings.warn('For decode bin (' + dec_bin_ii + ') bin time (' + dec_bin_timestamp +
                              ') there are multiple possible values for missing bins ' + missing_bins_list +
                              ', which is not allowed.')


            results.append(np.concatenate([obv_in_bin, [dec_bin_timestamp, num_spikes, dec_bin_ii]]))
            for missing_ii in range(int(max(missing_bins_list))):
                results.append(np.concatenate([global_prob_no_spike, [dec_bin_timestamp+((missing_ii+1)*time_bin_size),
                                                                      0, dec_bin_ii+missing_ii+1]]))

        likelihoods = pd.DataFrame(np.vstack(results),
                                   columns=enc_settings.pos_col_names+['timestamp', 'num_spikes', 'dec_bin'])
        return likelihoods

    @staticmethod
    def calc_posterior(likelihoods, transition_mat, enc_settings: EncodeSettings):
        """
        
        Args:
            likelihoods (pd.DataFrame): The evaluated likelihood function per time bin, from calc_likelihood(...).
            transition_mat (np.array): The point process state transition matrix.
            enc_settings (EncodeSettings): Realtime encoding settings.

        Returns (pd.DataFrame): The decoded posteriors per time bin estimating the animal's location.

        """
        likelihoods_pos = likelihoods.loc[:, pos_col_format(0, enc_settings.pos_num_bins):
                                          pos_col_format(enc_settings.pos_num_bins-1,
                                                         enc_settings.pos_num_bins)]

        posteriors = OfflinePPDecoder._posterior_from_numpy(likelihoods_pos.values,
                                                            transition_mat, enc_settings.pos_num_bins,
                                                            enc_settings.pos_bin_delta)

        # posteriors = pp_cy.calc_posterior_cy(likelihoods_pos.values.copy(order='C'),
        #                                      transition_mat.copy(order='C'), enc_settings.pos_num_bins,
        #                                      enc_settings.pos_bin_delta)

        # copy observ DataFrame and replace with likelihoods, preserving other columns

        posteriors_df = pd.DataFrame(posteriors, index=likelihoods.index,
                                     columns=enc_settings.pos_col_names)

        posteriors_df['num_spikes'] = likelihoods['num_spikes']
        posteriors_df['dec_bin'] = likelihoods['dec_bin']

        return posteriors_df

    @staticmethod
    def _posterior_from_numpy(likelihoods, transition_mat, pos_num_bins, pos_delta):

        last_posterior = np.ones(pos_num_bins)

        posteriors = np.zeros(likelihoods.shape)

        for like_ii, like in enumerate(likelihoods):
            posteriors[like_ii, :] = like * np.matmul(transition_mat, np.nan_to_num(last_posterior))
            posteriors[like_ii, :] = posteriors[like_ii, :] / (np.nansum(posteriors[like_ii, :]) *
                                                               pos_delta)
            #last_posterior = posteriors[like_ii, :]

            # velocity mask - reset posterior after masked encoding bins 
            if np.isnan(like).all():
                last_posterior = np.ones(pos_num_bins)

            else:
                last_posterior = posteriors[like_ii, :]

        # copy observ DataFrame and replace with likelihoods, preserving other columns

        return posteriors
