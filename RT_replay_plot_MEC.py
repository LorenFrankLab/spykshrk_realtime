# python script to plot replay assignment results from daily behavior sessions
# written by MEC from notebooks written by AKG and JG
# 5-12-20

#cell 1
# Setup and import packages
import sys
import os
#import pdb
from datetime import datetime, date

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#import loren_frank_data_processing as lfdp
#from loren_frank_data_processing import Animal

#import trodes2SS
#import sungod_util
import realtime_analysis_util

#from spykshrk.franklab.data_containers import RippleTimes, pos_col_format, Posteriors

#from spykshrk.franklab.pp_decoder.pp_clusterless import OfflinePPEncoder, OfflinePPDecoder

def main(realtime_rec, path_out):
    print(datetime.now())
    today = str(date.today())

    # 1. load real-time merged rec file
    realtime_rec_file = realtime_rec
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
    print('loaded real-time records')

    # 1b. merge stim_lockout and stim_message

    stim_message_file = stim_message
    stim_lockout_file = stim_lockout

    realtime_ripple_starts = stim_lockout_file[stim_lockout_file['lockout_state']==1]

    lockout_message_joined = pd.merge(realtime_ripple_starts, stim_message_file, how='inner',
                                      left_on='lockout_num', right_on = 'ripple_number',
                                      suffixes=('_rip', '_msg'))
    lockout_message_joined['rip_space'] = (lockout_message_joined['timestamp'].diff())/30
    print('number of replays:',lockout_message_joined.shape[0])

    # 2. summarize session
    # including: spikes total, dropped spikes, replay arm histogram

    plt.figure(figsize=(4,4))
    plt.scatter([0,1,2],[0,0,0])

    plt.figtext(0.5, 0.8, "Total spikes", ha="center", va="bottom", size="medium",color="k")
    plt.figtext(0.5, 0.75, encoder_data.shape[0], ha="center", va="bottom", size="medium",color="k")
    plt.figtext(0.5, 0.65, "Dropped spikes", ha="center", va="bottom", size="medium",color="k")
    plt.figtext(0.5, 0.6, decoder_missed_spikes.shape[0], ha="center", va="bottom", size="medium",color="k")

    plt.ylim(0,10)
    plt.xlim(0,10)

    spike_count_image_name = os.path.join(path_out, 'spike_count.pdf')
    plt.savefig(spike_count_image_name)
    plt.close()
    print('saved number of spikes. encoder:',encoder_data.shape[0],
          'dropped spikes:',decoder_missed_spikes.shape[0])

    replay_histogram = lockout_message_joined.copy()
    #replay_histogram = replay_histogram[replay_histogram['taskState']==2]
    plt.figure(figsize=(6,4))
    plt.hist(replay_histogram['posterior_max_arm'].values,bins=np.arange(0,101))
    plt.title('Real-time replay assignments')

    replay_histogram_name = os.path.join(path_out, 'replay_histogram.pdf')
    plt.savefig(replay_histogram_name)
    plt.close()
    print('saved histogram of replay arms')

    # 3. extract replay arm assignments from stim_message and plot
    replay_message_file = stim_message
    replays_to_plot = np.zeros((len(replay_message_file),4))

    arm_1_count = 0
    arm_2_count = 0
    other_replay_count = 0

    for i in np.arange(len(replay_message_file)):
        if (replay_message_file[i:i+1]['posterior_max_arm'].values == 1 and
            replay_message_file[i:i+1]['taskState'].values == 1):
            arm_1_count += 1
            replays_to_plot[i,0] = replay_message_file[i:i+1]['bin_timestamp'].values
            replays_to_plot[i,1] = replay_message_file[i:i+1]['taskState'].values
            replays_to_plot[i,2] = arm_1_count
            replays_to_plot[i,3] = 1
        elif (replay_message_file[i:i+1]['posterior_max_arm'].values == 2 and
              replay_message_file[i:i+1]['taskState'].values == 1):
            arm_2_count += 1
            replays_to_plot[i,0] = replay_message_file[i:i+1]['bin_timestamp'].values
            replays_to_plot[i,1] = replay_message_file[i:i+1]['taskState'].values
            replays_to_plot[i,2] = arm_2_count
            replays_to_plot[i,3] = 2
        elif replay_message_file[i:i+1]['taskState'].values == 1:
            other_replay_count += 1
            replays_to_plot[i,0] = replay_message_file[i:i+1]['bin_timestamp'].values
            replays_to_plot[i,1] = replay_message_file[i:i+1]['taskState'].values
            replays_to_plot[i,2] = other_replay_count
            replays_to_plot[i,3] = 99

    replays_to_plot_df = pd.DataFrame(data=replays_to_plot,columns=('bin_timestamp','taskState',
                                                                    'replay_count','replay_arm'))

    plt.figure(figsize=(6,6))
    plt.scatter(replays_to_plot_df[replays_to_plot_df['replay_arm']==1]['bin_timestamp'].values/30000,
                replays_to_plot_df[replays_to_plot_df['replay_arm']==1]['replay_count'].values,s=5,c='g')
    plt.scatter(replays_to_plot_df[replays_to_plot_df['replay_arm']==2]['bin_timestamp'].values/30000,
                replays_to_plot_df[replays_to_plot_df['replay_arm']==2]['replay_count'].values,s=5,c='r')
    plt.scatter(replays_to_plot_df[replays_to_plot_df['replay_arm']==99]['bin_timestamp'].values/30000,
                replays_to_plot_df[replays_to_plot_df['replay_arm']==99]['replay_count'].values,s=5,c='k')
    plt.figtext(0.2, 0.84, "Arm 1", ha="center", va="bottom", size="medium",color="g")
    plt.figtext(0.2, 0.8, "Arm 2", ha="center", va="bottom", size="medium",color="r")
    plt.figtext(0.2, 0.76, "Others", ha="center", va="bottom", size="medium",color="k")
    plt.xlabel('Time')
    plt.title('Replay count during session')

    plot_image_name = os.path.join(path_out, 'Replay_arm1_v_arm2.pdf')

    plt.savefig(plot_image_name)
    plt.close()
    print('saved posterior plot')

    # 4. plot posterior for all arm1 or arm2 replays

    # which files to use
    stim_message_file = stim_message
    decoder_data_file = decoder_data

    for index, timestamp in enumerate(stim_message_file['bin_timestamp']):
        if (stim_message_file[index:index+1]['posterior_max_arm'].values > 0 and 
            stim_message_file[index:index+1]['posterior_max_arm'].values < 3):

            posterior_to_plot = decoder_data_file[(decoder_data_file['bin_timestamp'] > timestamp-30*300) & 
                                                (decoder_data_file['bin_timestamp'] < timestamp+30*300)]
            posterior_to_plot = posterior_to_plot.reset_index()
            posterior_only_merged = posterior_to_plot.iloc[:,26:164]
            
            # start of ripple: lockout_message_joined: timestamp
            ripple_start = posterior_to_plot.index[(posterior_to_plot['bin_timestamp'].values >
                                                   lockout_message_joined[index:index+1]['timestamp'].values) &
                                                   (posterior_to_plot['bin_timestamp'].values <
                                                   lockout_message_joined[index:index+1]['timestamp'].values+250)][0]
            # end of ripple: stim_message: bin_timestamp
            ripple_end = posterior_to_plot.index[posterior_to_plot['bin_timestamp'].values ==
                                                 stim_message_file[index:index+1]['bin_timestamp'].values][0]

            # shortcut message arm
            max_arm = stim_message_file[index:index+1]['posterior_max_arm'].values[0]
            
            # shortcut message
            shortcut_message = stim_message_file[index:index+1]['shortcut_message_sent'].values[0]        

            # taskState
            taskState = stim_message_file[index:index+1]['taskState'].values[0]        

            #heatmap of posterior
            post_heatmap = posterior_only_merged.transpose()
            post_heatmap = post_heatmap.iloc[::-1]

            plt.figure(figsize=(8,4))
            plt.title(f'RT posterior, TaskState: {taskState} Rip msg: {index} Max arm: {max_arm} Shortcut: {shortcut_message}')
            ax = (sns.heatmap(post_heatmap,vmin=0, vmax=0.7))
            ##gap lines need to be inverse of where you would expect
            ax.hlines([136-11,136-27,136-43,136-59,136-75,136-91,136-107,136-123], lw=1, color='w',*ax.get_xlim())
            ax.scatter(np.arange(0,posterior_to_plot.shape[0]),136-posterior_to_plot['real_pos'].values,s=1,alpha=0.5,color='cyan')
            ## could take delay into account here: add delay/5
            ax.scatter(ripple_start,136-11,s=30,color='red',marker='x')
            ax.scatter(ripple_end,136-11,s=30,color='yellow',marker='x')

            # final step: save the figure
            plt.savefig(f'{path_out}RT_posterior_ripple_message_{index}.pdf')
            plt.close()

    # 4. calculate max for real-time posteriors
    post_error = decoder_data.copy()

    #post_error.drop(columns=['rec_ind','bin_timestamp','wall_time','velocity','real_pos',
    #                         'raw_x','raw_y','smooth_x','smooth_y','next_bin',
    #                         'spike_count','ripple','ripple_number','ripple_length',
    #                         'shortcut_message','box','arm1','arm2','arm3','arm4','arm5',
    #                         'arm6','arm7','arm8'], inplace=True)

    post_error = post_error.iloc[:,25:162]
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
    error_realtime = post_error.copy()
    error_realtime = error_realtime[error_realtime['linvel_flat']>4]
    print('decode movement times',error_realtime.shape)

    realtime_analysis_util.non_local_error(error_realtime,binned_arm_coords_realtime)
  
    # real-time local error
    realtime_analysis_util.local_error(error_realtime,binned_arm_coords_realtime)


    print("End of script!")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', action='store', dest='realtime_rec', help='Real-time rec filename') 
    parser.add_argument('-o', action='store', dest='path_out', help='Path to output')
    results = parser.parse_args()
    main(results.realtime_rec,results.path_out)

