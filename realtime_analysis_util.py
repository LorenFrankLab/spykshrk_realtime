# daily anlaysis functions for real-time experiment
import numpy as np

# function to cacluate non-local error (in fraction of time bins)
def non_local_error(posterior_with_error,arm_coords):
    box = posterior_with_error[(posterior_with_error['linpos_flat']>=arm_coords[0][0]) & (posterior_with_error['linpos_flat']<=arm_coords[0][1])]
    arm1 = posterior_with_error[(posterior_with_error['linpos_flat']>=arm_coords[1][0]) & (posterior_with_error['linpos_flat']<=arm_coords[1][1])]
    arm2 = posterior_with_error[(posterior_with_error['linpos_flat']>=arm_coords[2][0]) & (posterior_with_error['linpos_flat']<=arm_coords[2][1])]
    arm3 = posterior_with_error[(posterior_with_error['linpos_flat']>=arm_coords[3][0]) & (posterior_with_error['linpos_flat']<=arm_coords[3][1])]
    arm4 = posterior_with_error[(posterior_with_error['linpos_flat']>=arm_coords[4][0]) & (posterior_with_error['linpos_flat']<=arm_coords[4][1])]
    arm5 = posterior_with_error[(posterior_with_error['linpos_flat']>=arm_coords[5][0]) & (posterior_with_error['linpos_flat']<=arm_coords[5][1])]
    arm6 = posterior_with_error[(posterior_with_error['linpos_flat']>=arm_coords[6][0]) & (posterior_with_error['linpos_flat']<=arm_coords[6][1])]
    arm7 = posterior_with_error[(posterior_with_error['linpos_flat']>=arm_coords[7][0]) & (posterior_with_error['linpos_flat']<=arm_coords[7][1])]
    arm8 = posterior_with_error[(posterior_with_error['linpos_flat']>=arm_coords[8][0]) & (posterior_with_error['linpos_flat']<=arm_coords[8][1])]
    print(arm1.shape[0])
    print(posterior_with_error.shape)

    # fraction of bins with remote error, also substract time when posterior max is in box
    box_remote_error = ((box.shape[0] - 
                         box[(box['posterior_max']>=arm_coords[0][0]) & (box['posterior_max']<=arm_coords[0][1])].shape[0])/box.shape[0])
    arm1_remote_error = ((arm1.shape[0] - 
                          arm1[(arm1['posterior_max']>=arm_coords[1][0]) & (arm1['posterior_max']<=arm_coords[1][1])].shape[0] - 
                          arm1[arm1['posterior_max']<=arm_coords[0][1]].shape[0])/arm1.shape[0])
    arm2_remote_error = ((arm2.shape[0] - 
                          arm2[(arm2['posterior_max']>=arm_coords[2][0]) & (arm2['posterior_max']<=arm_coords[2][1])].shape[0] - 
                         arm2[arm2['posterior_max']<=arm_coords[0][1]].shape[0])/arm2.shape[0])
    arm3_remote_error = ((arm3.shape[0] - 
                          arm3[(arm3['posterior_max']>=arm_coords[3][0]) & (arm3['posterior_max']<=arm_coords[3][1])].shape[0] - 
                         arm3[arm3['posterior_max']<=arm_coords[0][1]].shape[0])/arm3.shape[0])
    arm4_remote_error = ((arm4.shape[0] - 
                          arm4[(arm4['posterior_max']>=arm_coords[4][0]) & (arm4['posterior_max']<=arm_coords[4][1])].shape[0] - 
                         arm4[arm4['posterior_max']<=arm_coords[0][1]].shape[0])/arm4.shape[0])
    arm5_remote_error = ((arm5.shape[0] - 
                          arm5[(arm5['posterior_max']>=arm_coords[5][0]) & (arm5['posterior_max']<=arm_coords[5][1])].shape[0] - 
                         arm5[arm5['posterior_max']<=arm_coords[0][1]].shape[0])/arm5.shape[0])
    arm6_remote_error = ((arm6.shape[0] - 
                          arm6[(arm6['posterior_max']>=arm_coords[6][0]) & (arm6['posterior_max']<=arm_coords[6][1])].shape[0] - 
                         arm6[arm6['posterior_max']<=arm_coords[0][1]].shape[0])/arm6.shape[0])
    arm7_remote_error = ((arm7.shape[0] - 
                          arm7[(arm7['posterior_max']>=arm_coords[7][0]) & (arm7['posterior_max']<=arm_coords[7][1])].shape[0] - 
                         arm7[arm7['posterior_max']<=arm_coords[0][1]].shape[0])/arm7.shape[0])
    arm8_remote_error = ((arm8.shape[0] - 
                          arm8[(arm8['posterior_max']>=arm_coords[8][0]) & (arm8['posterior_max']<=arm_coords[8][1])].shape[0] - 
                         arm8[arm8['posterior_max']<=arm_coords[0][1]].shape[0])/arm8.shape[0])

    #print error values
    print('individual remote error fractions:',
          box_remote_error,arm1_remote_error,arm2_remote_error,
          arm3_remote_error,arm4_remote_error,arm5_remote_error,
          arm6_remote_error,arm7_remote_error,arm8_remote_error)
    box_frac = box.shape[0]/posterior_with_error.shape[0]
    arm1_frac = arm1.shape[0]/posterior_with_error.shape[0]
    arm2_frac = arm2.shape[0]/posterior_with_error.shape[0]
    arm3_frac = arm3.shape[0]/posterior_with_error.shape[0]
    arm4_frac = arm4.shape[0]/posterior_with_error.shape[0]
    arm5_frac = arm5.shape[0]/posterior_with_error.shape[0]
    arm6_frac = arm6.shape[0]/posterior_with_error.shape[0]
    arm7_frac = arm7.shape[0]/posterior_with_error.shape[0]
    arm8_frac = arm8.shape[0]/posterior_with_error.shape[0]

    # fraction of arm-only time
    arm1_frac1 = arm1.shape[0]/(posterior_with_error.shape[0]*(1-box_frac))
    arm2_frac1 = arm2.shape[0]/(posterior_with_error.shape[0]*(1-box_frac))
    arm3_frac1 = arm3.shape[0]/(posterior_with_error.shape[0]*(1-box_frac))
    arm4_frac1 = arm4.shape[0]/(posterior_with_error.shape[0]*(1-box_frac))
    arm5_frac1 = arm5.shape[0]/(posterior_with_error.shape[0]*(1-box_frac))
    arm6_frac1 = arm6.shape[0]/(posterior_with_error.shape[0]*(1-box_frac))
    arm7_frac1 = arm7.shape[0]/(posterior_with_error.shape[0]*(1-box_frac))
    arm8_frac1 = arm8.shape[0]/(posterior_with_error.shape[0]*(1-box_frac))

    #weighted average for box + each arm
    print('fraction remote error including box time:',
          (box_remote_error*box_frac + arm1_remote_error*arm1_frac +arm2_remote_error*arm2_frac +
           arm3_remote_error*arm3_frac +arm4_remote_error*arm4_frac +arm5_remote_error*arm5_frac +
           arm6_remote_error*arm6_frac +arm7_remote_error*arm7_frac +arm8_remote_error*arm8_frac))
    # weighted average no box
    print('fraction remote error arms only:',
          (arm1_remote_error*arm1_frac1 +arm2_remote_error*arm2_frac1 +arm3_remote_error*arm3_frac1 +
           arm4_remote_error*arm4_frac1 +arm5_remote_error*arm5_frac1 +arm6_remote_error*arm6_frac1 +
           arm7_remote_error*arm7_frac1 +arm8_remote_error*arm8_frac1))

# function to cacluate local error (in cm)
def local_error(posterior_with_error,arm_coords):
	# define box and each arm
    box = posterior_with_error[(posterior_with_error['linpos_flat']>=arm_coords[0][0]) & (posterior_with_error['linpos_flat']<=arm_coords[0][1])]
    arm1 = posterior_with_error[(posterior_with_error['linpos_flat']>=arm_coords[1][0]) & (posterior_with_error['linpos_flat']<=arm_coords[1][1])]
    arm2 = posterior_with_error[(posterior_with_error['linpos_flat']>=arm_coords[2][0]) & (posterior_with_error['linpos_flat']<=arm_coords[2][1])]
    arm3 = posterior_with_error[(posterior_with_error['linpos_flat']>=arm_coords[3][0]) & (posterior_with_error['linpos_flat']<=arm_coords[3][1])]
    arm4 = posterior_with_error[(posterior_with_error['linpos_flat']>=arm_coords[4][0]) & (posterior_with_error['linpos_flat']<=arm_coords[4][1])]
    arm5 = posterior_with_error[(posterior_with_error['linpos_flat']>=arm_coords[5][0]) & (posterior_with_error['linpos_flat']<=arm_coords[5][1])]
    arm6 = posterior_with_error[(posterior_with_error['linpos_flat']>=arm_coords[6][0]) & (posterior_with_error['linpos_flat']<=arm_coords[6][1])]
    arm7 = posterior_with_error[(posterior_with_error['linpos_flat']>=arm_coords[7][0]) & (posterior_with_error['linpos_flat']<=arm_coords[7][1])]
    arm8 = posterior_with_error[(posterior_with_error['linpos_flat']>=arm_coords[8][0]) & (posterior_with_error['linpos_flat']<=arm_coords[8][1])]
    print(arm1.shape[0])
    print(posterior_with_error.shape)
    	
	# fraction of each arm dataframe where max position is local
    box_local = box[(box['posterior_max']>=arm_coords[0][0]) & (box['posterior_max']<=arm_coords[0][1])]
    arm1_local = arm1[(arm1['posterior_max']>=arm_coords[1][0]) & (arm1['posterior_max']<=arm_coords[1][1])]
    arm2_local = arm2[(arm2['posterior_max']>=arm_coords[2][0]) & (arm2['posterior_max']<=arm_coords[2][1])]
    arm3_local = arm3[(arm3['posterior_max']>=arm_coords[3][0]) & (arm3['posterior_max']<=arm_coords[3][1])]
    arm4_local = arm4[(arm4['posterior_max']>=arm_coords[4][0]) & (arm4['posterior_max']<=arm_coords[4][1])]
    arm5_local = arm5[(arm5['posterior_max']>=arm_coords[5][0]) & (arm5['posterior_max']<=arm_coords[5][1])]
    arm6_local = arm6[(arm6['posterior_max']>=arm_coords[6][0]) & (arm6['posterior_max']<=arm_coords[6][1])]
    arm7_local = arm7[(arm7['posterior_max']>=arm_coords[7][0]) & (arm7['posterior_max']<=arm_coords[7][1])]
    arm8_local = arm8[(arm8['posterior_max']>=arm_coords[8][0]) & (arm8['posterior_max']<=arm_coords[8][1])]

    box_local_error = np.median(box_local['error_cm'].values)
    arm1_local_error = np.median(arm1_local['error_cm'].values)
    arm2_local_error = np.median(arm2_local['error_cm'].values)
    arm3_local_error = np.median(arm3_local['error_cm'].values)
    arm4_local_error = np.median(arm4_local['error_cm'].values)
    arm5_local_error = np.median(arm5_local['error_cm'].values)
    arm6_local_error = np.median(arm6_local['error_cm'].values)
    arm7_local_error = np.median(arm7_local['error_cm'].values)
    arm8_local_error = np.median(arm8_local['error_cm'].values)

    print('local error median each arm:',arm1_local_error,arm2_local_error,arm3_local_error,
          arm4_local_error,arm5_local_error,arm6_local_error,
          arm7_local_error,arm8_local_error,)

    # total bins with local error
    local_bin_frac = (np.around(arm1_local.shape[0]/arm1.shape[0],decimals=2),
                      np.around(arm2_local.shape[0]/arm2.shape[0],decimals=2),
                      np.around(arm3_local.shape[0]/arm3.shape[0],decimals=2),
                      np.around(arm4_local.shape[0]/arm4.shape[0],decimals=2),
                      np.around(arm5_local.shape[0]/arm5.shape[0],decimals=2),
                      np.around(arm6_local.shape[0]/arm6.shape[0],decimals=2),
                      np.around(arm7_local.shape[0]/arm7.shape[0],decimals=2),
                      np.around(arm8_local.shape[0]/arm8.shape[0],decimals=2))
    print('fraction each arm local:',local_bin_frac)
    #weighted average for each arm
    # add 1 here becuase of divide by 0 error
    local_total_bins = (1 + arm1_local.shape[0]+arm2_local.shape[0]+
                        arm3_local.shape[0]+
                        arm4_local.shape[0]+arm5_local.shape[0]+
                        arm6_local.shape[0]+
                        arm7_local.shape[0]+arm8_local.shape[0])
    print('weighted average for all arms:',(arm1_local_error*(arm1_local.shape[0]/local_total_bins)+
     arm2_local_error*(arm2_local.shape[0]/local_total_bins)+
     arm3_local_error*(arm3_local.shape[0]/local_total_bins)+
     arm4_local_error*(arm4_local.shape[0]/local_total_bins)+
     arm5_local_error*(arm5_local.shape[0]/local_total_bins)+
     arm6_local_error*(arm6_local.shape[0]/local_total_bins)+
     arm7_local_error*(arm7_local.shape[0]/local_total_bins)+
     arm8_local_error*(arm8_local.shape[0]/local_total_bins)))

