import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from functools import partial


class LinearDecodeError:

    @staticmethod
    def conv_center_pos(pos, arm_coord):
        """
        Compute distance from center arm
        """
        if (pos >= arm_coord[0][0]) and (pos <= arm_coord[0][1]):
            return pos
        elif (pos >= arm_coord[1][0]) and (pos <= arm_coord[1][1]):
            return arm_coord[1][1]-pos+arm_coord[0][1]
        elif (pos >= arm_coord[2][0]) and (pos <= arm_coord[2][1]):
            return arm_coord[2][1]-pos+arm_coord[0][1]
        else:
            warnings.warn("Position {} not valid for center arm coordinate.".format(pos))
            return -pos

    @staticmethod
    def conv_left_pos(pos, arm_coord):
        if pos < arm_coord[0][1]:
            return arm_coord[0][1]-pos+arm_coord[1][1]-arm_coord[1][0]
        elif (pos >= arm_coord[1][0]) and (pos <= arm_coord[1][1]):
            return pos-arm_coord[1][0]
        elif (pos >= arm_coord[2][0]) and (pos <= arm_coord[2][1]):
            return arm_coord[2][1]-pos+arm_coord[1][1]-arm_coord[1][0]
        else:
            warnings.warn("Position {} not valid for left arm coordinate.".format(pos))
            return -pos

    @staticmethod
    def conv_right_pos(pos, arm_coord):
        if pos < arm_coord[0][1]:
            return arm_coord[0][1]-pos+arm_coord[2][1]-arm_coord[2][0]
        elif (pos >= arm_coord[1][0]) and (pos <= arm_coord[1][1]):
            return arm_coord[1][1]-pos+arm_coord[2][1]-arm_coord[2][0]
        elif (pos >= arm_coord[2][0]) and (pos <= arm_coord[2][1]):
            return pos-arm_coord[2][0]
        else:
            warnings.warn("Position {} not valid for right arm coordinate.".format(pos))
            return -pos

    @staticmethod
    def conv_arm_pos(arm_dec_est, arm_coordinates, conv_func, relative_well_label):
        new_arm_dec_est = pd.DataFrame()
        new_arm_dec_est.loc[:, 'est_pos'] = arm_dec_est['est_pos'].map(partial(conv_func,
                                                                               arm_coord=arm_coordinates))
        new_arm_dec_est.loc[:, 'real_pos'] = arm_dec_est[relative_well_label]

        return new_arm_dec_est

    @staticmethod
    def calc_error_for_plot(dec_est):
        dec_est.loc[:, 'error'] = dec_est['est_pos'] - dec_est['real_pos']
        dec_est.loc[:, 'abs_error'] = np.abs(dec_est['error'])
        dec_est.loc[:, 'plt_error_up'] = dec_est['error']
        dec_est.loc[dec_est['error'] < 0, 'plt_error_up'] = 0
        dec_est.loc[:, 'plt_error_down'] = dec_est['error']
        dec_est.loc[dec_est['error'] > 0, 'plt_error_down'] = 0
        dec_est.loc[:, 'plt_error_down'] = np.abs(dec_est['plt_error_down'])

        return dec_est

    def calc_error_table(self, lin_obj, dec_est_pos, arm_coordinates, vel_thresh):
        """
        
        Args:
            pos_data_bins: Expects columns lin_vel_center 
            dec_est_pos: 
            arm_coordinates: 
            vel_thresh: 

        Returns:

        """
        # Reindex and join real position (linpos) to the decode estimated position table
        dec_est_and_linpos = dec_est_pos.join(lin_obj)

        # Select rows only when velocity meets criterion
        dec_est_and_linpos = dec_est_and_linpos[np.abs(dec_est_and_linpos['lin_vel_center']) >= vel_thresh]

        # Separate out each arm's position
        center_dec_est_merged = dec_est_and_linpos[dec_est_and_linpos['seg_idx'] == 1]
        left_dec_est_merged = dec_est_and_linpos[(dec_est_and_linpos['seg_idx'] == 2) |
                                                 (dec_est_and_linpos['seg_idx'] == 3)]
        right_dec_est_merged = dec_est_and_linpos[(dec_est_and_linpos['seg_idx'] == 4) |
                                                  (dec_est_and_linpos['seg_idx'] == 5)]

        # Apply "closest well centric" tranform to each arm's data
        center_dec_est = self.conv_arm_pos(center_dec_est_merged, arm_coordinates, self.conv_center_pos, 'well_center')

        left_dec_est = self.conv_arm_pos(left_dec_est_merged, arm_coordinates, self.conv_left_pos, 'well_left')

        right_dec_est = self.conv_arm_pos(right_dec_est_merged, arm_coordinates, self.conv_right_pos, 'well_right')

        center_dec_est = self.calc_error_for_plot(center_dec_est)
        center_dec_est.columns = pd.MultiIndex.from_product([['center'], center_dec_est.columns])

        left_dec_est = self.calc_error_for_plot(left_dec_est)
        left_dec_est.columns = pd.MultiIndex.from_product([['left'], left_dec_est.columns])

        right_dec_est = self.calc_error_for_plot(right_dec_est)
        right_dec_est.columns = pd.MultiIndex.from_product([['right'], right_dec_est.columns])

        dec_error = pd.concat([center_dec_est, left_dec_est, right_dec_est])

        dec_error.sort_index(inplace=True)

        return dec_error


