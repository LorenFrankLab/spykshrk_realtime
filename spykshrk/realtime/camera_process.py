import os
import struct
import numpy as np
import math
from mpi4py import MPI
from spykshrk.realtime import realtime_base, realtime_logging, binary_record, datatypes
from spykshrk.realtime.simulator import simulator_process

from spykshrk.realtime.datatypes import SpikePoint, LinearPosPoint, CameraModulePoint
from spykshrk.realtime.realtime_base import ChannelSelection, TurnOnDataStream
from spykshrk.realtime.tetrode_models import kernel_encoder
import spykshrk.realtime.rst.RSTPython as RST

# for now this script just holds the two classes: LinearPositionAssignment and VelocityCalculator
# the functions in these classed are called in encoder_process and decoder_process

class LinearPositionAssignment:
    def __init__(self):
        self.segment = 0
        self.segment_pos = 0
        self.shift_linear_distance_by_arm_dictionary = dict()
        #this line runs the arm_shift_dict during __init__
        self.arm_shift_dictionary()
        self.box_correction_count = 1

    def arm_shift_dictionary(self):
        # max_pos: 8 arm - 136 | 4 arm - 72
        # 0-6 = box, 7 = arm1 ... 14 = arm8
        # 6-9-19: seems to work as expected! matches offline linearization!
        # 8-15-19: updated for new track geometry
        # old way: 0 = home->rip/wait, 1-8 = rip/wait->arms, 9-16 = outer arms

        # old way with split box
        #hardcode_armorder = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16] #add progressive stagger in this order

        # new way with 8 parallel segments for box
        hardcode_armorder = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] #add progressive stagger in this order

        hardcode_shiftamount = 4 # add this stagger to sum of previous shifts (was 20 for 1cm)
        # for now set all arm lengths to 60 for 1cm (12 for 5cm)
        linearization_arm_length = 12

        # Define dictionary for shifts for each arm segment

        # with this setup max position is 136
        for arm in hardcode_armorder: # for each outer arm
            # not using inner vs outer box for remy
            # if inner box, do nothing
            #if arm == 0:
            #   temporary_variable_shift = 0

            # if outer box segments add inner box
            #elif arm < 9 and arm > 0:
            #   temporary_variable_shift = 4

            # replace inner and outer box with 8 parallel segments

            # toggle this for 8 vs 4 arms
            if arm < 8:
            #if arm < 4:
               temporary_variable_shift = 0

            # for first arm replace linearization_arm_length with 7 for the box
            # old segments for box, set this to 9, new parallel segments for box, set to 8

            # toggle this for 4 vs 8 arms
            elif arm == 8:
            #elif arm == 4:
               temporary_variable_shift = hardcode_shiftamount + 8
               #temporary_variable_shift = hardcode_shiftamount + 8 + self.shift_linear_distance_by_arm_dictionary[hardcode_armorder[arm - 1]]

            else: # if arms 2-8, shift with gap
               temporary_variable_shift = hardcode_shiftamount + 12 + self.shift_linear_distance_by_arm_dictionary[hardcode_armorder[arm - 1]]

            # # 4 arms with single segment between home and wait well
            # # home to wait
            # if arm == 0:
            #     temporary_variable_shift = 0

            # # outer half of box
            # elif arm > 0 and arm < 5:
            #     temporary_variable_shift = 5

            # # first outer arm
            # elif arm == 5:
            #     temporary_variable_shift = hardcode_shiftamount + 8

            # # outer arms
            # else:
            #     temporary_variable_shift = hardcode_shiftamount + 12 + self.shift_linear_distance_by_arm_dictionary[hardcode_armorder[arm - 1]]

            self.shift_linear_distance_by_arm_dictionary[arm] = temporary_variable_shift
            #print(self.shift_linear_distance_by_arm_dictionary)

        return self.shift_linear_distance_by_arm_dictionary

    def assign_position(self, segment, segment_pos):
        self.assigned_pos = 0
        #print('segment',segment,'location',np.around(segment_pos,decimals=2))

        # now we can use the good linearization, so box is 8 bins with 4 inner (segment 0) and 4 outer (segments 1-8)
        # bin position here - use math.ceil to round UP for arms and math.floor to round down for box

        #if segment == 0:
        #    self.assigned_pos = math.floor(segment_pos*4 + self.shift_linear_distance_by_arm_dictionary[segment])
        #elif segment > 0 and segment < 9:
        #    self.assigned_pos = math.floor(segment_pos*4 + self.shift_linear_distance_by_arm_dictionary[segment])

        # for linearization with just 8 parallel segments for box
        # fixed so that box is 9 bins long (this matches 45cm at 5cm/bin)
        # fixed so that any values in box where segment_pos = 1, get set back to bin 8, need bin 9 to be empty

        # for 8 arms
        if segment < 8:
            self.assigned_pos = math.floor(segment_pos*9 + self.shift_linear_distance_by_arm_dictionary[segment])
            if self.assigned_pos == 9:
                self.box_correction_count += 1
                self.assigned_pos = 8
                #print('edge of box position binning correction')
            if self.assigned_pos == -1:
                self.assigned_pos = 0
                #print('position was -1')
            if self.box_correction_count % 1000 == 0:
                print('edge of box pos correction count',self.box_correction_count)
        else:
            self.assigned_pos = math.ceil(segment_pos*12 + self.shift_linear_distance_by_arm_dictionary[segment])

        # for 4 arms with multiple paths from home
        #if segment < 4:
        #    self.assigned_pos = math.floor(segment_pos*9 + self.shift_linear_distance_by_arm_dictionary[segment])
        #    if self.assigned_pos == 9:
        #        self.assigned_pos = 8
        #        #print('edge of box position binning correction')
        #    if self.assigned_pos == -1:
        #        self.assigned_pos = 0
        #        #print('position was -1')
        #else:
        #    self.assigned_pos = math.ceil(segment_pos*12 + self.shift_linear_distance_by_arm_dictionary[segment])

        # # 4 arms with single segment between home and wait
        # if segment == 0:
        #     self.assigned_pos = math.floor(segment_pos*6 + self.shift_linear_distance_by_arm_dictionary[segment])
        #     if self.assigned_pos == 6:
        #         self.assigned_pos = 5
        # elif segment > 0 and segment < 5:
        #     self.assigned_pos = math.floor(segment_pos*4 + self.shift_linear_distance_by_arm_dictionary[segment])
        #     if self.assigned_pos == 9:
        #         self.assigned_pos = 8
        # else:
        #     self.assigned_pos = math.ceil(segment_pos*12 + self.shift_linear_distance_by_arm_dictionary[segment])

        return self.assigned_pos

class VelocityCalculator:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.lastx = 0
        self.lasty = 0
        self.smooth_x = 0
        self.smooth_y = 0

        #this is the number of speed or position measurements used to smooth - original = 30
        self.NSPEED_FILT_POINTS = 6
        self.speed = [0] * self.NSPEED_FILT_POINTS
        self.speedFilt = [0] * self.NSPEED_FILT_POINTS
        self.x_pos = [0] * self.NSPEED_FILT_POINTS
        self.x_pos_Filt = [0] * self.NSPEED_FILT_POINTS
        self.y_pos = [0] * self.NSPEED_FILT_POINTS
        self.y_pos_Filt = [0] * self.NSPEED_FILT_POINTS


        #this is a half-gaussian kernel used to smooth the instanteous speed
        #self.speedFilterValues = [0.0393,0.0392,0.0391,0.0389,0.0387,0.0385,0.0382,0.0379,
        #0.0375,0.0371,0.0367,0.0362,0.0357,0.0352,0.0347,0.0341,0.0334,0.0328,0.0321,
        #0.0315,0.0307,0.0300,0.0293,0.0285,0.0278,0.0270,0.0262,0.0254,0.0246,0.0238]

        # new filter for 6 points instead of 30
        # for this filter need velocity cut-off of about 6 cm/sec
        self.speedFilterValues = [0.21491511, 0.20760799, 0.1891253, 0.16161616, 0.12894907, 0.09778637]

        # new filter for 8 points instead of 30
        #self.speedFilterValues = [0.14753615, 0.14606078, 0.14148716, 0.13425789, 0.12540572, 0.11404544,
        #                          0.10194748, 0.08925937]

        for i in np.arange(0,self.NSPEED_FILT_POINTS,1):
            self.speedFilt[i] = self.speedFilterValues[i]
            self.x_pos_Filt[i] = self.speedFilterValues[i]
            self.y_pos_Filt[i] = self.speedFilterValues[i]

        self.ind = self.NSPEED_FILT_POINTS - 1;
        self.x_ind = self.NSPEED_FILT_POINTS - 1;
        self.y_ind = self.NSPEED_FILT_POINTS - 1;

    # this smoothes x position - okay this works, but the positions really dont match well...
    def smooth_x_position(self, x):
        # i think this should go above in init, so it remembers old position each bin - nope
        self.smooth_x = 0

        i = 0
        tmpind = 0
        cmperpx = 0.2

        self.x_pos[self.x_ind] = x
        #print(self.x_pos)
        # this is filling up fine, but x_pos_filt[i] is always 0 ????

        for i in np.arange(0,self.NSPEED_FILT_POINTS,1):
            tmpind = (self.x_ind + i) % self.NSPEED_FILT_POINTS
            #print('x pos filt',self.x_pos_Filt)
            self.smooth_x = self.smooth_x + self.x_pos[tmpind]*self.x_pos_Filt[i]

        self.x_ind = self.x_ind - 1

        if self.x_ind < 0:
            self.x_ind = self.NSPEED_FILT_POINTS - 1

        #print('smooth x',self.smooth_x)
        return self.smooth_x

    # this smoothes y position
    def smooth_y_position(self, y):
        # i think this should go above in init, so it remembers old position each bin - nope
        self.smooth_y = 0

        i = 0
        tmpind = 0
        cmperpx = 0.2

        self.y_pos[self.y_ind] = y

        for i in np.arange(0,self.NSPEED_FILT_POINTS,1):
            tmpind = (self.y_ind + i) % self.NSPEED_FILT_POINTS
            self.smooth_y = self.smooth_y + self.y_pos[tmpind]*self.y_pos_Filt[i]

        self.y_ind = self.y_ind - 1

        if self.y_ind < 0:
            self.y_ind = self.NSPEED_FILT_POINTS - 1

        return self.smooth_y

    # this is the velocity calculator
    def calculator(self, x, y):
        #print(x,y)
        self.smoothSpeed = 0
        i = 0
        tmpind = 0
        #need to bring in network.pxpercm from trodes
        #cmperpx = 1/network.pxpercm
        cmperpx = 0.2

        # note: for remy cmperpx should be 0.2
        # it seems like the speed is still pretty high with jittering of headstage...
        # maybe this is because positon isnt smoothed??

        # this line calcualates distance between the last 2 points
        self.speed[self.ind] = ((x * cmperpx - self.lastx) * (x * cmperpx - self.lastx) +
                      (y * cmperpx - self.lasty) * (y * cmperpx - self.lasty))
        #print(x,y,self.lastx,self.lasty,network.pxpercm,self.speed[0])

        # this is distance / time - because 1/0.03 = 30
        if self.speed[self.ind] != 0:
            self.speed[self.ind] = np.sqrt(self.speed[self.ind])*30
            #print('raw speed',np.around(self.speed[self.ind],decimals=2),'x',np.around(x,decimals=2),
            #      'y',np.around(y,decimals=2))

        self.lastx = x * cmperpx
        self.lasty = y * cmperpx

        for i in np.arange(0,self.NSPEED_FILT_POINTS,1):
            tmpind = (self.ind + i) % self.NSPEED_FILT_POINTS
            #print('speed filt',self.speedFilt)
            self.smoothSpeed = self.smoothSpeed + self.speed[tmpind]*self.speedFilt[i]

        self.ind = self.ind - 1

        if self.ind < 0:
            self.ind = self.NSPEED_FILT_POINTS - 1

        #print('speed',self.smoothSpeed)
        return self.smoothSpeed

    # this is the velocity calculator with no smoothing
    # should use this with smoothed position
    def calculator_no_smooth(self, x, y):
        #print(x,y)
        self.smoothSpeed = 0
        #i = 0
        #tmpind = 0
        #need to bring in network.pxpercm from trodes
        #cmperpx = 1/network.pxpercm
        cmperpx = 0.2

        # note: for remy cmperpx should be 0.2
        # it seems like the speed is still pretty high with jittering of headstage...
        # maybe this is because positon isnt smoothed??

        # this line calcualates distance between the last 2 points
        self.speed = ((x * cmperpx - self.lastx) * (x * cmperpx - self.lastx) +
                      (y * cmperpx - self.lasty) * (y * cmperpx - self.lasty))
        #print(x,y,self.lastx,self.lasty,network.pxpercm,self.speed[0])

        # this is distance / time - because 1/0.03 = 30
        if self.speed != 0:
            self.speed = np.sqrt(self.speed)*30
            #print('raw speed',np.around(self.speed[self.ind],decimals=2),'x',np.around(x,decimals=2),
            #      'y',np.around(y,decimals=2))

        self.lastx = x * cmperpx
        self.lasty = y * cmperpx

        #for i in np.arange(0,self.NSPEED_FILT_POINTS,1):
        #    tmpind = (self.ind + i) % self.NSPEED_FILT_POINTS
        #    #print('speed filt',self.speedFilt)
        #    self.smoothSpeed = self.smoothSpeed + self.speed[tmpind]*self.speedFilt[i]

        #self.ind = self.ind - 1

        #if self.ind < 0:
        #    self.ind = self.NSPEED_FILT_POINTS - 1

        ##print('speed',self.smoothSpeed)
        return self.speed

#initialize VelocityCalculator
velCalc = VelocityCalculator()

#initialize LinearPositionAssignment
linPosAssign = LinearPositionAssignment()
