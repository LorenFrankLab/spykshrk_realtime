from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer, QElapsedTimer
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QGridLayout,
                               QLineEdit, QGroupBox, QHBoxLayout, QDialog,
                               QPushButton, QLabel, QSpinBox, QSlider, QStatusBar,
                               QFileDialog, QMessageBox, QRadioButton, QTextEdit)
from spykshrk.realtime.realtime_base import (RealtimeProcess, TerminateMessage,
                                MPIMessageTag, TimeSyncInit, SetupComplete)
from spykshrk.realtime.realtime_logging import PrintableMessage
import logging
import pyqtgraph as pg
import numpy as np
from matplotlib import cm
import seaborn as sns
from mpi4py import MPI

DEFAULT_SEND_PARAMS = {
    "target_arm" : 1,
    "posterior_threshold" : 0.5,
    "num_above_thresh" : 1,
    "max_center_well_dist" : 1,
    "ripple_threshold" : 3,
    "conditioning_ripple_threshold" : 3,
    "ripple_velocity_threshold" : 10,
    "encoding_velocity_threshold" : 10,
    "shortcut_message_on" : False,
    "instructive_task" : False,
    "reward_mode" : "replay"
}

DEFAULT_GUI_PARAMS = {
    "colormap" : "rocket"
}

class GuiMainParameterMessage(PrintableMessage):
    def __init__(
        self, target_arm:int, posterior_threshold:float,
        num_above_threshold:int, max_center_well_distance:float,
        ripple_velocity_threshold:float, shortcut_message_on:bool,
        instructive_task:bool, reward_mode:str):

        self.target_arm = target_arm
        self.posterior_threshold = posterior_threshold
        self.num_above_threshold = num_above_threshold
        self.max_center_well_distance = max_center_well_distance
        self.ripple_velocity_threshold = ripple_velocity_threshold
        self.shortcut_message_on = shortcut_message_on
        self.instructive_task = instructive_task
        self.reward_mode = reward_mode

class GuiRippleParameterMessage(PrintableMessage):
    def __init__(self, ripple_threshold:float, conditioning_ripple_threshold:float):
        self.ripple_threshold = ripple_threshold
        self.conditioning_ripple_threshold = conditioning_ripple_threshold

class GuiEncoderParameterMessage(PrintableMessage):
    def __init__(self, encoding_velocity_threshold:float):
        self.encoding_velocity_threshold = encoding_velocity_threshold

class GuiDecoderParameterMessage(PrintableMessage):
    def __init__(self, encoding_velocity_threshold:float):
        self.encoding_velocity_threshold = encoding_velocity_threshold

def show_message(parent, text, *, kind=None):
    if kind is None:
        kind = QMessageBox.NoIcon
    elif kind == "question":
        kind = QMessageBox.Question
    elif kind == "information":
        kind = QMessageBox.Information
    elif kind == "warning":
        kind = QMessageBox.Warning
    elif kind == "critical":
        kind = QMessageBox.Critical
    else:
        msg = QMessageBox(parent)
        msg.setText(f"Invalid message kind '{kind}' specified")
        msg.setIcon(QMessageBox.Critical)
        msg.addButton(QMessageBox.Ok)
        msg.exec_()
        return
    
    msg = QMessageBox(parent)
    msg.setText(text)
    msg.setIcon(kind)
    msg.addButton(QMessageBox.Ok)
    msg.exec_()


class Dialog(QDialog):

    def __init__(self, parent, comm, rank, config):
        super().__init__(parent)
        self.comm = comm
        self.rank = rank
        self.config = config
        self.setWindowTitle("Parameters")

        # messages that will be sent to other processes. their state
        # is initialized during setup and mutated when the user presses
        # the send button
        self.main_params = GuiMainParameterMessage(
            None, None, None, None, None, None, None, None
        )
        self.ripple_params = GuiRippleParameterMessage(
            None, None
        )
        self.encoder_params = GuiEncoderParameterMessage(
            None
        )
        self.decoder_params = GuiDecoderParameterMessage(
            None
        )

        # Add widgets with the following convention:
        # 1. Instantiate
        # 2. Set any shortcuts
        # 3. Set any attributes e.g. tool tips
        # 4. Connections
        # 5. Enable or disable
        # 6. Add to layout
        layout = QGridLayout(self)
        self.setup_target_arm(layout)
        self.setup_post_thresh(layout)
        self.setup_n_above_threshold(layout)
        self.setup_max_center_well_distance(layout)
        self.setup_ripple_thresh(layout)
        self.setup_conditioning_ripple_thresh(layout)
        self.setup_ripple_vel_thresh(layout)
        self.setup_encoding_vel_thresh(layout)
        self.setup_shortcut_message(layout)
        self.setup_instructive_task(layout)
        self.setup_reward_mode(layout)

        self.timer = QTimer()
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.send_all_params)

    def setup_target_arm(self, layout):
        self.target_arm_label = QLabel(self.tr("Target Arm"))
        layout.addWidget(self.target_arm_label, 0, 0)

        self.target_arm_edit = QLineEdit()
        layout.addWidget(self.target_arm_edit, 0, 1)

        self.target_arm_button = QPushButton(self.tr("Update"))
        self.target_arm_button.pressed.connect(self.check_target_arm)
        layout.addWidget(self.target_arm_button, 0, 2)

        try:
            value = int(self.config['ripple_conditioning']['replay_target_arm'])
        except KeyError:
            value = int(DEFAULT_SEND_PARAMS['target_arm'])
        self.target_arm_edit.setText(str(value))
        self.main_params.target_arm = value

    def setup_post_thresh(self, layout):
        self.post_label = QLabel(self.tr("Posterior Threshold"))
        layout.addWidget(self.post_label, 1, 0)
        
        self.post_edit = QLineEdit()
        layout.addWidget(self.post_edit, 1, 1)

        self.post_thresh_button = QPushButton(self.tr("Update"))
        self.post_thresh_button.pressed.connect(self.check_post_thresh)
        layout.addWidget(self.post_thresh_button, 1, 2)

        try:
            value = float(self.config['ripple_conditioning']['posterior_sum_threshold'])
        except KeyError:
            value = float(DEFAULT_SEND_PARAMS['posterior_threshold'])
        self.post_edit.setText(str(value))
        self.main_params.posterior_threshold = value

    def setup_n_above_threshold(self, layout):
        self.n_above_label = QLabel(self.tr("Num. Tetrodes Above Threshold"))
        layout.addWidget(self.n_above_label, 2, 0)
        
        self.n_above_edit = QLineEdit()
        layout.addWidget(self.n_above_edit, 2, 1)

        self.n_above_button = QPushButton(self.tr("Update"))
        self.n_above_button.pressed.connect(self.check_n_above)
        layout.addWidget(self.n_above_button, 2, 2)

        try:
            value = int(self.config['ripple']['RippleParameterMessage']['n_above_thresh'])
        except KeyError:
            value = int(DEFAULT_SEND_PARAMS['num_above_thresh'])
        self.n_above_edit.setText(str(value))
        self.main_params.num_above_threshold = value

    def setup_max_center_well_distance(self, layout):
        self.max_center_well_label = QLabel(self.tr("Max Center Well Distance"))
        layout.addWidget(self.max_center_well_label, 3, 0)
        
        self.max_center_well_edit = QLineEdit()
        layout.addWidget(self.max_center_well_edit, 3, 1)

        self.max_center_well_button = QPushButton(self.tr("Update"))
        self.max_center_well_button.pressed.connect(self.check_max_center_well)
        layout.addWidget(self.max_center_well_button, 3, 2)

        try:
            value = float(self.config['ripple_conditioning']['max_center_well_dist'])
        except KeyError:
            value = float(DEFAULT_SEND_PARAMS['max_center_well_dist'])
        self.max_center_well_edit.setText(str(value))
        self.main_params.max_center_well_distance = value

    def setup_ripple_thresh(self, layout):
        self.rip_thresh_label = QLabel(self.tr("Ripple Threshold"))
        layout.addWidget(self.rip_thresh_label, 4, 0)
        
        self.rip_thresh_edit = QLineEdit()
        layout.addWidget(self.rip_thresh_edit, 4, 1)

        self.rip_thresh_button = QPushButton(self.tr("Update"))
        self.rip_thresh_button.pressed.connect(self.check_rip_thresh)
        layout.addWidget(self.rip_thresh_button, 4, 2)

        try:
            value = float(self.config['ripple']['RippleParameterMessage']['ripple_threshold'])
        except KeyError:
            value = float(DEFAULT_SEND_PARAMS['ripple_threshold'])
        self.rip_thresh_edit.setText(str(value))
        self.ripple_params.ripple_threshold = value

    def setup_conditioning_ripple_thresh(self, layout):
        self.cond_rip_thresh_label = QLabel(self.tr("Conditioning Ripple Threshold"))
        layout.addWidget(self.cond_rip_thresh_label, 5, 0)
        
        self.cond_rip_thresh_edit = QLineEdit()
        layout.addWidget(self.cond_rip_thresh_edit, 5, 1)

        self.cond_rip_thresh_button = QPushButton(self.tr("Update"))
        self.cond_rip_thresh_button.pressed.connect(self.check_cond_rip_thresh)
        layout.addWidget(self.cond_rip_thresh_button, 5, 2)

        try:
            value = float(self.config['ripple_conditioning']['condition_rip_thresh'])
        except KeyError:
            value = float(DEFAULT_SEND_PARAMS['conditioning_ripple_threshold'])
        self.cond_rip_thresh_edit.setText(str(value))
        self.ripple_params.conditioning_ripple_threshold = value

    def setup_ripple_vel_thresh(self, layout):
        self.ripple_vel_thresh_label = QLabel(self.tr("Ripple Velocity Threshold"))
        layout.addWidget(self.ripple_vel_thresh_label, 6, 0)
        
        self.ripple_vel_thresh_edit = QLineEdit()
        layout.addWidget(self.ripple_vel_thresh_edit, 6, 1)

        self.ripple_vel_thresh_button = QPushButton(self.tr("Update"))
        self.ripple_vel_thresh_button.pressed.connect(self.check_ripple_vel_thresh)
        layout.addWidget(self.ripple_vel_thresh_button, 6, 2)

        try:
            value = float(self.config['ripple_conditioning']['ripple_detect_velocity'])
        except KeyError:
            value = float(DEFAULT_SEND_PARAMS['ripple_velocity_threshold'])
        self.ripple_vel_thresh_edit.setText(str(value))
        self.main_params.ripple_velocity_threshold = value

    def setup_encoding_vel_thresh(self, layout):
        self.encoding_vel_thresh_label = QLabel(self.tr("Encoding Velocity Threshold"))
        layout.addWidget(self.encoding_vel_thresh_label, 7, 0)
        
        self.encoding_vel_thresh_edit = QLineEdit()
        layout.addWidget(self.encoding_vel_thresh_edit, 7, 1)

        self.encoding_vel_thresh_button = QPushButton(self.tr("Update"))
        self.encoding_vel_thresh_button.pressed.connect(self.check_encoding_vel_thresh)
        layout.addWidget(self.encoding_vel_thresh_button, 7, 2)

        try:
            value = float(self.config['encoder']['vel'])
        except KeyError:
            value = float(DEFAULT_SEND_PARAMS['encoding_velocity_threshold'])
        self.encoding_vel_thresh_edit.setText(str(value))
        self.encoder_params.encoding_velocity_threshold = value
        self.decoder_params.encoding_velocity_threshold = value

    def setup_shortcut_message(self, layout):
        self.shortcut_label = QLabel(self.tr("Shortcut Message"))
        layout.addWidget(self.shortcut_label, 8, 0)
        
        self.shortcut_on = QRadioButton(self.tr("ON"))        
        self.shortcut_off = QRadioButton(self.tr("OFF"))
        shortcut_layout = QHBoxLayout()
        shortcut_layout.addWidget(self.shortcut_on)
        shortcut_layout.addWidget(self.shortcut_off)
        shortcut_group_box = QGroupBox()
        shortcut_group_box.setLayout(shortcut_layout)
        layout.addWidget(shortcut_group_box, 8, 1)

        self.shortcut_message_button = QPushButton(self.tr("Update"))
        self.shortcut_message_button.pressed.connect(self.check_shortcut)
        layout.addWidget(self.shortcut_message_button, 8, 2)

        try:
            if bool(self.config['ripple_conditioning']['shortcut_msg_on']):
                self.shortcut_on.setChecked(True)
                self.main_params.shortcut_message_on = True
            else:
                self.shortcut_off.setChecked(True)
                self.main_params.shortcut_message_on = False
        except KeyError:
            if DEFAULT_SEND_PARAMS['shortcut_msg_on']:
                self.shortcut_on.setChecked(True)
                self.main_params.shortcut_message_on = True
            else:
                self.shortcut_off.setChecked(True)
                self.main_params.shortcut_message_on = False

    def setup_instructive_task(self, layout):
        self.instructive_task_label = QLabel(self.tr("Instructive Task"))
        layout.addWidget(self.instructive_task_label, 9, 0)

        self.instructive_task_on = QRadioButton(self.tr("ON"))        
        self.instructive_task_off = QRadioButton(self.tr("OFF"))
        instructive_task_layout = QHBoxLayout()
        instructive_task_layout.addWidget(self.instructive_task_on)
        instructive_task_layout.addWidget(self.instructive_task_off)
        instructive_task_group_box = QGroupBox()
        instructive_task_group_box.setLayout(instructive_task_layout)
        layout.addWidget(instructive_task_group_box, 9, 1)

        self.instructive_task_button = QPushButton(self.tr("Update"))
        self.instructive_task_button.pressed.connect(self.check_instructive_task)
        layout.addWidget(self.instructive_task_button, 9, 2)

        try:
            if bool(self.config['ripple_conditioning']['instructive']):
                self.instructive_task_on.setChecked(True)
                self.main_params.instructive_task = True
            else:
                self.instructive_task_off.setChecked(True)
                self.main_params.instructive_task = False
        except KeyError:
            if DEFAULT_SEND_PARAMS["instructive_task"]:
                self.instructive_task_on.setChecked(True)
                self.main_params.instructive_task = True
            else:
                self.instructive_task_off.setChecked(True)
                self.main_params.instructive_task = False
        
        # Comment out when instructive task needed
        self.instructive_task_on.setEnabled(False)
        self.instructive_task_off.setEnabled(False)
        self.instructive_task_button.setEnabled(False)

    def setup_reward_mode(self, layout):
        self.reward_mode_label = QLabel(self.tr("Reward Mode"))
        layout.addWidget(self.reward_mode_label, 10, 0)

        self.reward_mode_conditioning_ripples = QRadioButton(self.tr("Conditioning ripples"))        
        self.reward_mode_replay = QRadioButton(self.tr("Replay"))
        reward_mode_layout = QHBoxLayout()
        reward_mode_layout.addWidget(self.reward_mode_conditioning_ripples)
        reward_mode_layout.addWidget(self.reward_mode_replay)
        reward_mode_group_box = QGroupBox()
        reward_mode_group_box.setLayout(reward_mode_layout)
        layout.addWidget(reward_mode_group_box, 10, 1)

        self.reward_mode_button = QPushButton(self.tr("Update"))
        self.reward_mode_button.pressed.connect(self.check_reward_mode)
        layout.addWidget(self.reward_mode_button, 10, 2)

        try:
            if self.config["reward_mode"] == "conditioning_ripples":
                self.reward_mode_conditioning_ripples.setChecked(True)
                self.main_params.reward_mode = "conditioning_ripples"
            elif self.config["reward_mode"] == "replay":
                self.reward_mode_replay.setChecked(True)
                self.main_params.reward_mode = "replay"
            else:
                show_message(
                    self,
                    f"Invalid reward mode \"{self.config['reward_mode']}\" listed in config file. "
                    "Setting to default 'replay'",
                    kind="critical"
                )
                self.main_params.reward_mode = "replay"
        except KeyError:
            if DEFAULT_SEND_PARAMS['reward_mode'] == "conditioning_ripples":
                self.reward_mode_conditioning_ripples.setChecked(True)
                self.main_params.reward_mode = "conditioning_ripples"
            elif DEFAULT_SEND_PARAMS['reward_mode'] == "replay":
                self.reward_mode_replay.setChecked(True)
                self.main_params.reward_mode = "replay"
            else:
                show_message(
                    self,
                    f"Invalid reward mode \"{DEFAULT_SEND_PARAMS['reward_mode']}\" "
                    "listed as default parameter. Setting reward mode to replay",
                    kind="critical"
                )
                self.reward_mode_conditioning_ripples.setChecked(True)
                self.main_params.reward_mode = "replay"
 
    def check_target_arm(self):

        target_arm = self.target_arm_edit.text()
        valid_num_arms = len(self.config["encoder"]["arm_coords"]) - 1
        try:
            target_arm = float(target_arm)
            _, rem = divmod(target_arm)
            show_error = False
            if rem != 0:
                show_error = True
            if target_arm not in list(range(1, valid_num_arms + 1)):
                show_error = True

            if show_error:      
                show_message(
                    self,
                    f"Target arm has to be an INTEGER between 1 and {valid_num_arms}, inclusive",
                    kind="critical")
            else:
                self.main_params.target_arm = int(target_arm)
                self.send_main_params()
                show_message(
                    self,
                    f"Updated - Target arm value: {int(target_arm)}",
                    kind="information")
        except:
            show_message(
                self,
                f"Target arm has to be an INTEGER between 1 and {self.valid_num_arms}, inclusive",
                kind="critical")

    def check_post_thresh(self):

        post_thresh = self.post_edit.text()
        try:
            post_thresh = float(post_thresh)
            if post_thresh < 0:
                show_message(
                    self,
                    "Posterior threshold cannot be a negative number",
                    kind="critical")
            elif post_thresh >= 1:
                show_message(
                    self,
                    "Posterior threshold must be less than 1",
                    kind="critical")
            else:
                self.main_params.posterior_threshold =post_thresh
                self.send_main_params()
                show_message(
                    self,
                    f"Message sent - Posterior threshold value: {post_thresh}",
                    kind="information")
        except:
            show_message(
                self,
                "Posterior threshold must be a non-negative number in the range [0, 1)",
                kind="critical")

    def check_n_above(self):
        max_n_above = len(self.config["trodes_network"]["ripple_tetrodes"])
        n_above = self.n_above_edit.text()
        try:
            n_above = float(n_above)
            _, rem = divmod(n)
            show_error = False
            if rem != 0:
                show_error = True
            if n_above not in list(range(1, max_n_above + 1)):
                show_error = True

            if show_error:
                show_message(
                    self,
                    "Number of tetrodes above threshold must be an integer value "
                    f"between 1 and {max_n_above}, inclusive",
                    kind="critical")
            else:
                self.main_params.num_above_threshold = int(n_above)
                self.send_main_params()
                show_message(
                    self,
                    f"Message sent - n above threshold value: {n_above}",
                    kind="information")
        except:
            show_message(
                self,
                "Number of tetrodes above threshold must be an integer value "
                "between 1 and max_n_above, inclusive",
                kind="critical")

    def check_max_center_well(self):
        # unbounded?
        dist = self.max_center_well_edit.text()
        try:
            dist = float(dist)
            if dist < 0:
                show_message(
                    self,
                    f"Max center well distance cannot be negative",
                    kind="critical")
            else:
                self.main_params.max_center_well_distance = dist
                self.send_main_params()
                show_message(
                    self,
                    f"Message sent - Max center well distance (cm) value: {dist}",
                    kind="information")
        except:
            show_message(
                self,
                f"Max center well distance must be a non-negative value",
                kind="critical")

    def check_rip_thresh(self):

        rip_thresh = self.rip_thresh_edit.text()
        try:
            rip_thresh = float(rip_thresh)
            if rip_thresh < 0:
                show_message(self, "Ripple threshold cannot be negative", kind="warning")
            
            else:
                self.ripple_params.ripple_threshold = rip_thresh
                self.send_ripple_params()
                show_message(
                    self,
                    f"Message sent - Ripple threshold value: {rip_thresh}",
                    kind="information")
        except:
            show_message(
                self,
                "Ripple threshold must be a non-negative number",
                kind="critical")

    def check_cond_rip_thresh(self):
        cond_rip_thresh = self.cond_rip_thresh_edit.text()
        try:
            cond_rip_thresh = float(cond_rip_thresh)
            if cond_rip_thresh < 0:
                show_message(
                    self,
                    "Conditioning ripple threshold cannot be negative",
                    kind="warning")
            
            else:
                self.ripple_params.conditioning_ripple_threshold = cond_rip_thresh
                self.send_ripple_params()
                show_message(
                    self,
                    f"Message sent - Conditioning ripple threshold value: {cond_rip_thresh}",
                    kind="information")
        except:
            show_message(
                self,
                "Conditioning ripple threshold must be a non-negative number",
                kind="critical")
    
    def check_ripple_vel_thresh(self):
        
        ripple_vel_thresh = self.ripple_vel_thresh_edit.text()
        try:
            ripple_vel_thresh = float(ripple_vel_thresh)
            if ripple_vel_thresh < 0:
                show_message(
                    self,
                    "Ripple velocity threshold cannot be a negative number",
                    kind="critical")
            else:
                self.main_params.ripple_velocity_threshold = ripple_vel_thresh
                self.send_main_params()
                show_message(
                    self,
                    f"Message sent - Ripple velocity threshold value: {vel_thresh}",
                    kind="information")
        except:
            show_message(
                self,
                "Ripple velocity threshold must be a non-negative number",
                kind="critical")

    def check_encoding_vel_thresh(self):
        
        encoding_vel_thresh = self.encoding_vel_thresh_edit.text()
        try:
            encoding_vel_thresh = float(encoding_vel_thresh)
            if encoding_vel_thresh < 0:
                show_message(
                    self,
                    "Encoding velocity threshold cannot be a negative number",
                    kind="critical")
            else:
                self.encoder_params.encoding_velocity_threshold = encoding_vel_thresh
                self.decoder_params.encoding_velocity_threshold = encoding_vel_thresh
                self.send_encoder_params()
                self.send_decoder_params()
                show_message(
                    self,
                    f"Message sent - Encoding velocity threshold value: {vel_thresh}",
                    kind="information")
        except:
            show_message(
                self,
                "Encoding velocity threshold must be a non-negative number",
                kind="critical")

    def check_shortcut(self):

        shortcut_on_checked = self.shortcut_on.isChecked()
        shortcut_off_checked = self.shortcut_off.isChecked()
        if shortcut_on_checked or shortcut_off_checked:
            if shortcut_on_checked:
                self.main_params.shortcut_message_on = True
                self.send_main_params()
                show_message(
                    self,
                    "Message sent - Set shortcut ON",
                    kind="information")
            else:
                self.main_params.shortcut_message_on = False
                self.send_main_params()
                show_message(
                    self,
                    "Message sent - Set shortcut OFF",
                    kind="information")
        else:
            show_message(
                self,
                "Neither button is selected. Doing nothing.",
                kind="information")


    def check_instructive_task(self):

        instructive_task_on = self.instructive_task_on.isChecked()
        instructive_task_off = self.instructive_task_off.isChecked()
        if instructive_task_on or instructive_task_off:
            if instructive_task_on:
                self.main_params.instructive_task = True
                self.send_main_params()
                show_message(
                    self,
                    "Instructive task is currently not being used. Doing nothing.",
                    kind="information")
            else:
                self.main_params.instructive_task = False
                self.send_main_params()
                show_message(
                    self,
                    "Instructive task is currently not being used. Doing nothing.",
                    kind="information")
        else:
            show_message(
                self,
                "Neither button is selected. Doing nothing.",
                kind="information")

    def check_reward_mode(self):
        reward_mode_conditioning_ripples = self.reward_mode_conditioning_ripples.isChecked()
        reward_mode_replay = self.reward_mode_replay.isChecked()
        if reward_mode_conditioning_ripples or reward_mode_replay:
            if reward_mode_conditioning_ripples:
                self.main_params.reward_mode = "conditioning_ripples"
                self.send_main_params()
                show_message(
                    self,
                    "Message sent - Set reward mode to conditioning ripples",
                    kind="information")
            else:
                self.main_params.reward_mode = "replay"
                self.send_main_params()
                show_message(
                    self,
                    "Message sent - Set reward mode to replay",
                    kind="information")
        else:
            show_message(
                self,
                "Neither button is selected. Doing nothing.",
                kind="information")

    def send_main_params(self):
        self.comm.send(self.main_params, dest=self.config['rank']['supervisor'])

    def send_ripple_params(self):
        for rank in self.config['rank']['ripples']:
            self.comm.send(self.ripple_params, dest=rank)
    
    def send_encoder_params(self):
        for rank in self.config['rank']['encoders']:
            self.comm.send(self.encoder_params, dest=rank)

    def send_decoder_params(self):
        for rank in self.config['rank']['decoder']:
            self.comm.send(self.decoder_params, dest=rank)

    def send_all_params(self):
        self.send_main_params()
        self.send_ripple_params()
        self.send_encoder_params()
        self.send_decoder_params()

    def run(self):
        self.timer.start()

    def closeEvent(self, event):
        show_message(
            self,
            "Cannot close while main window is still open",
            kind="warning")
        event.ignore()


class DecodingResultsWindow(QMainWindow):

    def __init__(self, comm, rank, config):
        super().__init__()
        self.comm = comm
        self.rank = rank
        self.config = config
        
        self.setWindowTitle("Decoder Output")
        self.graphics_widget = pg.GraphicsLayoutWidget()
        self.setCentralWidget(self.graphics_widget)
        
        self.parameters_dialog = Dialog(self, comm, rank, config)
        self.parameters_dialog.move(
            self.pos().x() + self.frameGeometry().width() + 30, self.pos().y())

        self.timer = QTimer()
        self.timer.setInterval(0)
        self.timer.timeout.connect(self.update)

        self.elapsed_timer = QElapsedTimer()
        self.refresh_msec = 30 # allow for option in config

        # approximately 2 secs for 6 msec bins. should really make this more flexible
        # in the config
        self.num_time_bins = 333

        num_plots = len(self.config["rank"]["decoder"])
        self.plots = [None] * num_plots
        self.plot_datas = [ [] for _ in range(num_plots)]
        self.images = [None] * num_plots
        self.posterior_datas = [None] * num_plots
        self.posterior_datas_ind = [0] * num_plots

        self.decoder_rank_ind_mapping = {}
        B = self.config["encoder"]["position"]["bins"]
        N = self.num_time_bins
        self.posterior_buff = np.zeros(B)
        for ii, rank in enumerate(self.config["rank"]["decoder"]):
            self.decoder_rank_ind_mapping[rank] = ii
            self.posterior_datas[ii] = np.zeros((B, N))

        # plot decoder lines
        for ii in range(num_plots):
            self.plots[ii] = self.graphics_widget.addPlot(
                ii, 0, 1, 1,
                labels={'left':'Position bin', 'bottom':'Time (sec)'})
            coords = self.config["encoder"]["arm_coords"]
            for lb, ub in coords:
                self.plot_datas[ii].append(
                    pg.PlotDataItem(
                        np.ones(self.num_time_bins) * lb, pen='w', width=10
                    )
                )
                self.plots[ii].addItem(self.plot_datas[ii][-1])
                self.plot_datas[ii].append(
                    pg.PlotDataItem(
                        np.ones(self.num_time_bins) * (ub+1), pen='w', width=10
                    )
                )
                self.plots[ii].addItem(self.plot_datas[ii][-1])
            
            # scale axes to time coordinates - make this user settable
            # rework this later to be more flexible
            x_axis = self.plots[ii].getAxis("bottom")
            ticks = np.linspace(0, N, 5)
            tick_labels = [str(np.round(tick*0.006, decimals=2)) for tick in ticks]
            x_axis.setTicks([[ (tick, tick_label) for (tick, tick_label) in zip(ticks, tick_labels)]])
            
            self.plots[ii].setMenuEnabled(False)
            self.images[ii] = pg.ImageItem(border=None)
            self.images[ii].setZValue(-100)
            self.plots[ii].addItem(self.images[ii])     
        
        self.init_colormap()

        self.req_cmd = self.comm.irecv(
            source=self.config["rank"]["supervisor"],
            tag=MPIMessageTag.COMMAND_MESSAGE)
        self.req_data = self.comm.Irecv(
            buf=self.posterior_buff,
            tag=MPIMessageTag.DATA_FOR_GUI
        )
        self.mpi_status = MPI.Status()

        self.ok_to_terminate = False

    def init_colormap(self):
        cmap = sns.color_palette(DEFAULT_GUI_PARAMS["colormap"], as_cmap=True)
        try:
            colormap = self.config["gui"]["colormap"]
            try:
                cmap = sns.color_palette(colormap, as_cmap=True)
            except:
                show_message(
                    self,
                    f"Colormap {colormap} could not be found, using default",
                    kind="information")
        except KeyError:
            pass

        cmap._init()
        lut = (cmap._lut * 255).view(np.ndarray)

        for image in self.images:
            image.setLookupTable(lut)
            x, y = self.posterior_datas[0].shape
            image.setImage(np.random.rand(x, y).T)

    def show_all(self):
        self.show()
        self.parameters_dialog.show()

    def update(self):
        # check for incoming messages and data
        req_cmd_ready, cmd_message = self.req_cmd.test()
        if req_cmd_ready:
            self.process_command(cmd_message)
            self.req_cmd = self.comm.irecv(
                source=self.config["rank"]["supervisor"],
                tag=MPIMessageTag.COMMAND_MESSAGE)

        req_data_ready, data_message = self.req_data.test(status=self.mpi_status)
        if req_data_ready:
            self.process_new_data()
            self.req_data = self.comm.Irecv(
                buf=self.posterior_buff,
                tag=MPIMessageTag.DATA_FOR_GUI
            )

        if self.elapsed_timer.elapsed() > self.refresh_msec:
            self.elapsed_timer.start()
            self.update_data()

    def process_command(self, message):
        if isinstance(message, SetupComplete):
            show_message(
                self,
                "All processes have finished setup. After closing this popup, "
                "hit record or play to start decoding.",
                kind="information"
            )
        elif isinstance(message, TerminateMessage):
            self.ok_to_terminate = True
            show_message(
                self,
                "Processes have terminated, closing GUI",
                kind="information"
            )
            self.close()
        elif isinstance(message, TimeSyncInit):
            # do nothing but still need to place barrier so the other processes
            # can proceed
            self.comm.Barrier()
        else:
            show_message(
                self,
                f"Message type {type(message)} received from main process, ignoring",
                kind="information"
            )

    def process_new_data(self):
        sender = self.mpi_status.source
        ind = self.decoder_rank_ind_mapping[sender]
        self.posterior_datas[ind][:, self.posterior_datas_ind[ind]] = self.posterior_buff.copy()
        self.posterior_datas_ind[ind] = (self.posterior_datas_ind[ind] + 1) % self.num_time_bins

    def update_data(self):
        for ii in range(len(self.plots)):
            self.posterior_datas[ii][np.isnan(self.posterior_datas[ii])] = 0
            self.images[ii].setImage(self.posterior_datas[ii].T * 255)
    
    def run(self):
        self.elapsed_timer.start()
        self.timer.start()
        self.parameters_dialog.run()

    def closeEvent(self, event):
        if not self.ok_to_terminate:
            show_message(
                self,
                "Processes not finished running. Closing GUI is disabled",
                kind="critical")
            event.ignore()
        else:
            super().closeEvent(event)

class GuiProcess(RealtimeProcess):

    def __init__(self, comm, rank, config):
        super().__init__(comm, rank, config)
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        self.app = app
        self.main_window = DecodingResultsWindow(comm, rank, config)
        self.comm.Barrier()

    def main_loop(self):
        self.main_window.show_all()
        self.main_window.run()
        self.app.exec()
        self.class_log.info("GUI process finished main loop")

if __name__ == "__main__":
    app = QApplication([])
    import json
    config = json.load(open('../../config/mossy_percy.json' ,'r'))
    win = DecodingResultsWindow(0, 1, config)
    win.show_all()
    app.exec_()
