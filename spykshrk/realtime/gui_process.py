from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer, QElapsedTimer
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QGridLayout,
                               QLineEdit, QGroupBox, QHBoxLayout, QDialog,
                               QPushButton, QLabel, QSpinBox, QSlider, QStatusBar,
                               QFileDialog, QMessageBox, QRadioButton, QTextEdit)
from spykshrk.realtime.realtime_base import (RealtimeProcess, TerminateMessage,
                                MPIMessageTag, TimeSyncInit)
import logging
import pyqtgraph as pg
import numpy as np
from matplotlib import cm
from mpi4py import MPI

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

        self.valid_num_arms = len(config["encoder"]["arm_coords"]) - 1

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
        self.setup_position_limit(layout)
        self.setup_max_center_well_distance(layout)
        self.setup_ripple_thresh(layout)
        self.setup_conditioning_ripple_thresh(layout)
        self.setup_vel_thresh(layout)
        self.setup_shortcut_message(layout)
        self.setup_instructive_task(layout)
        self.setup_ripple_cond_only(layout)

    def setup_target_arm(self, layout):
        self.target_arm_label = QLabel(self.tr("Target Arm"))
        layout.addWidget(self.target_arm_label, 0, 0)
        
        self.target_arm_edit = QLineEdit()
        layout.addWidget(self.target_arm_edit, 0, 1)

        self.target_arm_button = QPushButton(self.tr("Send"))
        self.target_arm_button.pressed.connect(self.check_target_arm)
        layout.addWidget(self.target_arm_button, 0, 2)

    def setup_post_thresh(self, layout):
        self.post_label = QLabel(self.tr("Posterior Threshold"))
        layout.addWidget(self.post_label, 1, 0)
        
        self.post_edit = QLineEdit()
        layout.addWidget(self.post_edit, 1, 1)

        self.post_thresh_button = QPushButton(self.tr("Send"))
        self.post_thresh_button.pressed.connect(self.check_post_thresh)
        layout.addWidget(self.post_thresh_button, 1, 2)

    def setup_n_above_threshold(self, layout):
        self.n_above_label = QLabel(self.tr("Num. Tetrodes Above Threshold"))
        layout.addWidget(self.n_above_label, 2, 0)
        
        self.n_above_edit = QLineEdit()
        layout.addWidget(self.n_above_edit, 2, 1)

        self.n_above_button = QPushButton(self.tr("Send"))
        self.n_above_button.pressed.connect(self.check_n_above)
        layout.addWidget(self.n_above_button, 2, 2)

    def setup_position_limit(self, layout):
        self.pos_limit_label = QLabel(self.tr("Position limit"))
        layout.addWidget(self.pos_limit_label, 3, 0)
        
        self.pos_limit_edit = QLineEdit()
        layout.addWidget(self.pos_limit_edit, 3, 1)

        self.pos_limit_button = QPushButton(self.tr("Send"))
        self.pos_limit_button.pressed.connect(self.check_pos_limit)
        layout.addWidget(self.pos_limit_button, 3, 2)

    def setup_max_center_well_distance(self, layout):
        self.max_center_well_label = QLabel(self.tr("Max center well distance"))
        layout.addWidget(self.max_center_well_label, 4, 0)
        
        self.max_center_well_edit = QLineEdit()
        layout.addWidget(self.max_center_well_edit, 4, 1)

        self.max_center_well_button = QPushButton(self.tr("Send"))
        self.max_center_well_button.pressed.connect(self.check_max_center_well)
        layout.addWidget(self.max_center_well_button, 4, 2)

    def setup_ripple_thresh(self, layout):
        self.rip_thresh_label = QLabel(self.tr("Ripple Threshold"))
        layout.addWidget(self.rip_thresh_label, 5, 0)
        
        self.rip_thresh_edit = QLineEdit()
        layout.addWidget(self.rip_thresh_edit, 5, 1)

        self.rip_thresh_button = QPushButton(self.tr("Send"))
        self.rip_thresh_button.pressed.connect(self.check_rip_thresh)
        layout.addWidget(self.rip_thresh_button, 5, 2)

    def setup_conditioning_ripple_thresh(self, layout):
        self.cond_rip_thresh_label = QLabel(self.tr("Conditioning Ripple Threshold"))
        layout.addWidget(self.cond_rip_thresh_label, 6, 0)
        
        self.cond_rip_thresh_edit = QLineEdit()
        layout.addWidget(self.cond_rip_thresh_edit, 6, 1)

        self.cond_rip_thresh_button = QPushButton(self.tr("Send"))
        self.cond_rip_thresh_button.pressed.connect(self.check_cond_rip_thresh)
        layout.addWidget(self.cond_rip_thresh_button, 6, 2)

    def setup_vel_thresh(self, layout):
        self.vel_thresh_label = QLabel(self.tr("Velocity Threshold"))
        layout.addWidget(self.vel_thresh_label, 7, 0)
        
        self.vel_thresh_edit = QLineEdit()
        layout.addWidget(self.vel_thresh_edit, 7, 1)

        self.vel_thresh_button = QPushButton(self.tr("Send"))
        self.vel_thresh_button.pressed.connect(self.check_vel_thresh)
        layout.addWidget(self.vel_thresh_button, 7, 2)

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

        self.shortcut_message_button = QPushButton(self.tr("Send"))
        self.shortcut_message_button.pressed.connect(self.check_shortcut)
        layout.addWidget(self.shortcut_message_button, 8, 2)

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

        self.instructive_task_button = QPushButton(self.tr("Send"))
        self.instructive_task_button.pressed.connect(self.check_instructive_task)
        layout.addWidget(self.instructive_task_button, 9, 2)

    def setup_ripple_cond_only(self, layout):
        self.rip_cond_only_label = QLabel(self.tr("Ripple Conditioning Only"))
        layout.addWidget(self.rip_cond_only_label, 10, 0)

        self.rip_cond_only_on = QRadioButton(self.tr("YES"))        
        self.rip_cond_only_off = QRadioButton(self.tr("NO"))
        rip_cond_only_layout = QHBoxLayout()
        rip_cond_only_layout.addWidget(self.rip_cond_only_on)
        rip_cond_only_layout.addWidget(self.rip_cond_only_off)
        rip_cond_only_group_box = QGroupBox()
        rip_cond_only_group_box.setLayout(rip_cond_only_layout)
        layout.addWidget(rip_cond_only_group_box, 10, 1)

        self.rip_cond_only_button = QPushButton(self.tr("Send"))
        self.rip_cond_only_button.pressed.connect(self.check_rip_cond_only)
        layout.addWidget(self.rip_cond_only_button, 10, 2)

    def check_target_arm(self):

        target_arm = self.target_arm_edit.text()
        try:
            target_arm = float(target_arm)
            _, rem = divmod(target_arm)
            show_error = False
            if rem != 0:
                show_error = True
            if target_arm not in list(range(1, self.valid_num_arms + 1)):
                show_error = True

            if show_error:      
                show_message(
                    self,
                    f"Target arm has to be an INTEGER between 1 and {self.valid_num_arms}, inclusive",
                    kind="critical")
            else:
                # send message -- main
                show_message(
                    self,
                    f"Message sent - Target arm value: {int(target_arm)}",
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
                # send out value -- main
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
                    "between 1 and max_n_above, inclusive",
                    kind="critical")
            else:
                # send message -- main
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

    def check_pos_limit(self):
        coords = np.array(self.config["encoder"]["arm_coords"])
        min_pos = np.min(coords)
        max_pos = np.max(coords)
        pos_limit = self.pos_limit_edit.text()
        try:
            pos_limit = float(pos_limit)
            _, rem = divmod(pos_limit)
            show_error = False
            if rem != 0:
                show_error = True
            if pos_limit not in list(range(min_pos, max_pos + 1)):
                show_error = True

            if show_error:
                show_message(
                    self,
                    f"Position limit must be an integer between {min_pos} and {max_pos}, inclusive",
                    kind="critical")
            else:
                # send message - main
                show_message(
                    self,
                    f"Message sent - Position limit value: {pos_limit}",
                    kind="information")
        except:
            show_message(
                self,
                f"Position limit must be an integer between {min_pos} and {max_pos}, inclusive",
                kind="critical")

    def check_max_center_well(self):
        # unbounded?
        dist = self.max_center_well_edit.text()
        try:
            dist = float(dist)
            if dist < 0:
                show_error(
                    self,
                    f"Max center well distance cannot be negative",
                    kind="critical")
                return

            # send to main
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
                # send message -- ripple
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
                # send message -- ripple
                show_message(
                    self,
                    f"Message sent - Conditioning ripple threshold value: {cond_rip_thresh}",
                    kind="information")
        except:
            show_message(
                self,
                "Conditioning ripple threshold must be a non-negative number",
                kind="critical")
    
    def check_vel_thresh(self):
        
        vel_thresh = self.vel_thresh_edit.text()
        try:
            vel_thresh = float(vel_thresh)
            if vel_thresh < 0:
                show_message(
                    self,
                    "Velocity threshold cannot be a negative number",
                    kind="critical")
            else:
                # send out message -- main, encoder, decoder
                show_message(
                    self,
                    f"Message sent - Velocity threshold value: {vel_thresh}",
                    kind="information")
        except:
            show_message(
                self,
                "Velocity threshold must be a non-negative number",
                kind="critical")

    def check_shortcut(self):

        shortcut_on_checked = self.shortcut_on.isChecked()
        shortcut_off_checked = self.shortcut_off.isChecked()
        if shortcut_on_checked or shortcut_off_checked:
            if shortcut_on_checked:
                # send message -- main
                show_message(
                    self,
                    "Message sent - Set shortcut ON",
                    kind="information")
            else:
                # send message -- main
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
                show_message(
                    self,
                    "Instructive task is currently not being used",
                    kind="information")
            else:
                show_message(
                    self,
                    "Instructive task is currently not being used",
                    kind="information")
        else:
            show_message(
                self,
                "Neither button is selected. Doing nothing.",
                kind="information")

    def check_rip_cond_only(self):
        rip_cond_only_on_checked = self.rip_cond_only_on.isChecked()
        rip_cond_only_off_checked = self.rip_cond_only_off.isChecked()
        if rip_cond_only_on_checked or rip_cond_only_off_checked:
            if rip_cond_only_on_checked:
                # send message -- main (bool)
                show_message(
                    self,
                    "Message sent - Set ripple conditioning only to YES",
                    kind="information")
            else:
                # send message -- main (bool)
                show_message(
                    self,
                    "Message sent - Set ripple conditioning only to NO",
                    kind="information")
        else:
            show_message(
                self,
                "Neither button is selected. Doing nothing.",
                kind="information")

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
            self.plots[ii] = self.graphics_widget.addPlot(ii, 0, 1, 1)
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
                        np.ones(self.num_time_bins) * ub, pen='w', width=10
                    )
                )
                self.plots[ii].addItem(self.plot_datas[ii][-1])
            self.images[ii] = pg.ImageItem(border=None)

            colormap = cm.get_cmap("Spectral_r") # allow setting in config
            colormap._init()
            lut = (colormap._lut * 255).view(np.ndarray)
            
            self.images[ii].setLookupTable(lut)
            self.images[ii].setZValue(-100)
            self.plots[ii].addItem(self.images[ii])

        self.req_cmd = self.comm.irecv(
            source=self.config["rank"]["supervisor"],
            tag=MPIMessageTag.COMMAND_MESSAGE)
        self.req_data = self.comm.Irecv(
            buf=self.posterior_buff,
            tag=MPIMessageTag.DATA_FOR_GUI
        )
        self.mpi_status = MPI.Status()

        self.ok_to_terminate = False

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
        if isinstance(message, TerminateMessage):
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
