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
        self.setup_ripple_thresh(layout)
        self.setup_post_thresh(layout)
        self.setup_target_arm(layout)
        self.setup_vel_thresh(layout)
        self.setup_shortcut_message(layout)
        self.setup_instructive_task(layout)

    def setup_ripple_thresh(self, layout):
        self.rip_thresh_label = QLabel(self.tr("Ripple Threshold"))
        layout.addWidget(self.rip_thresh_label, 0, 0)
        
        self.rip_thresh_edit = QLineEdit()
        layout.addWidget(self.rip_thresh_edit, 0, 1)

        self.rip_thresh_button = QPushButton(self.tr("Send"))
        self.rip_thresh_button.pressed.connect(self.check_rip_thresh)
        layout.addWidget(self.rip_thresh_button, 0, 2)

    def setup_post_thresh(self, layout):
        self.post_label = QLabel(self.tr("Posterior Threshold"))
        layout.addWidget(self.post_label, 1, 0)
        
        self.post_edit = QLineEdit()
        layout.addWidget(self.post_edit, 1, 1)

        self.post_thresh_button = QPushButton(self.tr("Send"))
        self.post_thresh_button.pressed.connect(self.check_post_thresh)
        layout.addWidget(self.post_thresh_button, 1, 2)

    def setup_target_arm(self, layout):
        self.target_arm_label = QLabel(self.tr("Target Arm"))
        layout.addWidget(self.target_arm_label, 2, 0)
        
        self.target_arm_edit = QLineEdit()
        layout.addWidget(self.target_arm_edit, 2, 1)

        self.target_arm_button = QPushButton(self.tr("Send"))
        self.target_arm_button.pressed.connect(self.check_target_arm)
        layout.addWidget(self.target_arm_button, 2, 2)

    def setup_vel_thresh(self, layout):
        self.vel_thresh_label = QLabel(self.tr("Velocity Threshold"))
        layout.addWidget(self.vel_thresh_label, 3, 0)
        
        self.vel_thresh_edit = QLineEdit()
        layout.addWidget(self.vel_thresh_edit, 3, 1)

        self.vel_thresh_button = QPushButton(self.tr("Send"))
        self.vel_thresh_button.pressed.connect(self.check_vel_thresh)
        layout.addWidget(self.vel_thresh_button, 3, 2)

    def setup_shortcut_message(self, layout):
        self.shortcut_label = QLabel(self.tr("Shortcut Message"))
        layout.addWidget(self.shortcut_label, 4, 0)
        
        self.shortcut_on = QRadioButton(self.tr("ON"))        
        self.shortcut_off = QRadioButton(self.tr("OFF"))
        shortcut_layout = QHBoxLayout()
        shortcut_layout.addWidget(self.shortcut_on)
        shortcut_layout.addWidget(self.shortcut_off)
        shortcut_group_box = QGroupBox()
        shortcut_group_box.setLayout(shortcut_layout)
        layout.addWidget(shortcut_group_box, 4, 1)

        self.shortcut_message_button = QPushButton(self.tr("Send"))
        self.shortcut_message_button.pressed.connect(self.check_shortcut)
        layout.addWidget(self.shortcut_message_button, 4, 2)

    def setup_instructive_task(self, layout):
        self.instructive_task_label = QLabel(self.tr("Instructive Task"))
        layout.addWidget(self.instructive_task_label, 5, 0)

        self.instructive_task_on = QRadioButton(self.tr("ON"))        
        self.instructive_task_off = QRadioButton(self.tr("OFF"))
        instructive_task_layout = QHBoxLayout()
        instructive_task_layout.addWidget(self.instructive_task_on)
        instructive_task_layout.addWidget(self.instructive_task_off)
        instructive_task_group_box = QGroupBox()
        instructive_task_group_box.setLayout(instructive_task_layout)
        layout.addWidget(instructive_task_group_box, 5, 1)

        self.instructive_task_button = QPushButton(self.tr("Send"))
        self.instructive_task_button.pressed.connect(self.check_instructive_task)
        layout.addWidget(self.instructive_task_button, 5, 2)

    def check_rip_thresh(self):

        rip_thresh = self.rip_thresh_edit.text()
        try:
            rip_thresh = float(rip_thresh)
            if rip_thresh < 0:
                show_message(self, "Ripple threshold cannot be negative", kind="warning")
            
            else:
                # send message -- ripple and main
                show_message(
                    self,
                    f"Ripple threshold value of {rip_thresh} sent",
                    kind="information")
        except:
            show_message(
                self,
                "Ripple threshold must be a non-negative number",
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
                    f"Posterior threshold value of {post_thresh} sent",
                    kind="information")
        except:
            show_message(
                self,
                "Posterior threshold must be a non-negative number in the range [0, 1)",
                kind="critical")

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
            if target_arm < 1:
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
                    f"Target arm value of {int(target_arm)} sent",
                    kind="information")
        except:
            show_message(
                self,
                f"Target arm has to be an INTEGER between 1 and {self.valid_num_arms}, inclusive",
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
                    f"Velocity threshold value of {vel_thresh} sent",
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
                    "Shortcut on message sent",
                    kind="information")
            else:
                # send message -- main
                show_message(
                    self,
                    "Shortcut off message sent",
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

    def closeEvent(self, event):
        show_message(
            self,
            "Cannot close while main window is still open",
            kind="warning")
        event.ignore()


    def check_for_updates(self):
        # where we look for messages from other processes
        # and eventually display data
        pass


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

    def start_timer(self):
        self.timer2 = QTimer()
        self.timer2.setInterval(5000)
        self.timer2.timeout.connect(self.close)
        self.timer2.start()

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
