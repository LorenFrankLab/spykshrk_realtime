import math
import matplotlib.pyplot as plt
from matplotlib import patches
import functools
import pandas as pd
import numpy as np
import holoviews as hv
from holoviews.operation.datashader import datashade, aggregate, shade, regrid, dynspread
from holoviews.operation import decimate
import datashader as ds

import functools

from spykshrk.franklab.data_containers import pos_col_format, Posteriors, LinearPosition, \
    EncodeSettings, StimLockout, RippleTimes

idx = pd.IndexSlice

class MissingDataError(RuntimeError):
    pass


class VisualizationConfigError(RuntimeError):
    pass


class DecodeVisualizer:

    def __init__(self, posteriors: Posteriors, enc_settings: EncodeSettings,
                 linpos: LinearPosition=None, riptimes: RippleTimes=None, relative=False):
        self.time_dim_name = 'time (s)'
        self.pos_dim_name = 'linpos (cm)'
        self.val_dim_name = 'probability'

        self.posteriors = posteriors
        self.linpos = linpos
        self.riptimes = riptimes
        if riptimes is not None:
            self.posteriors.apply_time_event(riptimes, event_mask_name='ripple_grp')
        self.linflat = linpos.get_mapped_single_axis()
        if riptimes is not None:
            self.linflat.apply_time_event(riptimes, event_mask_name='ripple_grp')
        self.enc_settings = enc_settings
        self.relative = relative
        self.post_img = hv.Image(np.flip(self.posteriors.get_distribution_view().values.T, axis=0),
                                 bounds=(self.posteriors.get_time_start(), self.posteriors.get_pos_start(),
                                         self.posteriors.get_time_end(),
                                         self.posteriors.get_pos_end()),
                                 kdims=[self.time_dim_name, self.pos_dim_name], vdims=[self.val_dim_name])
        self.post_img = self.post_img.redim(probability={'range': (0, 0.3)})

    def plot_decode_image(self, time, x_range=None, y_range=None, plt_range=10):

        sel_range = [time, time+plt_range]

        if (x_range is None):
            x_range = [time, time+plt_range]

        if (y_range is None):
            y_range = [0, self.enc_settings.pos_bins[-1]]

        x_range = list(x_range)
        #x_range[0] -= plt_range
        #x_range[1] += plt_range
        x_range[0] = max(x_range[0], self.posteriors.get_time_start())
        x_range[1] = min(x_range[1], self.posteriors.get_time_end())

        # post_time = self.posteriors.index.get_level_values('time')
        # img_sel = self.posteriors.get_distribution_view().query("(time > @x_range[0]) & (time < @x_range[1])").values.T
        # img_sel = np.flip(img_sel, axis=0)

        # img = hv.Image(img_sel, bounds=(x_range[0], self.enc_settings.pos_bins[0],
        #                                 x_range[1],
        #                                 self.enc_settings.pos_bins[-1]),
        #                kdims=['time (sec)', 'linpos (cm)'], vdims=['probability'],)

        # img = img.redim(probability={'range': (0, 0.3)})
        # img.extents = (sel_range[0], 0, sel_range[1], self.enc_settings.pos_bins[-1])
        self.post_img.extents = (sel_range[0], 0, sel_range[1], self.enc_settings.pos_bins[-1])
        self.post_img.relabel('posteriors')
        rgb = shade(regrid(self.post_img, aggregator='mean', dynamic=False,
                           x_range=x_range, y_range=y_range), cmap=plt.get_cmap('hot'),
                    normalization='linear', dynamic=False)
        # rgb = shade(regrid(self.post_img, aggregator='mean', dynamic=False,
        #                    x_range=x_range, y_range=y_range, y_sampling=1, x_sampling=0.001),
        #             cmap=plt.get_cmap('hot'), normalization='linear', dynamic=False)

        rgb.extents = (sel_range[0], 0, sel_range[1], self.enc_settings.pos_bins[-1])
        return rgb

    def highlight_ripples(self, time, x_range=None, y_range=None, plt_range=10):

        def rect(starttime, endtime, pos_min, pos_max):
            return {('x', 'y'): [(starttime, pos_min), (endtime, pos_min), (endtime, pos_max), (starttime, pos_max)]}

        boxes = [rect(entry.starttime, entry.endtime, self.enc_settings.pos_bins[0], self.enc_settings.pos_bins[-1])
                 for entry in self.riptimes.itertuples()]
        poly = hv.Polygons(boxes, group='events', label='ripples')
        poly.extents = (time, self.enc_settings.pos_bins[0], time+plt_range, self.enc_settings.pos_bins[-1])
        return poly

    def plot_arm_boundaries(self, time, x_range=None, y_range=None, plt_range=None):
        if plt_range is None:
            plt_range = self.posteriors.get_time_total() - time
        if x_range is None:
            x_range = (time, time + plt_range)
        if y_range is None:
            y_range = self.posteriors.get_pos_range()

        lines = hv.Overlay()

        for arm in self.enc_settings.arm_coordinates:
            for bound in arm:

                #line = hv.Curve((x_range, [bound]*2)).opts(style={'line_dash': 'dashed', 'line_color': 'grey'})
                line = hv.Curve((x_range, [bound]*2),
                                extents=(x_range[0], None, x_range[1], None),
                                group='arm_bound').opts(style={'color': '#AAAAAA',
                                                               'line_dash': 'dashed',
                                                               'line_color': 'grey',
                                                               'linestyle': '--'})
                lines *= line

        return lines

    def plot_linear_pos(self, time, x_range=None, y_range=None, plt_range=10):
        linflat_sel_data = self.linflat['linpos_flat'].values
        linflat_sel_time = self.linflat.index.get_level_values('time')
        pos = hv.Points((linflat_sel_time, linflat_sel_data), kdims=[self.time_dim_name, self.pos_dim_name],
                        extents=(time, None, time + plt_range, None), label=('linpos', 'Linear Position'))

        return pos

    def plot_all(self, time=None, x_range=None, y_range=None, plt_range=10):
        out = hv.Overlay()

        img = self.plot_decode_image(time, x_range, y_range, plt_range)
        out *= img
        pos = self.plot_linear_pos(time, x_range, y_range, plt_range)
        out *= pos
        arms = self.plot_arm_boundaries(time, x_range, y_range, plt_range)
        out *= arms
        if self.riptimes is not None:
            rips = self.highlight_ripples(time, x_range, y_range, plt_range)
            out *= rips
        return out

    def plot_all_dynamic(self, stream, slide=10, plt_range=10, values=None):

        if values is None:
            values = np.arange(self.posteriors.index.get_level_values('time')[0],
                               self.posteriors.index.get_level_values('time')[-1],
                               slide)

        dmap = hv.DynamicMap(functools.partial(self.plot_all, plt_range=plt_range),
                             kdims=hv.Dimension('time', values=values),
                             streams=[stream])
        return dmap

    def plot_ripple(self, rip_ind):
        rip_post = self.posteriors.query('ripple_grp == @rip_ind').get_distribution_view()
        rip_time = rip_post.index.get_level_values('time')
        rip_img = hv.Image(np.flip(rip_post.values.T, axis=0),
                           bounds=(0, self.enc_settings.pos_bins[0],
                                                      rip_time[-1]-rip_time[0], self.enc_settings.pos_bins[-1]),
                           kdims=[self.time_dim_name, self.pos_dim_name], vdims=[self.val_dim_name],
                           label='ripple_decode: ', group=str(rip_ind))

        rip_img = rip_img.redim(probability={'range': (0, 0.3)})

        return rip_img

    def plot_ripple_linflat(self, rip_ind):
        linflat_ripple = self.linflat.query('ripple_grp == @rip_ind')
        linflat_ripple_time = (np.array(linflat_ripple.index.get_level_values('time')) -
                               self.riptimes.query('event == @rip_ind')['starttime'].values)
        plt = hv.Points((linflat_ripple_time, linflat_ripple['linpos_flat'].values),
                        kdims=[self.time_dim_name, self.pos_dim_name], label=('linpos', 'linear position'))
        #plt = plt.opts(style={'marker': '*', 'color': '#AAAAFF', 'size': 14})
        return plt

    def plot_ripple_all(self, rip_ind):
        rip_plt = self.plot_ripple(rip_ind)
        x_range = rip_plt.range(self.time_dim_name)
        y_range = rip_plt.range(self.pos_dim_name)
        arm_plt = self.plot_arm_boundaries(time=x_range[0], x_range=x_range, y_range=y_range)
        lin_plt = self.plot_ripple_linflat(rip_ind)
        return hv.Overlay([rip_plt, arm_plt, lin_plt], label='ripple_decode: ', group=str(rip_ind))

    def plot_ripple_dynamic(self):
        dmap = hv.DynamicMap(self.plot_ripple_all,
                             kdims=hv.Dimension('rip_ind', range=(0, self.riptimes.get_num_events())))

        return dmap

    def plot_ripple_grid(self, num_x=3, num_y=None, return_list=True):
        if num_y is None:
            num_y = int(math.ceil(len(self.riptimes) / float(num_x)))

        plt_grp_size = num_x * num_y
        plt_num_grps = int(math.ceil(self.riptimes.get_num_events()/plt_grp_size))
        plot_list = []

        rip_ids = self.riptimes.index.get_level_values('event')
        rip_plt_grps = [[rip_ids[(grp_id * plt_grp_size) + entry_ind] for entry_ind in
                         range(min([len(rip_ids) - plt_grp_size * grp_id, plt_grp_size]))]
                         for grp_id in range(plt_num_grps)]

        if (len(rip_plt_grps) > 1) and (return_list is False):
            raise VisualizationConfigError("Can't set return_list=False when grid cannot hold all plots.")

        if return_list:
            for rip_plt_grp in rip_plt_grps:
                lay = hv.Layout([self.plot_ripple_all(rip_id) for rip_id in rip_plt_grp]).cols(num_x)

                plot_list.append(lay)

            return plot_list

        else:
            lay = hv.Layout({rip_id: self.plot_ripple_all(rip_id) for rip_id in rip_plt_grps[0]}).cols(num_x)
            return lay

    @staticmethod
    def plot_stim_lockout(ax, stim_lock: StimLockout, plt_range, plt_height):

        stim_lock_sel = stim_lock.get_range_sec(*plt_range)

        for stim_item in stim_lock_sel['time'].itertuples():
            pat = patches.Rectangle(xy=(stim_item.on, 0), width=(stim_item.off - stim_item.on), height=plt_height,
                                    fill=True, alpha=0.5, color='gray', zorder=3)

            ax.add_patch(pat)

        return ax


class DecodeErrorVisualizer:

    arms = ['center', 'left', 'right']

    def __init__(self, error_table):
        self.error_table = error_table

        self.arm_error_tables = {}

        maxlist = []
        minlist = []

        for arm in self.arms:

            arm_table = error_table.loc[:, idx[arm, ['real_pos', 'plt_error_up', 'plt_error_down']]]
            arm_table.columns = arm_table.columns.droplevel(0)
            arm_table = arm_table.reset_index(level=['time']).reindex(columns=['time','real_pos', 'plt_error_up',
                                                                               'plt_error_down'])

            self.arm_error_tables[arm] = hv.Table(arm_table.dropna(), kdims='time',
                                                  vdims=['real_pos', 'plt_error_down', 'plt_error_up'])

            maxlist.append(max(self.arm_error_tables[arm]['real_pos'] + self.arm_error_tables[arm]['plt_error_up']))
            minlist.append(min(self.arm_error_tables[arm]['real_pos'] + self.arm_error_tables[arm]['plt_error_down']))

        self.min_pos = min(minlist)
        self.max_pos = max(maxlist)

    def plot_arms_error(self, start_time=None, interval=None, lookahead=0, lookbehind=0):

        error_plots = {}
        real_pos_plots = {}
        joint_pos_plots = {}

        for arm, arm_table in self.arm_error_tables.items():

            if not (start_time is None or interval is None):
                #arm_table = arm_table[start_time: start_time+interval]
                pass

            if len(arm_table) == 0:
                # dummy table
                arm_table = hv.Table([[start_time,self.max_pos,0,0]], kdims=['time'],
                                     vdims=['real_pos','plt_error_down','plt_error_up'])
            error_plots[arm] = hv.ErrorBars(arm_table, kdims='time',
                                            vdims=['real_pos', 'plt_error_down', 'plt_error_up'],
                                            extents=(start_time, self.min_pos, start_time+interval, self.max_pos))
            error_plots[arm].redim(time={'range': (start_time, start_time+interval)})
            real_pos_plots[arm] = arm_table.to.points(kdims=['time', 'real_pos'], vdims=['real_pos'],
                                                      extents=(start_time, self.min_pos,
                                                               start_time+interval, self.max_pos))
            #joint_pos_plots[arm] = real_pos_plots[arm] * error_plots[arm]

        errorbar_overlay = hv.NdOverlay(error_plots, kdims='arm')
        errorbars = hv.NdOverlay(error_plots, kdims='arm')
        points = hv.NdOverlay(real_pos_plots, kdims='arm')

        return errorbars * points

    def plot_arms_error_dmap(self, slide_interval, plot_interval=None):

        if plot_interval is None:
            plot_interval = slide_interval

        dmap = hv.DynamicMap(functools.partial(self.plot_arms_error, interval=plot_interval),
                             kdims=[hv.Dimension('start_time',
                                                 values=np.arange(self.error_table.index.get_level_values('time')[0],
                                                                  self.error_table.index.get_level_values('time')[-1],
                                                                  slide_interval))])

        return dmap
