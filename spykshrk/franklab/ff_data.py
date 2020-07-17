import os
from glob import glob
from scipy.io import loadmat

class FFAnimalInfo:
    """ animalinfo - stores details of the single animal to process and generates
    file lists that points to the data to process.

    """

    def __init__(self, base_dir, name, days, epochs, tetrodes, timescale=10000,
                 new_data=True):
        """ init function

        Args:
            base_dir: root directory of animal data
            name: name of animal
            days: array of which days of data to process
            tetrodes: array of which tetrodes to process
            epochs: list of epochs for encoding

        """
        self.base_dir = base_dir
        self.name = name
        self.days = days
        self.epochs = epochs
        self.tetrodes = tetrodes
        self.timescale = timescale
        self.times = self._parse_times()
        self.new_data = new_data

    def _parse_times(self):
        """ parse_time - stores time data (times.mat) for each day to make sure
        each day is synchronized.  Should only be called from __init__

        Stores only the times for each epoch, 0 indexed.

        """
        times = {}
        for day in self.days:
            day_path = os.path.join(self.base_dir, self.name,
                                    '%s%02d/times.mat' % (self.name, day))
            mat = loadmat(day_path)
            time_ranges = mat['ranges']
            times.setdefault(day, time_ranges[1::, :].astype('int32').tolist())
        return times

    def _get_spike_paths(self):
        """ get_spike_paths - returns a list of paths that points to the spike
        data ([animalname][day]-[tetrode].mat) of each tetrode and day.

        This function uses the matlab data structure because the reference dataset
        being used is Frank, whose raw data was lost.

        """
        path_list = []
        for day in self.days:
            day_path = os.path.join(self.base_dir, self.name,
                                    '%s%02d' % (self.name, day))
            for tet in self.tetrodes:
                tet_path_glob = os.path.join(day_path, '%02d*' % tet)
                tet_paths = glob(tet_path_glob)

                # only keep matching directories
                for tet_path in tet_paths:
                    if not os.path.isdir(tet_path):
                        del tet_paths[tet_paths.index(tet_path)]

                # Directory sanity checks
                if len(tet_paths) < 1:
                    print(('WARNING: %s day %02d does not have file for tetrode %02d,' +
                          ' skipping tetrode (%s)') %
                          (self.name, day, tet, tet_path_glob))
                    continue
                elif len(tet_paths) > 1:
                    print(('WARNING: %s day %02d has multiple directories for tetrode %02d,' +
                          'by default using first entry\n(%s)') %
                          (self.name, day, tet, tet_paths))

                spike_data_path = os.path.join(tet_paths[0], '%s%02d-%02d.mat'
                                               % (self.name, day, tet))

                path_list.append((day, tet, spike_data_path))

        return path_list

    def _get_eeg_paths(self):
        """ returns a list of paths that points to the eeg data ([tetrode]-###.eeg) of each tetrode and day.

        This function uses the older \*.eeg because the reference dataset is frank.

        """
        path_list = []

        for day in self.days:
            day_path = os.path.join(self.base_dir, self.name, '%s%02d'
                                    % (self.name, day))
            for tet in self.tetrodes:
                tet_path_glob = os.path.join(day_path, '%02d*.eeg' % tet)
                tet_paths = glob(tet_path_glob)
                if len(tet_paths) < 1:
                    print(('WARNING: %s day %02d does not have eeg file for tetrode %02d,' +
                           ' skipping tetrode (%s)')
                          % (self.name, day, tet, tet_path_glob))
                    continue
                elif len(tet_paths) > 1:
                    print(('WARNING: %s day %02d has multiple eeg files for tetrode %02d,' +
                           'by default using first entry\n(%s)')
                          % (self.name, day, tet, tet_paths))
                path_list.append((day, tet, tet_paths[0]))
        return path_list

    def _get_pos_paths(self):
        """ get_pos_paths - returns a list of paths that points to the pos
        data (matclust_[animalname][day]-[tetrode].mat) of each tetrode and day.

        This function uses the already processed matclust data to extract position
        because the raw data of the reference dataset used (frank) was lost.

        """
        path_list = []
        for day in self.days:
            day_path = os.path.join(self.base_dir, self.name, '%s%02d'
                                    % (self.name, day))
            for tet in self.tetrodes:
                tet_path_glob = os.path.join(day_path, '%02d*' % tet, '%s%02d-%02d_params.mat'
                                             % (self.name, day, tet))
                tet_paths = glob(tet_path_glob)

                # directory sanity check
                if len(tet_paths) < 1:
                    print(('WARNING: %s day %02d does not have file for tetrode %02d,' +
                           ' skipping tetrode (%s)')
                          % (self.name, day, tet, tet_path_glob))
                    continue
                elif len(tet_paths) > 1:
                    print(('WARNING: %s day %02d has multiple directories for tetrode %02d,' +
                           'by default using first entry\n(%s)')
                          % (self.name, day, tet, tet_paths))

                path_list.append((day, tet, tet_paths[0]))
        return path_list

    def _get_posmat_paths(self):
        """ get_pathmat_paths - returns a list of paths that points to the post processed pos
        data ([post dir]/[3 char name prefix]pos[day].mat) of each day.

        This function uses the already processed video position data to extract position
        because the raw data of the reference dataset used (frank) was lost.

        """
        path_list = []
        for day in self.days:
            anim_prefix = self.name[0:3].title()
            day_glob = os.path.join(self.base_dir, anim_prefix.title()[0:3],
                                    '%slinpos%02d.mat' % (self.name[0:3].lower(), day))
            day_path = glob(day_glob)
            # directory sanity check
            if len(day_path) < 1:
                print('WARNING: %s day %02d does not have file %slinpos%02d.mat'
                      % (self.name, day, self.name[0:3].lower(), day))
                continue
            elif len(day_path) > 1:
                print(('WARNING: %s day %02d has multiple directories %slinpos%02d.mat' +
                       'which should not happen...')
                      % (self.name, day, self.name[0:3].lower(), day))

            path_list.append((day, day_path[0]))
        return path_list

    def _get_ripplecons_paths(self):

        path_list = []
        for day in self.days:
            anim_prefix = self.name[0:3].title()
            day_glob = os.path.join(self.base_dir, anim_prefix.title()[0:3],
                                    '%sripplescons%02d.mat' % (self.name[0:3].lower(), day))
            day_path = glob(day_glob)
            # directory sanity check
            if len(day_path) < 1:
                print('WARNING: %s day %02d does not have file %sripplescons%02d.mat'
                      % (self.name, day, self.name[0:3].lower(), day))
                continue
            elif len(day_path) > 1:
                print(('WARNING: %s day %02d has multiple directories %sripplescons%02d.mat' +
                       'which should not happen...')
                      % (self.name, day, self.name[0:3].lower(), day))

            path_list.append((day, day_path[0]))
        return path_list

    def _num_epochs(self, day):
        time_day = self.times[day]
        return np.size(time_day, 0)

    def _calc_epoch_state(self, day, cur_time):
        """ Returns the encoded epoch that the time belongs to.  If the time
        is not in an epoch or if the epoch is not being encoded, return -1

        Args:
            day: The day to lookup
            cur_time: The current time to lookup

        """
        time = self.times[day]
        for ii in self.epochs:
            if time[ii][0] <= cur_time <= time[ii][1]:
                return ii
        return -1

    def _get_epoch_time_range(self, day, epoch):
        time = self.times[day]
        if epoch not in self.epochs:
            raise ConfigurationError('Epoch requested not an '
                                     'epoch being processed')

        return time[epoch][0], time[epoch][1]
