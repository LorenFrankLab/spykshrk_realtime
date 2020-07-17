import sys
import logging
import logging.config
import trodes.FSData.fsFilters as fsFilters
import trodes.FSData.nspike_data as nspike
import trodes.FSData.sim_databuffer as sim_data
import franklab.franklab_data as fld
import trodes.FSData.franklab_datastream as fldstream

sys.path.append('../Modules/FSGui')
import fsShared

# import trodes.FSData.fsShared as fsShared


class SimulatorSupervisor:
    def __init__(self, sim_databuffer, num_ntrodes):
        self.class_log = logging.getLogger(name='{}.{}'.format(self.__class__.__module__, self.__class__.__name__))
        self.sim_databuffer = sim_databuffer
        self.num_ntrodes = num_ntrodes

        self.rtf = fsFilters.RTFilterManager(self, self.num_ntrodes)

        # Setup record keeper
        self.rtf.create_record_manager(label='fsdata sim', save_dir='/opt/data36/daliu/trodes/debug_scratch',
                                       file_prefix='fsdata_sim_test', file_postfix='fsdatarec')
        self.rtf.start_recording()

        # Setup ripple filter parameters
        self.rtf.ripple_filter.param.ripCoeff1 = 1.2
        self.rtf.ripple_filter.param.ripCoeff2 = 0.2
        self.rtf.ripple_filter.param.ripple_threshold = 5
        self.rtf.ripple_filter.param.sampDivisor = 10000
        self.rtf.ripple_filter.param.n_above_thresh = 2
        self.rtf.ripple_filter.param.lockoutTime = 7500
        self.rtf.ripple_filter.param.detectNoRippleTime = 60000
        self.rtf.ripple_filter.param.dioGatePort = 1
        self.rtf.ripple_filter.param.detectNoRipples = False
        self.rtf.ripple_filter.param.dioGate = 2
        self.rtf.ripple_filter.param.enabled = True
        self.rtf.ripple_filter.param.useCustomBaseline = True
        self.rtf.ripple_filter.param.updateCustomBaseline = False

        # Setup spatial filter parameters
        self.rtf.spatial_filter.param.lowerLeftX = 0
        self.rtf.spatial_filter.param.lowerLeftY = 0
        self.rtf.spatial_filter.param.upperRightX = 10000
        self.rtf.spatial_filter.param.upperRightY = 10000
        self.rtf.spatial_filter.param.minSpeed = 0
        self.rtf.spatial_filter.param.maxSpeed = 4
        self.rtf.spatial_filter.param.cmPerPix = 1
        self.rtf.spatial_filter.param.lockoutTime = 0
        self.rtf.spatial_filter.param.enabled = True

        self.rtf.stim_enabled = True

    def main_loop(self):
        for data in self.sim_databuffer():
            dataclientinfo = fsShared.DataClientInfo()
            try:
                dataclientinfo.nTrodeIndex = int(data.ntrode_index)
            except AttributeError:
                pass
            self.rtf.processdata(data_client_info=dataclientinfo, data=data)


def init_nspike_anim():
    anim_dir = '/opt/data36/daliu/other/mkarlsso/'

    timescale = 10000

    animal_name = 'bond'
    days = [4]
    tetrodes = [5, 11, 12, 14, 19]
    tetrodes_ca1 = [5, 11, 12, 14, 19]

    epoch_encode = [1]
    new_data = True
    anim = nspike.AnimalInfo(animal_dir=anim_dir,
                             animal_name=animal_name,
                             days=days,
                             tetrodes=tetrodes,
                             tetrodes_ca1=tetrodes_ca1,
                             epoch_encode=epoch_encode,
                             timescale=timescale,
                             new_data=new_data)
    return anim


def init_nspike_datastream(anim):

    eeg = nspike.EEGDataStream(anim, 100)
    # spk = nspike.SpkDataStream(anim, 100)
    # pos = nspike.LinPosMatDataStream(anim, 1000)
    pos = nspike.PosMatDataStream(anim, 1000)

    databuffer = sim_data.SimDataBuffer([eeg(), pos()])

    return databuffer


def init_franklab_datastream():
    anim = fld.FrankAnimalInfo('/opt/data36/jason/', 'kanye')
    lfp_data = fld.LFPData(anim, '20160426')
    lfp_stream = fldstream.LFPDataStream(lfp_data=lfp_data, epoch=2, ntrodes=list(range(1, 5)))

    pos_data = fld.RawPosData(anim, '20160426')
    pos_stream = fldstream.RawPosDataStream(pos_data=pos_data, epoch=2, recon_type='online')

    databuffer = sim_data.SimDataBuffer([lfp_stream(), pos_stream()])

    return databuffer


def main(argv = None):

    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'simple': {
                'format': '%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s',
                'datefmt': '%H:%M:%S',
            },
        },
        'handlers': {
            'console': {
                'level': 'INFO',
                'class': 'logging.StreamHandler',
                'formatter': 'simple',
            },
            'debug_file_handler': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'DEBUG',
                'formatter': 'simple',
                'filename': 'debug.log',
                'maxBytes': 10485760,
                'backupCount': 20,
                'encoding': 'utf8',
            }
        },
        'loggers': {
            '': {
                'handlers': ['console', 'debug_file_handler'],
                'level': 'NOTSET',
                'propagate': True,
            }
        }
    })

    # anim = init_nspike_anim()
    # databuffer = init_nspike_datastream(anim)

    databuffer = init_franklab_datastream()

    sim_super = SimulatorSupervisor(sim_databuffer=databuffer, num_ntrodes=4)
    sim_super.main_loop()


if __name__ == '__main__':
    main()
