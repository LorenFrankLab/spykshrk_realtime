from zmq import ZMQError
import xml.etree.ElementTree as ET

from spykshrk.realtime.datatypes import Datatypes
from spykshrk.realtime.realtime_base import DataSourceReceiver
from spykshrk.realtime.datatypes import LFPPoint, SpikePoint, CameraModulePoint
from trodesnetwork.socket import SourceSubscriber

def get_ntrode_inds(config, ntrode_ids):
    # ntrode_ids should be a list of integers
    inds_to_extract = []
    xmltree = ET.parse(config["trodes"]["config_file"])
    root = xmltree.getroot()
    for ii, ntrode in enumerate(root.iter("SpikeNTrode")):
        ntid = int(ntrode.get("id"))
        if ntid in ntrode_ids:
            inds_to_extract.append(ii)

    return inds_to_extract

class TrodesNetworkDataReceiver(DataSourceReceiver):
    def __init__(self, comm, rank, config, datatype):
        if not datatype in (
            Datatypes.LFP,
            Datatypes.SPIKES,
            Datatypes.LINEAR_POSITION
        ):
            raise TypeError(f"Invalid datatype {datatype}")
        super().__init__(comm, rank, config, datatype)

        if self.datatype == Datatypes.LFP:
            self.datapoint = LFPPoint(None, None, None, None)
            self.sub_obj = SourceSubscriber('source.lfp')
        elif self.datatype == Datatypes.SPIKES:
            self.datapoint = SpikePoint(None, None, None)
            self.sub_obj = SourceSubscriber('source.spikes')
        else:
            self.sub_obj = SourceSubscriber('source.position')
            self.datapoint = CameraModulePoint(None, None, None, None, None)
        
        self.start = False
        self.stop = False

        self.ntrode_ids = [] # only applicable for spikes and LFP
        self.inds_to_extract = None # only applicable for LFP
        self.ntrode_id_ind = 0 # only applicable for LFP

        self.temp_data = None

    def __next__(self):
        if self.stop:
            raise StopIteration()

        if not self.start:
            return None

        if self.datatype == Datatypes.LFP and self.is_subbed_multiple:
            # extracted all the channels from a single LFP packet so reset index.
            if self.ntrode_id_ind % self.n_subbed == 0:
                self.ntrode_id_ind = 0
            else:
                self.class_log.info(f"ntrode_id_ind is {self.ntrode_id_ind}")
                ind = self.inds_to_extract[self.ntrode_id_ind]
                ntid = self.ntrode_ids[self.ntrode_id_ind]
                self.datapoint.update_data(self.temp_data, ind, ntid, ind)
                self.ntrode_id_ind += 1
                self.class_log.debug("Retuning multi sample LFP")
                return self.datapoint, None
        
        try:
            self.temp_data = self.sub_obj.receive(noblock=True)
            
            if self.datatype == Datatypes.LFP:
                
                ind = self.inds_to_extract[self.ntrode_id_ind]
                ntid = self.ntrode_ids[self.ntrode_id_ind]
                self.datapoint.update_data(self.temp_data, ind, ntid, ind)
                if self.is_subbed_multiple:
                    self.class_log.debug("Multiple")
                    self.ntrode_id_ind += 1
                self.class_log.debug("Returning LFP")
                return self.datapoint, None

            elif self.datatype == Datatypes.SPIKES:
                
                ntid = self.temp_data['nTrodeId']
                if ntid in self.ntrode_ids:
                    self.datapoint.update_data(self.temp_data)
                    return self.datapoint, None
                else:
                    return None
            
            else:

                self.datapoint.update_data(self.temp_data)
                return self.datapoint, None

        except ZMQError:
            return None

    def register_datatype_channel(self, channel):
        ntrode_id = channel
        if self.datatype in (Datatypes.LFP, Datatypes.SPIKES):
            if not ntrode_id in self.ntrode_ids:
                self.ntrode_ids.append(ntrode_id)
            else:
                self.class_log.debug("Already streaming from ntrode id {ntrode_id}")
        else:
            self.class_log.debug("Already set up to stream position, doing nothing")
            return
        
        if self.datatype == Datatypes.LFP:
            self.inds_to_extract = get_ntrode_inds(self.config, self.ntrode_ids)

        self.class_log.debug(
            f"Set up to stream from ntrode ids {self.ntrode_ids}")

    def start_all_streams(self):
        self.start = True
        self.class_log.debug("Datastream activated")

    def stop_all_streams(self):
        self.start = False

    def stop_iterator(self):
        self.stop = True

    @property
    def n_subbed(self):
        return len(self.ntrode_ids)

    @property
    def is_subbed_multiple(self):
        return len(self.ntrode_ids) > 1