import xml.etree.ElementTree as ET
import numpy as np

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

def get_network_address(config):
    xmltree = ET.parse(config["trodes"]["config_file"])
    root = xmltree.getroot()
    
    network_config = root.find("NetworkConfiguration")
    if network_config is None:
        raise ValueError("NetworkConfiguration section not defined")

    try:
        address = network_config.attrib["trodesHost"]
        port = network_config.attrib["trodesPort"]
    except KeyError:
        return None

    return "tcp://" + address + ":" + port

def get_sampling_rates(config):
    # if the relevant sections don't exist in the workspace file,
    # use these defaults
    fs = 30000
    fs_lfp = 1500

    xmltree = ET.parse(config["trodes"]["config_file"])
    root = xmltree.getroot()

    hardware_config = root.find("HardwareConfiguration")
    fs = hardware_config.attrib["samplingRate"]

    # lfp sampling rate might be defined in various places
    try_network_config = False
    global_config = root.find("GlobalConfiguration")
    try:
        fs_lfp = global_config.attrib["lfpRate"]
    except KeyError:
        try_network_config = True

    if try_network_config:
        network_config = root.find("NetworkConfiguration")
        if network_config is not None:
            try:
                fs_lfp = network_config.attrib["lfpRate"]
            except KeyError:
                pass

    # if the lfp rate isn't found anywhere, it'll be the default
    # global sampling rate is always defined though
    return float(fs), float(fs_lfp)


def normalize_to_probability(distribution):
    '''Ensure the distribution integrates to 1 so that it is a probability
    distribution
    '''
    return distribution / np.nansum(distribution)
