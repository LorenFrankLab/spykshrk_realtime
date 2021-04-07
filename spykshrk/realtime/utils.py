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

def normalize_to_probability(distribution):
    '''Ensure the distribution integrates to 1 so that it is a probability
    distribution
    '''
    return distribution / np.nansum(distribution)
