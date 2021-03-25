from zmq import ZMQError

import time
import numpy as np
import matplotlib.pyplot as plt

import trodesnetwork as tn

def main(server_address, N, test_duration):
    t0 = time.time()

    lfp_ts = 0
    spike_ts = 0

    ts_diffs = np.zeros(N)
    ind = 0
    ct = 0

    spike_recv = tn.SourceSubscriber(
        'source.waveforms',
        server_address=server_address)
    lfp_recv = tn.SourceSubscriber(
        'source.lfp',
        server_address=server_address)
    
    while time.time() - t0 < test_duration:
        try:
            lfp_data = lfp_recv.receive(noblock=True)
            lfp_ts = lfp_data['localTimestamp']
        except ZMQError:
            pass

        try:
            spike_data = spike_recv.receive(noblock=True)
            spike_ts = spike_data['localTimestamp']

            if ind == N:
                ts_diffs = np.hstack((ts_diffs, np.zeros(N)))

            # compare timestamp of currently received spike against
            # most recently received LFP timestamp
            ts_diffs[ind] = (lfp_ts - spike_ts) / 30
            ind = (ind + 1) % ts_diffs.shape[0]
            ct += 1
        except ZMQError:
            pass

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.set_title("Delay between spike and LFP (Trodes) timestamp")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Count")
    ax.hist(ts_diffs[:ct], bins=100, range=[0, 100])
    plt.show()

if __name__ == "__main__":
    server_address = "tcp://127.0.0.1:49152" # for Trodes server
    N = 50000 # starting number of elements in buffer
    test_duration = 30 # seconds
    main(server_address, N, test_duration)