import numpy as np


class FusionStats:
    def __init__(self):
        self.send_freq = []
        self.prev_send_id = 0
        self.lag_hist = {}

    def reset(self):
        self.prev_send_id = 0

    def update_send_freq(self, cur_send_id):
        self.send_freq.append(cur_send_id - self.prev_send_id)
        self.prev_send_id = cur_send_id

    def update_lag_hist(self, key, value):
        if not key in self.lag_hist:
            self.lag_hist[key] = 0
        self.lag_hist[key] += value

    def print(self):
        send_freq = np.asarray(self.send_freq, dtype=int)

        lag_hist_str = '{'
        for key, value in self.lag_hist.items():
            lag_hist_str += '({}:{}) '.format(key, value)
        lag_hist_str += '}'

        print("FusionStat: SendFreq(mean/std)=({}/{}), LagHist={}".format(send_freq.mean(),
                                                                          send_freq.std(), lag_hist_str))
