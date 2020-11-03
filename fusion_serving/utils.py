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

        print("FusionStat: SendFreq=({mean}/{std}), LagHist={}".format(send_freq.mean(),
                                                                       send_freq.std(), lag_hist_str))


class EarlyDiscardStats:
    def __init__(self):
        self.tot_windows = 0
        self.n_sent_windows = 0
        self._init_frames_sent()
        self.frames_sent_percent = []
        self.windows_sent_percent = []

    def _init_frames_sent(self):
        self.frames_sent = np.zeros(10000)
        self.max_frame_num = -1

    def reset(self):
        # Compute percentages
        windows_sent_percent = self._compute_windows_percent()
        self.windows_sent_percent.append(windows_sent_percent)
        frames_sent_percent = self._compute_frames_percent()
        self.frames_sent_percent.append(frames_sent_percent)

        # Init again
        self.tot_windows = 0
        self.n_sent_windows = 0
        self._init_frames_sent()

    def update_n_sent_windows(self, value):
        assert value == 0 or value == 1
        self.n_sent_windows += value
        self.tot_windows += 1

    def update_frames_sent(self, window_meta, sent):
        if sent:
            self.frames_sent[window_meta.start_frame:window_meta.end_frame] = 1.0
        if self.max_frame_num < window_meta.end_frame:
            self.max_frame_num = window_meta.end_frame

    def _compute_frames_percent(self):
        n_frames = self.max_frame_num + 1
        if n_frames == 0:
            return None
        n_sent = 0
        for i in range(n_frames):
            if self.frames_sent[i] == 1.0:
                n_sent += 1
        return (n_sent/n_frames)

    def _compute_windows_percent(self):
        if self.tot_windows == 0:
            return None
        return (self.n_sent_windows / self.tot_windows)

    def print(self):
        print("Windows percent per video:  ", self.windows_sent_percent)
        print("Frames percent per video: ", self.frames_sent_percent)
