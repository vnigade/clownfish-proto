import sys
# from .methods import kalman as Kalman
from .methods import exponential_smoothing as ES
from .async_model import AsyncModel
import time
from .sliding_window import SlidingWindow, WindowMeta
from .filter import Filter
from .utils import EarlyDiscardStats
import torch.nn.functional as F
import torch
import threading


def get_fusion_method(opt):
    method = None
    if opt.fusion_method == 'kalman':
        method = Kalman.generate_method(opt)
    elif opt.fusion_method == 'exponential_smoothing':
        method = ES.generate_method(opt)
    return method


def softmax(data):
    data = torch.FloatTensor(data)
    data = F.softmax(data, dim=-1)
    return data.cpu().detach().numpy()


def sigmoid(data):
    data = torch.FloatTensor(data)
    data = F.sigmoid(data)
    return data.cpu().detach().numpy()


class FusionModel():
    def __init__(self, opt, local_model=None, remote_model=None, callback=None):
        self.local_model = local_model
        self.remote_model = remote_model
        if remote_model is None or local_model is None:
            raise NotImplemented(
                'Not implement when local and remote models are None')

        self.sliding_window = SlidingWindow(opt)

        if callback:  # async processing
            self.remote_model.register_callback(self._handle_window_output)
            self._callback = callback
        self.remote_model.set_cache(False)
        self.local_model.set_compression_level(1)
        self.outstanding_frames = []
        self.outstanding_windows = []
        self.received_window_outputs = {}
        self.lock = threading.Lock()
        self.discard_threshold = 0.5
        self.windows_stats = EarlyDiscardStats()
        self.last_window_to_remote = None

    def reset(self):
        self.sliding_window.reset()
        self.windows_stats.reset()
        assert self.local_model is not None, "Local model cannot be None"
        assert self.remote_model is not None, "Remote model cannot be None"
        self.local_model.reset()
        self.remote_model.reset()
        self.last_window_to_remote = None

        return

    def close(self):
        self.windows_stats.print()
        pass

    def _handle_window_output(self, window_output):
        window_id = window_output[0][2]
        # print("Handling window output", window_id)
        with self.lock:
            self.received_window_outputs[window_id] = window_output
            if window_id != self.outstanding_windows[0].id:
                return
            while len(self.outstanding_windows):
                if self.outstanding_windows[0].id in self.received_window_outputs:
                    window_meta = self.outstanding_windows[0]
                    window_output = self.received_window_outputs[window_meta.id]
                    _, output, output_size = window_output
                    # print("Handling window output:", window_meta)
                    # print("Outstanding frames:", self.outstanding_frames)
                    output = self._handle_non_fusion(window_meta, output)

                    frame_id = window_meta.end_frame
                    for i in range(self.sliding_window.window_stride):
                        if len(self.outstanding_frames) == 0:
                            break
                        assert self.outstanding_frames[0][0] == (frame_id + i), print("Expected outstanding frame id {}"
                                                                                      " but received id {}".format(self.outstanding_frames[0][0], (frame_id + i)))
                        self._callback((frame_id+i, output, output_size))
                        del self.outstanding_frames[0]

                    del self.outstanding_windows[0]
                    del self.received_window_outputs[window_meta.id]
                else:
                    break

        # print("Handling window output exit", window_output[0][2], window_id)
        return

    def _handle_initial_frames(self, frame_id):
        with self.lock:
            assert frame_id == self.outstanding_frames[0][0], print(
                "Initial frames mismatch")
            del self.outstanding_frames[0]
            self._callback((frame_id, None, None))

    def _handle_non_fusion(self, window_meta, output):
        if output is not None:
            output = softmax(output)
        return output

    def predict_window(self, window_meta, input):
        output = self.local_model(window_meta, input)
        output_sigmoid = sigmoid(output[1])[0]
        # print("Output shape enter", window_meta, output_sigmoid)
        if output_sigmoid >= self.discard_threshold:
            assert window_meta.id == self.local_model.window_meta_cache.id
            cached_input = self.local_model.encoded_cache_list
            if self.last_window_to_remote and (self.last_window_to_remote.id + 1) == window_meta.id:
                start_idx = self.sliding_window.window_size - self.sliding_window.window_stride
                cached_input = cached_input[start_idx:]
            # self.remote_model(window_meta, input)
            self.remote_model.predict(
                window_meta, cached_input, needs_serialization=False)
            self.last_window_to_remote = window_meta
            self.windows_stats.update_n_sent_windows(1)
            self.windows_stats.update_frames_sent(window_meta, sent=True)
        else:
            output = (output[0], None, None)
            self._handle_window_output(output)
            self.windows_stats.update_n_sent_windows(0)
            self.windows_stats.update_frames_sent(window_meta, sent=False)
        # print("Output shape exit", window_meta, output_sigmoid.shape)
        return

    def predict_frame(self, frame_id, frame):
        self.sliding_window.append(frame_id, frame)
        window, window_meta = self.sliding_window.next()

        with self.lock:
            self.outstanding_frames.append((frame_id, window_meta))

        if window is not None:
            with self.lock:
                self.outstanding_windows.append(window_meta)
            self.predict_window(window_meta=window_meta, input=window)
        else:
            handle_this_frame = False
            with self.lock:
                if self.outstanding_frames[0][0] == frame_id:
                    handle_this_frame = True
            if handle_this_frame:
                self._handle_initial_frames(frame_id)

        # if frame_id < self.sliding_window.window_size:
            # Initial frames do not have any results. So, reply immediately
        #    self._handle_initial_frames(frame_id)

        return

    def __call__(self, window_id, input):
        return self.predict_window(window_id, input)
