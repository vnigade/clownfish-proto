import sys
# from .methods import kalman as Kalman
from .methods import exponential_smoothing as ES
from .async_model import AsyncModel
import time
from .sliding_window import SlidingWindow, WindowMeta
from .filter import Filter
from .utils import FusionStats
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


class FusionModel():
    def __init__(self, opt, local_model=None, remote_model=None, callback=None):
        self.local_model = local_model
        self.remote_model = remote_model
        if remote_model is None and local_model is None:
            raise NotImplemented(
                'Not implement when local and remote models are None')

        self.fusion_enabled = False
        if self.fusion_enabled:
            self.fusion_method = get_fusion_method(opt)
            self.async_remote_model = AsyncModel(remote_model, num_workers=1)
            self.filter = Filter(opt)

        self.prev_output = (None, None)
        self.sliding_window = SlidingWindow(opt)

        self.stats = FusionStats()
        if callback:  # async processing
            self.local_model.register_callback(self._handle_window_output)
            self._callback = callback
        self.outstanding_frames = []
        self.outstanding_windows = []
        self.outstanding_inputs = []
        self.lock = threading.Lock()

    def reset(self):
        self.prev_output = (None, None)
        self.sliding_window.reset()
        self.stats.reset()
        if self.fusion_enabled:
            # Filter
            self.filter.reset()
            # Fusion method
            self.fusion_method.reset()
            # Local model
            self.local_model.reset()
            # Remote model
            self.async_remote_model.reset()
        else:
            assert self.local_model is not None, "Local model cannot be None"
            self.local_model.reset()

        return

    def close(self):
        if self.fusion_enabled:
            self.async_remote_model.close()
            self.stats.print()

    def _handle_window_output(self, window_output):
        with self.lock:
            window_meta, output, output_size = window_output
            # print("Handling window output:", window_meta)
            # print("Outstanding frames:", self.outstanding_frames)
            window_meta = WindowMeta(
                window_meta[0], window_meta[1], window_meta[2])
            assert self.outstanding_windows[0].id == window_meta.id, print("Expected outstanding windows id {} but"
                                                                           " Received id {}". format(self.outstanding_windows[0].id, window_meta.id))

            if self.fusion_enabled:
                output = self._handle_fusion(window_meta, output)
            else:
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

    def _handle_initial_frames(self, frame_id):
        with self.lock:
            assert frame_id == self.outstanding_frames[0][0], print(
                "Initial frames mismatch")
            del self.outstanding_frames[0]
            self._callback((frame_id, None, None))

    def _handle_non_fusion(self, window_meta, output):
        output_softmax = softmax(output)
        if self.prev_output[0] is not None:
            assert (self.prev_output[0].id + 1) == window_meta.id
            # Perform moving average
            input1 = self.prev_output[1]
            input2 = output_softmax
            output = 0.5 * input1 + 0.5 * input2
        else:
            output = output_softmax

        self.prev_output = (window_meta, output)
        # self.prev_output = (None, None)
        return output

    def _handle_fusion(self, window_meta, local_result):

        local_result_softmax = softmax(local_result)

        # compute similarity
        self.fusion_method.compute_similarity(
            window_meta.id, local_result, local_result_softmax)

        # Filter based on the similarity measure
        # NOTE: self.lock is already taken
        window_meta_input, input = self.outstanding_inputs[0]
        assert window_meta_input.id == window_meta.id
        del self.outstanding_inputs[0]
        rho = self.fusion_method.similarity.get_similarity(window_meta.id)
        self._filter(window_meta, input, scheme='transition', rho=rho)

        # Update
        remote_window_id, remote_output = self._update(window_meta)

        # Fuse
        output, output_size = self.fusion_method.fuse(
            window_meta.id, local_result_softmax, remote_window_id, remote_output)
        return output

    def _filter(self, window_meta, input, scheme='periodic', rho=0.0):
        '''
        Identify window to send it to the remote.
        '''
        if self.filter.should_send(window_meta.id, scheme, rho):
            # print("Filter: sending window", window_meta.id, scheme, rho)
            self.async_remote_model.predict(window_meta, input)
            if self.stats:
                self.stats.update_send_freq(window_meta.id)

    def _update(self, cur_window_meta):
        '''
        Update history if we have received the remote score.
        '''
        # Check if remote model prediction is available
        remote_output = self.async_remote_model.next_output()
        remote_frame2window_id = -1
        if remote_output is not None:
            remote_frame2window_id = remote_output[0][0] // self.sliding_window.window_stride
            print("Update: received result from remote for {} at {}".format(
                remote_frame2window_id, cur_window_meta.id))
            remote_result = remote_output[1]
            remote_result_softmax = softmax(remote_output[1])
            if remote_frame2window_id < cur_window_meta.id:
                # Received delayed result. So, correct the history
                self.fusion_method.reinforce(
                    cur_window_meta.id, remote_frame2window_id, remote_result_softmax)
                lag = cur_window_meta.id - remote_frame2window_id
                remote_output = None
                remote_frame2window_id = -1
            else:
                # No delay in the remote predictions
                assert remote_frame2window_id == cur_window_meta.id
                lag = 0
                print("No delayed result...")
                remote_output = remote_result_softmax

            if self.stats is not None:
                self.stats.update_lag_hist(lag, 1)

        return remote_frame2window_id, remote_output

    def _fusion_model(self, window_meta, input):
        '''
        window_meta: Tuple (start_frame_id, end_frame_id, window_id)
        '''
        # Filter
        self._filter(window_meta, input, scheme='periodic')

        # Local predict
        with self.lock:
            self.outstanding_inputs.append((window_meta, input))
        self.local_model(window_meta, input)

    def predict_window(self, window_meta, input):
        if self.fusion_enabled:
            self._fusion_model(window_meta, input)
        elif self.local_model:
            self.local_model(window_meta, input)
        elif self.remote_model:
            self.remote_model(window_meta, input)

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

        if frame_id < self.sliding_window.window_size:
            # Initial frames do not have any results. So, reply immediately
            self._handle_initial_frames(frame_id)

        return

    def __call__(self, window_id, input):
        return self.predict_window(window_id, input)
