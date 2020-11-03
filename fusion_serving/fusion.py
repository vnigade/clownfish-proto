import sys
# from .methods import kalman as Kalman
from .methods import exponential_smoothing as ES
from .async_model import AsyncModel
import time
from .sliding_window import SlidingWindow
from .filter import Filter
from .utils import FusionStats
import torch.nn.functional as F
import torch


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
    def __init__(self, opt, local_model=None, remote_model=None):
        self.local_model = local_model
        self.remote_model = remote_model
        if remote_model is None and local_model is None:
            raise NotImplemented(
                'Not implement when local and remote models are None')

        self.fusion_enabled = (remote_model and local_model)
        if self.fusion_enabled:
            self.fusion_method = get_fusion_method(opt)
            self.async_remote_model = AsyncModel(remote_model, num_workers=1)
            self.filter = Filter(opt)

        self.prev_output = (None, None)
        self.sliding_window = SlidingWindow(opt)

        self.stats = FusionStats()

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

    def _filter(self, window_meta, input, scheme='periodic', rho=0.0):
        '''
        Identify window to send it to the remote.
        '''
        if self.filter.should_send(window_meta.id, scheme, rho):
            # print("Filter: sending window", window_meta.id, scheme, rho)
            self.async_remote_model.predict(window_meta, input)
            if self.stats:
                self.stats.update_send_freq(window_meta.id)

    def _update(self, cur_window_meta, input):
        '''
        Update history if we have received the remote score.
        '''
        # Check if remote model prediction is available
        remote_output = self.async_remote_model.next_output()
        # TODO: This is only for testing.
        # if remote_output is not None and (cur_window_meta.id - remote_output[0][2]) > 20:
        #     remote_output =  None # This is to test computation time of very long lag.

        remote_frame2window_id = -1
        if remote_output is not None:
            remote_frame2window_id = remote_output[0][0] // self.sliding_window.window_stride
            # print("Update: received result from remote for {} at {}".format(remote_frame2window_id, cur_window_meta.id))
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

    def _predict_window(self, window_meta, input):
        '''
        window_meta: Tuple (start_frame_id, end_frame_id, window_id)
        '''
        # Filter
        self._filter(window_meta, input, scheme='periodic')

        # Local predict
        local_output = self.local_model(window_meta, input)
        local_frame2window_id = local_output[0][0] // self.sliding_window.window_stride
        assert local_frame2window_id == window_meta.id, print("Local window ID {} is not equal to window_id {}".format(
            local_frame2window_id, window_meta.id))
        local_result = local_output[1]
        local_result_softmax = softmax(local_result)

        # compute similarity
        self.fusion_method.compute_similarity(
            window_meta.id, local_result, local_result_softmax)

        # Filter based on the similarity measure
        rho = self.fusion_method.similarity.get_similarity(window_meta.id)
        self._filter(window_meta, input, scheme='transition', rho=rho)

        # Update
        remote_window_id, remote_output = self._update(window_meta, input)

        # Fuse
        output, output_size = self.fusion_method.fuse(
            window_meta.id, local_result_softmax, remote_window_id, remote_output)
        return window_meta, output, output_size

    def predict_window(self, window_meta, input):
        if self.fusion_enabled:
            output = self._predict_window(window_meta, input)
        elif self.local_model:
            output = self.local_model(window_meta, input)
        elif self.remote_model:
            output = self.remote_model(window_meta, input)

        if not self.fusion_enabled:
            output_softmax = softmax(output[1])
            output = (output[0], output_softmax, output[2])
            if self.prev_output[0] is not None:
                # Perform moving average
                input1 = self.prev_output[0]
                input2 = output[1]
                output_ma = 0.5 * input1 + 0.5 * input2
                output = (output[0], output_ma, output[2])
        return output

    def predict_frame(self, frame_id, frame):
        self.sliding_window.append(frame_id, frame)
        window, window_meta = self.sliding_window.next()
        if window is None:
            output = self.prev_output[0]
            output_size = self.prev_output[1]
        else:
            _, output, output_size = self.predict_window(
                window_meta=window_meta, input=window)
            self.prev_output = (output, output_size)

        return frame_id, output, output_size

    def __call__(self, window_id, input):
        return self.predict_window(window_id, input)
