
import numpy as np
from .similarity import Similarity

KEY = 0
VALUE = 1


class ExponentialSmoothing():
    def __init__(self, opt):
        self.history = {}
        self.local_results_softmax = {}
        self.accumulated_alpha = {}
        self.sim_method = opt.sim_method
        self.ensemble_weight = 0.33
        self.init(opt)

    def init(self, opt):
        self.similarity = Similarity(opt.sim_method, opt)

    def reset(self):
        self.history.clear()
        self.local_results_softmax.clear()
        self.accumulated_alpha.clear()
        self.similarity.reset()

    def ensemble(self, local_result, remote_result):
        ensemble = self.ensemble_weight * local_result + \
            (1.0 - self.ensemble_weight)*remote_result
        return ensemble

    def get_corelation_coefficent(self, id, **kwargs):
        rho = self.similarity.get_similarity(id, **kwargs)
        threshold = 0.5
        alpha = rho
        assert id > 0
        if self.accumulated_alpha[id-1] < threshold:
            alpha = 0.5

        self.accumulated_alpha[id] = self.accumulated_alpha[id-1] * alpha
        # print("CorrCoeff: Accumulated alpha prev {} and cur {}".format(self.accumulated_alpha[id-1], self.accumulated_alpha[id]))

        return alpha

    def compute_similarity(self, window_id, local_result, local_result_softmax):

        # raw score
        if window_id != 0:
            prev_window_id = self.local_results_prev[KEY]
            assert window_id == (prev_window_id + 1)
            vec1 = self.local_results_prev[VALUE]
            vec2 = local_result

            # softmax score
            vec1_softmax = self.local_results_softmax[window_id-1]
            vec2_softmax = local_result_softmax

            # compute similarity
            kwargs = {'current_window_id': window_id, 'vec1': vec1, 'vec2': vec2, 'vec1_softmax': vec1_softmax,
                      'vec2_softmax': vec2_softmax}
        else:
            kwargs = {}
        rho = self.similarity.get_similarity(window_id, **kwargs)
        # print("Similarity: {} for {}".format(rho, window_id))
        # save local results
        self.local_results_prev = (window_id, local_result)
        self.local_results_softmax[window_id] = local_result_softmax
        return rho

    def fuse(self, current_window_id, local_result_softmax, remote_window_id, remote_result_softmax):

        # print("Fuse Score: current_window_id remote_window_id", current_window_id, remote_window_id)
        perceived = None
        # We received remote score faster.
        if remote_window_id == current_window_id:
            assert remote_result_softmax is not None
            perceived = self.ensemble(
                local_result_softmax, remote_result_softmax)
            # print("Fuse Score: Remote is avaible")

        if len(self.history) == 0:
            # Init list of states
            if remote_window_id == current_window_id:
                assert perceived is not None
                self.accumulated_alpha[current_window_id] = np.max(perceived)
            else:
                perceived = local_result_softmax
                self.accumulated_alpha[current_window_id] = 0.0

            self.history[current_window_id] = perceived
            self.local_results_softmax[current_window_id] = local_result_softmax
            return perceived, perceived.shape

        # print("Fuse input", type(local_result))
        alpha = self.get_corelation_coefficent(id=current_window_id)

        if remote_window_id == current_window_id:
            assert perceived is not None
            self.accumulated_alpha[current_window_id] = np.max(perceived)
            # print("Fuse score: remote infleunce {}", self.accumulated_alpha[current_window_id])
        else:
            prev_perceived = self.history[current_window_id - 1]
            perceived = prev_perceived * alpha + \
                (1 - alpha) * local_result_softmax

        self.history[current_window_id] = perceived
        self.local_results_softmax[current_window_id] = local_result_softmax

        return perceived, perceived.shape

    def reinforce(self, current_window_id, remote_window_id, remote_result_softmax):

        # print("Update scores for ", current_window_id, remote_window_id)
        assert current_window_id > remote_window_id

        # Update history for the remote result
        local_result_softmax = self.local_results_softmax[remote_window_id]
        perceived = self.ensemble(local_result_softmax, remote_result_softmax)
        self.history[remote_window_id] = perceived
        self.accumulated_alpha[remote_window_id] = np.max(perceived)
        # print("Update score: accumulated alpha {}".format(self.accumulated_alpha[remote_window_id]))

        # Update remaining history using predict model
        for idx in range(remote_window_id + 1, current_window_id):
            local_result_softmax = self.local_results_softmax[idx]

            # Calculate alpha
            alpha = self.get_corelation_coefficent(id=idx)

            # Update history
            perceived = self.history[idx - 1] * \
                alpha + (1 - alpha) * local_result_softmax
            self.history[idx] = perceived

        return


def generate_method(opt):
    return ExponentialSmoothing(opt)
