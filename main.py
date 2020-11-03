from model_serving_client import ModelServingClient as ModelClient
from opts import parse_opts
import frame_manager.generator as FM
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
import numpy as np
import time
import os
from fusion_serving.fusion import FusionModel
from ResNets_3D_PyTorch.utils import AverageMeter
from ResNets_3D_PyTorch.utils import Logger
from utils import EvaluationMetric
import copy
from imutils.video import FPS
from common import utils
from common.fps import FPSLogger
import threading


def update_accuracy(label, label_name, output, window_size, video_id, frame_id):
    global target_length
    factor = 2

    if frame_id <= window_size:
        return

    if len(label) > 0:
        target_action = label[0]
    else:
        target_action = -1

    if output is not None:
        pred_action = np.argmax(output)
        pred_detection = 1
    else:
        pred_action = -1
        pred_detection = 0

    if target_action != -1:
        # Skip first few frames in accuracy calculation.
        target_length += 1
        if target_length > (window_size // factor):
            eval_metric.update(pred_action, target_action)
            detection_metric.update(pred_detection, 1)
        # print("TEST", video_id, frame_id, target_action+1, pred_action+1)

    else:
        target_length = 0
        detection_metric.update(pred_detection, 0)

    # print("TEST", video_id, frame_id, target_action+1, np.argmax(output)+1)


class FrameOutputHandler:
    """
    Output handler called as a callback when prediction output for the 
    frame is available.
    """

    def __init__(self, opt, fusion_stats=None, fps_logger=True):
        self.fusion_stats = fusion_stats
        self.window_size = opt.window_size
        self.outstanding_frames = []
        self._lock = threading.Lock()
        self._cv_lock = threading.Condition(self._lock)
        self.fps_logger = None
        if fps_logger:
            self.fps_logger = FPSLogger(timer=5)
            self.fps_logger.start()

    def append(self, video_meta):
        with self._lock:
            self.outstanding_frames.append(video_meta)

    def _wait_for_outstanding(self):
        self._cv_lock.acquire()
        try:
            while len(self.outstanding_frames) > 0:
                self._cv_lock.wait()
        finally:
            self._cv_lock.release()

    def start(self):
        self.outstanding_frames = []
        if self.fps_logger:
            self.fps_logger.start_logging()

    def stop(self):
        self._wait_for_outstanding()
        if self.fps_logger:
            self.fps_logger.stop_logging()
            self.fps_logger.save()

    def close(self):
        if self.fps_logger:
            self.fps_logger.close()
            self.fps_logger.join()

    def __call__(self, frame_output):
        frame_id, output, output_size = frame_output
        try:
            self._cv_lock.acquire()
            # print("Frame id output:", frame_id, output)
            # Get frame metadata
            video_meta = self.outstanding_frames[0]
            assert video_meta['start_frame'] == video_meta['end_frame'], print(
                "Start and End frames are not same")
            assert frame_id == video_meta['start_frame'], print(
                "Frame id {} not equal to video meta start frame {}".format(frame_id, video_meta))

            if self.fps_logger:
                self.fps_logger.update()

            # Prediction stats
            start = video_meta['frame_stats']['prediction_start_time']
            end = time.time()
            end2end_time = (end - start) * (1000)
            dataread_time = video_meta['frame_stats']['ReadingTime']

            # Update accuracy
            frame_id = video_meta['start_frame']
            update_accuracy(video_meta['label'], video_meta['label_name'], output,
                            self.window_size, video_meta['video_id'], frame_id)

            # Update stats
            if self.fusion_stats:
                self.fusion_stats.update(
                    {"PredictionTime": end2end_time, "ReadingTime": dataread_time})

            del self.outstanding_frames[0]
            self._cv_lock.notify()
        finally:
            self._cv_lock.release()


if __name__ == '__main__':
    opt = parse_opts()
    args_dict = opt.__dict__
    print("------------------------------------")
    print(" Configurations:")
    for key in args_dict.keys():
        print("- {}: {}".format(key, args_dict[key]))
    print("------------------------------------")

    # Dataset generator.
    transform = transforms.Compose([transforms.Resize(opt.input_size)])
    transform = None

    opt_fm = copy.deepcopy(opt)
    opt_fm.window_size = 1
    opt_fm.window_stride = 1
    frame_manager = FM.get_data_generator(opt_fm, transform)

    # Evaluation metric
    eval_metric = EvaluationMetric(opt)
    detection_metric = EvaluationMetric(opt)

    # Metric Logger
    logger = Logger(os.path.join(opt.log_path, "accuracy.log"), [
                    'Video', 'Average Accuracy', 'F1 Score', 'Detection Precision', 'Detection Recall'])

    # stats
    fusion_stats = None
    if opt.enable_stats:
        fusion_stats = utils.Stats(
            ['ReadingTime', 'PredictionTime'], stats_freq=1, stats_name="fusion")

    # Output handler
    output_handler = FrameOutputHandler(opt, fusion_stats, fps_logger=True)

    # Model serving clients
    edge_model = ModelClient(host=opt.edge_host, port_number=opt.edge_port,
                             stats=opt.enable_stats, stats_name="edge_client")

    cloud_model = ModelClient(host=opt.cloud_host, port_number=opt.cloud_port,
                              stats=opt.enable_stats, stats_name="cloud_client")
    fusion_model = FusionModel(
        opt, local_model=edge_model, remote_model=cloud_model, callback=output_handler)

    try:
        fusion_model.reset()
        prev_video = None
        start_time_dr = time.time()
        output_handler.start()
        for i, (video_meta, video_data) in enumerate(frame_manager):
            # print("Data reading time..", (time.time() - read_time) * 1000)
            end_time_dr = time.time()
            dataread_time = (end_time_dr - start_time_dr) * 1000
            video = video_meta['video_id']
            # This is a new video and thus new sequence. It can only occur during testing.
            if prev_video is not None and prev_video != video:
                output_handler.stop()
                logger.log({'Video': prev_video,
                            'Average Accuracy': eval_metric.get_accuracy(),
                            'F1 Score': eval_metric.get_f1_score(),
                            'Detection Precision': detection_metric.get_precision_score(),
                            'Detection Recall': detection_metric.get_recall_score()})

                # Reset for new video
                fusion_model.reset()
                output_handler.start()

            video_meta['frame_stats'] = {
                'ReadingTime': dataread_time, 'prediction_start_time': time.time()}
            frame_id = video_meta['start_frame']
            assert video_meta['start_frame'] == video_meta['end_frame']
            frame = video_data
            output_handler.append(video_meta)
            fusion_model.predict_frame(frame_id, frame)
            prev_video = video

            start_time_dr = time.time()
    except KeyboardInterrupt:
        pass

    output_handler.stop()
    output_handler.close()
    fusion_model.reset()
    # Last video stats
    logger.log({'Video': prev_video,
                'Average Accuracy': eval_metric.get_accuracy(),
                'F1 Score': eval_metric.get_f1_score(),
                'Detection Precision': detection_metric.get_precision_score(),
                'Detection Recall': detection_metric.get_recall_score()})

    logger.log({'Video': 'Total',
                'Average Accuracy': eval_metric.get_accuracy(),
                'F1 Score': eval_metric.get_f1_score(),
                'Detection Precision': detection_metric.get_precision_score(),
                'Detection Recall': detection_metric.get_recall_score()})
    fusion_model.close()
    print("Done!")
