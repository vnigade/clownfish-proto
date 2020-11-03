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


def update_accuracy(label, label_name, output, window_size, video_id, frame_id):
    global target_length
    factor = 2

    if frame_id <= window_size:
        return

    if len(label) > 0:
        target_action = label[0]
    else:
        target_action = -1

    if target_action != -1 and output is not None:
        # output can be None for first window_size frames.
        # Skip first few frames in accuracy calculation.
        target_length += 1
        if target_length > (window_size // factor):
            pred_action = np.argmax(output)
            eval_metric.update(pred_action, target_action)
        # print("TEST", video_id, frame_id, target_action+1, np.argmax(output)+1)
    else:
        target_length = 0

    # print("TEST", video_id, frame_id, target_action+1, np.argmax(output)+1)


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

    # Model serving clients
    edge_model = ModelClient(host=opt.edge_host, port_number=opt.edge_port,
                             stats=opt.enable_stats, stats_name="edge_client")

    cloud_model = None
    if opt.cloud_host != "":
        cloud_model = ModelClient(host=opt.cloud_host, port_number=opt.cloud_port,
                                  stats=opt.enable_stats, stats_name="cloud_client")
    fusion_model = FusionModel(
        opt, local_model=edge_model, remote_model=cloud_model)

    # Evaluation metric
    eval_metric = EvaluationMetric(opt)
    fps_logger = FPSLogger(timer=5)
    fps_logger.start()
    # fps_logger = None

    # Metric Logger
    logger = Logger(os.path.join(opt.log_path, "accuracy.log"), [
                    'Video', 'Average Accuracy', 'F1 Score'])

    # stats
    fusion_stats = None
    if opt.enable_stats:
        fusion_stats = utils.Stats(
            ['ReadingTime', 'PredictionTime'], stats_freq=1, stats_name="fusion")

    try:
        fusion_model.reset()
        prev_video = None
        if fps_logger:
            fps_logger.start_logging()

        start_time_dr = time.time()
        for i, (video_meta, video_data) in enumerate(frame_manager):
            # print("Data reading time..", (time.time() - read_time) * 1000)
            end_time_dr = time.time()
            dataread_time = (end_time_dr - start_time_dr) * 1000
            video = video_meta['video_id']
            # This is a new sequence and can only occur during testing
            if prev_video is not None and prev_video != video:
                if fps_logger:
                    fps_logger.stop_logging()
                    fps_logger.save()

                logger.log({'Video': prev_video,
                            'Average Accuracy': eval_metric.get_accuracy(),
                            'F1 Score': eval_metric.get_f1_score()})

                # Reset for new video
                fusion_model.reset()
                if fps_logger:
                    fps_logger.start_logging()

            start = time.time()
            frame_id = video_meta['start_frame']
            assert video_meta['start_frame'] == video_meta['end_frame']
            frame = video_data
            window_meta, output, output_size = fusion_model.predict_frame(
                frame_id, frame)
            if fps_logger:
                fps_logger.update()
            end = time.time()
            end2end_time = (end - start) * (1000)
            prev_video = video
            update_accuracy(
                video_meta['label'], video_meta['label_name'], output, opt.window_size, video_meta['video_id'], frame_id)
            # Capture stats
            if fusion_stats:
                fusion_stats.update(
                    {"PredictionTime": end2end_time, "ReadingTime": dataread_time})
            start_time_dr = time.time()
    except KeyboardInterrupt:
        pass

    if fps_logger:
        fps_logger.stop_logging()
        fps_logger.save()
        fps_logger.close()
        fps_logger.join()

    # Last video stats
    logger.log({'Video': prev_video,
                'Average Accuracy': eval_metric.get_accuracy(),
                'F1 Score': eval_metric.get_f1_score()})

    logger.log({'Video': 'Total',
                'Average Accuracy': eval_metric.get_accuracy(),
                'F1 Score': eval_metric.get_f1_score()})
    fusion_model.close()
    print("Done!")
