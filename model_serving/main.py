import sys
from model_serving import ModelServingServicer as ModelServer

from opts import parse_opts
from model import generate_model, get_transforms
from ResNets_3D_PyTorch.utils import AverageMeter
from ResNets_3D_PyTorch.utils import Logger
import os
from common import utils

if __name__ == '__main__':
    opt = parse_opts()
    args_dict = opt.__dict__
    print("------------------------------------")
    print(" Configurations:")
    for key in args_dict.keys():
        print("- {}: {}".format(key, args_dict[key]))
    print("------------------------------------")

    # assert opt.resume_path
    model = generate_model(opt)
    print(model)
    transforms = get_transforms(opt)
    stats = None
    if opt.enable_stats:
        stats = utils.Stats(["preprocessing_time", "prediction_time", "postprocessing_time"],
                            stats_name="model_server")
    server = ModelServer(model, opt.port_number, opt.workers,
                         transforms=transforms, stats=stats)
    server.start_server()
