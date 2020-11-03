import sys

# NOTE: If for some reason, your PYTHONPATH doesn't work. Comment out the following two lines
# by setting `root_path` to the absolute path of `clownfish-proto` repo.
# root_path = "~/clownfish-proto"
# sys.path.insert(0, root_path)

import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as F
from ResNets_3D_PyTorch.mean import get_mean
from ResNets_3D_PyTorch.temporal_transforms import LoopPadding
from ResNets_3D_PyTorch.spatial_transforms import Compose, Normalize, Scale, CenterCrop, ToTensor
from ResNets_3D_PyTorch import model as resnet_model
import torch
import os
from multiprocessing.pool import ThreadPool
import multiprocessing as mp


def load_model(opt, model):
    if os.path.isfile(opt.resume_path):
        print(("=> loading checkpoint '{}'".format(opt.resume_path)))
        checkpoint = torch.load(opt.resume_path)
        res = [val for key, val in checkpoint['state_dict'].items()
               if 'module' in key]
        # if opt.no_cuda:
        if len(res) == 0:
            model.module.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['state_dict'])
    else:
        print(("Checkpoint file {} does not exist".format(opt.resume_path)))

    return model


def generate_model(opt):
    opt.n_classes = opt.num_classes
    opt.pretrain_path = None
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    model, _ = resnet_model.generate_model(opt)

    if opt.resume_path:
        model = load_model(opt, model)

    model.eval().cuda()
    if opt.use_tensorrt:
        from torch2trt import torch2trt
        input = torch.ones((1, 3, 32, 112, 112)).cuda()
        model = torch2trt(model, [input])

    # Inference optimzations
    cudnn.benchmarks = True
    cudnn.enabled = True
    return model


_spatial_transforms = None
_temporal_transforms = None
_mp_pool = None


def transforms(input):
    global _spatial_transforms, _temporal_transforms, _mp_pool
    # num_indices = list(range(input.shape[0]))
    num_indices = list(range(len(input)))
    # num_indices = _temporal_transforms(num_indices) # don't need to pad temporally

    # clip = [_spatial_transforms(F.to_pil_image(input[i])) for i in num_indices]
    if _mp_pool is None:
        clip = [_spatial_transforms(input[i]) for i in num_indices]
    else:
        clip = _mp_pool.map(_spatial_transforms, input)
        # futures = [_mp_pool.submit(_spatial_transforms, input[i]) for i in num_indices]
        # clip = [future.result() for future in futures]
    clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
    clip = clip.unsqueeze(0)
    return clip


def get_transforms(opt):
    global _spatial_transforms, _temporal_transforms, _mp_pool
    _SCALE_SIZE = opt.sample_size
    _CROP_SIZE = opt.sample_size
    _TEMPORAL_SIZE = opt.sample_duration
    mean = get_mean(1, dataset='activitynet')
    _NORM_METHOD = Normalize(mean, [1, 1, 1])
    _spatial_transforms = Compose([
        Scale(_SCALE_SIZE),
        CenterCrop(_CROP_SIZE),
        ToTensor(1), _NORM_METHOD
    ])
    _temporal_transforms = LoopPadding(_TEMPORAL_SIZE)
    _mp_pool = ThreadPool(mp.cpu_count()//2)
    return transforms
