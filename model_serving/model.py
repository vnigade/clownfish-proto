# from models import eco_lite as ECO
# from models import i3d as I3D
from models import resnet as ResNet


def generate_model(opt):
    if opt.model == "ECO":
        model = ECO.generate_model(opt)
    elif opt.model == "I3D":
        model = I3D.generate_model(opt)
    elif opt.model == "resnet" or opt.model == "resnext" or opt.model == 'mobilenet':
        model = ResNet.generate_model(opt)
    return model


def get_transforms(opt):
    if opt.model == "ECO":
        transforms = ECO.get_transforms(opt)
    elif opt.model == "I3D":
        transforms = I3D.get_transforms(opt)
    elif opt.model == "resnet" or opt.model == "resnext" or opt.model == 'mobilenet':
        transforms = ResNet.get_transforms(opt)
    return transforms
