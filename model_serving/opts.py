import argparse


def parse_opts():
    parser = argparse.ArgumentParser(
        description="Model Serving")
    parser.add_argument('--dataset', type=str, choices=['PKUMMD'])
    parser.add_argument('--model', type=str,
                        choices=['ECO', 'I3D', 'resnet', 'resnext'])
    parser.add_argument('--modality', type=str,
                        choices=['RGB', 'Flow', 'JOINT'])
    parser.add_argument('--num_classes', type=int, default=0)
    # ========================= Model Configs ==========================
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--model_depth", type=int)
    parser.add_argument("--resnet_shortcut", default='B',
                        type=str, help='Shortcut type of resnet (A | B)')
    parser.add_argument("--resnext_cardinality", default=32,
                        type=int, help='ResNeXt cardinality')
    parser.add_argument("--sample_size", default=112,
                        type=int, help='Height and width of inputs')
    parser.add_argument("--sample_duration", default=16,
                        type=int, help='Temporal width i.e. number of frames')
    parser.add_argument("--no_cuda", action='store_true',
                        help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=False)
    parser.add_argument("--resume_path", type=str, default="")
    parser.add_argument("--gpus", nargs='+', type=int, default=None)
    parser.add_argument('--no_cuda_predict', action='store_true',
                        help='If true, cuda is not used during predict')
    parser.set_defaults(no_cuda_predict=False)

    # ========================= Monitor Configs ==========================
    parser.add_argument('--print-freq', '-p', default=20, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--enable_stats', action='store_true',
                        help='enable stats collection such as prediction time')
    parser.set_defaults(enable_stats=False)
    parser.add_argument('--stats_freq', default=5, type=int,
                        help='Stats loggging frequency')
    parser.add_argument("--stats_path", type=str, default="")

    # ========================= Runtime Configs ==========================
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of serving workers (default: 4)')
    parser.add_argument('--port_number', type=int, default=9999)
    parser.add_argument('--use_tensorrt', action='store_true',
                        help='Optimize model using TensorRT')
    parser.set_defaults(use_tensorrt=False)

    args = parser.parse_args()

    return args
