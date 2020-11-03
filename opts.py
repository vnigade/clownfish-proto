import argparse


def parse_opts():
    parser = argparse.ArgumentParser(description="Clownfish framework")
    parser.add_argument('--dataset', type=str,
                        choices=['PKUMMD'])
    parser.add_argument('--n_classes', type=int, default=51)
    parser.add_argument('--dataset_path', type=str, default="")
    parser.add_argument('--dataset_annotation', type=str)
    parser.add_argument('--window_size', type=int, default=16)
    parser.add_argument('--window_stride', type=int, default=4)
    parser.add_argument('--input_size', type=int, default=224)
    # ========================= Fusion ==========================
    parser.add_argument('--fusion_method', type=str,
                        default="exponential_smoothing")
    parser.add_argument('--sim_method', type=str, default='fix_ma',
                        choices=['fix_ma', 'cosine', 'siminet'])
    parser.add_argument('--siminet_path', type=str, default='')
    parser.add_argument('--filter_interval', type=int, default=5)
    parser.add_argument('--transition_threshold', type=float, default=0.5)
    # ========================= ModelServer ==========================
    parser.add_argument('--edge_host', type=str, default="localhost")
    parser.add_argument('--cloud_host', type=str, default="localhost")
    parser.add_argument('--edge_port', type=int, default=9999)
    parser.add_argument('--cloud_port', type=int, default=9998)
    # ========================= Monitor Configs ==========================
    parser.add_argument('--print-freq', '-p', default=20, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--enable_stats', action='store_true',
                        help='enable stats collection such as prediction time')
    parser.set_defaults(enable_stats=False)
    # ========================= Runtime Configs ==========================
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--log_path', type=str, default="")

    args = parser.parse_args()

    return args
