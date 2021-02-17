import argparse

import torch

from inv_cooking.training.utils import _BaseModule


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=str)
    return parser


if __name__ == '__main__':
    """
    Example usage:
    
    ```
    python checkpoint_to_cpu.py /path/to/checkpoint/best.ckpt
    ```
    """
    args = get_argument_parser().parse_args()
    checkpoint = torch.load(args.checkpoint_path)
    _BaseModule.recursively_move_to_cpu(checkpoint)
    torch.save(checkpoint, args.checkpoint_path)
