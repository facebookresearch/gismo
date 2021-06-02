"""
Take a lightning checkpoint and convert it to a image encoder checkpoint
by extracting the weights of the image encoder
"""
import argparse

import torch


def lightning_checkpoint_to_image_encoder_checkpoint(input_path: str, output_path: str):
    cp = torch.load(input_path, map_location='cpu')
    state_dict = cp["state_dict"]
    state_dict = {
        k[len('model.image_encoder.pretrained_net.'):]: v
        for k, v in state_dict.items()
        if k.startswith("model.image_encoder.pretrained_net.")
    }
    torch.save(state_dict, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str)
    parser.add_argument('-o', '--output', type=str)
    args = parser.parse_args()
    lightning_checkpoint_to_image_encoder_checkpoint(args.input, args.output)
