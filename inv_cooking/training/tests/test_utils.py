from torch import nn

from inv_cooking.training.utils import _BaseModule


def test_checkpoint_to_cpu():
    model = nn.Linear(100, 10).cuda()
    checkpoint = {
        "state_dict": model.state_dict()
    }
    _BaseModule.checkpoint_to_cpu(checkpoint)
    for param_name in checkpoint["state_dict"].keys():
        assert model.state_dict()[param_name].is_cuda
        assert not checkpoint["state_dict"][param_name].is_cuda
