import torch

from inv_cooking.models.ingredients_predictor.utils import label2_k_hots


def test_label2_k_hots():
    vocab_size = 20
    max_num_labels = 10

    pad_value = vocab_size - 1
    labels = torch.tensor(
        [
            list(range(max_num_labels)) + [pad_value],
            list(range(max_num_labels, max_num_labels * 2)) + [pad_value],
        ]
    )

    expected = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    )
    out = label2_k_hots(labels, pad_value=pad_value, remove_eos=False)
    assert expected.equal(out), "padding should be removed"

    expected = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    )
    out = label2_k_hots(labels, pad_value=pad_value, remove_eos=True)
    assert expected.equal(out), "padding and end of sequence should be removed"
