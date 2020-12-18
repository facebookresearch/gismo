import torch

from inv_cooking.models.ingredients_predictor.utils import predictions_to_indices


def test_predictions_to_indices_bce():
    label_logits = torch.tensor([
        [5.0, -10.0, 3.0, 2.0, 1.0, 15.0],
        [5.0, -10.0, 3.0, 2.0, 1.0, -5.0],
    ])
    label_probs = torch.sigmoid(label_logits)
    vocab_size = label_probs.size(1)

    # Test without cardinality prediction
    expected = torch.tensor([[5, 0], [0, 2]])
    out = predictions_to_indices(
        label_probs=label_probs,
        max_num_labels=2,
        pad_value=vocab_size - 1,
        threshold=0.5,
        cardinality_prediction=None,
        which_loss="bce",
    )
    assert expected.equal(out)

    # Test with cardinality prediction
    expected = torch.tensor([[5, 0, 5, 5], [0, 2, 5, 5]])
    out = predictions_to_indices(
        label_probs=label_probs,
        max_num_labels=4,
        pad_value=vocab_size - 1,
        threshold=0.5,
        cardinality_prediction=torch.tensor([2, 3]),
        which_loss="bce",
    )
    assert expected.equal(out)

def test_predictions_to_indices_td_loss():
    label_logits = torch.tensor([
        [5.0, -10.0, 3.0, 2.0, 1.0, 6.0],
        [5.0, -10.0, 3.0, 2.0, 1.0, -5.0],
    ])
    label_probs = torch.nn.functional.softmax(label_logits, dim=-1)
    vocab_size = label_probs.size(1)

    expected = torch.tensor([[5, 0, 5], [0, 2, 5]])
    out = predictions_to_indices(
        label_probs=label_probs,
        max_num_labels=3,
        pad_value=vocab_size - 1,
        threshold=0.9,
        cardinality_prediction=None,
        which_loss="td",
    )
    assert expected.equal(out), "5 is the padding value"

