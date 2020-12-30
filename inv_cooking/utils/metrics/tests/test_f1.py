import torch

from inv_cooking.utils.metrics.f1 import DistributedF1


def test_f1_perfect_match():
    ingr_vocab_size = 5

    for remove_eos in [True, False]:
        for which_f1 in ["i_f1", "o_f1", "c_f1"]:
            metric = DistributedF1(
                which_f1=which_f1,
                pad_value=ingr_vocab_size - 1,
                remove_eos=remove_eos,
                dist_sync_on_step=True,
            )
            pred = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
            gt = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
            metric.update(pred, gt)
            output = metric.compute()
            assert output == torch.tensor(1.0)


def test_f1_imperfect_match():
    ingr_vocab_size = 10

    pr = torch.tensor([[1, 3, 6, 7], [6, 6, 8, 9]])
    gt = torch.tensor([[0, 3, 4, 9], [3, 4, 7, 9]])

    """
    Equivalent to (0 is EOS, 0 is PAD, both removed):
    
    pred[0] tensor([1, 0, 1, 0, 0, 1, 1, 0])
    pred[1] tensor([0, 0, 0, 0, 0, 1, 0, 1])
    
    true[0] tensor([0, 0, 1, 1, 0, 0, 0, 0])
    true[1] tensor([0, 0, 1, 1, 0, 0, 1, 0])
    """

    """
    Computed by sample:
    - the first sample has precision 0.25 (1 match over 4)
    - the first sample has recall 0.5 (0 and 9 are EOS and PAD)
    - the first sample has F1-score (2 * P * R / (P + R)) = 0.3333
    - the second sample has precision 0.0 (no match)
    - the second sample has recall 0.0 (no match)
    - the second sample has F1-score 0
    => the i_f1 is the average of the the two F1 score: 0.1666
    """
    i_f1 = DistributedF1("i_f1", pad_value=ingr_vocab_size - 1, remove_eos=True)
    i_f1.update(pr, gt)
    assert torch.allclose(torch.tensor(0.1667), i_f1.compute(), atol=1e-4)

    """
    Computed globally:
    - the precision is (1 match over 6): 0.1666
    - the recall is (1 match over 5): 0.2
    - the F1-score is (2 * P * R / (P + R)) = 0.1818
    """
    o_f1 = DistributedF1("o_f1", pad_value=ingr_vocab_size - 1, remove_eos=True)
    o_f1.update(pr, gt)
    assert torch.allclose(torch.tensor(0.1818), o_f1.compute(), atol=1e-4)

    """
    Computed by category and then averaged:
    - categories for which there are no match (2 of them) and no ground truth will get 1
    - the category with the match will get 0.666 (precision is 1, recall is 0.5)    
    tensor([0, 1, 0.6666, 0, 1, 0, 0, 0]).mean() => 0.3333
    """
    c_f1 = DistributedF1("c_f1", pad_value=ingr_vocab_size - 1, remove_eos=True)
    c_f1.update(pr, gt)
    assert torch.allclose(torch.tensor(0.3333), c_f1.compute(), atol=1e-4)
