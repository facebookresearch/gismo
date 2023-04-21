# Copyright (c) Meta Platforms, Inc. and affiliates 
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from .permutation_invariant_criterion import (
    BiPartiteAssignmentCriterion,
    ChamferDistanceCriterion,
    ChamferDistanceType,
    PooledBinaryCrossEntropy,
    chamfer_cross_entropy_distance,
    knn_cross_entropy,
)


def test_knn_cross_entropy():
    targets = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ]
    )
    probs = torch.tensor(
        [
            [[0.05, 0.9, 0.05], [0.9, 0.05, 0.05]],
            [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05]],
            [[0.9, 0.05, 0.05], [0.05, 0.05, 0.9]],
        ]
    )
    expected = -5 * torch.log(torch.tensor(0.9)) - torch.log(torch.tensor(0.05))
    output = knn_cross_entropy(targets.cuda(), probs.cuda())
    assert torch.allclose(expected.cuda(), output.cuda(), atol=1e-4)
    print(chamfer_cross_entropy_distance(probs.cuda(), targets.cuda()))


def test_ChamferDistance_shapes():
    torch.manual_seed(0)
    batch_size = 2
    max_num_ingredients = 3
    vocab_size = 5

    # Don't predict a pad value
    logits = torch.randn(
        size=(batch_size, max_num_ingredients + 1, vocab_size - 1)
    ).cuda()
    target = torch.randint(
        low=0, high=vocab_size - 1, size=(batch_size, max_num_ingredients + 1)
    ).cuda()
    target[:, -1] = 0  # EOS at the end for every sample
    target[-1, -2] = 0  # Adding EOS on step before the end
    target[-1, -1] = vocab_size - 1  # Adding padding after the EOS

    for distanceType in [
        ChamferDistanceType.l2,
        ChamferDistanceType.cross_entropy,
        ChamferDistanceType.unilateral_cross_entropy,
    ]:
        criterion = ChamferDistanceCriterion(
            eos_value=0, pad_value=vocab_size - 1, distanceType=distanceType
        )
        losses = criterion(logits, target)
        print(losses)


def test_BiPartiteAssignmentCriterions_shapes():
    torch.manual_seed(0)
    batch_size = 21
    max_num_ingredients = 20
    vocab_size = 1487
    pad_value = vocab_size - 1
    criterion = BiPartiteAssignmentCriterion(eos_value=0, pad_value=pad_value)
    logits = torch.randn(
        size=(batch_size, max_num_ingredients + 1, vocab_size - 1)
    )  # No pad value predicted
    target = torch.randint(
        low=1, high=pad_value, size=(batch_size, max_num_ingredients + 1)
    )  # Create valid ingr

    # Try all possibilities of EOS position
    for i in range(batch_size):
        target[i, i] = 0  # Adding EOS before the end
        target[i, i + 1 :] = pad_value  # Adding padding after the EOS

    losses = criterion(logits.cuda(), target.cuda())
    assert "label_loss" in losses
    assert "eos_loss" in losses
    print(losses)


def test_SetPooledCrossEntropy_shapes():
    torch.manual_seed(0)
    batch_size = 2
    max_num_ingredients = 20
    vocab_size = 1487

    criterion = PooledBinaryCrossEntropy(eos_value=0, pad_value=vocab_size)
    logits = torch.randn(size=(batch_size, max_num_ingredients + 1, vocab_size))
    target = torch.randint(
        low=0, high=vocab_size, size=(batch_size, max_num_ingredients + 1)
    )
    losses = criterion(logits, target)

    assert 2 == len(losses)
    assert torch.allclose(losses["label_loss"], torch.tensor(0.0833), atol=1e-4)
    assert torch.allclose(losses["eos_loss"], torch.tensor(0.0003), atol=1e-4)


def test_SetPooledCrossEntropy_output():
    torch.manual_seed(0)
    vocab_size = 5

    criterion = PooledBinaryCrossEntropy(eos_value=0, pad_value=vocab_size)

    # Input in which we predict two times the first ingredient
    # and one time the second ingredient, then EOS
    logits = torch.FloatTensor(
        [[[1, 11, 1, 1, 1], [1, 10, 1, 1, 1], [1, 1, 10, 1, 1], [11, 1, 1, 1, 1]]]
    )

    # If the target is predict 1, 2 and 3
    target = torch.LongTensor([[1, 2, 3, 0]])
    losses = criterion(logits, target)
    assert torch.allclose(losses["label_loss"], torch.tensor(2.2503), atol=1e-4)
    assert torch.allclose(losses["eos_loss"], torch.tensor(0.0001), atol=1e-4)

    # If the target is predict 1 and 2
    target = torch.LongTensor([[1, 2, 0, 0]])
    losses = criterion(logits, target)
    assert torch.allclose(losses["label_loss"], torch.tensor(2.2502), atol=1e-4)
    assert torch.allclose(losses["eos_loss"], torch.tensor(2.2502), atol=1e-4)


def test_SetPooledCrossEntropy_is_invariant_to_order():
    torch.manual_seed(0)
    batch_size = 2
    max_num_ingredients = 20
    vocab_size = 1487
    eos_token = 0

    criterion = PooledBinaryCrossEntropy(eos_value=0, pad_value=vocab_size)
    for _ in range(10):
        logits = torch.randn(size=(batch_size, max_num_ingredients + 1, vocab_size))
        target = torch.randint(
            low=1, high=vocab_size, size=(batch_size, max_num_ingredients + 1)
        )
        target[:, -1] = eos_token
        losses = criterion(logits, target)

        new_target = target.clone()
        new_target[:, :-1] = target[:, torch.randperm(max_num_ingredients)]
        new_losses = criterion(logits, new_target)

        assert torch.allclose(losses["label_loss"], new_losses["label_loss"])
        assert torch.allclose(losses["eos_loss"], new_losses["eos_loss"])
