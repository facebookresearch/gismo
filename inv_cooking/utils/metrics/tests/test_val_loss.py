import torch

from inv_cooking.utils.metrics.val_loss import DistributedValLosses


def test_distributed_validation_loss():
    weights = {"label_loss": 0.75, "recipe_loss": 0.25}

    # Include all metrics in case of 'monitor_ingr_losses'
    metric = DistributedValLosses(weights=weights, monitor_ingr_losses=True)
    fake_training(metric)
    output = metric.compute()
    assert output == {
        'total_loss': torch.tensor(2.7500),
        'label_loss': torch.tensor(2.5000),
        'recipe_loss': torch.tensor(3.5000)}

    # Exclude ingredient metrics in case of 'monitor_ingr_losses=False'
    metric = DistributedValLosses(weights=weights, monitor_ingr_losses=False)
    fake_training(metric)
    output = metric.compute()
    assert output == {
        'total_loss': torch.tensor(0.8750),
        'recipe_loss': torch.tensor(3.5000)}


def fake_training(metric):
    metric.update({
        "label_loss": torch.tensor([1.0, 2.0, 3.0]),
        "recipe_loss": torch.tensor([2.0, 3.0, 4.0]),
        "n_samples": 3,
    })
    metric.update({
        "label_loss": torch.tensor([4.0]),
        "recipe_loss": torch.tensor([5.0]),
        "n_samples": 1,
    })
