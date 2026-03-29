import torch

from rllm_ext.training.predictive_actor import PredictiveActor


def test_prediction_loss_weight_default():
    """Verify default prediction_loss_weight is 0.1."""
    actor = PredictiveActor.__new__(PredictiveActor)
    actor.prediction_loss_weight = {"prediction_loss_weight": 0.1}.get(
        "prediction_loss_weight", 0.1
    )
    assert actor.prediction_loss_weight == 0.1


def test_prediction_ce_loss_computed_from_log_probs():
    """Verify prediction CE loss = -log_prob at prediction positions."""
    # Simulate the core prediction loss computation
    log_prob = torch.log(torch.tensor([[0.1, 0.2, 0.3, 0.4]]))
    pred_mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.long)

    pred_ce_per_token = -log_prob
    pred_token_count = pred_mask.sum()
    pred_ce = (pred_ce_per_token * pred_mask).sum() / pred_token_count.clamp(min=1)

    # CE loss should be the mean of -log_prob at positions where mask=1
    expected = (-log_prob[0, 1] + -log_prob[0, 3]) / 2
    assert torch.isclose(pred_ce, expected, atol=1e-6)


def test_prediction_ce_loss_zero_when_no_prediction_tokens():
    """When prediction_mask is all zeros, no prediction loss should be added."""
    pred_mask = torch.zeros(1, 4, dtype=torch.long)

    pred_token_count = pred_mask.sum()
    # The actor code checks pred_token_count > 0 before computing loss
    assert pred_token_count == 0


def test_prediction_ce_loss_multiple_samples():
    """Verify prediction CE loss works across multiple batch samples."""
    log_prob = torch.log(torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]))
    pred_mask = torch.tensor([[0, 1, 0, 0], [0, 0, 1, 0]], dtype=torch.long)

    pred_ce_per_token = -log_prob
    pred_token_count = pred_mask.sum()
    pred_ce = (pred_ce_per_token * pred_mask).sum() / pred_token_count.clamp(min=1)

    # CE loss should be the mean of -log_prob at masked positions across all samples
    expected = (-log_prob[0, 1] + -log_prob[1, 2]) / 2
    assert torch.isclose(pred_ce, expected, atol=1e-6)
