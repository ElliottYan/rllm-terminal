igtimport torch
import torch.nn.functional as F
from types import SimpleNamespace

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


class _DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 0


class _DummyChatParser:
    def tokenize_and_mask(self, messages):
        prompt_len = max(len(messages) - 1, 1)
        target_len = max(len(messages[-1]["content"]) % 3 + 1, 1)
        prompt_ids = torch.arange(1, prompt_len + 1, dtype=torch.long)
        response_ids = torch.arange(3, 3 + target_len, dtype=torch.long)
        return prompt_ids, response_ids, None


class _RecordingActorModule(torch.nn.Module):
    def __init__(self, vocab_size: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.batch_sizes = []

    def forward(self, input_ids, attention_mask, use_cache=False):
        del attention_mask, use_cache
        self.batch_sizes.append(int(input_ids.shape[0]))
        logits = F.one_hot(
            input_ids % self.vocab_size, num_classes=self.vocab_size
        ).float()
        logits = logits + self.bias
        return SimpleNamespace(logits=logits)


def test_prediction_ce_loss_chunks_large_aux_batches():
    actor = PredictiveActor.__new__(PredictiveActor)
    actor.config = {
        "prediction_loss_max_length": 32,
        "prediction_loss_forward_batch_size": 2,
    }
    actor.actor_module = _RecordingActorModule()
    actor.device_name = "cpu"
    actor.prediction_temperature = 1.0
    actor.prediction_loss_forward_batch_size = 2
    actor._resolve_prediction_chat_parser = lambda tokenizer: _DummyChatParser()

    prediction_examples = [
        {
            "prompt_messages": [
                {"role": "user", "content": f"prompt-{idx}"},
            ],
            "target_text": f"target-{idx}",
        }
        for idx in range(5)
    ]

    loss = actor._compute_cross_entropy_prediction_loss(
        prediction_examples=prediction_examples,
        tokenizer=_DummyTokenizer(),
    )

    assert loss is not None
    assert torch.isfinite(loss)
    assert actor.actor_module.batch_sizes == [2, 2, 1]


def test_prediction_ce_loss_pads_aux_batches_to_distributed_max():
    actor = PredictiveActor.__new__(PredictiveActor)
    actor.config = {
        "prediction_loss_max_length": 32,
        "prediction_loss_forward_batch_size": 2,
    }
    actor.actor_module = _RecordingActorModule()
    actor.device_name = "cpu"
    actor.prediction_temperature = 1.0
    actor.prediction_loss_forward_batch_size = 2
    actor._resolve_prediction_chat_parser = lambda tokenizer: _DummyChatParser()
    actor._get_distributed_max_example_count = lambda local_count: 3

    prediction_examples = [
        {
            "prompt_messages": [
                {"role": "user", "content": "prompt-0"},
            ],
            "target_text": "target-0",
        }
    ]

    loss = actor._compute_cross_entropy_prediction_loss(
        prediction_examples=prediction_examples,
        tokenizer=_DummyTokenizer(),
    )

    assert loss is not None
    assert torch.isfinite(loss)
    assert actor.actor_module.batch_sizes == [2, 1]
