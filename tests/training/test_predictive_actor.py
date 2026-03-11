import torch
import torch.nn as nn

from rllm_ext.training.predictive_actor import PredictiveActor


class _DummyTokenizer:
    name_or_path = "dummy-tokenizer"
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False):
        parts = []
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            parts.append(f"<{role}>{content}</{role}>")

        if add_generation_prompt:
            parts.append("<assistant>")

        rendered = "".join(parts)
        if tokenize:
            return self.encode(rendered, add_special_tokens=False)
        return rendered

    def encode(self, text, add_special_tokens=False):
        return [3 + (ord(ch) % 17) for ch in text]

    def __call__(self, texts, add_special_tokens=False, truncation=False, max_length=None):
        if isinstance(texts, str):
            texts = [texts]

        token_ids = []
        for text in texts:
            ids = [3 + (ord(ch) % 17) for ch in text]
            if truncation and max_length is not None:
                ids = ids[:max_length]
            token_ids.append(ids)

        return {"input_ids": token_ids}


class _DummyCausalLM(nn.Module):
    def __init__(self, vocab_size: int = 64, hidden_size: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        hidden = self.embed(input_ids)
        logits = self.proj(hidden)
        return type("DummyOutput", (), {"logits": logits})


def _build_predictive_actor_for_test() -> PredictiveActor:
    actor = PredictiveActor.__new__(PredictiveActor)
    actor.actor_module = _DummyCausalLM()
    actor.tokenizer = _DummyTokenizer()
    actor.config = {"prediction_loss_max_length": 64}
    actor.device_name = "cpu"
    actor.prediction_temperature = 1.0
    actor.prediction_loss_type = "cross_entropy"
    return actor


def test_prediction_loss_microbatch_computes_trainable_loss():
    actor = _build_predictive_actor_for_test()

    prediction_targets = [
        {
            "examples": [
                {
                    "prompt_messages": [
                        {"role": "user", "content": "solve it"},
                        {"role": "assistant", "content": "<tool_call>calculator</tool_call>"},
                        {"role": "user", "content": "predict the tool output"},
                    ],
                    "target_text": "<prediction>result=10</prediction>",
                }
            ]
        },
        {
            "examples": [
                {
                    "prompt_messages": [
                        {"role": "user", "content": "check status"},
                        {"role": "assistant", "content": "<tool_call>status</tool_call>"},
                        {"role": "user", "content": "predict the tool output"},
                    ],
                    "target_text": "<prediction>status: ok</prediction>",
                }
            ]
        },
    ]
    response_mask = torch.ones((2, 4), dtype=torch.long)

    loss = actor._compute_prediction_loss_microbatch(prediction_targets, response_mask)

    assert loss is not None
    assert loss.requires_grad
    assert torch.isfinite(loss).item()

    loss.backward()
    grad_norm = actor.actor_module.embed.weight.grad.abs().sum().item()
    assert grad_norm > 0


def test_prediction_loss_microbatch_returns_none_for_invalid_targets():
    actor = _build_predictive_actor_for_test()

    prediction_targets = [
        {"examples": []},
        {"examples": [{"prompt_messages": [], "target_text": "   "}]},
    ]
    response_mask = torch.ones((2, 4), dtype=torch.long)

    loss = actor._compute_prediction_loss_microbatch(prediction_targets, response_mask)

    assert loss is None
