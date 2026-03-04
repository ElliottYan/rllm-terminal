# Prediction Auxiliary Loss - Usage Guide

This guide explains how to use the prediction auxiliary loss feature in rLLM training.

## Overview

The prediction auxiliary loss feature adds a direct loss signal for tool output prediction accuracy, complementing the reward-based approach.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Pipeline                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. Workflow generates trajectories                         │
│     └─ PredictiveToolWorkflow collects prediction data      │
│                                                               │
│  2. AgentWorkflowEngine transforms to DataProto             │
│     └─ PredictiveAgentWorkflowEngine adds prediction_targets│
│                                                               │
│  3. Trainer orchestrates PPO training                       │
│     └─ PredictiveAgentWorkflowTrainer uses custom actor     │
│                                                               │
│  4. Actor computes losses                                   │
│     └─ PredictiveActor adds prediction loss to PPO loss     │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Installation

The extension is part of `rllm_ext`. All components are isolated from core `rllm`.

```python
from rllm_ext.training import PredictiveAgentWorkflowTrainer
from rllm_ext.workflows import PredictiveToolWorkflow
```

## Configuration

Add prediction loss configuration to your Hydra config:

```yaml
# config.yaml
rllm:
  prediction_loss:
    enabled: true          # Enable prediction auxiliary loss
    weight: 0.1            # Loss weight (default: 0.1)
    loss_type: cross_entropy  # Loss type (currently placeholder)
    temperature: 1.0       # Temperature for loss computation

  workflow:
    n_parallel_tasks: 128
    retry_limit: 3
```

## Usage Example

### Basic Usage

```python
from rllm.data import DatasetRegistry
from rllm.rewards import math_reward_fn
from rllm_ext.training import PredictiveAgentWorkflowTrainer
from rllm_ext.workflows import PredictiveToolWorkflow, PredictionConfig

# Load datasets
train_dataset = DatasetRegistry.load_dataset("deepscaler_math", "train")
test_dataset = DatasetRegistry.load_dataset("aime2024", "test")

# Configure workflow with prediction enabled
workflow_args = {
    "agent_cls": "PredictiveToolAgent",  # or import and use class directly
    "env_cls": "PredictiveToolEnvironment",
    "agent_args": {
        "tools": ["python"],
        "parser_name": "qwen",
    },
    "env_args": {
        "tools": ["python"],
        "reward_fn": math_reward_fn,
    },
    "max_steps": 10,
    "prediction_cfg": {
        "enabled": True,
        "max_tokens": 256,
        "add_prediction_to_messages": True,
    },
}

# Create trainer with prediction loss support
trainer = PredictiveAgentWorkflowTrainer(
    workflow_class=PredictiveToolWorkflow,
    workflow_args=workflow_args,
    config=config,  # Your hydra config
    train_dataset=train_dataset,
    val_dataset=test_dataset,
)

# Train
trainer.train()
```

### Advanced: Custom Loss Implementation

The current implementation includes a placeholder for the actual loss computation.
To implement the full prediction loss:

1. **Edit** `rllm_ext/training/predictive_actor.py`:
   ```python
   def _compute_prediction_loss_microbatch(self, prediction_targets, response_mask):
       # 1. Tokenize prediction_texts and actual_texts
       pred_texts = [t["prediction"] for t in prediction_targets if t["has_prediction"]]
       actual_texts = [t["actual"] for t in prediction_targets if t["has_prediction"]]

       # 2. Get model logits for predictions
       # ... tokenize and forward pass ...

       # 3. Compute cross-entropy loss
       # loss = F.cross_entropy(...)

       # 4. Return aggregated loss
       return loss
   ```

2. **Alternative loss types**:
   - Cross-entropy (next-token prediction)
   - MSE (embedding similarity)
   - Contrastive loss
   - BLEU-style similarity

## Data Flow

### 1. Workflow Step (During Rollout)

```python
# In PredictiveToolWorkflow.run():
step.info["rllm_ext.prediction_text"] = prediction_text
step.info["rllm_ext.prediction_raw_text"] = prediction_raw_text
step.info["rllm_ext.prediction_reasoning"] = prediction_reasoning
```

### 2. Engine Transformation (Rollout → Training)

```python
# In PredictiveAgentWorkflowEngine.transform_results_for_verl():
prediction_targets = [
    {
        "prediction": "The answer is 42",
        "actual": "The result is 42",
        "prediction_raw_text": "full model output...",
        "prediction_reasoning": "step-by-step reasoning...",
        "has_prediction": True,
    },
    # ... more steps
]

# Added to DataProto.non_tensor_batch["prediction_targets"]
```

### 3. Actor Loss Computation (During Training)

```python
# In PredictiveActor.update_policy():
policy_loss = pg_loss - entropy_loss * entropy_coeff

# Add prediction auxiliary loss
prediction_loss = self._compute_prediction_loss_microbatch(...)
policy_loss = policy_loss + prediction_loss * prediction_loss_weight

loss.backward()
```

## Metrics

The following metrics are logged during training:

| Metric | Description |
|--------|-------------|
| `train/prediction_loss` | Raw prediction loss (before weighting) |
| `train/prediction_loss_weight` | Configured loss weight |
| `train/total_prediction_loss` | Weighted prediction loss |
| `actor/pg_loss` | PPO policy gradient loss |
| `actor/entropy_loss` | Entropy regularization loss |

## Troubleshooting

### Loss is always 0

- Check that `prediction_loss.enabled: true` in config
- Verify that steps have predictions (check `has_prediction` in data)
- Ensure `prediction_loss_weight > 0`

### Prediction data not found

- Verify `PredictiveToolWorkflow` is used (not base `Workflow`)
- Check `prediction_cfg.enabled: true` in workflow args
- Ensure `PredictiveAgentWorkflowEngine` is used in trainer

### Training slower than expected

- Prediction loss computation adds overhead
- Consider `prediction_loss_weight: 0.05` or lower
- Enable only for fine-tuning, not pre-training

## Implementation Notes

### Design Decisions

1. **Isolation**: All changes in `rllm_ext/`, no core `rllm` modifications
2. **Optional**: Feature can be disabled via config
3. **Extensible**: Easy to add new loss types

### Future Improvements

- [ ] Implement actual cross-entropy loss with tokenizer
- [ ] Add configurable loss types (MSE, contrastive, etc.)
- [ ] Optimize batch processing for prediction loss
- [ ] Add unit tests for prediction loss computation

## References

- Original workflow: `rllm/workflows/simple_workflow.py`
- PPO trainer: `rllm/trainer/verl/agent_workflow_trainer.py`
- Actor module: `verl/verl/workers/actor/dp_actor.py`
