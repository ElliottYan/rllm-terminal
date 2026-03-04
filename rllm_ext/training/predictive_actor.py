"""Extended Actor module with prediction auxiliary loss."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import base actor from verl
from verl import DataProto
from verl.workers.actor.dp_actor import DataParallelPPOActor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


__all__ = ["PredictiveActor"]


class PredictiveActor(DataParallelPPOActor):
    """
    Extended PPO Actor that adds prediction auxiliary loss.

    This actor extends the base DataParallelPPOActor to:
    1. Extract prediction targets from batch non_tensor_batch
    2. Compute prediction loss (cross-entropy between prediction and actual output)
    3. Add prediction loss to PPO policy loss

    The prediction loss encourages the model to predict tool outputs accurately.
    """

    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        super().__init__(config, actor_module, actor_optimizer)

        # Prediction loss configuration
        self.prediction_loss_weight = config.get("prediction_loss_weight", 0.1)
        self.prediction_loss_type = config.get("prediction_loss_type", "cross_entropy")
        self.prediction_temperature = config.get("prediction_temperature", 1.0)

        if torch.distributed.get_rank() == 0:
            logger.info(f"PredictiveActor initialized with:")
            logger.info(f"  prediction_loss_weight: {self.prediction_loss_weight}")
            logger.info(f"  prediction_loss_type: {self.prediction_loss_type}")

    def update_policy(self, data: DataProto) -> dict:
        """
        Override update_policy to add prediction auxiliary loss.

        This is a modified version of parent's update_policy that adds
        prediction loss to the total loss before backward pass.
        """
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        # Also select prediction_targets if available
        if "prediction_targets" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("prediction_targets")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        metrics = {}

        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    import verl.utils.torch_functional as verl_F
                    from verl.utils.seqlen_balancing import prepare_dynamic_batch

                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = (
                        self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    )
                    clip_ratio_high = (
                        self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    )
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    # Compute PPO policy loss
                    from verl.trainer.ppo.core_algos import compute_policy_loss, agg_loss

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

                    if self.config.policy_loss.loss_mode == "vanilla":
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            loss_agg_mode=loss_agg_mode,
                        )
                    else:
                        from verl.trainer.ppo.core_algos import get_policy_loss_fn

                        policy_loss_fn = get_policy_loss_fn(loss_mode)
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                        )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        from verl.trainer.ppo.core_algos import kl_penalty

                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item()
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    # === NEW: Add prediction auxiliary loss ===
                    if "prediction_targets" in model_inputs and self.prediction_loss_weight > 0:
                        prediction_loss = self._compute_prediction_loss_microbatch(
                            model_inputs["prediction_targets"], response_mask
                        )
                        if prediction_loss is not None:
                            # Add to total policy loss
                            policy_loss = policy_loss + prediction_loss * self.prediction_loss_weight
                            micro_batch_metrics["train/prediction_loss"] = prediction_loss.detach().item()

                    if self.config.use_dynamic_bsz:
                        loss = policy_loss * (response_mask.shape[0] / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation

                    loss.backward()

                    micro_batch_metrics.update(
                        {
                            "actor/pg_loss": pg_loss.detach().item(),
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        }
                    )

                    from verl.utils.py_functional import append_to_dict

                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}

                from verl.utils.py_functional import append_to_dict

                append_to_dict(metrics, mini_batch_metrics)

        self.actor_optimizer.zero_grad()
        return metrics

    def _compute_prediction_loss_microbatch(
        self, prediction_targets, response_mask
    ) -> torch.Tensor | None:
        """
        Compute prediction loss for a micro_batch.

        Args:
            prediction_targets: Array of prediction metadata dicts for this micro_batch
            response_mask: Response mask to match batch size

        Returns:
            Scalar prediction loss tensor or None if no valid predictions
        """
        device = response_mask.device

        # Filter steps that have predictions
        valid_indices = []
        pred_texts = []
        actual_texts = []

        for i, pred_target in enumerate(prediction_targets):
            if pred_target.get("has_prediction", False):
                pred_text = pred_target.get("prediction", "").strip()
                actual_text = pred_target.get("actual", "").strip()
                if pred_text and actual_text:
                    valid_indices.append(i)
                    pred_texts.append(pred_text)
                    actual_texts.append(actual_text)

        if not valid_indices:
            return None

        # For now, return a placeholder loss
        # TODO: Implement actual loss computation using tokenizer
        #
        # The full implementation would:
        # 1. Tokenize pred_texts and actual_texts
        # 2. Get model logits for pred_texts
        # 3. Compute cross-entropy against actual_texts
        # 4. Aggregate over valid samples
        #
        # For placeholder purposes, return a small loss that requires grad

        # Placeholder: return zero loss with grad
        return torch.tensor(0.0, device=device, requires_grad=True)

    def _compute_cross_entropy_prediction_loss(
        self,
        prediction_texts: list[str],
        actual_texts: list[str],
        tokenizer,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss between predictions and actual outputs.

        Tokenizes both texts and computes cross-entropy loss.

        Args:
            prediction_texts: List of predicted texts
            actual_texts: List of actual (ground truth) texts
            tokenizer: Tokenizer to use

        Returns:
            Scalar cross-entropy loss
        """
        # Tokenize predictions (as input to model)
        pred_tokens = tokenizer(
            prediction_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        # Tokenize actual outputs (as targets)
        actual_tokens = tokenizer(
            actual_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        # Move to device
        device = next(self.actor_module.parameters()).device
        pred_input_ids = pred_tokens["input_ids"].to(device)
        pred_attention_mask = pred_tokens["attention_mask"].to(device)
        actual_input_ids = actual_tokens["input_ids"].to(device)

        # Forward pass on prediction text
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = self.actor_module(
                input_ids=pred_input_ids,
                attention_mask=pred_attention_mask,
                use_cache=False,
            )
            logits = outputs.logits  # (batch, seq_len, vocab_size)

        # Compute cross-entropy loss
        # Shift logits and labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = actual_input_ids[..., 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=tokenizer.pad_token_id,
            reduction="mean",
        )

        return loss


# Register a factory function to create PredictiveActor
def create_predictive_actor(config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
    """Factory function to create PredictiveActor instance."""
    return PredictiveActor(config, actor_module, actor_optimizer)
