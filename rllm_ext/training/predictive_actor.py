"""Extended Actor module with prediction auxiliary loss."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

# Import base actor from verl
from verl import DataProto
from verl.workers.actor.dp_actor import DataParallelPPOActor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


__all__ = ["PredictiveActor"]


class PredictiveActor(DataParallelPPOActor):
    """
    Extended PPO Actor that adds prediction auxiliary loss.

    Prediction tokens are identified by a ``prediction_mask`` in the batch.
    CE loss is computed directly from the log_probs of the PPO forward pass
    (no separate forward pass needed).

    The prediction_mask is produced by PredictiveAgentWorkflowEngine and marks
    tokens that correspond to prediction assistant turns (``rllm_prediction: True``).
    """

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        super().__init__(config, actor_module, actor_optimizer)

        # Prediction loss configuration
        self.prediction_loss_weight = config.get("prediction_loss_weight", 0.1)

        if torch.distributed.get_rank() == 0:
            logger.info("PredictiveActor initialized with:")
            logger.info(f"  prediction_loss_weight: {self.prediction_loss_weight}")

    def update_policy(self, data: DataProto) -> dict:
        """
        Override update_policy to add prediction auxiliary loss.

        Prediction CE loss = -log_prob at prediction positions, computed from
        the same forward pass used for PPO.
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
        non_tensor_select_keys = (
            ["multi_modal_inputs"] if has_multi_modal_inputs else []
        )

        # Also select prediction_mask if available
        if "prediction_mask" in data.batch.keys():
            select_keys.append("prediction_mask")

        data = data.select(
            batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys
        )

        # Split to make minibatch iterator for updating the actor
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        metrics = {}

        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    from verl.utils.seqlen_balancing import prepare_dynamic_batch

                    max_token_len = (
                        self.config.ppo_max_token_len_per_gpu
                        * self.ulysses_sequence_parallel_size
                    )
                    micro_batches, _ = prepare_dynamic_batch(
                        mini_batch, max_token_len=max_token_len
                    )
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size
                        // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(
                        self.config.ppo_micro_batch_size_per_gpu
                    )

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = (
                        self.config.clip_ratio_low
                        if self.config.clip_ratio_low is not None
                        else clip_ratio
                    )
                    clip_ratio_high = (
                        self.config.clip_ratio_high
                        if self.config.clip_ratio_high is not None
                        else clip_ratio
                    )
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                    )

                    # Compute PPO policy loss
                    from verl.trainer.ppo.core_algos import (
                        compute_policy_loss,
                        agg_loss,
                    )

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

                    if self.config.policy_loss.loss_mode == "vanilla":
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = (
                            compute_policy_loss(
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
                        )
                    else:
                        from verl.trainer.ppo.core_algos import get_policy_loss_fn

                        policy_loss_fn = get_policy_loss_fn(loss_mode)
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = (
                            policy_loss_fn(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                advantages=advantages,
                                response_mask=response_mask,
                                loss_agg_mode=loss_agg_mode,
                                config=self.config,
                            )
                        )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(
                            loss_mat=entropy,
                            loss_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                        )
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        from verl.trainer.ppo.core_algos import kl_penalty

                        kld = kl_penalty(
                            logprob=log_prob,
                            ref_logprob=ref_log_prob,
                            kl_penalty=self.config.kl_loss_type,
                        )
                        kl_loss = agg_loss(
                            loss_mat=kld,
                            loss_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                        )

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item()
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    # === Prediction auxiliary loss (reuses PPO forward pass) ===
                    if (
                        "prediction_mask" in model_inputs
                        and self.prediction_loss_weight > 0
                    ):
                        pred_mask = model_inputs[
                            "prediction_mask"
                        ]  # (bsz, response_length)
                        pred_token_count = pred_mask.sum()
                        if pred_token_count > 0:
                            # CE loss = -log_prob at prediction positions
                            pred_ce_per_token = -log_prob  # (bsz, response_length)
                            pred_ce = (
                                pred_ce_per_token * pred_mask
                            ).sum() / pred_token_count.clamp(min=1)
                            policy_loss = (
                                policy_loss + pred_ce * self.prediction_loss_weight
                            )
                            micro_batch_metrics["train/prediction_loss"] = (
                                pred_ce.detach().item()
                            )

                    if self.config.use_dynamic_bsz:
                        loss = policy_loss * (
                            response_mask.shape[0] / self.config.ppo_mini_batch_size
                        )
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


# Register a factory function to create PredictiveActor
def create_predictive_actor(
    config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None
):
    """Factory function to create PredictiveActor instance."""
    return PredictiveActor(config, actor_module, actor_optimizer)
