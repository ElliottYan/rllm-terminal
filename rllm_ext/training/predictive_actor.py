"""Extended Actor module with prediction auxiliary loss."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from verl import DataProto
    from verl.workers.actor.dp_actor import DataParallelPPOActor
except ImportError:
    DataProto = Any

    class DataParallelPPOActor:  # pragma: no cover - test fallback only
        pass

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


__all__ = ["PredictiveActor"]


class PredictiveActor(DataParallelPPOActor):
    """
    Extended PPO Actor that adds prediction auxiliary loss.

    Prediction loss can be computed in two ways:
    1. From ``prediction_mask`` when prediction turns are present in the sampled transcript.
    2. From ``prediction_targets`` when prediction turns were intentionally omitted
       from the live transcript and must be reconstructed from step metadata.
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
        self.prediction_loss_type = config.get("prediction_loss_type", "cross_entropy")
        self.prediction_temperature = config.get("prediction_temperature", 1.0)
        self.prediction_loss_forward_batch_size = max(
            int(config.get("prediction_loss_forward_batch_size", 8)),
            1,
        )

        if torch.distributed.get_rank() == 0:
            logger.info("PredictiveActor initialized with:")
            logger.info(f"  prediction_loss_weight: {self.prediction_loss_weight}")
            logger.info(f"  prediction_loss_type: {self.prediction_loss_type}")
            logger.info(
                "  prediction_loss_forward_batch_size: %s",
                self.prediction_loss_forward_batch_size,
            )

    def update_policy(self, data: DataProto) -> dict:
        """
        Override update_policy to add prediction auxiliary loss.

        Prediction CE loss is taken from the PPO forward pass when prediction
        turns are present in the transcript, and falls back to a separate
        supervised forward pass when prediction turns were intentionally kept
        out of the rollout context.
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
        if "prediction_targets" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("prediction_targets")

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

                    prediction_loss = None
                    if self.prediction_loss_weight > 0:
                        pred_mask = model_inputs.get("prediction_mask")
                        if pred_mask is not None:
                            pred_token_count = pred_mask.sum()
                            if pred_token_count > 0:
                                pred_ce_per_token = -log_prob
                                prediction_loss = (
                                    pred_ce_per_token * pred_mask
                                ).sum() / pred_token_count.clamp(min=1)

                        if (
                            prediction_loss is None
                            and "prediction_targets" in model_inputs
                        ):
                            prediction_loss = self._compute_prediction_loss_microbatch(
                                model_inputs["prediction_targets"]
                            )

                        if prediction_loss is not None:
                            policy_loss = (
                                policy_loss
                                + prediction_loss * self.prediction_loss_weight
                            )
                            micro_batch_metrics["train/prediction_loss"] = (
                                prediction_loss.detach().item()
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

    def _compute_prediction_loss_microbatch(
        self, prediction_targets
    ) -> torch.Tensor | None:
        """
        Compute prediction loss for a micro-batch.

        Uses separate supervised examples only when prediction turns were omitted
        from the PPO transcript and therefore cannot be trained from
        ``prediction_mask`` on the main rollout forward pass.
        """
        prediction_examples = []

        for pred_target in prediction_targets:
            if not isinstance(pred_target, dict):
                continue

            examples = pred_target.get("examples", [])
            if not isinstance(examples, list):
                continue

            for example in examples:
                if not isinstance(example, dict):
                    continue

                prompt_messages = example.get("prompt_messages")
                target_text = example.get("target_text", "")

                if not isinstance(prompt_messages, list):
                    continue

                target_text = str(target_text).strip()
                if not target_text:
                    continue

                prediction_examples.append(
                    {
                        "prompt_messages": prompt_messages,
                        "target_text": target_text,
                    }
                )

        if not prediction_examples:
            return None

        if self.prediction_loss_type != "cross_entropy":
            if not getattr(self, "_warned_unsupported_prediction_loss", False):
                logger.warning(
                    "Unsupported prediction_loss_type=%s, skip prediction loss (supported: cross_entropy).",
                    self.prediction_loss_type,
                )
                self._warned_unsupported_prediction_loss = True
            return None

        tokenizer = self._resolve_prediction_tokenizer()
        if tokenizer is None:
            if not getattr(self, "_warned_missing_prediction_tokenizer", False):
                logger.warning(
                    "Prediction loss skipped because tokenizer is unavailable on PredictiveActor/actor_module."
                )
                self._warned_missing_prediction_tokenizer = True
            return None

        return self._compute_cross_entropy_prediction_loss(
            prediction_examples=prediction_examples,
            tokenizer=tokenizer,
        )

    def _resolve_prediction_tokenizer(self) -> Any | None:
        """Resolve tokenizer from actor first, then actor module wrappers."""
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is not None:
            return tokenizer

        actor_module = getattr(self, "actor_module", None)
        if actor_module is None:
            return None

        tokenizer = getattr(actor_module, "tokenizer", None)
        if tokenizer is not None:
            return tokenizer

        wrapped = getattr(actor_module, "module", None)
        if wrapped is None:
            return None
        return getattr(wrapped, "tokenizer", None)

    def _resolve_prediction_chat_parser(self, tokenizer):
        from rllm.parser.chat_template_parser import ChatTemplateParser

        parser = getattr(self, "_prediction_chat_parser", None)
        if parser is not None and getattr(parser, "tokenizer", None) is tokenizer:
            return parser

        try:
            parser = ChatTemplateParser.get_parser(tokenizer)
        except Exception:
            parser = ChatTemplateParser(tokenizer)

        self._prediction_chat_parser = parser
        return parser

    def _get_distributed_max_example_count(self, local_count: int) -> int:
        """
        Synchronize the auxiliary-example count across ranks.

        Some ranks can legitimately have zero prediction targets in a PPO
        micro-batch while others still have several. We must still run the same
        number of auxiliary forwards/backwards on every rank, otherwise FSDP/NCCL
        can hang because collectives no longer line up.
        """
        if (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
        ):
            return int(local_count)

        device = next(self.actor_module.parameters()).device
        count_tensor = torch.tensor(
            [int(local_count)],
            dtype=torch.long,
            device=device,
        )
        torch.distributed.all_reduce(
            count_tensor, op=torch.distributed.ReduceOp.MAX
        )
        return int(count_tensor.item())

    def _compute_cross_entropy_prediction_loss(
        self,
        prediction_examples: list[dict[str, Any]],
        tokenizer,
    ) -> torch.Tensor | None:
        """
        Compute CE loss for predicting actual output from prediction-step chat context.
        """
        max_total_len = int(self.config.get("prediction_loss_max_length", 512))
        max_total_len = max(max_total_len, 8)
        chat_parser = self._resolve_prediction_chat_parser(tokenizer)

        input_id_lists: list[list[int]] = []
        label_id_lists: list[list[int]] = []

        for example in prediction_examples:
            prompt_messages = example.get("prompt_messages")
            target_text = example.get("target_text", "")
            if not isinstance(prompt_messages, list):
                continue

            target_message = {"role": "assistant", "content": target_text}

            try:
                prompt_ids, response_ids, _ = chat_parser.tokenize_and_mask(
                    prompt_messages + [target_message]
                )
            except Exception:
                continue

            prompt_ids = prompt_ids.tolist()
            response_ids = response_ids.tolist()

            if not response_ids:
                continue

            if len(response_ids) >= max_total_len:
                prompt_ids = []
                response_ids = response_ids[:max_total_len]
            else:
                max_prompt_len = max_total_len - len(response_ids)
                if len(prompt_ids) > max_prompt_len:
                    prompt_ids = prompt_ids[-max_prompt_len:]

            input_ids = prompt_ids + response_ids
            label_ids = [-100] * len(prompt_ids) + response_ids

            if not input_ids or not any(label != -100 for label in label_ids):
                continue

            input_id_lists.append(input_ids)
            label_id_lists.append(label_ids)

        if not input_id_lists:
            local_example_count = 0
        else:
            local_example_count = len(input_id_lists)

        target_example_count = self._get_distributed_max_example_count(
            local_example_count
        )
        if target_example_count <= 0:
            return None

        pad_token_id = tokenizer.pad_token_id
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            pad_token_id = eos_token_id if eos_token_id is not None else 0

        if local_example_count < target_example_count:
            pad_count = target_example_count - local_example_count
            # Pad with dummy rows so every rank executes the same number of
            # auxiliary forward/backward graphs. Dummy rows carry no supervised
            # labels, so they contribute zero gradient.
            input_id_lists.extend([[pad_token_id]] * pad_count)
            label_id_lists.extend([[-100]] * pad_count)

        device = next(self.actor_module.parameters()).device
        total_loss_sum = None
        total_token_count = 0
        chunk_size = min(
            self.prediction_loss_forward_batch_size,
            target_example_count,
        )

        # Prediction targets can explode when a whole multi-step trajectory is
        # reconstructed into one supervised micro-batch. Run them in smaller
        # chunks so actor update does not create one giant auxiliary forward.
        for chunk_start in range(0, len(input_id_lists), chunk_size):
            chunk_inputs = input_id_lists[chunk_start : chunk_start + chunk_size]
            chunk_labels = label_id_lists[chunk_start : chunk_start + chunk_size]
            max_seq_len = max(len(ids) for ids in chunk_inputs)

            padded_inputs = []
            padded_labels = []
            padded_attn_mask = []
            for input_ids, label_ids in zip(chunk_inputs, chunk_labels):
                pad_len = max_seq_len - len(input_ids)
                padded_inputs.append(input_ids + [pad_token_id] * pad_len)
                padded_labels.append(label_ids + [-100] * pad_len)
                padded_attn_mask.append([1] * len(input_ids) + [0] * pad_len)

            input_ids = torch.tensor(padded_inputs, dtype=torch.long, device=device)
            attention_mask = torch.tensor(
                padded_attn_mask, dtype=torch.long, device=device
            )
            labels = torch.tensor(padded_labels, dtype=torch.long, device=device)

            with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
                outputs = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                logits = outputs.logits
                if self.prediction_temperature != 1.0:
                    logits = logits / self.prediction_temperature

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            valid_token_count = int((shift_labels != -100).sum().item())
            if valid_token_count == 0:
                zero_loss = shift_logits.sum() * 0
                total_loss_sum = (
                    zero_loss
                    if total_loss_sum is None
                    else total_loss_sum + zero_loss
                )
                continue

            loss_sum = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            total_loss_sum = (
                loss_sum if total_loss_sum is None else total_loss_sum + loss_sum
            )
            total_token_count += valid_token_count

        if total_loss_sum is None:
            return None

        return total_loss_sum / max(total_token_count, 1)


# Register a factory function to create PredictiveActor
def create_predictive_actor(
    config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None
):
    """Factory function to create PredictiveActor instance."""
    return PredictiveActor(config, actor_module, actor_optimizer)
