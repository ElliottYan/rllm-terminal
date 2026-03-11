from __future__ import annotations

from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker


class PredictiveActorRolloutRefWorker(ActorRolloutRefWorker):
    """FSDP worker that swaps in ``PredictiveActor`` on the remote worker."""

    def _maybe_inject_predictive_actor(self):
        if not getattr(self, "_is_actor", False):
            return

        prediction_loss_weight = self.config.actor.get("prediction_loss_weight", 0)
        if prediction_loss_weight <= 0:
            return

        from rllm_ext.training.predictive_actor import PredictiveActor

        predictive_actor = PredictiveActor(
            config=self.config.actor,
            actor_module=self.actor_module_fsdp,
            actor_optimizer=self.actor_optimizer,
        )
        if hasattr(self, "tokenizer"):
            predictive_actor.tokenizer = self.tokenizer
        self.actor = predictive_actor

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        self._maybe_inject_predictive_actor()


class AsyncPredictiveActorRolloutRefWorker(AsyncActorRolloutRefWorker):
    """Async FSDP worker that swaps in ``PredictiveActor`` on the remote worker."""

    def _maybe_inject_predictive_actor(self):
        if not getattr(self, "_is_actor", False):
            return

        prediction_loss_weight = self.config.actor.get("prediction_loss_weight", 0)
        if prediction_loss_weight <= 0:
            return

        from rllm_ext.training.predictive_actor import PredictiveActor

        predictive_actor = PredictiveActor(
            config=self.config.actor,
            actor_module=self.actor_module_fsdp,
            actor_optimizer=self.actor_optimizer,
        )
        if hasattr(self, "tokenizer"):
            predictive_actor.tokenizer = self.tokenizer
        self.actor = predictive_actor

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        self._maybe_inject_predictive_actor()
