# Predictive Tool Sweep

This directory contains a reproducible sweep setup based on `exp_scripts/base.sh`.

## Files

- `base.env`: shared sweep defaults (resource settings, runtime flags).
- `matrix.tsv`: per-run sweep grid.
- `run_sweep.sh`: sequential sweep launcher.

## Usage

Run all rows in `matrix.tsv`:

```bash
bash exp_scripts/sweeps/predictive_tool/run_sweep.sh
```

Dry run (print commands only):

```bash
DRY_RUN=1 bash exp_scripts/sweeps/predictive_tool/run_sweep.sh
```

Run one specific row:

```bash
ONLY_RUN_ID=p11_pred_sim bash exp_scripts/sweeps/predictive_tool/run_sweep.sh
```

Resume from row index (1-based, excluding header/comments):

```bash
START_FROM_INDEX=4 bash exp_scripts/sweeps/predictive_tool/run_sweep.sh
```

## Matrix Column Notes

- `run_id`: unique run identifier.
- `enable_prediction`: maps to `enable_prediction=...`.
- `enable_similarity_reward`: maps to `enable_similarity_reward=...`.
- `add_prediction_to_messages`: maps to `prediction_cfg.add_prediction_to_messages=...`.
- `simple_tir`: maps to `prediction_cfg.simple_tir=...`.
- `prediction_max_tokens`: maps to `prediction_cfg.max_tokens=...`.
- `prediction_loss_weight`: maps to `+actor_rollout_ref.actor.prediction_loss_weight=...`.
- `total_epochs`: maps to `trainer.total_epochs=...`.
- `max_steps`: maps to `rllm.agent.max_steps=...`.
- `extra_overrides`: optional `;` separated Hydra overrides, `-` means none.
