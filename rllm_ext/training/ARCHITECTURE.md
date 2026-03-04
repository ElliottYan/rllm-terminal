# Prediction Auxiliary Loss - Architecture & Connection

## 核心问题：如何确保使用自定义的 Engine 和 Trainer？

### 连接流程

```
用户代码
    │
    ├─ 使用 PredictiveAgentWorkflowTrainer (不是 AgentWorkflowPPOTrainer)
    │
    ↓
PredictiveAgentWorkflowTrainer.init_workers()
    │
    ├─ [Step 1] _configure_prediction_loss()
    │   └─ 添加 prediction_loss_weight 到 config.actor
    │
    ├─ [Step 2] super().init_workers()
    │   └─ AgentWorkflowPPOTrainer.init_workers()
    │       ├─ super().init_workers()  # RayPPOTrainer
    │       │   └─ 创建 actor_rollout_wg (使用 DataParallelPPOActor)
    │       │
    │       └─ 创建 AgentWorkflowEngine  # 会被替换
    │
    ├─ [Step 3] _replace_workflow_engine()
    │   ├─ 创建 PredictiveAgentWorkflowEngine
    │   ├─ 初始化 worker pool
    │   └─ 替换 self.agent_execution_engine
    │
    └─ [Step 4] _inject_predictive_actor()
        ├─ 提取现有的 actor_module 和 actor_optimizer
        ├─ 创建 PredictiveActor (配置了 prediction_loss_weight)
        └─ 替换 actor_rollout_wg.actor
```

## 关键验证点

### ✅ 1. Trainer 层级

```python
from rllm_ext.training import PredictiveAgentWorkflowTrainer

trainer = PredictiveAgentWorkflowTrainer(
    workflow_class=PredictiveToolWorkflow,  # 自定义 workflow
    config=config,  # config.rllm.prediction_loss.enabled = True
    ...
)
```

**验证**：
- `type(trainer)` == `PredictiveAgentWorkflowTrainer`
- `type(trainer.__class__.__bases__[0])` == `AgentWorkflowPPOTrainer`

### ✅ 2. Engine 层级

```python
# 在 init_workers() 后
assert isinstance(trainer.agent_execution_engine, PredictiveAgentWorkflowEngine)
```

**验证**：
- `type(trainer.agent_execution_engine)` == `PredictiveAgentWorkflowEngine`
- 该 engine 的 `transform_results_for_verl()` 会添加 `prediction_targets`

### ✅ 3. Actor 层级

```python
# 在 init_workers() 后
assert isinstance(trainer.actor_rollout_wg.actor, PredictiveActor)
```

**验证**：
- `type(trainer.actor_rollout_wg.actor)` == `PredictiveActor`
- 该 actor 的 `update_policy()` 会计算 prediction loss

## 数据流验证

### Rollout 阶段 (Engine 负责)

```python
# trainer.generate_trajectories() 调用
trainer.agent_execution_engine.execute_tasks_verl(batch)
    ↓
# PredictiveAgentWorkflowEngine.transform_results_for_verl()
batch.non_tensor_batch["prediction_targets"] = [...]  # 添加了！
```

**验证**：
```python
batch = trainer.generate_trajectories(test_batch)
assert "prediction_targets" in batch.non_tensor_batch
```

### 训练阶段 (Actor 负责)

```python
# trainer.fit_agent() 调用
trainer.actor_rollout_wg.update_actor(batch)
    ↓
# PredictiveActor.update_policy()
policy_loss = pg_loss + prediction_loss * weight  # 添加了！
loss.backward()
```

**验证**：
```python
metrics = trainer.actor_rollout_wg.update_actor(batch)
assert "train/prediction_loss" in metrics
```

## 调试技巧

### 1. 打印类型信息

```python
# 在训练开始前
print(f"Trainer type: {type(trainer).__name__}")
print(f"Engine type: {type(trainer.agent_execution_engine).__name__}")
print(f"Actor type: {type(trainer.actor_rollout_wg.actor).__name__}")
print(f"Actor config: {trainer.config.actor}")
```

期望输出：
```
Trainer type: PredictiveAgentWorkflowTrainer
Engine type: PredictiveAgentWorkflowEngine
Actor type: PredictiveActor
Actor config: {..., 'prediction_loss_weight': 0.1, ...}
```

### 2. 检查数据

```python
# 在 generate_trajectories 后
batch = trainer.generate_trajectories(test_batch)
print(f"Has prediction_targets: {'prediction_targets' in batch.non_tensor_batch}")

if "prediction_targets" in batch.non_tensor_batch:
    targets = batch.non_tensor_batch["prediction_targets"]
    print(f"Prediction targets length: {len(targets)}")
    print(f"First target: {targets[0]}")
```

### 3. 检查 Metrics

```python
# 在训练循环中
metrics = trainer.actor_rollout_wg.update_actor(batch)
print(f"Metrics keys: {metrics.keys()}")

if "train/prediction_loss" in metrics:
    print(f"✓ Prediction loss computed: {metrics['train/prediction_loss']}")
else:
    print("✗ Prediction loss NOT computed")
```

## 常见问题

### Q1: Engine 没有被替换？

**原因**：`_replace_workflow_engine()` 没有被调用

**解决**：检查 `init_workers()` 中的调用顺序

### Q2: Actor 没有被替换？

**原因**：`prediction_loss_weight` = 0

**解决**：在 config 中设置：
```yaml
rllm:
  prediction_loss:
    enabled: true
    weight: 0.1  # > 0
```

### Q3: prediction_targets 为空？

**原因**：
1. Workflow 不是 `PredictiveToolWorkflow`
2. `prediction_cfg.enabled` = False
3. 所有 step 都没有 tool call (被 simple_tir 过滤)

**解决**：检查 workflow 配置

### Q4: Prediction loss 始终为 0？

**原因**：`_compute_prediction_loss_microbatch()` 返回 placeholder

**解决**：实现实际的 loss 计算逻辑

## 配置检查清单

```yaml
# config.yaml
rllm:
  # 1. 启用 prediction loss
  prediction_loss:
    enabled: true
    weight: 0.1

  # 2. Workflow 配置
  workflow:
    n_parallel_tasks: 128
    retry_limit: 3

# 3. 使用自定义 trainer
trainer:
  # 在 Python 代码中指定，不是配置
```

```python
# train.py
from rllm_ext.training import PredictiveAgentWorkflowTrainer
from rllm_ext.workflows import PredictiveToolWorkflow

workflow_args = {
    "prediction_cfg": {
        "enabled": True,  # 必须启用
        ...
    }
}

trainer = PredictiveAgentWorkflowTrainer(
    workflow_class=PredictiveToolWorkflow,  # 使用自定义 workflow
    workflow_args=workflow_args,
    ...
)
```

## 总结

| 层级 | 类名 | 职责 | 如何验证 |
|------|------|------|---------|
| Trainer | `PredictiveAgentWorkflowTrainer` | 组装所有组件 | `type(trainer)` |
| Engine | `PredictiveAgentWorkflowEngine` | 收集 prediction 数据 | `type(trainer.agent_execution_engine)` |
| Actor | `PredictiveActor` | 计算 prediction loss | `type(trainer.actor_rollout_wg.actor)` |

只有**三者都被正确替换**，prediction auxiliary loss 才能生效！
