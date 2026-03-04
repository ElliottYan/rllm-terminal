# Prediction Auxiliary Loss - 完整调用路径与架构图

## 🎯 核心目标

**为 rLLM 的 PPO 训练添加 prediction auxiliary loss**，让模型直接优化工具输出预测的准确性。

---

## 📍 当前代码的调用路径

### 用户入口（你的训练脚本）

```python
# examples/math_tool/train_math_with_tool_prediction_workflow.py

from rllm_ext import PredictiveAgentTrainer  # ← 关键改动：从 rllm_ext 导入
from rllm_ext.workflows import PredictiveToolWorkflow

use_predictive_trainer = True
if use_predictive_trainer is True:
    AgentTrainer = PredictiveAgentTrainer  # ← 使用自定义 Trainer

trainer = AgentTrainer(
    workflow_class=PredictiveToolWorkflow,
    workflow_args={...},
    config=config,
    train_dataset=train_dataset,
    val_dataset=test_dataset,
)
trainer.train()
```

---

## 🔍 详细调用链

### 第一阶段：初始化 (Init)

```
trainer.train()
    ↓
PredictiveAgentTrainer.train()
    ↓
TaskRunner.run()  # [Ray Remote Actor]
    ↓
PredictiveTaskRunner.run()  # ← 使用的是我们的扩展版本！
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 1: 创建 Trainer                                        │
├─────────────────────────────────────────────────────────────┤
│ if config.rllm.workflow.use_workflow:                      │
│     trainer = PredictiveAgentWorkflowTrainer(...)          │
│     # 注意：这是 AgentWorkflowPPOTrainer 的扩展版            │
│ else:                                                       │
│     trainer = AgentPPOTrainer(...)                          │
└─────────────────────────────────────────────────────────────┘
    ↓
trainer.init_workers()
    ↓
PredictiveAgentWorkflowTrainer.init_workers()
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: 初始化工作流（4个步骤）                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ 1️⃣ _configure_prediction_loss()                            │
│    └─ 读取 config.rllm.prediction_loss                       │
│    └─ 设置 config.actor.prediction_loss_weight = 0.1        │
│                                                              │
│ 2️⃣ super().init_workers()                                   │
│    └─ AgentWorkflowPPOTrainer.init_workers()                │
│       ├─ 创建 actor_rollout_wg (使用 DataParallelPPOActor)  │
│       ├─ 创建 critic_wg                                      │
│       ├─ 创建 ref_policy_wg                                 │
│       └─ 创建 AgentWorkflowEngine                            │
│                                                              │
│ 3️⃣ _replace_workflow_engine()                              │
│    └─ 创建 PredictiveAgentWorkflowEngine                     │
│    └─ 初始化 worker pool                                     │
│    └─ 替换 self.agent_execution_engine                       │
│                                                              │
│ 4️⃣ _inject_predictive_actor()                               │
│    if prediction_loss_weight > 0:                           │
│       └─ 创建 PredictiveActor                               │
│       └─ 替换 actor_rollout_wg.actor                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
    ↓
初始化完成！现在：
- self.agent_execution_engine = PredictiveAgentWorkflowEngine
- self.actor_rollout_wg.actor = PredictiveActor
```

### 第二阶段：数据生成 (Rollout)

```
trainer.fit_agent()
    ↓
for batch in train_dataloader:
    ↓
    final_gen_batch_output = self.generate_trajectories(batch)
        ↓
    self.agent_execution_engine.execute_tasks_verl(batch)
        ↓
    PredictiveAgentWorkflowEngine.execute_tasks_verl()
        ↓
        await self.execute_tasks(tasks)
            ↓
        workflow = await workflow_queue.get()  # 获取 PredictiveToolWorkflow
            ↓
        episode = await workflow.run(task=task, uid=uid)
            ↓
        PredictiveToolWorkflow.run()
            ↓
        for step_idx in range(max_steps):
            # 1️⃣ 生成 action
            output = await rollout_engine.get_model_response(messages)
            action = agent.update_from_model(response)

            # 2️⃣ 生成 prediction（关键！）
            if prediction_cfg.enabled:
                prediction_prompt = build_prediction_prompt(action)
                pred_output = await rollout_engine.get_model_response(prediction_prompt)

                # 存储到 step.info
                step.info["rllm_ext.prediction_text"] = prediction_text
                step.info["rllm_ext.prediction_raw_text"] = prediction_raw_text
                step.info["rllm_ext.prediction_reasoning"] = prediction_reasoning

            # 3️⃣ 执行 action
            next_obs, reward, done, info = env.step(action)
                ↓
            PredictiveToolEnvironment.step()
                ↓
            # 计算 similarity reward（可选）
            if similarity_config.enabled:
                similarity_reward = compute_prediction_similarity_reward(
                    prediction_text, actual_outputs, similarity_config
                )
                reward += similarity_reward  # ← 通过 reward 影响 RL

            # 4️⃣ 更新 step
            cur_step = agent.get_current_state()
            cur_step.reward = reward
            cur_step.info.update(step_info)
            # prediction 信息已经在 step.info 中了！

        return episode  # 包含多个 steps，每个 step 有 prediction 信息
            ↓
    return self.transform_results_for_verl(episodes, task_ids)
        ↓
    PredictiveAgentWorkflowEngine.transform_results_for_verl()
        ↓
    for episode in episodes:
        for trajectory in episode.trajectories:
            for step in trajectory.steps:
                # 🎯 关键：收集 prediction 数据
                pred_text = step.info.get("rllm_ext.prediction_text")
                actual_output = get_actual_from_observation(step.observation)

                prediction_targets.append({
                    "prediction": pred_text,
                    "actual": actual_output,
                    "has_prediction": pred_text is not None,
                })

    # 调用父类生成基础 batch
    batch = super().transform_results_for_verl(episodes, task_ids)

    # 🎯 关键：添加 prediction_targets 到 batch
    batch.non_tensor_batch["prediction_targets"] = np.array(prediction_targets)

    return batch
        ↓
return final_gen_batch_output  # DataProto with prediction_targets!
```

### 第三阶段：训练更新 (Training)

```
batch = final_gen_batch_output  # 包含 prediction_targets
    ↓
# 计算 advantages
batch = compute_advantage(batch, ...)
    ↓
# 更新 actor
actor_output = self.actor_rollout_wg.update_actor(batch)
    ↓
PredictiveActor.update_policy(data=batch)  # ← 使用的是我们的扩展 Actor！
    ↓
for epoch in ppo_epochs:
    for mini_batch in data.split(ppo_mini_batch_size):
        for micro_batch in mini_batch.split(micro_batch_size):
            # 1️⃣ 前向传播
            entropy, log_prob = self._forward_micro_batch(micro_batch)

            # 2️⃣ 计算 PPO loss
            pg_loss = compute_policy_loss(log_prob, old_log_prob, advantages, ...)
            policy_loss = pg_loss - entropy_loss * entropy_coeff

            # 3️⃣ 添加 KL loss（如果启用）
            if use_kl_loss:
                kl_loss = kl_penalty(log_prob, ref_log_prob)
                policy_loss = policy_loss + kl_loss * kl_coef

            # 🎯 关键：添加 prediction auxiliary loss
            if "prediction_targets" in micro_batch and prediction_loss_weight > 0:
                prediction_loss = self._compute_prediction_loss_microbatch(
                    prediction_targets, response_mask
                )
                # TODO: 实现 actual loss 计算（目前是 placeholder）
                # prediction_loss = compute_actual_loss(pred_texts, actual_texts)

                policy_loss = policy_loss + prediction_loss * prediction_loss_weight
                metrics["train/prediction_loss"] = prediction_loss

            # 4️⃣ 反向传播
            loss.backward()  # ← gradient 包含了 prediction loss 的贡献！

    return metrics
        ↓
grad_norm = self._optimizer_step()  # 更新模型参数
```

---

## 🏗️ 组件架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Training Script                      │
│  examples/math_tool/train_math_with_tool_prediction_workflow.py │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ↓ 使用
┌─────────────────────────────────────────────────────────────────┐
│                     PredictiveAgentTrainer                       │
│              (rllm_ext/training/predictive_agent_trainer.py)     │
│                                                                   │
│  职责：Drop-in replacement for AgentTrainer                      │
│  关键：使用 PredictiveTaskRunner (Ray actor)                     │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ↓ 创建
┌─────────────────────────────────────────────────────────────────┐
│                  PredictiveAgentWorkflowTrainer                  │
│         (rllm_ext/training/predictive_agent_workflow_trainer.py)  │
│                                                                   │
│  职责：扩展 AgentWorkflowPPOTrainer，注入自定义组件               │
│  关键方法：                                                       │
│    - init_workers(): 替换 engine 和 actor                        │
│    - _configure_prediction_loss(): 配置参数                       │
│    - _replace_workflow_engine(): 替换为 PredictiveAgentWorkflowEngine │
│    - _inject_predictive_actor(): 替换为 PredictiveActor           │
└──────────────────────┬────────────────────────────┬──────────────┘
                       │                            │
                       ↓ 创建                      ↓ 创建
┌──────────────────────────────────┐  ┌──────────────────────────────────┐
│  PredictiveAgentWorkflowEngine   │  │       PredictiveActor             │
│ (rllm_ext/training/              │  │ (rllm_ext/training/              │
│  predictive_agent_workflow_      │  │  predictive_actor.py)            │
│  engine.py)                      │  │                                   │
│                                  │  │ 职责：扩展 DataParallelPPOActor  │
│ 职责：收集 prediction 数据       │  │                                   │
│                                  │  │ 关键方法：                       │
│ 关键方法：                       │  │   - update_policy(): 在 PPO loss  │
│ - transform_results_for_verl():  │  │       基础上添加 prediction loss │
│   添加 prediction_targets        │  │   - _compute_prediction_loss_    │
│   到 DataProto                   │  │     microbatch(): 计算 loss      │
└──────────────────────────────────┘  └──────────────────────────────────┘
                       │
                       ↓ 使用
┌─────────────────────────────────────────────────────────────────┐
│                    PredictiveToolWorkflow                        │
│          (rllm_ext/workflows/predictive_tool_workflow.py)        │
│                                                                   │
│  职责：在每个 tool call 后添加 prediction sub-step               │
│                                                                   │
│  关键方法：                                                       │
│    - run(): action → predict → env.step                          │
│    - _build_prediction_prompt(): 构建 prediction prompt          │
│    - _extract_tagged_block(): 解析 prediction 结果              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 数据流图

```
┌───────────────────────────────────────────────────────────────┐
│  Step 1: Rollout - 生成数据和 prediction                       │
└───────────────────────────────────────────────────────────────┘

Prompt (用户问题)
    ↓
Model → Action (tool call)
    ↓
Model → Prediction (预测工具输出)
    ├─ prediction_text: "The answer is 42"
    └─ prediction_reasoning: "The code calculates..."
    ↓
Environment → Tool Execution → Actual Output: "The result is 42"
    ↓
Step 对象:
{
    action: tool_call,
    observation: {tool_outputs: {"output": "The result is 42"}},
    info: {
        "rllm_ext.prediction_text": "The answer is 42",
        "rllm_ext.prediction_raw_text": "...",
        "rllm_ext.prediction_reasoning": "..."
    },
    reward: similarity_reward  # 可选
}

┌───────────────────────────────────────────────────────────────┐
│  Step 2: Transform - 转换为 DataProto                         │
└───────────────────────────────────────────────────────────────┘

Episode (多个 Steps)
    ↓
PredictiveAgentWorkflowEngine.transform_results_for_verl()
    ↓
提取 prediction 数据:
prediction_targets = [
    {
        "prediction": "The answer is 42",
        "actual": "The result is 42",
        "has_prediction": true
    },
    ...
]
    ↓
DataProto:
{
    batch: {
        "input_ids": ...,
        "responses": ...,
        "attention_mask": ...
    },
    non_tensor_batch: {
        "prediction_targets": [...],  # ← 关键！
        "step_ids": [...],
        ...
    }
}

┌───────────────────────────────────────────────────────────────┐
│  Step 3: Training - 计算 loss 并反向传播                      │
└───────────────────────────────────────────────────────────────┘

DataProto (with prediction_targets)
    ↓
PredictiveActor.update_policy()
    ↓
Forward pass:
    - model(input_ids) → logits
    - log_prob = log_softmax(logits, responses)
    ↓
Compute PPO loss:
    - pg_loss = PPO_CLIP(ratio * advantages)
    - entropy_loss = -entropy * coeff
    - kl_loss = KL(log_prob || ref_log_prob)
    ↓
Compute Prediction loss:  # 🎯 新增
    - prediction_loss = compute_loss(pred_texts, actual_texts)
    ↓
Total loss:
    policy_loss = pg_loss - entropy_loss + kl_loss
                + prediction_loss * weight  # ← 添加！
    ↓
loss.backward()  # ← gradient 流向所有参数
    ↓
Update: optimizer.step()
```

---

## ✅ 验证检查点

在训练开始前，你可以添加这些检查来确保一切正确连接：

```python
# 在 trainer.train() 之前添加
trainer = PredictiveAgentTrainer(...)

# 检查 1: Trainer 类型
assert isinstance(trainer, PredictiveAgentTrainer), \
    f"Wrong trainer type: {type(trainer)}"

# 检查 2: Config 配置
assert hasattr(trainer.config, 'rllm'), "Config missing rllm section"
assert hasattr(trainer.config.rllm, 'prediction_loss'), \
    "Config missing prediction_loss section"

# 这些检查需要在 init_workers() 之后进行
# 所以在实际运行时，你需要检查日志输出
```

**期望的日志输出：**

```
PredictiveAgentWorkflowEngine injected successfully
PredictiveActor initialized with:
  prediction_loss_weight: 0.1
  prediction_loss_type: cross_entropy
```

**Metrics 输出：**

训练时应该能看到：
- `train/prediction_loss`: prediction loss 的原始值
- `actor/pg_loss`: PPO policy gradient loss
- `actor/entropy_loss`: entropy regularization
- `actor/kl_loss`: KL divergence loss

---

## 📊 完整组件清单

| 组件 | 文件路径 | 职责 |
|------|---------|------|
| `PredictiveAgentTrainer` | `rllm_ext/training/predictive_agent_trainer.py` | 高层 API，替换 `AgentTrainer` |
| `PredictiveTaskRunner` | 同上 | Ray actor，使用 `PredictiveAgentWorkflowTrainer` |
| `PredictiveAgentWorkflowTrainer` | `rllm_ext/training/predictive_agent_workflow_trainer.py` | 训练器，替换 engine 和 actor |
| `PredictiveAgentWorkflowEngine` | `rllm_ext/training/predictive_agent_workflow_engine.py` | 收集 prediction 数据 |
| `PredictiveActor` | `rllm_ext/training/predictive_actor.py` | 计算 prediction loss |
| `PredictiveToolWorkflow` | `rllm_ext/workflows/predictive_tool_workflow.py` | 生成 prediction |
| `PredictiveToolAgent` | `rllm_ext/agents/predictive_tool_agent.py` | 存储 prediction |
| `PredictiveToolEnvironment` | `rllm_ext/environments/predictive_tool_env.py` | 可选：similarity reward |

---

## 🎯 总结

**一句话概括：**
> 通过在 Rollout 阶段收集 prediction 数据，在 Training 阶段将其作为 auxiliary loss 添加到 PPO objective 中。

**关键替换点：**
1. `AgentTrainer` → `PredictiveAgentTrainer`
2. `AgentWorkflowEngine` → `PredictiveAgentWorkflowEngine`
3. `DataParallelPPOActor` → `PredictiveActor`

**数据流：**
```
Rollout: Prediction → Step.info → DataProto.prediction_targets
Training: prediction_targets → prediction_loss → total_loss → backward → update
```
