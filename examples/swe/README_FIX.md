# SWE Data Preparation 修复说明

## 问题分析

### 1. train.parquet 的 key 问题

**原因**：之前的 `process_fn` 只是简单返回原始数据，没有提取和标准化关键字段。

**现象**：
- 保存的数据缺少 `question`、`ground_truth`、`data_source` 等标准字段
- 训练脚本期望这些标准字段，导致后续使用时出错

### 2. train_verl.parquet 读取问题

**原因**：`DatasetRegistry.apply_verl_postprocessing()` 会创建嵌套结构：
```python
{
    "prompt": [{"role": "user", "content": "placeholder"}],  # 列表嵌套字典
    "reward_model": {"style": "rule", "ground_truth": None},  # 嵌套字典
    "extra_info": entry,  # 可能包含复杂嵌套
}
```

**现象**：
- Pandas 在将这种嵌套结构保存到 parquet 时可能出现问题
- 读取时可能因为数据类型不一致或嵌套过深而失败
- 特别是当 `extra_info` 包含列表、字典等复杂结构时

## 解决方案

### 方案：确保传入 DatasetRegistry 的数据都是可序列化的简单类型

在 `prepare_swe_data.py` 中的 `process_fn` 里：

1. **提取所有关键字段**：确保包含 `question`、`ground_truth`、`data_source` 等标准字段

2. **将复杂结构序列化为 JSON 字符串**：
   ```python
   "FAIL_TO_PASS": json.dumps(row_dict.get("FAIL_TO_PASS", [])),
   "PASS_TO_PASS": json.dumps(row_dict.get("PASS_TO_PASS", [])),
   "metadata": json.dumps(row_dict),
   ```

3. **保留必要的字典结构字段**：
   - `index` 和 `uid` 等 Verl 需要的字段保持为顶级字段
   - 复杂嵌套结构序列化为 JSON 字符串
   - 确保所有值都是简单类型（str, int, float, bool, None）

### 为什么不修改 rllm/data/dataset.py？

`apply_verl_postprocessing` 是 rLLM 框架的核心方法，被所有数据集使用。修改它会影响其他数据集的处理逻辑。正确的做法是在每个数据准备脚本中确保传入的数据格式正确。

## 使用方法

### 1. 重新生成数据

```bash
cd examples/swe
python prepare_swe_data.py
```

### 2. 验证数据

```python
import pandas as pd
import json

# 读取普通 parquet
df = pd.read_parquet("path/to/train.parquet")
print(df.columns)
print(df.iloc[0]['question'])  # 应该包含 problem_statement

# 读取 verl parquet
df_verl = pd.read_parquet("path/to/train_verl.parquet")
print(df_verl.columns)  # 应该有 prompt, reward_model, extra_info

# extra_info 中应该包含所有原始字段
first_row = df_verl.iloc[0]
extra_info = first_row['extra_info']
print(extra_info['question'])
print(extra_info['instance_id'])
```

### 3. 检查数据完整性

```bash
python test_prepare_swe.py
```

## 数据格式说明

### train.parquet 格式
```python
{
    "question": str,              # problem_statement
    "ground_truth": str,          # 空字符串（SWE 任务通过 patch 评估）
    "data_source": str,           # "swe"
    "index": int,                 # 数据索引（Verl 数据集需要）
    "uid": str,                   # 唯一标识符
    "instance_id": str,           # 任务唯一标识
    "repo": str,                  # 仓库名
    "base_commit": str,           # 基础提交
    "patch": str,                 # 修复补丁
    "test_patch": str,            # 测试补丁
    "problem_statement": str,     # 问题描述
    "hints_text": str,            # 提示信息
    "created_at": str,            # 创建时间
    "version": str,               # 版本
    "environment_setup_commit": str,  # 环境设置提交
    "FAIL_TO_PASS": str,          # JSON 字符串，失败->通过的测试
    "PASS_TO_PASS": str,          # JSON 字符串，通过->通过的测试
}
```

### train_verl.parquet 格式
由 `apply_verl_postprocessing` 自动生成，包含：
- `prompt`: 占位符提示（JSON 字符串格式）
- `reward_model`: 奖励模型配置（JSON 字符串格式）
- `extra_info`: 上述所有字段的字典（包含 `index`, `uid` 等 Verl 需要的字段）

**重要**：`extra_info` 字段必须是字典格式，因为 Verl 的数据集加载器会直接访问 `extra_info.index` 等字段。如果序列化为 JSON 字符串，Verl 会无法访问这些字段。

## 参考

参考 `scripts/data/swe_dataset.py` 的实现，它采用了另一种方式：
- 直接创建训练格式的数据
- 使用 pandas 直接保存，不通过 DatasetRegistry
- 将 extra_info 保存为 JSON 字符串

如果你的场景需要直接用于训练（不需要 DatasetRegistry 的管理功能），可以参考那个实现。
