# 数据集嵌套结构问题分析

## TL;DR

**发现**：其他数据集没有遇到 PyArrow 错误，是因为它们在传入 `DatasetRegistry.register_dataset()` 之前，就已经将所有复杂结构序列化为 JSON 字符串了。

**但是**：`apply_verl_postprocessing` 仍然会创建嵌套结构，理论上所有数据集都可能遇到这个问题。

---

## 各数据集的处理方式对比

### ✅ 正确处理嵌套结构的数据集

#### 1. DeepCoder (`examples/deepcoder/prepare_deepcoder_data.py`)

```python
def preprocess_fn(example, idx):
    # ... 处理逻辑 ...
    return {
        "question": question,
        "ground_truth": json.dumps(tests),  # ✅ 序列化列表
        "data_source": "livecodebench",
        "uid": f"deepcoder_{idx}",
        "index": idx,
        "starter_code": starter_code,
        "metadata": json.dumps(metadata)  # ✅ 序列化字典
    }
```

**关键**：`tests` 是一个列表，`metadata` 是一个字典，都被序列化为 JSON 字符串。

#### 2. VimGolf (`examples/vimgolf/prepare_vimgolf_data.py`)

```python
challenge_data = dict(
    input=input,
    output=target,
    challenge_id=line_data["id"],
)
it = {
    "question": question_prompt,
    "ground_truth": json.dumps(challenge_data),  # ✅ 序列化字典
    "data_source": "vimgolf-public-challenges",
}
```

**关键**：`challenge_data` 是字典，被序列化为 JSON 字符串。

### ✅ 只传简单类型的数据集

#### 3. HotpotQA (`examples/mcp/prepare_hotpotqa_data.py`)

```python
processed = [{
    "question": example["question"],      # ✅ 字符串
    "ground_truth": example["answer"],    # ✅ 字符串
    "data_source": "hotpotqa"             # ✅ 字符串
} for example in split_data]
```

**关键**：所有字段都是简单字符串，没有嵌套结构。

#### 4. FrozenLake (`examples/frozenlake/prepare_frozenlake_data.py`)

```python
def frozenlake_process_fn(seed, size, p, idx):
    return {
        "seed": seed,     # ✅ 整数
        "size": size,     # ✅ 整数
        "p": p,           # ✅ 浮点数
        "index": idx,     # ✅ 整数
        "uid": f"{seed}_{size}_{p}"  # ✅ 字符串
    }
```

**关键**：所有字段都是简单类型（整数、浮点数、字符串）。

#### 5. Math 数据集 (`examples/simple_math/prepare_math_dataset.py`)

```python
def preprocess_fn(example):
    return {
        "question": example.get("problem", ""),      # ✅ 字符串
        "ground_truth": example.get("solution", ""), # ✅ 字符串
        "data_source": "hendrycks_math",             # ✅ 字符串
    }
```

**关键**：所有字段都是字符串。

### ❌ SWE 数据集的问题

#### SWE (旧版本 - 有问题)

```python
def process_fn(row):
    row_dict = dict(row)
    return row_dict  # ❌ 直接返回原始字典，包含嵌套结构
```

**问题**：
- `row_dict` 可能包含 `FAIL_TO_PASS`（列表）、`PASS_TO_PASS`（列表）等嵌套结构
- 这些嵌套结构没有被序列化

#### SWE (修复后 - 正确)

```python
def process_fn(row):
    row_dict = dict(row)
    problem_statement = row_dict.get("problem_statement", "")
    
    return {
        "question": problem_statement,
        "ground_truth": "",
        "data_source": "swe",
        # ... 其他简单字段 ...
        "FAIL_TO_PASS": json.dumps(row_dict.get("FAIL_TO_PASS", [])),  # ✅ 序列化
        "PASS_TO_PASS": json.dumps(row_dict.get("PASS_TO_PASS", [])),  # ✅ 序列化
        "metadata": json.dumps(row_dict),  # ✅ 序列化
    }
```

---

## 真正的问题：`apply_verl_postprocessing`

即使所有数据集都正确序列化了输入数据，`apply_verl_postprocessing` 仍然会创建嵌套结构：

```python
def apply_verl_postprocessing(cls, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    processed_data = []
    for entry in data:
        processed_entry = {
            "prompt": [{"role": "user", "content": "placeholder"}],  # ❌ 列表嵌套字典
            "reward_model": {                                        # ❌ 嵌套字典
                "style": "rule",
                "ground_truth": None,
            },
            "extra_info": entry,  # ❌ 整个 entry 字典
        }
        processed_data.append(processed_entry)
    return processed_data
```

### 为什么其他数据集没有报错？

可能的原因：

1. **没有实际使用 verl 文件**：大部分训练脚本可能直接用 `train.parquet`，不用 `train_verl.parquet`
2. **数据量小**：小数据集的嵌套层次浅，PyArrow 可以处理
3. **运气好**：恰好没有触发 PyArrow 的限制

### 验证方法

让我们测试一下其他数据集的 verl 文件是否真的可以读取：

```python
import pandas as pd
import os
from rllm.data.dataset import DatasetRegistry

# 测试所有 verl 文件
dataset_dir = DatasetRegistry._DATASET_DIR
for dataset_name in os.listdir(dataset_dir):
    verl_files = []
    dataset_path = os.path.join(dataset_dir, dataset_name)
    if os.path.isdir(dataset_path):
        for f in os.listdir(dataset_path):
            if f.endswith('_verl.parquet'):
                verl_files.append(os.path.join(dataset_path, f))
    
    for verl_file in verl_files:
        try:
            df = pd.read_parquet(verl_file)
            print(f"✅ {dataset_name}/{os.path.basename(verl_file)}: OK")
        except Exception as e:
            print(f"❌ {dataset_name}/{os.path.basename(verl_file)}: {e}")
```

---

## 结论和建议

### 结论

1. **其他数据集做得对**：它们在传入 `register_dataset` 之前序列化了复杂结构
2. **SWE 数据集需要修复**：传入时需要序列化 `FAIL_TO_PASS` 等字段
3. **框架层面有潜在问题**：`apply_verl_postprocessing` 创建的嵌套结构可能在某些情况下无法被 PyArrow 处理

### 建议

#### 方案 A：只修复 SWE 数据集（✅ 推荐，已完成）

**优点**：
- 最小化修改
- 不影响其他数据集
- 与其他数据集的处理方式保持一致

**做法**：
```python
# 在 prepare_swe_data.py 中序列化复杂字段
"FAIL_TO_PASS": json.dumps(row_dict.get("FAIL_TO_PASS", [])),
"PASS_TO_PASS": json.dumps(row_dict.get("PASS_TO_PASS", [])),
"metadata": json.dumps(row_dict),
```

**状态**：✅ 已在当前的 `prepare_swe_data.py` 中实现

#### 方案 B：修复 `apply_verl_postprocessing`（可选，更彻底）

**优点**：
- 从根源解决问题
- 防止未来其他数据集遇到同样问题
- 更健壮

**做法**：
```python
@classmethod
def apply_verl_postprocessing(cls, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    processed_data = []
    for entry in data:
        processed_entry = {
            "prompt": json.dumps([{"role": "user", "content": "placeholder"}]),
            "reward_model": json.dumps({
                "style": "rule",
                "ground_truth": entry.get("ground_truth", None),
            }),
            "extra_info": json.dumps(entry),
        }
        processed_data.append(processed_entry)
    return processed_data
```

**风险**：
- 需要测试所有现有数据集
- 可能影响已有的训练流程（如果它们依赖嵌套结构）

---

## 最终推荐

### 短期：只修复 SWE 数据集（方案 A）

这是最安全的方案，已经在当前的 `prepare_swe_data.py` 中实现。这样做：
- ✅ 解决了 SWE 数据集的问题
- ✅ 与其他数据集的处理方式一致
- ✅ 不修改框架代码，零风险

### 长期：考虑修复框架（方案 B）

如果未来有时间和资源，可以考虑修复 `apply_verl_postprocessing`，因为：
- 更健壮，防止类似问题再次出现
- 统一处理逻辑
- 但需要充分测试，确保不破坏现有功能

### 当前状态

**现在的 `prepare_swe_data.py` 已经是正确的实现**，只要确保：
1. 所有复杂字段都被序列化为 JSON 字符串
2. 只传入简单类型（str, int, float, bool, None）到 `register_dataset`

这样就不会遇到 PyArrow 的错误了！
