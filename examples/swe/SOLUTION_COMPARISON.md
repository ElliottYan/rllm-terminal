# SWE 数据准备问题解决方案对比

## 错误原因

```
pyarrow.lib.ArrowNotImplementedError: Nested data conversions not implemented for chunked array outputs
```

这个错误发生在读取 `train_verl.parquet` 时，因为 `apply_verl_postprocessing` 创建了嵌套结构：

```python
{
    "prompt": [{"role": "user", "content": "placeholder"}],  # 列表嵌套字典
    "reward_model": {"style": "rule", "ground_truth": None},  # 嵌套字典
    "extra_info": entry,  # 嵌套字典（包含所有字段）
}
```

PyArrow 无法处理这种深度嵌套的结构。

---

## 方案 1：修复 `apply_verl_postprocessing`（✅ 推荐）

### 修改内容

修改 `rllm/data/dataset.py` 中的 `apply_verl_postprocessing` 方法，将嵌套结构序列化为 JSON 字符串：

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

### 优点

1. ✅ **一劳永逸**：修复后，所有数据集都不会再遇到这个问题
2. ✅ **保持一致性**：所有数据集继续使用 `DatasetRegistry`
3. ✅ **正确的修复**：在框架层面解决 bug
4. ✅ **向后兼容**：读取数据时 `json.loads()` 即可还原

### 缺点

1. ❌ 需要修改核心框架代码
2. ❌ 需要重新生成所有已存在的 verl parquet 文件

### 使用方式

```python
from rllm.data.dataset import DatasetRegistry

# 注册数据集（和之前一样）
train_dataset = DatasetRegistry.register_dataset("R2E_Gym_Subset", train_data, "train")

# 加载数据
dataset = DatasetRegistry.load_dataset('R2E_Gym_Subset', 'train')

# 读取 verl 文件（需要反序列化）
import pandas as pd
import json

verl_path = dataset.get_verl_data_path()
df = pd.read_parquet(verl_path)

# 解析 JSON 字段
prompt = json.loads(df.iloc[0]['prompt'])
reward_model = json.loads(df.iloc[0]['reward_model'])
extra_info = json.loads(df.iloc[0]['extra_info'])
```

---

## 方案 2：不使用 DatasetRegistry（⚠️ 替代方案）

### 修改内容

创建新脚本 `prepare_swe_data_direct.py`，直接保存 parquet 文件，不使用 `DatasetRegistry`。

### 优点

1. ✅ **不修改框架**：不触碰核心代码
2. ✅ **完全控制**：自己决定数据格式
3. ✅ **参考实现**：与 `scripts/data/swe_dataset.py` 一致

### 缺点

1. ❌ **失去统一管理**：数据不在 `dataset_registry.json` 中注册
2. ❌ **无法使用 DatasetRegistry API**：不能用 `load_dataset()` 等方法
3. ❌ **需要手动管理路径**：训练时需要指定完整路径
4. ❌ **不一致**：与其他数据集的使用方式不同

### 使用方式

```bash
# 生成数据
cd examples/swe
python prepare_swe_data_direct.py --output_dir ../../data/swe
```

```python
# 读取数据（直接用 pandas）
import pandas as pd
import json

df = pd.read_parquet("data/swe/R2E_Gym_Subset_train.parquet")

# 解析 JSON 字段
prompt = json.loads(df.iloc[0]['prompt'])
reward_model = json.loads(df.iloc[0]['reward_model'])
extra_info = json.loads(df.iloc[0]['extra_info'])
```

---

## 对比总结

| 维度 | 方案 1：修复框架 | 方案 2：直接保存 |
|------|----------------|----------------|
| **解决问题** | ✅ 彻底解决 | ✅ 绕过问题 |
| **修改框架** | ❌ 需要 | ✅ 不需要 |
| **统一管理** | ✅ 是 | ❌ 否 |
| **使用便利** | ✅ 高 | ⚠️ 中 |
| **长期维护** | ✅ 好 | ⚠️ 需额外维护 |
| **影响范围** | ⚠️ 所有数据集 | ✅ 仅 SWE |

---

## 建议

### 🎯 推荐方案 1

**理由：**

1. 这是一个**框架 bug**，不是使用问题
2. PyArrow 无法处理嵌套结构是已知限制
3. 修复后所有数据集受益，不会再出现这个错误
4. 序列化为 JSON 字符串是标准做法（参考 `scripts/data/swe_dataset.py`）

**修复后需要做的：**

```bash
# 1. 删除旧的 verl 文件（如果存在）
rm -rf rllm/data/datasets/*/train_verl.parquet
rm -rf rllm/data/datasets/*/test_verl.parquet

# 2. 重新生成数据
cd examples/swe
python prepare_swe_data.py
```

### 🔄 何时使用方案 2

仅在以下情况下使用：

- 你确实不能/不想修改框架代码
- 你只需要临时处理 SWE 数据
- 你的训练脚本已经适配了直接读取 parquet 的方式

---

## 技术细节

### 为什么嵌套结构会导致问题？

Parquet 是列式存储格式，对嵌套结构的支持有限：

1. **简单嵌套**（list<int>）：支持
2. **struct 类型**（一层字典）：支持
3. **复杂嵌套**（list<struct<...>>）：部分支持
4. **深度嵌套**（dict<list<dict<...>>>）：❌ 不支持

`apply_verl_postprocessing` 创建的结构属于第 4 类，PyArrow 无法处理。

### 为什么序列化为 JSON 字符串可以解决？

```python
# 原来：复杂嵌套（PyArrow 无法处理）
{"prompt": [{"role": "user", "content": "..."}]}

# 现在：简单字符串（PyArrow 完全支持）
{"prompt": '{"role": "user", "content": "..."}'}
```

字符串是 parquet 的基本类型，完全支持。

---

## 验证修复

修复后，运行测试：

```bash
cd examples/swe
python test_prepare_swe.py
```

应该看到：

```
✓ Successfully read parquet file
✓ All JSON fields are valid
```

而不是：

```
✗ ArrowNotImplementedError: Nested data conversions not implemented
```
