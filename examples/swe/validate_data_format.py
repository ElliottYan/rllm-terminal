"""
验证 SWE 数据格式是否符合 Verl 的要求
"""
import json
import pandas as pd
import os
from rllm.data.dataset import DatasetRegistry

def validate_data_format():
    """验证数据格式"""
    print("="*60)
    print("SWE 数据格式验证")
    print("="*60)

    dataset_dir = DatasetRegistry._DATASET_DIR

    if not os.path.exists(dataset_dir):
        print("❌ 数据集目录不存在，请先运行 prepare_swe_data.py")
        return

    # 查找 SWE 数据集
    swe_datasets = []
    for item in os.listdir(dataset_dir):
        if 'swe' in item.lower() or 'r2e' in item.lower():
            swe_datasets.append(item)

    if not swe_datasets:
        print("❌ 未找到 SWE 数据集")
        return

    print(f"找到 {len(swe_datasets)} 个 SWE 数据集: {swe_datasets}")

    for dataset_name in swe_datasets:
        print(f"\n{'='*40}")
        print(f"验证数据集: {dataset_name}")
        print(f"{'='*40}")

        dataset_path = os.path.join(dataset_dir, dataset_name)

        # 检查普通 parquet 文件
        train_parquet = os.path.join(dataset_path, "train.parquet")
        if os.path.exists(train_parquet):
            print(f"\n✅ 验证 train.parquet")

            try:
                df = pd.read_parquet(train_parquet)
                print(f"   行数: {len(df)}")
                print(f"   列数: {len(df.columns)}")

                # 检查必需字段
                required_fields = ['question', 'ground_truth', 'data_source', 'index', 'uid']
                missing_fields = [f for f in required_fields if f not in df.columns]
                if missing_fields:
                    print(f"   ❌ 缺少必需字段: {missing_fields}")
                else:
                    print(f"   ✅ 所有必需字段都存在")

                # 检查复杂字段是否已序列化
                complex_fields = ['FAIL_TO_PASS', 'PASS_TO_PASS']
                for field in complex_fields:
                    if field in df.columns:
                        sample_value = df[field].iloc[0] if len(df) > 0 else None
                        if sample_value and isinstance(sample_value, str):
                            try:
                                json.loads(sample_value)
                                print(f"   ✅ {field} 已正确序列化为 JSON 字符串")
                            except json.JSONDecodeError:
                                print(f"   ❌ {field} 不是有效的 JSON 字符串")
                        else:
                            print(f"   ⚠️  {field} 不是字符串类型")

            except Exception as e:
                print(f"   ❌ 读取失败: {e}")
        else:
            print(f"\n❌ train.parquet 不存在")

        # 检查 verl parquet 文件
        verl_parquet = os.path.join(dataset_path, "train_verl.parquet")
        if os.path.exists(verl_parquet):
            print(f"\n✅ 验证 train_verl.parquet")

            try:
                df_verl = pd.read_parquet(verl_parquet)
                print(f"   行数: {len(df_verl)}")
                print(f"   列数: {len(df_verl.columns)}")

                if len(df_verl) > 0:
                    first_row = df_verl.iloc[0]

                    # 检查 prompt 字段（应该是 JSON 字符串）
                    if 'prompt' in df_verl.columns:
                        prompt = first_row['prompt']
                        if isinstance(prompt, str):
                            try:
                                prompt_data = json.loads(prompt)
                                print(f"   ✅ prompt 字段正确序列化为 JSON")
                            except json.JSONDecodeError:
                                print(f"   ❌ prompt 字段不是有效的 JSON")
                        else:
                            print(f"   ❌ prompt 字段不是字符串")

                    # 检查 reward_model 字段（应该是 JSON 字符串）
                    if 'reward_model' in df_verl.columns:
                        reward_model = first_row['reward_model']
                        if isinstance(reward_model, str):
                            try:
                                reward_data = json.loads(reward_model)
                                print(f"   ✅ reward_model 字段正确序列化为 JSON")
                            except json.JSONDecodeError:
                                print(f"   ❌ reward_model 字段不是有效的 JSON")
                        else:
                            print(f"   ❌ reward_model 字段不是字符串")

                    # 检查 extra_info 字段（应该是字典，可以访问 .get()）
                    if 'extra_info' in df_verl.columns:
                        extra_info = first_row['extra_info']
                        if isinstance(extra_info, dict):
                            # Verl 期望能访问 extra_info.index
                            index = extra_info.get('index', None)
                            uid = extra_info.get('uid', None)
                            if index is not None and uid is not None:
                                print(f"   ✅ extra_info 是字典，可以访问 index 和 uid")
                                print(f"      index: {index}, uid: {uid}")
                            else:
                                print(f"   ❌ extra_info 缺少 index 或 uid 字段")
                        else:
                            print(f"   ❌ extra_info 不是字典类型 (类型: {type(extra_info)})")

            except Exception as e:
                print(f"   ❌ 读取失败: {e}")
        else:
            print(f"\n❌ train_verl.parquet 不存在")

    print(f"\n{'='*60}")
    print("验证完成")
    print(f"{'='*60}")

if __name__ == "__main__":
    validate_data_format()
