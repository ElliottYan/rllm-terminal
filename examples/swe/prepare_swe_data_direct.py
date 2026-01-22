"""
SWE 数据准备脚本 - 直接保存 parquet 版本
不使用 DatasetRegistry，避免 verl postprocessing 的嵌套结构问题
"""
import json
import os
import pandas as pd
from datasets import load_dataset

import rllm
from rllm.agents.system_prompts import SWE_SYSTEM_PROMPT, SWE_USER_PROMPT

# Get the directory for rLLM repo
RLLM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(rllm.__file__)))

SWE_DATASETS = [
    "R2E-Gym/R2E-Gym-Subset",
    "R2E-Gym/R2E-Gym-Lite",
    "R2E-Gym/R2E-Gym-V1",
    "R2E-Gym/SWE-Bench-Lite",
    "R2E-Gym/SWE-Bench-Verified",
    "r2e-edits/SweSmith-RL-Dataset",
]


def prepare_swe_data_direct(output_dir=None):
    """
    直接准备 SWE 数据并保存为 parquet，不使用 DatasetRegistry。
    
    这种方式避免了 apply_verl_postprocessing 的嵌套结构问题。
    
    Args:
        output_dir: 输出目录，默认为 rllm_dir/data/swe
    
    Returns:
        dict: 保存的文件路径字典
    """
    if output_dir is None:
        output_dir = os.path.join(RLLM_DIR, "data/swe")
    
    os.makedirs(output_dir, exist_ok=True)
    saved_files = {}
    
    def make_process_fn():
        def process_fn(row):
            row_dict = dict(row)
            problem_statement = row_dict.get("problem_statement", "")
            
            # 创建训练格式的数据（与 scripts/data/swe_dataset.py 一致）
            # 所有嵌套结构都序列化为 JSON 字符串
            return {
                "data_source": "swe",
                "prompt": json.dumps([
                    {"role": "system", "content": SWE_SYSTEM_PROMPT},
                    {"role": "user", "content": SWE_USER_PROMPT.format(problem_statement=problem_statement)}
                ]),
                "ability": "swe",
                "reward_model": json.dumps({"style": "rule", "ground_truth": ""}),
                "extra_info": json.dumps(row_dict),
            }
        
        return process_fn
    
    process_fn = make_process_fn()
    
    for dataset_name in SWE_DATASETS:
        print(f"Processing dataset: {dataset_name}")
        try:
            # Load the dataset dictionary (which contains splits like 'train' or 'test')
            dataset_splits = load_dataset(dataset_name)
        except Exception as e:
            print(f"Failed to load dataset {dataset_name}: {e}")
            continue
        
        output_name_base = dataset_name.split("/")[-1].replace("-", "_")
        
        # Determine which split exists ('train' or 'test')
        if "train" in dataset_splits:
            split_name = "train"
            split_data = dataset_splits["train"]
        elif "test" in dataset_splits:
            split_name = "test"
            split_data = dataset_splits["test"]
        else:
            print(f"Skipping {dataset_name} as it contains neither 'train' nor 'test' split.")
            continue
        
        print(f"Using '{split_name}' split for {dataset_name}")
        
        # Process the data from the identified split
        processed_data = [process_fn(row) for row in split_data]
        
        # Create DataFrame and save to a single parquet file
        df = pd.DataFrame(processed_data)
        output_filepath = os.path.join(output_dir, f"{output_name_base}_{split_name}.parquet")
        df.to_parquet(output_filepath, index=False)
        
        saved_files[f"{output_name_base}_{split_name}"] = output_filepath
        print(f"✅ Saved {len(df)} records to {output_filepath}")
        print(f"   File size: {os.path.getsize(output_filepath) / (1024 * 1024):.2f} MB")
    
    return saved_files


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare SWE datasets directly without DatasetRegistry")
    parser.add_argument("--output_dir", default=None, help="Output directory for parquet files")
    args = parser.parse_args()
    
    print("="*70)
    print("SWE Data Preparation (Direct Mode)")
    print("="*70)
    print("\n这个脚本直接保存 parquet 文件，不使用 DatasetRegistry")
    print("优点：避免 apply_verl_postprocessing 的嵌套结构问题")
    print("缺点：不会在 dataset_registry.json 中注册\n")
    
    saved_files = prepare_swe_data_direct(args.output_dir)
    
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"Successfully saved {len(saved_files)} files:")
    for name, path in saved_files.items():
        print(f"  - {name}: {path}")
    
    print("\n如何使用这些数据：")
    print("```python")
    print("import pandas as pd")
    print("import json")
    print("")
    print("# 读取数据")
    for name, path in list(saved_files.items())[:1]:  # 显示第一个示例
        print(f"df = pd.read_parquet('{path}')")
        break
    print("")
    print("# 解析 JSON 字段")
    print("first_row = df.iloc[0]")
    print("prompt = json.loads(first_row['prompt'])")
    print("reward_model = json.loads(first_row['reward_model'])")
    print("extra_info = json.loads(first_row['extra_info'])")
    print("```")
