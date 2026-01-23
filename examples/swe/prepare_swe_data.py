import json

from datasets import load_dataset

from rllm.data.dataset import DatasetRegistry

SWE_DATASETS = [
    "R2E-Gym/R2E-Gym-Subset",
    "R2E-Gym/R2E-Gym-Lite",
    "R2E-Gym/R2E-Gym-V1",
    "R2E-Gym/SWE-Bench-Lite",
    "R2E-Gym/SWE-Bench-Verified",
    "r2e-edits/SweSmith-RL-Dataset",
]


def prepare_swe_data():
    """
    Prepare and register SWE datasets for training and testing.
    
    This function processes SWE datasets and saves them in a format suitable for training.
    Each row is converted to have standardized fields that can be serialized to parquet.

    Returns:
        tuple: (train_datasets, test_datasets) - lists of registered datasets
    """

    def make_process_fn():
        def process_fn(row):
            row_dict = dict(row)
            problem_statement = row_dict.get("problem_statement", "")

            # Create a flat structure with all fields as simple types (strings, numbers)
            # Complex structures are serialized to JSON strings for parquet compatibility
            return {
                # Core fields expected by the training pipeline
                "question": problem_statement,
                "ground_truth": "",  # SWE tasks are evaluated by patch application, not simple string match
                "data_source": "swe",

                # Required fields for Verl dataset indexing
                "index": row_dict.get("index", 0),  # Verl expects this in extra_info.index
                "uid": row_dict.get("instance_id", ""),  # Unique identifier

                # SWE-specific fields
                "instance_id": row_dict.get("instance_id", ""),
                "repo": row_dict.get("repo", ""),
                "base_commit": row_dict.get("base_commit", ""),
                "patch": row_dict.get("patch", ""),
                "test_patch": row_dict.get("test_patch", ""),
                "problem_statement": problem_statement,
                "hints_text": row_dict.get("hints_text", ""),
                "created_at": str(row_dict.get("created_at", "")),
                "version": str(row_dict.get("version", "")),
                "environment_setup_commit": row_dict.get("environment_setup_commit", ""),

                # Serialize complex fields to JSON strings for parquet compatibility
                "FAIL_TO_PASS": json.dumps(row_dict.get("FAIL_TO_PASS", [])),
                "PASS_TO_PASS": json.dumps(row_dict.get("PASS_TO_PASS", [])),
            }

        return process_fn

    process_fn = make_process_fn()
    train_datasets = []
    test_datasets = []

    for dataset_name in SWE_DATASETS:
        print(f"Processing dataset: {dataset_name}")
        try:
            # Load the dataset dictionary (which contains splits like 'train' or 'test')
            dataset_splits = load_dataset(dataset_name)
        except Exception as e:
            print(f"Failed to load dataset {dataset_name}: {e}")
            continue

        dataset_key = dataset_name.split("/")[-1].replace("-", "_")

        # Process train split if it exists
        if "train" in dataset_splits:
            print(f"Processing 'train' split for {dataset_name}")
            train_data = [process_fn(row) for row in dataset_splits["train"]]
            train_dataset = DatasetRegistry.register_dataset(f"{dataset_key}", train_data, "train")
            train_datasets.append(train_dataset)
            print(f"Registered train dataset with {len(train_data)} examples")

        # Process test split if it exists
        if "test" in dataset_splits:
            print(f"Processing 'test' split for {dataset_name}")
            test_data = [process_fn(row) for row in dataset_splits["test"]]
            test_dataset = DatasetRegistry.register_dataset(f"{dataset_key}", test_data, "test")
            test_datasets.append(test_dataset)
            print(f"Registered test dataset with {len(test_data)} examples")

        # If neither train nor test exists, use the first available split as train
        if "train" not in dataset_splits and "test" not in dataset_splits:
            available_splits = list(dataset_splits.keys())
            if available_splits:
                split_name = available_splits[0]
                print(f"Using '{split_name}' split as train data for {dataset_name}")
                train_data = [process_fn(row) for row in dataset_splits[split_name]]
                train_dataset = DatasetRegistry.register_dataset(f"{dataset_key}", train_data, "train")
                train_datasets.append(train_dataset)
                print(f"Registered train dataset with {len(train_data)} examples")

    return train_datasets, test_datasets


if __name__ == "__main__":
    train_datasets, test_datasets = prepare_swe_data()
    print("\nSummary:")
    print(f"Total train datasets: {len(train_datasets)}")
    print(f"Total test datasets: {len(test_datasets)}")

    if train_datasets:
        print("Sample train example from first dataset:")
        print(train_datasets[0].get_data()[0])

    if test_datasets:
        print("Sample test example from first dataset:")
        print(test_datasets[0].get_data()[0])
