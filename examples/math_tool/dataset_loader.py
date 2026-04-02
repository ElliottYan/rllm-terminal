git import logging
import os

import polars as pl

from rllm.data.dataset import Dataset, DatasetRegistry

logger = logging.getLogger(__name__)


def load_dataset_with_path_override(config, name: str, split: str) -> Dataset | None:
    """Load from an explicit parquet path when configured, otherwise fall back to DatasetRegistry."""
    dataset_path_key = f"{split}_dataset_path"
    dataset_path = config.get(dataset_path_key)

    if dataset_path:
        dataset_path = os.path.abspath(os.path.expanduser(dataset_path))
        if not os.path.exists(dataset_path):
            logger.warning(f"Dataset file not found: {dataset_path}")
            return None

        data = pl.read_parquet(dataset_path).to_dicts()

        logger.info(f"Loaded dataset '{name}' split '{split}' from path '{dataset_path}' with {len(data)} examples.")

        return Dataset(data=data, name=name, split=split)

    return DatasetRegistry.load_dataset(name, split)
