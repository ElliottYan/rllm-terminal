"""
Test script to verify SWE data preparation and parquet file integrity.
"""
import json
import os
import pandas as pd
from prepare_swe_data import prepare_swe_data
from rllm.data.dataset import DatasetRegistry

def test_parquet_files():
    """Test if parquet files can be read correctly."""
    print("Testing SWE data preparation...")
    
    # Get the dataset directory
    dataset_dir = DatasetRegistry._DATASET_DIR
    
    # Find all SWE datasets
    swe_datasets = [d for d in os.listdir(dataset_dir) if 'swe' in d.lower() or 'r2e' in d.lower()]
    
    if not swe_datasets:
        print("No SWE datasets found. Please run prepare_swe_data() first.")
        return
    
    print(f"\nFound {len(swe_datasets)} SWE dataset(s): {swe_datasets}")
    
    for dataset_name in swe_datasets:
        dataset_path = os.path.join(dataset_dir, dataset_name)
        print(f"\n{'='*60}")
        print(f"Testing dataset: {dataset_name}")
        print(f"{'='*60}")
        
        # Find all parquet files
        parquet_files = [f for f in os.listdir(dataset_path) if f.endswith('.parquet')]
        
        for parquet_file in parquet_files:
            file_path = os.path.join(dataset_path, parquet_file)
            print(f"\nTesting file: {parquet_file}")
            print(f"File size: {os.path.getsize(file_path) / 1024:.2f} KB")
            
            try:
                # Try to read the parquet file
                df = pd.read_parquet(file_path)
                print(f"✓ Successfully read parquet file")
                print(f"  Rows: {len(df)}")
                print(f"  Columns: {list(df.columns)}")
                
                # Show sample data
                if len(df) > 0:
                    print(f"\nFirst row preview:")
                    first_row = df.iloc[0].to_dict()
                    for key, value in list(first_row.items())[:5]:  # Show first 5 fields
                        if isinstance(value, str) and len(value) > 100:
                            print(f"  {key}: {value[:100]}...")
                        else:
                            print(f"  {key}: {value}")
                    
                    # Check if verl file, try to parse JSON fields
                    if '_verl' in parquet_file:
                        print(f"\n  Validating Verl format...")
                        try:
                            prompt = json.loads(first_row['prompt'])
                            reward_model = json.loads(first_row['reward_model'])
                            extra_info = json.loads(first_row['extra_info'])
                            print(f"  ✓ All JSON fields are valid")
                            print(f"    - prompt type: {type(prompt)}")
                            print(f"    - reward_model type: {type(reward_model)}")
                            print(f"    - extra_info type: {type(extra_info)}")
                        except json.JSONDecodeError as e:
                            print(f"  ✗ JSON parsing error: {e}")
                
            except Exception as e:
                print(f"✗ Failed to read parquet file: {e}")
                import traceback
                traceback.print_exc()

def main():
    """Main test function."""
    print("SWE Data Preparation Test")
    print("="*60)
    
    # Test reading existing parquet files
    test_parquet_files()
    
    print("\n" + "="*60)
    print("Test completed!")
    print("="*60)

if __name__ == "__main__":
    main()
