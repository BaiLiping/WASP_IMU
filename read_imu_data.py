#!/usr/bin/env python3
"""
Script to read IMU data from txt files.
The data format is:
- Column 1: Timestamp
- Column 2: Data type (only ACC is valid)
- Column 3: X acceleration
- Column 4: Y acceleration
- Column 5: Z acceleration (ignored)
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path


def read_imu_data(filepath):
    """
    Read IMU data from a txt file.
    
    Parameters:
    -----------
    filepath : str or Path
        Path to the IMU data file
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing timestamp, x and y acceleration data
        Only rows with ACC data type are included
    """
    # Read the data file
    data = pd.read_csv(filepath, sep='\t', header=None, 
                       names=['timestamp', 'data_type', 'x', 'y', 'z'])
    
    # Filter only ACC data
    acc_data = data[data['data_type'] == 'ACC'].copy()
    
    # Drop the data_type and z columns as they're not needed
    acc_data = acc_data[['timestamp', 'x', 'y']]
    
    # Reset index
    acc_data.reset_index(drop=True, inplace=True)
    
    return acc_data


def process_all_files(data_dir='data'):
    """
    Process all txt files in the data directory.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the data files
        
    Returns:
    --------
    dict
        Dictionary with filename as key and DataFrame as value
    """
    data_path = Path(data_dir)
    all_data = {}
    
    # Process all txt files in the directory
    for file in data_path.glob('*.txt'):
        # Skip statistics.txt if it exists
        if file.name == 'statistics.txt':
            continue
            
        print(f"Processing {file.name}...")
        try:
            df = read_imu_data(file)
            all_data[file.stem] = df
            print(f"  - Loaded {len(df)} ACC samples")
        except Exception as e:
            print(f"  - Error processing {file.name}: {e}")
    
    return all_data


def main():
    """Main function to demonstrate usage."""
    
    # Example 1: Read a single file
    print("Example 1: Reading single file")
    print("-" * 40)
    
    single_file_data = read_imu_data('data/data1.txt')
    print(f"Shape: {single_file_data.shape}")
    print(f"Columns: {list(single_file_data.columns)}")
    print("\nFirst 5 rows:")
    print(single_file_data.head())
    print("\nBasic statistics:")
    print(single_file_data.describe())
    
    # Example 2: Process all files
    print("\n\nExample 2: Processing all files")
    print("-" * 40)
    
    all_data = process_all_files('data')
    
    print(f"\nLoaded {len(all_data)} data files")
    for filename, df in all_data.items():
        print(f"  {filename}: {len(df)} samples")
    
    # Example 3: Basic analysis
    print("\n\nExample 3: Basic analysis")
    print("-" * 40)
    
    for filename, df in all_data.items():
        print(f"\n{filename}:")
        print(f"  Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  Duration: {df['timestamp'].max() - df['timestamp'].min()} ms")
        print(f"  X acc - mean: {df['x'].mean():.4f}, std: {df['x'].std():.4f}")
        print(f"  Y acc - mean: {df['y'].mean():.4f}, std: {df['y'].std():.4f}")


if __name__ == "__main__":
    main()