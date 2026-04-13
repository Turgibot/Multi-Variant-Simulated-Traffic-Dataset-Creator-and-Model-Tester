#!/usr/bin/env python3
"""
Debug the file sorting in TemporalGraphDataset.
"""

import glob
import os
import re
from temporal_dataset import extract_step_number

def debug_file_sorting():
    """Debug the file sorting issue."""
    print("🔍 Debugging File Sorting")
    print("=" * 40)
    
    data_path = "/app/dataset"
    file_pattern = "*.pt"
    
    # Get files
    files = glob.glob(os.path.join(data_path, file_pattern))
    print(f"Total files found: {len(files)}")
    
    # Test extract_step_number on first few files
    print("\nTesting extract_step_number:")
    for i, file_path in enumerate(files[:10]):
        step_num = extract_step_number(file_path)
        print(f"  {os.path.basename(file_path)} -> {step_num}")
    
    # Test sorting
    print("\nTesting sorting:")
    sorted_files = sorted(
        files,
        key=lambda p: (extract_step_number(p), os.path.basename(p))
    )
    
    print(f"Sorted files (first 10):")
    for i, file_path in enumerate(sorted_files[:10]):
        step_num = extract_step_number(file_path)
        print(f"  {i}: {os.path.basename(file_path)} -> {step_num}")
    
    # Test with num_files and start_idx
    print("\nTesting with num_files=30, start_idx=0:")
    num_files = 30
    start_idx = 0
    selected_files = sorted_files[start_idx:start_idx + num_files]
    print(f"Selected files: {len(selected_files)}")
    
    if len(selected_files) > 0:
        print("First 5 selected files:")
        for i, file_path in enumerate(selected_files[:5]):
            step_num = extract_step_number(file_path)
            print(f"  {i}: {os.path.basename(file_path)} -> {step_num}")
    else:
        print("❌ No files selected!")
    
    return len(selected_files) > 0

if __name__ == "__main__":
    success = debug_file_sorting()
    if success:
        print("\n✅ File sorting works!")
    else:
        print("\n❌ File sorting failed!")
