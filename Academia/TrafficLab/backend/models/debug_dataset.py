#!/usr/bin/env python3
"""
Debug the dataset loading issue.
"""

import glob
import os

def debug_dataset_loading():
    """Debug why the dataset can't find the files."""
    print("🔍 Debugging Dataset Loading")
    print("=" * 40)
    
    data_path = "/app/dataset"
    file_pattern = "*.pt"
    
    print(f"Data path: {data_path}")
    print(f"File pattern: {file_pattern}")
    print(f"Path exists: {os.path.exists(data_path)}")
    
    if os.path.exists(data_path):
        print(f"Directory contents:")
        files = os.listdir(data_path)
        print(f"  Total files: {len(files)}")
        print(f"  First 10 files: {files[:10]}")
        
        # Test glob pattern
        glob_pattern = os.path.join(data_path, file_pattern)
        print(f"Glob pattern: {glob_pattern}")
        
        pt_files = glob.glob(glob_pattern)
        print(f"Glob results: {len(pt_files)} files found")
        print(f"First 5 .pt files: {pt_files[:5]}")
        
        # Test with step_ pattern
        step_pattern = os.path.join(data_path, "step_*.pt")
        step_files = glob.glob(step_pattern)
        print(f"Step pattern results: {len(step_files)} files found")
        print(f"First 5 step files: {step_files[:5]}")
        
        return len(pt_files) > 0
    else:
        print("❌ Data path does not exist!")
        return False

if __name__ == "__main__":
    success = debug_dataset_loading()
    if success:
        print("\n✅ Files found! Dataset should work.")
    else:
        print("\n❌ No files found! Dataset will fail.")
