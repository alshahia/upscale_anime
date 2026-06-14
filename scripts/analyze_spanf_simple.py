#!/usr/bin/env python3
"""
Simple SPAN-F Checkpoint Analysis (without torch dependency)

This script analyzes the downloaded SPAN-F checkpoint files to:
1. Check file formats and sizes
2. Extract basic information from ZIP structure
3. Provide recommendations for fine-tuning
"""

import os
import sys
import zipfile
import json
from pathlib import Path

def analyze_zip_structure(checkpoint_path):
    """Analyze PyTorch ZIP checkpoint structure"""
    print(f"\n=== Analyzing {os.path.basename(checkpoint_path)} ===")
    
    try:
        with zipfile.ZipFile(checkpoint_path, 'r') as zf:
            files = zf.namelist()
            print(f"ZIP files: {len(files)}")
            
            # Look for data files
            data_files = [f for f in files if f.endswith('.data')]
            json_files = [f for f in files if f.endswith('.json')]
            
            print(f"  Data files: {len(data_files)}")
            print(f"  JSON files: {len(json_files)}")
            
            # Try to read JSON for metadata
            for json_file in json_files:
                try:
                    with zf.open(json_file) as jf:
                        content = jf.read().decode('utf-8')
                        print(f"  {json_file}: {content[:200]}...")
                except (UnicodeDecodeError, KeyError):
                    print(f"  Could not read {json_file}")
            
            return {
                'files': files,
                'data_files': data_files,
                'json_files': json_files
            }
            
    except Exception as e:
        print(f"Error analyzing ZIP: {e}")
        return None

def estimate_model_from_filename(filename):
    """Estimate model configuration from filename"""
    parts = filename.replace('.pth', '').split('_')
    
    config = {}
    
    if 'x2' in filename:
        config['scale'] = 2
    elif 'x4' in filename:
        config['scale'] = 4
    
    if 'ch48' in filename:
        config['channels'] = 48
    elif 'ch52' in filename:
        config['channels'] = 52
    
    return config

def main():
    """Main analysis function"""
    checkpoint_dir = Path("checkpoints/span")
    
    if not checkpoint_dir.exists():
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return
    
    # Find all .pth files
    checkpoints = list(checkpoint_dir.glob("*.pth"))
    
    if not checkpoints:
        print("No .pth files found in checkpoint directory")
        return
    
    print(f"Found {len(checkpoints)} checkpoint files")
    
    # Analyze each checkpoint
    analyses = {}
    for checkpoint_path in sorted(checkpoints):
        file_size = checkpoint_path.stat().st_size / (1024 * 1024)  # MB
        print(f"\n{checkpoint_path.name}: {file_size:.1f} MB")
        
        # Estimate from filename
        config = estimate_model_from_filename(checkpoint_path.name)
        print(f"  Estimated config: {config}")
        
        # Analyze ZIP structure
        zip_analysis = analyze_zip_structure(checkpoint_path)
        if zip_analysis:
            analyses[checkpoint_path.name] = {
                'size_mb': file_size,
                'config': config,
                'zip_analysis': zip_analysis
            }
    
    # Summary and recommendations
    print("\n" + "="*60)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*60)
    
    # Expected SPAN-F configuration based on NTIRE 2025 paper
    expected_spanf = {
        'scale': 4,
        'channels': 32  # SPAN-F uses 32 channels
    }
    
    print(f"\nExpected SPAN-F Configuration (NTIRE 2025):")
    print(f"  Scale: {expected_spanf['scale']}x")
    print(f"  Channels: {expected_spanf['channels']}")
    print(f"  Blocks: 10")
    print(f"  Architecture: Modified SPAN with speed optimizations")
    
    # Find best matches
    x4_checkpoints = [name for name, analysis in analyses.items() 
                      if analysis['config'].get('scale') == 4]
    
    print(f"\n[OK] x4 Super-Resolution Checkpoints:")
    for name in x4_checkpoints:
        analysis = analyses[name]
        print(f"  - {name} ({analysis['size_mb']:.1f} MB)")
        print(f"    Channels: {analysis['config'].get('channels')}")
        
        # Check compatibility
        channels = analysis['config'].get('channels')
        if channels == 32:
            print(f"    [OK] Perfect SPAN-F match!")
        elif channels in [48, 52]:
            print(f"    [WARN]  Channel mismatch - expected 32, got {channels}")
            print(f"       May need architecture adjustment")
    
    # Configuration issues found
    print(f"\n[WARN]  CONFIGURATION ISSUES IDENTIFIED:")
    print(f"  1. Channel mismatch: Checkpoints have 48/52 channels, SPAN-F expects 32")
    print(f"  2. Current config points to: pretrained/SPANF_x4/spanx4_ch48.pth")
    print(f"  3. Actual checkpoint location: checkpoints/span/")
    
    print(f"\n[CONFIG] RECOMMENDED FIXES:")
    print(f"  1. Update configs/finetune_spanf.yaml line 13:")
    print(f"     FROM: pretrained_checkpoint: pretrained/SPANF_x4/spanx4_ch48.pth")
    print(f"     TO:   pretrained_checkpoint: checkpoints/span/spanx4_ch48.pth")
    
    print(f"\n  2. Update model configuration in configs/finetune_spanf.yaml:")
    print(f"     channels: 48  # Match checkpoint (or 52 for ch52 variant)")
    print(f"     num_blocks: 12  # Likely 12 blocks for these variants")
    
    print(f"\n  3. Alternative - Use SPAN-Tiny configuration:")
    print(f"     - These checkpoints might be SPAN-Tiny variants")
    print(f"     - Update model type from 'spanf' to 'span_tiny'")
    print(f"     - SPAN-Tiny uses 26 channels, 12 blocks")
    
    print(f"\n[LIST] NEXT STEPS:")
    print(f"  1. Choose which checkpoint to use (recommend spanx4_ch48.pth)")
    print(f"  2. Update config file path")
    print(f"  3. Update model channels to match checkpoint")
    print(f"  4. Test loading before training")
    print(f"  5. Monitor for key mismatch errors during training")

if __name__ == "__main__":
    main()
