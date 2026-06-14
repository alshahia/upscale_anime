#!/usr/bin/env python3
"""
Test NaN detection and recovery functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

import torch
import torch.nn as nn

def test_nan_handler():
    """Test NaN handler functionality."""
    print("TESTING NaN HANDLER")
    print("=" * 50)
    
    # Create a simple model
    model = nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters())
    
    # Create NaN handler
    from utils.nan_handler import NaNHandler
    nan_handler = NaNHandler(verbose=False)
    
    # Test normal tensor
    normal_tensor = torch.randn(5, 5)
    result = nan_handler.detect_nan_in_tensor(normal_tensor, 'normal')
    print('Normal tensor detection:', result['has_nan'])
    
    # Test NaN tensor
    nan_tensor = torch.full((5, 5), float('nan'))
    result = nan_handler.detect_nan_in_tensor(nan_tensor, 'nan')
    print('NaN tensor detection:', result['has_nan'])
    
    # Test model detection
    model_result = nan_handler.detect_nan_in_model(model)
    print('Model detection:', model_result['nan_detected'])
    
    # Test backup/restore
    backup_result = nan_handler.create_backup(model, optimizer, 100)
    print('Backup result:', backup_result['success'])
    
    # Get report
    report = nan_handler.get_nan_report()
    print('NaN report created:', len(report))
    
    print('[OK] NaN handler test completed')
    return True

if __name__ == "__main__":
    test_nan_handler()
