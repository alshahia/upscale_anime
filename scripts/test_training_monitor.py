#!/usr/bin/env python3
"""
Test training monitoring functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

import torch

def test_training_monitor():
    """Test training monitor functionality."""
    print("TESTING TRAINING MONITOR")
    print("=" * 50)
    
    from anime_sr.utils.training_monitor import TrainingMonitor, TrainingDashboard
    
    monitor = TrainingMonitor(verbose=False)
    
    # Simulate some training steps
    for i in range(10):
        loss = 1.0 - i * 0.05 + torch.randn(1).item() * 0.01
        grad_norm = 0.5 + torch.randn(1).item() * 0.1
        lr = 1e-4 * (0.95 ** (i // 10))
        
        monitor.log_step(i, loss, grad_norm, lr)
    
    # Get report
    report = monitor.get_stability_report()
    print("Training report created:", len(report))
    print("Overall health:", report['stability_indicators']['overall_health'])
    
    # Test dashboard
    dashboard = TrainingDashboard(monitor)
    dashboard_data = dashboard.get_dashboard_data()
    print("Dashboard data created:", len(dashboard_data))
    print("Current loss:", dashboard_data['current_metrics']['loss'])
    
    print("\n[OK] Training monitor test completed")
    return True

if __name__ == "__main__":
    test_training_monitor()
