#!/usr/bin/env python3
"""
Test script for the error handling and logging system.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from anime_sr.utils.error_handler import (
    ErrorHandler, ErrorSeverity, SuperResolutionError, 
    ModelLoadError, handle_errors, log_performance
)
from anime_sr.utils.logging_config import setup_logging, get_logger, log_system_info


def test_basic_error_handling():
    """Test basic error handling functionality."""
    print("🧪 Testing Basic Error Handling")
    print("=" * 40)
    
    # Initialize error handler
    error_handler = ErrorHandler(log_dir="test_logs")
    
    # Test custom exception
    try:
        raise ModelLoadError("Test model loading error", "MODEL_001", {"model": "test_model"})
    except Exception as e:
        error_handler.log_error(e, severity=ErrorSeverity.MEDIUM, context={"test": True})
    
    # Test regular exception
    try:
        raise ValueError("Test regular exception")
    except Exception as e:
        error_handler.log_error(e, severity=ErrorSeverity.HIGH, context={"test": True})
    
    # Print error summary
    summary = error_handler.get_error_summary()
    print(f"Total errors: {summary['total_errors']}")
    print(f"Errors by type: {summary['errors_by_type']}")
    print(f"Errors by severity: {summary['errors_by_severity']}")
    
    print("[OK] Basic error handling test completed")


def test_decorators():
    """Test error handling decorators."""
    print("\n🧪 Testing Error Handling Decorators")
    print("=" * 40)
    
    @handle_errors(ErrorSeverity.LOW)
    def test_function():
        raise ValueError("Test function error")
    
    @log_performance("test_operation")
    def test_performance_function():
        import time
        time.sleep(0.1)
        return "success"
    
    # Test error decorator
    try:
        test_function()
    except Exception as e:
        print(f"Caught decorated function error: {e}")
    
    # Test performance decorator
    result = test_performance_function()
    print(f"Performance function result: {result}")
    
    print("[OK] Decorators test completed")


def test_logging_system():
    """Test the logging system."""
    print("\n🧪 Testing Logging System")
    print("=" * 40)
    
    # Setup logging
    setup_logging(log_dir="test_logs", app_name="test_app")
    
    # Get loggers
    main_logger = get_logger("test_main")
    training_logger = get_logger("training")
    inference_logger = get_logger("inference")
    
    # Test logging
    main_logger.info("Main logger test message")
    training_logger.debug("Training logger debug message")
    inference_logger.warning("Inference logger warning message")
    
    # Test system info logging
    log_system_info()
    
    print("[OK] Logging system test completed")


def test_error_context():
    """Test error context manager."""
    print("\n🧪 Testing Error Context")
    print("=" * 40)
    
    from anime_sr.utils.error_handler import ErrorContext, MemoryContext
    
    # Test error context
    try:
        with ErrorContext("test_operation", {"param1": "value1"}):
            raise ValueError("Test error in context")
    except Exception as e:
        print(f"Caught context error: {e}")
    
    # Test memory context
    try:
        with MemoryContext(threshold_mb=1000, operation="test_memory"):
            pass  # Simulate some work
    except Exception as e:
        print(f"Memory context error: {e}")
    
    print("[OK] Error context test completed")


def test_error_report():
    """Test error report generation."""
    print("\n🧪 Testing Error Report")
    print("=" * 40)
    
    error_handler = ErrorHandler(log_dir="test_logs")
    
    # Generate some test errors
    error_handler.log_error(ValueError("Test error 1"), ErrorSeverity.LOW)
    error_handler.log_error(RuntimeError("Test error 2"), ErrorSeverity.MEDIUM)
    error_handler.log_error(Exception("Test error 3"), ErrorSeverity.HIGH)
    
    # Generate report
    report = error_handler.create_error_report()
    print("Error Report:")
    print(report[:500] + "..." if len(report) > 500 else report)
    
    # Save report
    error_handler.save_error_report()
    
    print("[OK] Error report test completed")


def test_validation():
    """Test input validation."""
    print("\n🧪 Testing Input Validation")
    print("=" * 40)
    
    error_handler = ErrorHandler(log_dir="test_logs")
    
    # Test valid input
    try:
        schema = {"name": str, "age": int}
        data = {"name": "test", "age": 25}
        error_handler.validate_input(data, schema, "test_validation")
        print("[OK] Valid input validation passed")
    except Exception as e:
        print(f"[ERROR] Valid input validation failed: {e}")
    
    # Test invalid input
    try:
        schema = {"name": str, "age": int}
        data = {"name": "test", "age": "invalid"}
        error_handler.validate_input(data, schema, "test_validation")
        print("[ERROR] Invalid input validation should have failed")
    except Exception as e:
        print(f"[OK] Invalid input validation correctly failed: {e}")
    
    print("[OK] Validation test completed")


def main():
    """Run all tests."""
    print("[LAUNCH] Starting Error Handling and Logging Tests")
    print("=" * 50)
    
    try:
        test_basic_error_handling()
        test_decorators()
        test_logging_system()
        test_error_context()
        test_error_report()
        test_validation()
        
        print("\n" + "=" * 50)
        print("[CELEBRATE] All tests completed successfully!")
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
