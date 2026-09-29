"""
Comprehensive error handling and logging utilities for the anime super-resolution project.
Provides structured error handling, logging, and debugging capabilities.
"""

import os
import sys
import logging
import logging.handlers
import traceback
import inspect
import functools
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
from .logging_config import LogLevel, LoggingConfig

# Custom exception classes
class SuperResolutionError(Exception):
    """Base exception for super-resolution errors."""
    def __init__(self, message: str, error_code: str = None, context: Dict[str, Any] = None):
        super().__init__(message)
        self.error_code = error_code
        self.context = context or {}
        self.timestamp = datetime.now()


class ModelLoadError(SuperResolutionError):
    """Error loading model."""
    pass


class InferenceError(SuperResolutionError):
    """Error during inference."""
    pass


class TrainingError(SuperResolutionError):
    """Error during training."""
    pass


class ConfigurationError(SuperResolutionError):
    """Error in configuration."""
    pass


class DataError(SuperResolutionError):
    """Error in data processing."""
    pass


class MemoryError(SuperResolutionError):
    """Memory-related error."""
    pass


class ValidationError(SuperResolutionError):
    """Validation error."""
    pass


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorHandler:
    """Comprehensive error handling and logging system."""
    
    def __init__(self, log_dir: Path = None, enable_file_logging: bool = True, 
                 enable_console_logging: bool = True, log_level: LogLevel = LogLevel.INFO):
        self.log_dir = log_dir or Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.enable_file_logging = enable_file_logging
        self.enable_console_logging = enable_console_logging
        self.log_level = log_level
        
        # Error statistics
        self.error_stats = {
            "total_errors": 0,
            "errors_by_type": {},
            "errors_by_severity": {},
            "recent_errors": [],
            "last_error_time": None
        }
        
        self._setup_logging()
        
        # Error callbacks
        self.error_callbacks = []
    
    def _setup_logging(self):
        """Set up structured logging."""
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        root_logger.addHandler(console_handler)
        
        # File handlers
        if self.enable_file_logging:
            # Main log file with rotation
            file_handler = logging.handlers.RotatingFileHandler(
                self.log_dir / "app.log",
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setLevel(LogLevel.DEBUG.value)
            file_handler.setFormatter(detailed_formatter)
            root_logger.addHandler(file_handler)
            
            # Error log file with rotation
            error_handler = logging.handlers.RotatingFileHandler(
                self.log_dir / "errors.log",
                maxBytes=5 * 1024 * 1024,  # 5MB
                backupCount=3,
                encoding='utf-8'
            )
            error_handler.setLevel(LogLevel.ERROR.value)
            error_handler.setFormatter(detailed_formatter)
            root_logger.addHandler(error_handler)
            
            # Debug log file with rotation
            debug_handler = logging.handlers.RotatingFileHandler(
                self.log_dir / "debug.log",
                maxBytes=20 * 1024 * 1024,  # 20MB
                backupCount=5,
                encoding='utf-8'
            )
            debug_handler.setLevel(LogLevel.DEBUG.value)
            debug_handler.setFormatter(detailed_formatter)
            root_logger.addHandler(debug_handler)
    
    def log_error(self, error: Exception, severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                  context: Dict[str, Any] = None, user_message: str = None, 
                  original_error: Exception = None):
        """Log an error with full context and proper chaining."""
        
        # Update statistics
        self.error_stats["total_errors"] += 1
        error_type = type(error).__name__
        self.error_stats["errors_by_type"][error_type] = self.error_stats["errors_by_type"].get(error_type, 0) + 1
        self.error_stats["errors_by_severity"][severity.value] = self.error_stats["errors_by_severity"].get(severity.value, 0) + 1
        
        # Create error record with enhanced context
        error_record = {
            "timestamp": datetime.now().isoformat(),
            "type": error_type,
            "message": str(error),
            "severity": severity.value,
            "context": context or {},
            "user_message": user_message,
            "traceback": traceback.format_exc(),
            "module": getattr(error, '__module__', 'unknown'),
            "function": getattr(error, '__qualname__', 'unknown'),
            "line": getattr(error, '__lineno__', 0),
            "original_error": str(original_error) if original_error else None
        }
        
        # Add to recent errors with context retention
        self.error_stats["recent_errors"].append(error_record)
        if len(self.error_stats["recent_errors"]) > 100:
            self.error_stats["recent_errors"].pop(0)
        
        # Log with appropriate level
        log_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }.get(severity, logging.ERROR)
        
        logger = logging.getLogger('super_resolution')
        logger.log(log_level, f"{severity.value.upper()}: {error_record}")
        
        # Call error callbacks if any
        for callback in self.error_callbacks:
            try:
                callback(error_record)
            except Exception as callback_error:
                logger.warning(f"Error in error callback: {callback_error}")
        
        # Return enhanced error record for further processing
        return error_record

    def get_logger_name(self, error: Exception) -> str:
        """Get appropriate logger name for error type."""
        error_type = type(error).__name__
        
        if error_type in ["ModelLoadError", "InferenceError", "TrainingError", "ConfigurationError", "DataError", "MemoryError", "ValidationError"]:
            return "super_resolution"
        else:
            return "general"
    
    def save_error_to_file(self, error_record: Dict[str, Any]):
        """Save error record to JSON file."""
        try:
            error_file = self.log_dir / "errors.json"
            
            # Load existing errors
            if error_file.exists():
                with open(error_file, 'r') as f:
                    existing_errors = json.load(f)
            else:
                existing_errors = []
            
            # Add new error
            existing_errors.append(error_record)
            
            # Keep only last 1000 errors
            if len(existing_errors) > 1000:
                existing_errors = existing_errors[-1000:]
            
            # Save back
            with open(error_file, 'w') as f:
                json.dump(existing_errors, f, indent=2, default=str)
                
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to save error to file: {e}")
    
    def handle_training_exception(self, func: Callable):
        """Decorator for training-specific exception handling."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except TrainingError as e:
                self.log_error(e, severity=ErrorSeverity.HIGH, context={
                    "function": func.__name__,
                    "training_stage": kwargs.get("stage", "unknown"),
                    "epoch": kwargs.get("epoch", "unknown")
                })
                raise
            except Exception as e:
                self.log_error(e, severity=ErrorSeverity.CRITICAL, context={
                    "function": func.__name__,
                    "training_stage": kwargs.get("stage", "unknown"),
                    "epoch": kwargs.get("epoch", "unknown")
                })
                raise
        return wrapper
    
    def handle_inference_exception(self, func: Callable):
        """Decorator for inference-specific exception handling."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except InferenceError as e:
                self.log_error(e, severity=ErrorSeverity.MEDIUM, context={
                    "function": func.__name__,
                    "input_shape": kwargs.get("input_shape", "unknown"),
                    "model_name": kwargs.get("model_name", "unknown")
                })
                raise
            except Exception as e:
                self.log_error(e, severity=ErrorSeverity.HIGH, context={
                    "function": func.__name__,
                    "input_shape": kwargs.get("input_shape", "unknown"),
                    "model_name": kwargs.get("model_name", "unknown")
                })
                raise
        return wrapper
    
    def add_error_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add error callback function."""
        self.error_callbacks.append(callback)
    
    def handle_exception(self, func, *args, **kwargs):
        """Decorator for exception handling with context."""
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Get calling context
                frame = inspect.currentframe()
                caller_frame = frame.f_back.f_back[1] if frame and frame.f_back else None
                
                context = {
                    'function': func.__name__,
                    'module': func.__module__,
                    'args': str(args)[:100],  # Truncate for logging
                    'kwargs': str(kwargs)[:100],  # Truncate for logging
                }
                
                if caller_frame:
                    context.update({
                        'caller_function': caller_frame.function,
                        'caller_module': caller_frame.module,
                        'caller_line': caller_frame.lineno
                    })
                
                # Log with full context
                error_record = self.log_error(
                    error=e, 
                    severity=ErrorSeverity.MEDIUM,
                    context=context,
                    user_message=f"Exception in {func.__name__}"
                )
                
                # Re-raise with proper chaining
                raise SuperResolutionError(
                    f"{func.__name__} failed: {str(e)}",
                    error_code=type(e).__name__,
                    context=error_record.get('context', {})
                ) from e
        
        return wrapper

    def get_error_summary(self):
        """Get error summary statistics."""
        return {
            "total_errors": self.error_stats["total_errors"],
            "errors_by_type": self.error_stats["errors_by_type"],
            "errors_by_severity": self.error_stats["errors_by_severity"],
            "recent_errors_count": len(self.error_stats["recent_errors"]),
            "last_error": self.error_stats["recent_errors"][-1] if self.error_stats["recent_errors"] else None
        }

    def clear_error_stats(self):
        """Clear error statistics."""
        self.error_stats = {
            "total_errors": 0,
            "errors_by_type": {},
            "errors_by_severity": {},
            "recent_errors": []
        }
    
    def validate_input(self, data: Any, schema: Dict[str, Any], context: str = "validation") -> bool:
        """Validate input data against schema."""
        try:
            # Basic validation logic
            for key, expected_type in schema.items():
                if key not in data:
                    raise ValidationError(f"Missing required field: {key}")
                
                if not isinstance(data[key], expected_type):
                    raise ValidationError(f"Invalid type for {key}: expected {expected_type.__name__}, got {type(data[key]).__name__}")
            
            return True
            
        except ValidationError as e:
            self.log_error(e, severity=ErrorSeverity.MEDIUM, context={"validation_context": context})
            raise
        except Exception as e:
            self.log_error(e, severity=ErrorSeverity.HIGH, context={"validation_context": context})
            raise ValidationError(f"Validation failed: {str(e)}")
    
    def check_memory_usage(self, threshold_mb: float = 8000) -> bool:
        """Check memory usage and log if threshold exceeded."""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb > threshold_mb:
                error = MemoryError(f"Memory usage exceeded threshold: {memory_mb:.1f}MB > {threshold_mb}MB")
                self.log_error(error, severity=ErrorSeverity.HIGH, context={
                    "memory_usage_mb": memory_mb,
                    "threshold_mb": threshold_mb
                })
                return False
            
            return True
            
        except Exception as e:
            self.log_error(e, severity=ErrorSeverity.LOW, context={"operation": "memory_check"})
            return True  # Assume OK if we can't check
    
    def log_performance_metrics(self, operation: str, duration: float, metrics: Dict[str, Any] = None):
        """Log performance metrics."""
        logger = logging.getLogger("performance")
        
        message = f"Operation: {operation} | Duration: {duration:.3f}s"
        if metrics:
            message += f" | Metrics: {json.dumps(metrics, default=str)}"
        
        logger.info(message)
        
        # Log slow operations as warnings
        if duration > 10.0:
            logger.warning(f"Slow operation detected: {operation} took {duration:.3f}s")
    
    def create_error_report(self) -> str:
        """Create comprehensive error report."""
        summary = self.get_error_summary()
        
        report = f"""
# Error Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- Total Errors: {summary['total_errors']}
- Recent Errors: {summary['recent_errors_count']}

## Errors by Type
"""
        
        for error_type, count in summary['errors_by_type'].items():
            report += f"- {error_type}: {count}\n"
        
        report += "\n## Errors by Severity\n"
        for severity, count in summary['errors_by_severity'].items():
            report += f"- {severity}: {count}\n"
        
        if summary['last_error']:
            report += f"\n## Last Error\n"
            report += f"Time: {summary['last_error']['timestamp']}\n"
            report += f"Type: {summary['last_error']['type']}\n"
            report += f"Message: {summary['last_error']['message']}\n"
            report += f"Severity: {summary['last_error']['severity']}\n"
        
        return report
    
    def save_error_report(self, output_path: str = None):
        """Save error report to file."""
        if output_path is None:
            output_path = self.log_dir / f"error_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        report = self.create_error_report()
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        logging.getLogger(__name__).info(f"Error report saved to: {output_path}")


# Global error handler instance
error_handler = ErrorHandler()


# Convenience decorators
def handle_errors(severity: ErrorSeverity = ErrorSeverity.MEDIUM):
    """Decorator for automatic error handling."""
    def decorator(func):
        return error_handler.handle_exception(func)
    return decorator


def handle_training_errors():
    """Decorator for training-specific error handling."""
    return error_handler.handle_training_exception


def handle_inference_errors():
    """Decorator for inference-specific error handling."""
    return error_handler.handle_inference_exception


def validate_input(schema: Dict[str, Any], context: str = "validation"):
    """Decorator for input validation."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Validate kwargs if schema provided
            if schema and kwargs:
                error_handler.validate_input(kwargs, schema, context)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def log_performance(operation_name: str = None):
    """Decorator for performance logging."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                op_name = operation_name or f"{func.__module__}.{func.__name__}"
                error_handler.log_performance_metrics(op_name, duration)
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                op_name = operation_name or f"{func.__module__}.{func.__name__}"
                error_handler.log_performance_metrics(op_name, duration, {"error": str(e)})
                raise
        return wrapper
    return decorator


def check_memory(threshold_mb: float = 8000):
    """Decorator for memory checking."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            error_handler.check_memory_usage(threshold_mb)
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Context managers
class ErrorContext:
    """Context manager for error handling."""
    
    def __init__(self, operation: str, context: Dict[str, Any] = None):
        self.operation = operation
        self.context = context or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if exc_type is not None:
            error_handler.log_error(
                exc_val,
                severity=ErrorSeverity.HIGH,
                context={
                    "operation": self.operation,
                    "duration": duration,
                    **self.context
                }
            )
        else:
            error_handler.log_performance_metrics(self.operation, duration, self.context)
        
        return False  # Don't suppress exceptions


class MemoryContext:
    """Context manager for memory monitoring."""
    
    def __init__(self, threshold_mb: float = 8000, operation: str = "unknown"):
        self.threshold_mb = threshold_mb
        self.operation = operation
        self.start_memory = None
    
    def __enter__(self):
        try:
            import psutil
            process = psutil.Process()
            self.start_memory = process.memory_info().rss / 1024 / 1024
        except ImportError:
            self.start_memory = 0
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            import psutil
            process = psutil.Process()
            end_memory = process.memory_info().rss / 1024 / 1024
            memory_delta = end_memory - self.start_memory
            
            if memory_delta > self.threshold_mb:
                error = MemoryError(f"Memory delta exceeded threshold in {self.operation}: {memory_delta:.1f}MB")
                error_handler.log_error(error, severity=ErrorSeverity.HIGH, context={
                    "operation": self.operation,
                    "memory_delta_mb": memory_delta,
                    "threshold_mb": self.threshold_mb
                })
        except ImportError:
            pass
        
        return False  # Don't suppress exceptions


# Utility functions
def setup_global_error_handling(log_dir: str = "logs"):
    """Set up global error handling."""
    global error_handler
    error_handler = ErrorHandler(log_dir)
    return error_handler


def log_function_call(func: Callable):
    """Decorator to log function calls."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={len(args)}, kwargs={len(kwargs)}")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Completed {func.__name__}")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            raise
    return wrapper


def create_error_alert(error_record: Dict[str, Any]):
    """Example error callback for alerts."""
    if error_record.get("severity") == ErrorSeverity.CRITICAL.value:
        # Send alert (email, Slack, etc.)
        logging.getLogger("alerts").critical(f"CRITICAL ERROR: {error_record['message']}")


# Initialize with default alert callback
error_handler.add_error_callback(create_error_alert)
