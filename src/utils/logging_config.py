"""
Centralized logging configuration for the anime super-resolution project.
Provides structured logging with multiple handlers and formatters.
"""

import os
import sys
import logging
import logging.handlers
import logging.config
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import platform
from enum import Enum


class LogLevel(Enum):
    """Standardized log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        if platform.system() != "Windows":
            # Add colors
            color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                           'filename', 'module', 'lineno', 'funcName', 'created', 
                           'msecs', 'relativeCreated', 'thread', 'threadName', 
                           'processName', 'process', 'exc_info', 'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)


class LoggingConfig:
    """Centralized logging configuration."""

    DEFAULT_LEVELS = {
        'root': LogLevel.INFO,
        'super_resolution': LogLevel.DEBUG,
        'training': LogLevel.INFO,
        'inference': LogLevel.INFO,
        'performance': LogLevel.INFO,
    }

    PRODUCTION_LEVELS = {
        'root': LogLevel.WARNING,
        'super_resolution': LogLevel.INFO,
        'training': LogLevel.INFO,
        'inference': LogLevel.INFO,
        'performance': LogLevel.WARNING,
    }

    def __init__(self, log_dir: str = "logs", app_name: str = "anime_sr"):
        self.log_dir = Path(log_dir)
        self.app_name = app_name
        self.log_dir.mkdir(exist_ok=True)
        
        # Logging configuration
        self.config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'detailed': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                    'datefmt': '%Y-%m-%d %H:%M:%S'
                },
                'simple': {
                    'format': '%(asctime)s - %(levelname)s - %(message)s',
                    'datefmt': '%H:%M:%S'
                },
                'colored': {
                    '()': ColoredFormatter,
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                    'datefmt': '%H:%M:%S'
                },
                'json': {
                    '()': JSONFormatter
                }
            },
            'handlers': {},
            'loggers': {
                '': {
                    'level': self.DEFAULT_LEVELS['root'].value,
                    'handlers': ['console', 'file']
                },
                'super_resolution': {
                    'level': self.DEFAULT_LEVELS['super_resolution'].value,
                    'handlers': ['console', 'file', 'error_file'],
                    'propagate': False
                },
                'training': {
                    'level': 'INFO',
                    'handlers': ['console', 'file', 'training_file'],
                    'propagate': False
                },
                'inference': {
                    'level': 'INFO',
                    'handlers': ['console', 'file', 'inference_file'],
                    'propagate': False
                },
                'performance': {
                    'level': 'INFO',
                    'handlers': ['performance_file'],
                    'propagate': False
                }
            }
        }
        
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Set up logging handlers."""
        
        # Console handler
        self.config['handlers']['console'] = {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'colored' if platform.system() != "Windows" else 'simple',
            'stream': 'ext://sys.stdout'
        }
        
        # Main file handler
        self.config['handlers']['file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': LogLevel.DEBUG.value,
            'formatter': 'detailed',
            'filename': str(self.log_dir / f"{self.app_name}.log"),
            'maxBytes': 10 * 1024 * 1024,  # 10MB
            'backupCount': 5,
            'encoding': 'utf-8'
        }
        
        # Error file handler
        self.config['handlers']['error_file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'ERROR',
            'formatter': 'detailed',
            'filename': str(self.log_dir / f"{self.app_name}_errors.log"),
            'maxBytes': 5 * 1024 * 1024,  # 5MB
            'backupCount': 3,
            'encoding': 'utf-8'
        }
        
        # Training file handler
        self.config['handlers']['training_file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': LogLevel.DEBUG.value,
            'formatter': 'json',
            'filename': str(self.log_dir / f"{self.app_name}_training.log"),
            'maxBytes': 20 * 1024 * 1024,  # 20MB
            'backupCount': 5,
            'encoding': 'utf-8'
        }
        
        # Inference file handler
        self.config['handlers']['inference_file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'INFO',
            'formatter': 'json',
            'filename': str(self.log_dir / f"{self.app_name}_inference.log"),
            'maxBytes': 10 * 1024 * 1024,  # 10MB
            'backupCount': 3,
            'encoding': 'utf-8'
        }
        
        # Performance file handler
        self.config['handlers']['performance_file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'INFO',
            'formatter': 'json',
            'filename': str(self.log_dir / f"{self.app_name}_performance.log"),
            'maxBytes': 5 * 1024 * 1024,  # 5MB
            'backupCount': 2,
            'encoding': 'utf-8'
        }
    
    def setup_logging(self, config_dict: Optional[Dict[str, Any]] = None):
        """Set up logging with configuration."""
        if config_dict:
            # Override with provided config
            self.config.update(config_dict)
        
        # Apply configuration
        logging.config.dictConfig(self.config)
        
        # Log setup completion
        logger = logging.getLogger(self.app_name)
        logger.info(f"Logging system initialized for {self.app_name}")
        logger.info(f"Log directory: {self.log_dir}")
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger instance."""
        return logging.getLogger(name)
    
    def add_file_handler(self, name: str, filename: str, level: str = "INFO", 
                        formatter: str = "detailed", max_bytes: int = 10*1024*1024, 
                        backup_count: int = 3):
        """Add a custom file handler."""
        handler_config = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': level,
            'formatter': formatter,
            'filename': str(self.log_dir / filename),
            'maxBytes': max_bytes,
            'backupCount': backup_count,
            'encoding': 'utf-8'
        }
        
        self.config['handlers'][name] = handler_config
        
        # Reconfigure logging
        logging.config.dictConfig(self.config)
    
    def set_level(self, logger_name: str, level: str):
        """Set logging level for a specific logger."""
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, level.upper()))
    
    def enable_debug_mode(self):
        """Enable debug mode for all loggers."""
        for logger_name in self.config['loggers']:
            self.set_level(logger_name, LogLevel.DEBUG.value)
    
    def disable_debug_mode(self):
        """Disable debug mode for all loggers."""
        for logger_name in self.config['loggers']:
            if logger_name == 'super_resolution':
                self.set_level(logger_name, LogLevel.DEBUG.value)
            else:
                self.set_level(logger_name, LogLevel.INFO.value)
    
    def set_production_mode(self):
        """Set production logging levels."""
        for logger_name, level in self.PRODUCTION_LEVELS.items():
            if logger_name in self.config['loggers']:
                self.set_level(logger_name, level.value)
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up old log files."""
        cutoff_date = datetime.now().timestamp() - (days_to_keep * 24 * 3600)
        
        for log_file in self.log_dir.glob("*.log*"):
            if log_file.stat().st_mtime < cutoff_date:
                try:
                    log_file.unlink()
                    logging.getLogger(__name__).info(f"Cleaned up old log file: {log_file}")
                except OSError as e:
                    logging.getLogger(__name__).warning(f"Failed to clean up {log_file}: {e}")
    
    def create_logger(self, name: str, level: str = "INFO", handlers: list = None):
        """Create a new logger with custom configuration."""
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        
        if handlers:
            for handler_name in handlers:
                if handler_name in self.config['handlers']:
                    # Create handler from config
                    handler_config = self.config['handlers'][handler_name]
                    handler = self._create_handler_from_config(handler_config)
                    logger.addHandler(handler)
        
        return logger
    
    def _create_handler_from_config(self, config: Dict[str, Any]) -> logging.Handler:
        """Create handler from configuration dict."""
        handler_class = config['class']
        
        if handler_class == 'logging.StreamHandler':
            handler = logging.StreamHandler()
        elif handler_class == 'logging.handlers.RotatingFileHandler':
            handler = logging.handlers.RotatingFileHandler(
                filename=config['filename'],
                maxBytes=config['maxBytes'],
                backupCount=config['backupCount'],
                encoding=config['encoding']
            )
        else:
            raise ValueError(f"Unsupported handler class: {handler_class}")
        
        # Set level
        if 'level' in config:
            handler.setLevel(getattr(logging, config['level'].upper()))
        
        # Set formatter
        if 'formatter' in config:
            formatter_name = config['formatter']
            if formatter_name in self.config['formatters']:
                formatter_config = self.config['formatters'][formatter_name]
                
                if '()' in formatter_config:
                    # Custom formatter class
                    formatter_class = formatter_config['()']
                    formatter = formatter_class()
                else:
                    # Standard formatter
                    formatter = logging.Formatter(
                        formatter_config['format'],
                        datefmt=formatter_config.get('datefmt')
                    )
                
                handler.setFormatter(formatter)
        
        return handler


# Global logging configuration instance
logging_config = LoggingConfig()


def setup_logging(log_dir: str = "logs", app_name: str = "anime_sr", 
                 config_dict: Optional[Dict[str, Any]] = None):
    """Set up logging system."""
    global logging_config
    logging_config = LoggingConfig(log_dir, app_name)
    logging_config.setup_logging(config_dict)
    return logging_config


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging_config.get_logger(name)


def log_system_info():
    """Log system information."""
    logger = get_logger("system")
    
    import platform
    import torch
    import psutil
    
    logger.info("=" * 50)
    logger.info("SYSTEM INFORMATION")
    logger.info("=" * 50)
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"GPU Count: {torch.cuda.device_count()}")
        logger.info(f"Current GPU: {torch.cuda.current_device()}")
        logger.info(f"GPU Name: {torch.cuda.get_device_name()}")
    
    # System resources
    memory = psutil.virtual_memory()
    logger.info(f"Total Memory: {memory.total / 1024**3:.1f}GB")
    logger.info(f"Available Memory: {memory.available / 1024**3:.1f}GB")
    logger.info(f"CPU Count: {psutil.cpu_count()}")
    
    logger.info("=" * 50)


def log_model_info(model, model_name: str = "model"):
    """Log model information."""
    logger = get_logger("model")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model: {model_name}")
    logger.info(f"Total Parameters: {total_params:,}")
    logger.info(f"Trainable Parameters: {trainable_params:,}")
    logger.info(f"Model Size: {total_params * 4 / 1024**2:.1f}MB")
    
    # Log model structure if available
    if hasattr(model, 'scale'):
        logger.info(f"Scale Factor: {model.scale}")
    if hasattr(model, 'channels'):
        logger.info(f"Channels: {model.channels}")
    if hasattr(model, 'num_blocks'):
        logger.info(f"Number of Blocks: {model.num_blocks}")


def log_training_progress(epoch: int, total_epochs: int, metrics: Dict[str, float], 
                         stage: str = "training"):
    """Log training progress."""
    logger = get_logger("training")
    
    progress = (epoch / total_epochs) * 100
    
    message = f"[{stage.upper()}] Epoch {epoch}/{total_epochs} ({progress:.1f}%)"
    
    for metric_name, metric_value in metrics.items():
        message += f" | {metric_name}: {metric_value:.4f}"
    
    logger.info(message)


def log_inference_result(input_path: str, output_path: str, processing_time: float, 
                        metrics: Dict[str, float] = None):
    """Log inference results."""
    logger = get_logger("inference")
    
    message = f"Inference completed"
    message += f" | Input: {input_path}"
    message += f" | Output: {output_path}"
    message += f" | Time: {processing_time:.3f}s"
    
    if metrics:
        for metric_name, metric_value in metrics.items():
            message += f" | {metric_name}: {metric_value:.4f}"
    
    logger.info(message)


def log_error_with_context(error: Exception, context: Dict[str, Any], 
                         severity: str = "ERROR"):
    """Log error with context."""
    logger = get_logger("error")
    
    message = f"{severity}: {type(error).__name__}: {str(error)}"
    if context:
        message += f" | Context: {json.dumps(context, default=str)}"
    
    logger.error(message, exc_info=True)


def create_performance_logger(name: str) -> logging.Logger:
    """Create a performance-specific logger."""
    return logging_config.create_logger(
        name=f"performance.{name}",
        level="INFO",
        handlers=["performance_file"]
    )


# Initialize logging system
def init_logging():
    """Initialize the logging system."""
    setup_logging()
    log_system_info()


# Convenience functions
def debug(message: str, **kwargs):
    """Log debug message."""
    get_logger("debug").debug(message, extra=kwargs)


def info(message: str, **kwargs):
    """Log info message."""
    get_logger("info").info(message, extra=kwargs)


def warning(message: str, **kwargs):
    """Log warning message."""
    get_logger("warning").warning(message, extra=kwargs)


def error(message: str, **kwargs):
    """Log error message."""
    get_logger("error").error(message, extra=kwargs)


def critical(message: str, **kwargs):
    """Log critical message."""
    get_logger("critical").critical(message, extra=kwargs)
