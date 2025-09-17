import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import json
import os

from .config import settings

class StructuredFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if hasattr(record, 'extra_data'):
            log_entry.update(record.extra_data)
        
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)

class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, '')
        reset_color = self.RESET if log_color else ''
        
        formatter = f"{log_color}%(asctime)s - %(name)s - %(levelname)s - %(message)s{reset_color}"
        return logging.Formatter(formatter).format(record)

def setup_logging(log_level: Optional[str] = None, log_file: Optional[str] = None):
    if log_level is None:
        log_level = settings.LOG_LEVEL
    
    if log_file is None:
        log_file = settings.LOG_FILE
    
    # Create logs directory if it doesn't exist
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = ColoredFormatter()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler with structured JSON output
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = StructuredFormatter()
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Error file handler for warnings and above
    error_log_file = log_file.replace('.log', '_errors.log')
    error_handler = logging.FileHandler(error_log_file)
    error_handler.setLevel(logging.WARNING)
    error_formatter = StructuredFormatter()
    error_handler.setFormatter(error_formatter)
    logger.addHandler(error_handler)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

def log_interaction(interaction_type: str, user_input: str, ai_response: str, 
                   session_id: str, additional_data: Optional[dict] = None):
    logger = get_logger("interaction_logger")
    
    log_data = {
        "interaction_type": interaction_type,
        "user_input": user_input[:500],  # Truncate long inputs
        "ai_response": ai_response[:500],  # Truncate long responses
        "session_id": session_id,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if additional_data:
        log_data.update(additional_data)
    
    logger.info("Interaction logged", extra={"extra_data": log_data})

def log_evaluation(question: str, answer: str, score: float, 
                  evaluation_details: dict, session_id: str):
    logger = get_logger("evaluation_logger")
    
    log_data = {
        "question": question[:200],
        "answer": answer[:500],
        "score": score,
        "evaluation_details": evaluation_details,
        "session_id": session_id,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    logger.info("Evaluation completed", extra={"extra_data": log_data})

def log_error(error_type: str, error_message: str, context: dict, 
             session_id: str, traceback: Optional[str] = None):
    logger = get_logger("error_logger")
    
    log_data = {
        "error_type": error_type,
        "error_message": error_message,
        "context": context,
        "session_id": session_id,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if traceback:
        log_data["traceback"] = traceback
    
    logger.error("Error occurred", extra={"extra_data": log_data}, exc_info=True)

def log_performance(operation: str, duration: float, success: bool, 
                     additional_metrics: Optional[dict] = None):
    logger = get_logger("performance_logger")
    
    log_data = {
        "operation": operation,
        "duration_seconds": duration,
        "success": success,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if additional_metrics:
        log_data.update(additional_metrics)
    
    if success:
        logger.info("Performance metric", extra={"extra_data": log_data})
    else:
        logger.warning("Performance issue", extra={"extra_data": log_data})

# Initialize logging on module import
setup_logging()