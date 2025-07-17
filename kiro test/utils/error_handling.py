import logging
import traceback
from functools import wraps
from typing import Any, Callable, Dict, Optional
from datetime import datetime
import json

class PipelineError(Exception):
    """Base exception for pipeline errors"""
    def __init__(self, message: str, stage: str = "", details: Dict[str, Any] = None):
        self.message = message
        self.stage = stage
        self.details = details or {}
        self.timestamp = datetime.now()
        super().__init__(self.message)

class DatabaseError(PipelineError):
    """Database-related errors"""
    pass

class VectorizationError(PipelineError):
    """Vectorization-related errors"""
    pass

class GNNError(PipelineError):
    """GNN training-related errors"""
    pass

class RAGError(PipelineError):
    """RAG system-related errors"""
    pass

class OrderingError(PipelineError):
    """Ordering system-related errors"""
    pass

def error_handler(stage: str = "", reraise: bool = True):
    """Decorator for handling and logging errors"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            try:
                return func(*args, **kwargs)
            except PipelineError:
                # Re-raise pipeline errors as-is
                if reraise:
                    raise
                return None
            except Exception as e:
                error_msg = f"Error in {stage or func.__name__}: {str(e)}"
                logger.error(error_msg)
                logger.debug(traceback.format_exc())
                
                if reraise:
                    raise PipelineError(
                        message=error_msg,
                        stage=stage or func.__name__,
                        details={"original_error": str(e)}
                    )
                return None
        return wrapper
    return decorator

def safe_execute(func: Callable, *args, default_return=None, **kwargs):
    """Safely execute a function with error handling"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Safe execution failed for {func.__name__}: {e}")
        return default_return

class ErrorReporter:
    """Centralized error reporting and tracking"""
    
    def __init__(self):
        self.errors = []
        self.logger = logging.getLogger(__name__)
    
    def report_error(self, error: Exception, context: Dict[str, Any] = None):
        """Report an error with context"""
        error_record = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "message": str(error),
            "context": context or {},
            "traceback": traceback.format_exc()
        }
        
        self.errors.append(error_record)
        self.logger.error(f"Error reported: {error_record}")
        
        # Keep only last 100 errors
        if len(self.errors) > 100:
            self.errors = self.errors[-100:]
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recent errors"""
        if not self.errors:
            return {"total_errors": 0, "recent_errors": []}
        
        error_types = {}
        for error in self.errors:
            error_type = error["error_type"]
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            "total_errors": len(self.errors),
            "error_types": error_types,
            "recent_errors": self.errors[-5:],  # Last 5 errors
            "most_common_error": max(error_types.items(), key=lambda x: x[1])[0] if error_types else None
        }
    
    def export_errors(self, filepath: str):
        """Export errors to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.errors, f, indent=2)
        self.logger.info(f"Errors exported to {filepath}")

# Global error reporter
error_reporter = ErrorReporter()