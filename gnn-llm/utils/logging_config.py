import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path

def setup_logging(log_level: str = "INFO", log_file: str = "logs/pipeline.log"):
    """Setup comprehensive logging configuration"""
    
    # Create logs directory if it doesn't exist
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # Error file handler
    error_log_file = str(log_dir / "errors.log")
    error_handler = logging.handlers.RotatingFileHandler(
        error_log_file,
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(error_handler)
    
    # Set specific logger levels
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('selenium').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    
    logging.info(f"Logging configured - Level: {log_level}, File: {log_file}")

class PipelineLogger:
    """Specialized logger for pipeline operations"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.start_time = None
        self.stage_times = {}
    
    def start_stage(self, stage_name: str):
        """Start timing a pipeline stage"""
        self.start_time = datetime.now()
        self.logger.info(f"Starting stage: {stage_name}")
    
    def end_stage(self, stage_name: str, success: bool = True, details: str = ""):
        """End timing a pipeline stage"""
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            self.stage_times[stage_name] = duration
            
            status = "completed" if success else "failed"
            log_msg = f"Stage '{stage_name}' {status} in {duration:.2f}s"
            if details:
                log_msg += f" - {details}"
            
            if success:
                self.logger.info(log_msg)
            else:
                self.logger.error(log_msg)
        
        self.start_time = None
    
    def log_metrics(self, metrics: dict, stage: str = ""):
        """Log metrics in a structured format"""
        prefix = f"[{stage}] " if stage else ""
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.logger.info(f"{prefix}{key}: {value}")
            else:
                self.logger.info(f"{prefix}{key}: {str(value)}")
    
    def get_timing_summary(self) -> str:
        """Get summary of stage timings"""
        if not self.stage_times:
            return "No timing data available"
        
        total_time = sum(self.stage_times.values())
        summary = f"Total pipeline time: {total_time:.2f}s\n"
        
        for stage, duration in self.stage_times.items():
            percentage = (duration / total_time) * 100
            summary += f"  {stage}: {duration:.2f}s ({percentage:.1f}%)\n"
        
        return summary.strip()