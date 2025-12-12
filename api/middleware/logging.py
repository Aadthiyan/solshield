#!/usr/bin/env python3
"""
Logging Middleware for FastAPI

This module provides comprehensive logging middleware for the API,
including request/response logging, error tracking, and performance monitoring.
"""

import time
import uuid
import logging
import structlog
from typing import Callable, Dict, Any
from fastapi import Request, Response
from fastapi.responses import JSONResponse
import json
from datetime import datetime
import traceback
import sys
from pathlib import Path

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

class APILogger:
    """Centralized API logger with structured logging"""
    
    def __init__(self, log_level: str = "INFO"):
        self.logger = structlog.get_logger("api")
        self.setup_file_logging()
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message, **kwargs)
    
    def setup_file_logging(self):
        """Setup file logging handlers"""
        # Create logs directory
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Setup file handlers
        file_handler = logging.FileHandler("logs/api.log")
        file_handler.setLevel(logging.INFO)
        
        error_handler = logging.FileHandler("logs/errors.log")
        error_handler.setLevel(logging.ERROR)
        
        # Setup formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        error_handler.setFormatter(formatter)
        
        # Add handlers to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        root_logger.addHandler(error_handler)
    
    def log_request(self, request: Request, request_id: str):
        """Log incoming request"""
        self.logger.info(
            "Request received",
            request_id=request_id,
            method=request.method,
            url=str(request.url),
            client_ip=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            content_type=request.headers.get("content-type")
        )
    
    def log_response(self, request_id: str, status_code: int, response_time: float, response_size: int = 0):
        """Log outgoing response"""
        self.logger.info(
            "Response sent",
            request_id=request_id,
            status_code=status_code,
            response_time=response_time,
            response_size=response_size
        )
    
    def log_error(self, request_id: str, error: Exception, context: Dict[str, Any] = None):
        """Log error with context"""
        self.logger.error(
            "Error occurred",
            request_id=request_id,
            error_type=type(error).__name__,
            error_message=str(error),
            traceback=traceback.format_exc(),
            context=context or {}
        )
    
    def log_performance(self, request_id: str, operation: str, duration: float, metadata: Dict[str, Any] = None):
        """Log performance metrics"""
        self.logger.info(
            "Performance metric",
            request_id=request_id,
            operation=operation,
            duration=duration,
            metadata=metadata or {}
        )

class LoggingMiddleware:
    """FastAPI middleware for request/response logging"""
    
    def __init__(self, app):
        self.app = app
        self.logger = APILogger()
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        request = Request(scope, receive)
        request_id = str(uuid.uuid4())
        
        # Add request ID to request state
        request.state.request_id = request_id
        
        # Log request
        self.logger.log_request(request, request_id)
        
        # Track active request
        performance_monitor.increment_active()
        
        # Track start time
        start_time = time.time()
        
        # Process request
        response_sent = False
        response_data = None
        
        async def send_wrapper(message):
            nonlocal response_sent, response_data
            
            if message["type"] == "http.response.start":
                # Log response start
                status_code = message["status"]
                response_time = time.time() - start_time
                
                self.logger.log_response(request_id, status_code, response_time)
                
                # Log performance metrics
                self.logger.log_performance(
                    request_id,
                    "request_processing",
                    response_time,
                    {"status_code": status_code}
                )
                
                response_sent = True
                
                # Record metrics
                performance_monitor.record_request(
                    response_time,
                    status_code < 400
                )
            
            elif message["type"] == "http.response.body":
                # Track response size
                if isinstance(response_data, dict):
                    response_size = len(json.dumps(response_data))
                else:
                    response_size = len(message.get("body", b""))
                
                self.logger.log_performance(
                    request_id,
                    "response_size",
                    response_size,
                    {"content_type": "application/json"}
                )
            
            await send(message)
        
        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as e:
            # Log error
            self.logger.log_error(request_id, e, {
                "method": request.method,
                "url": str(request.url),
                "client_ip": request.client.host if request.client else None
            })
            
            # Send error response
            error_response = JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "request_id": request_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            await error_response(scope, receive, send)
            
            # Record failed metric
            performance_monitor.record_request(0.0, False)
            
        finally:
            performance_monitor.decrement_active()

class ErrorHandler:
    """Centralized error handling for the API"""
    
    def __init__(self):
        self.logger = APILogger()
    
    def handle_validation_error(self, error, request_id: str) -> JSONResponse:
        """Handle Pydantic validation errors"""
        self.logger.log_error(request_id, error, {"error_type": "validation"})
        
        return JSONResponse(
            status_code=422,
            content={
                "error": "Validation error",
                "detail": str(error),
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    def handle_model_error(self, error, request_id: str) -> JSONResponse:
        """Handle model inference errors"""
        self.logger.log_error(request_id, error, {"error_type": "model_inference"})
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Model inference error",
                "detail": "Failed to analyze contract",
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    def handle_rate_limit_error(self, error, request_id: str) -> JSONResponse:
        """Handle rate limiting errors"""
        self.logger.log_error(request_id, error, {"error_type": "rate_limit"})
        
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "detail": "Too many requests",
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    def handle_generic_error(self, error, request_id: str) -> JSONResponse:
        """Handle generic errors"""
        self.logger.log_error(request_id, error, {"error_type": "generic"})
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": "An unexpected error occurred",
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

class PerformanceMonitor:
    """Performance monitoring for API endpoints"""
    
    def __init__(self):
        self.logger = APILogger()
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "max_response_time": 0.0,
            "min_response_time": float('inf'),
            "active_requests": 0
        }
        self.response_times = []
    
    def increment_active(self):
        """Increment active requests counter"""
        self.metrics["active_requests"] += 1
        
    def decrement_active(self):
        """Decrement active requests counter"""
        if self.metrics["active_requests"] > 0:
            self.metrics["active_requests"] -= 1
    
    def record_request(self, response_time: float, success: bool):
        """Record request metrics"""
        self.metrics["total_requests"] += 1
        
        if success:
            self.metrics["successful_requests"] += 1
        else:
            self.metrics["failed_requests"] += 1
        
        # Update response time metrics
        self.response_times.append(response_time)
        self.metrics["max_response_time"] = max(self.metrics["max_response_time"], response_time)
        self.metrics["min_response_time"] = min(self.metrics["min_response_time"], response_time)
        
        # Calculate average
        self.metrics["average_response_time"] = sum(self.response_times) / len(self.response_times)
        
        # Keep only last 1000 response times for memory efficiency
        if len(self.response_times) > 1000:
            self.response_times = self.response_times[-1000:]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.metrics.copy()
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "max_response_time": 0.0,
            "min_response_time": float('inf')
        }
        self.response_times = []

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

class RequestContext:
    """Request context manager for tracking request state"""
    
    def __init__(self, request_id: str):
        self.request_id = request_id
        self.start_time = time.time()
        self.logger = APILogger()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if exc_type:
            self.logger.log_error(self.request_id, exc_val, {
                "duration": duration,
                "exception_type": exc_type.__name__
            })
        else:
            self.logger.log_performance(
                self.request_id,
                "request_completed",
                duration
            )
    
    def log_operation(self, operation: str, duration: float, metadata: Dict[str, Any] = None):
        """Log a specific operation"""
        self.logger.log_performance(
            self.request_id,
            operation,
            duration,
            metadata
        )

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration for the API"""
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/api.log")
        ]
    )
    
    # Configure specific loggers
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("api").setLevel(logging.DEBUG)
    
    return APILogger(log_level)
