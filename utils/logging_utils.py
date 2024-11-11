import os
from datetime import datetime
import logging
import matplotlib.pyplot as plt
from typing import Any, Dict
import json

class Logger:
    def __init__(self, name: str, log_dir: str = "logs"):
        self.name = name
        
        # Create log directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(log_dir, f"{name}_{timestamp}")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(os.path.join(self.log_dir, "training.log"))
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        # Metrics storage
        self.metrics: Dict[str, Dict[int, float]] = {}

    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)

    def log_scalar(self, name: str, value: float, step: int):
        """Log a scalar value"""
        if name not in self.metrics:
            self.metrics[name] = {}
        self.metrics[name][step] = value
        
        # Save metrics to file
        metrics_file = os.path.join(self.log_dir, "metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f)

    def log_figure(self, name: str, figure: plt.Figure, step: int):
        """Save a matplotlib figure"""
        figure_dir = os.path.join(self.log_dir, "figures")
        os.makedirs(figure_dir, exist_ok=True)
        figure.savefig(os.path.join(figure_dir, f"{name}_{step}.png"))

    def close(self):
        """Close the logger"""
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler) 