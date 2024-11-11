import logging
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from typing import Optional

class Logger:
    def __init__(self, name: str, log_dir: str = "logs", use_tensorboard: bool = True):
        # Create log directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(log_dir, f"{name}_{timestamp}")
        os.makedirs(self.log_dir, exist_ok=True)

        # Set up file logging
        log_file = os.path.join(self.log_dir, "run.log")
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # TensorBoard setup
        self.writer = SummaryWriter(log_dir=self.log_dir) if use_tensorboard else None

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value to TensorBoard"""
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def log_histogram(self, tag: str, values, step: int):
        """Log a histogram to TensorBoard"""
        if self.writer:
            self.writer.add_histogram(tag, values, step)

    def log_figure(self, tag: str, figure, step: int):
        """Log a matplotlib figure to TensorBoard"""
        if self.writer:
            self.writer.add_figure(tag, figure, step)

    def close(self):
        if self.writer:
            self.writer.close() 