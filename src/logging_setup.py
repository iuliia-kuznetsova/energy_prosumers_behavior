'''
    Logging setup for the project.
'''

# ---------- Imports ---------- #
import os
import logging
from pathlib import Path
from pythonjsonlogger import logger

# ---------- Logging setup ---------- #
def setup_logging(name: str):
    '''
        Setup logging for the project.
        
        Each module gets its own separate logger with its own handlers.
        Logs are written to logs/{name}.json
    '''
    # Always write logs to <project_root>/logs/
    project_root = Path(__file__).resolve().parent.parent
    logs_dir = project_root / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Get a unique logger for this module
    logger.setLevel(logging.INFO)
    
    # Prevent logs from propagating to root logger
    logger.propagate = False
    
    # Only add handlers if they haven't been added yet
    if not logger.handlers:
        # Terminal output
        terminal_handler = logging.StreamHandler()
        terminal_handler.setLevel(logging.INFO)
        terminal_formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
        terminal_handler.setFormatter(terminal_formatter)
        logger.addHandler(terminal_handler)

        # JSON file output
        log_file = logs_dir / f'logs_{name}.json'
        json_handler = logging.FileHandler(str(log_file), mode='a')
        json_handler.setLevel(logging.INFO)

        json_formatter = jsonlogger.JsonFormatter(
            fmt='%(asctime)s %(name)s %(levelname)s %(message)s',
            rename_fields={'asctime': 'timestamp', 'levelname': 'level', 'name': 'module'}
        )
        json_handler.setFormatter(json_formatter)
        logger.addHandler(json_handler)

        # Initial logging
        logger.info(f'Logging to {log_file} setup completed')

    return logger