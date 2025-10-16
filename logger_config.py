# logger_config.py
import logging

class NewlineFormatter(logging.Formatter):
    def format(self, record):
        record.newline = '\n'
        return super().format(record)

def get_logger(name: str) -> logging.Logger:
    formatter = NewlineFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(funcName)s]%(newline)s'
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if not logger.hasHandlers():
        logger.addHandler(handler)
    return logger