import logging
from pathlib import Path


def get_logger(log_path):
    logger = logging.getLogger("tamper_detection")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # 文件处理器
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger