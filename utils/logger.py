import logging
from pathlib import Path
import sys
import io

def get_logger(log_path):
    logger = logging.getLogger("tamper_detection")
    if logger.handlers:
        return logger  # 已初始化过则直接返回，避免重复添加 handler

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # 确保日志目录存在
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    # 文件处理器，写入 utf-8
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    # 控制台处理器：包装 stdout，遇到无法编码字符时用替代（避免抛出 UnicodeEncodeError）
    try:
        stream = io.TextIOWrapper(sys.stdout.buffer, encoding=sys.stdout.encoding or "utf-8", errors="replace")
    except Exception:
        stream = sys.stdout
    console_handler = logging.StreamHandler(stream)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger