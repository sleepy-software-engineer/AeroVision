import logging
from datetime import datetime
from logging import StreamHandler


class ColorFormatter(logging.Formatter):
    COLOR_CODES = {
        "INFO": "\033[34m",
        "DEBUG": "\033[33m",
        "CRITICAL": "\033[31m",
        "RESET": "\033[0m",
    }

    def format(self, record: logging.LogRecord) -> str:
        log_time = datetime.now().strftime("%H:%M:%S")
        level_name = record.levelname
        color = self.COLOR_CODES.get(level_name, self.COLOR_CODES["RESET"])
        reset = self.COLOR_CODES["RESET"]
        formatted_message = f"{log_time} [{level_name}] {record.getMessage()}"
        return f"{color}{formatted_message}{reset}"


class LoggerFactory:
    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        handler = StreamHandler()
        handler.setLevel(logging.DEBUG)
        formatter = ColorFormatter()
        handler.setFormatter(formatter)
        if not logger.handlers:
            logger.addHandler(handler)
        return logger


if __name__ == "__main__":
    logger = LoggerFactory.get_logger("example")
    logger.info("This is an info message.")
    logger.debug("This is a debug message.")
    logger.critical("This is a critical message.")
