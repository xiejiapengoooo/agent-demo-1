import logging

logging.basicConfig(
    format="%(name)s %(levelname)-8s %(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_logger(label: str) -> logging.Logger:
    logger = logging.getLogger(f"[{label}]")
    logger.setLevel(logging.DEBUG)
    return logger


__all__ = ["get_logger"]
