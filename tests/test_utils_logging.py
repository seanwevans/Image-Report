import logging
from utils import logger, setup_logging


def test_setup_logging_clears_handlers(tmp_path):
    logger.handlers.clear()
    log_file = tmp_path / "log.txt"
    setup_logging(log_file=log_file)
    first = len(logger.handlers)
    setup_logging(log_file=log_file)
    second = len(logger.handlers)
    assert first == second == 2
