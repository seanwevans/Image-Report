import logging
from utils import setup_logging

logger = logging.getLogger(__name__)


def test_setup_logging_clears_handlers(tmp_path):
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    log_file = tmp_path / "log.txt"
    setup_logging(log_file=log_file)
    first = len(root_logger.handlers)
    setup_logging(log_file=log_file)
    second = len(root_logger.handlers)
    assert first == second == 2
