import pytest
from utils import parse_args


def test_nms_threshold_valid():
    args = parse_args(["input.jpg", "output.xml", "--nms-threshold", "0.5"])
    assert args.nms_threshold == 0.5


def test_nms_threshold_invalid():
    with pytest.raises(SystemExit):
        parse_args(["input.jpg", "output.xml", "--nms-threshold", "1.5"])

