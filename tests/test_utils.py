import logging
from pathlib import Path
from utils import parse_args


def test_parse_args_defaults(tmp_path):
    input_file = tmp_path / "image.jpg"
    input_file.write_text("data")
    output_file = tmp_path / "out.xml"
    args = parse_args([str(input_file), str(output_file)])

    assert args.input_path == input_file
    assert args.output_path == output_file
    assert args.log_file == input_file.with_suffix(".ir.log")
    assert args.log_level_console == "INFO"
    assert args.log_level_console_int == logging.INFO
    assert args.log_level_file == "DEBUG"
    assert args.log_level_file_int == logging.DEBUG


def test_parse_args_quiet(tmp_path):
    input_file = tmp_path / "image.jpg"
    input_file.write_text("data")
    output_file = tmp_path / "out.xml"
    args = parse_args([str(input_file), str(output_file), "--quiet"])

    assert args.log_level_console == "WARNING"
    assert args.log_level_console_int == logging.WARNING


def test_parse_args_directory_input(tmp_path):
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    output_dir = tmp_path / "out"
    args = parse_args([str(input_dir), str(output_dir)])

    assert args.log_file is None
