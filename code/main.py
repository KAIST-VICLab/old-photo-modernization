import argparse
from src.config.parser import parse_config
from src.experiment import init_experiment
from src.engine import build_engine
import os


def is_file_path(string):
    if os.path.isfile(string):
        return string
    else:
        raise FileNotFoundError


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--config_path",
        type=is_file_path,
        required=True,
        help="Path to experiment config file (YAML)",
    )
    args = arg_parser.parse_args()
    global_config = parse_config(args.config_path)
    init_experiment(global_config)
    engine = build_engine(global_config)
    engine.run()


if __name__ == "__main__":
    main()
