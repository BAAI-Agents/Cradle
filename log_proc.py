import argparse
import os

from cradle.log.logger import process_log_messages
from cradle.config import Config
from cradle.utils.file_utils import get_latest_directories_in_path

config = Config()


# Converts the latest log to markdown file for easy visualization, unless a specific log is specified
def main(args):

    path = args.logDir
    if path is None:
        run_dir = os.path.dirname(config.work_dir)
        path = get_latest_directories_in_path(run_dir, 2)[1]

    log = process_log_messages(path)

    with open(path + '/logs/cradle_log.md', 'w', encoding="utf-8") as f:
        f.write(log)


def get_args_parser():
    parser = argparse.ArgumentParser("Cradle Log Parser to Markdown")
    parser.add_argument("--logDir", type=str, default=None, help="The path to the log directory")
    return parser


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
