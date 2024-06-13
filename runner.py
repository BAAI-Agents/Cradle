import argparse
import importlib

from cradle.config import Config
from cradle.log import Logger

config = Config()
logger = Logger()

def main(args):

    entry = None

    if "skylines" in config.env_short_name.lower():
        runner_module = importlib.import_module('cradle.runner.skylines_runner')
        entry = getattr(runner_module, 'entry')

    elif "dealers" in config.env_short_name.lower():
        runner_module = importlib.import_module('cradle.runner.dealers_runner')
        entry = getattr(runner_module, 'entry')

    elif "rdr2" in config.env_short_name.lower():
        runner_module = importlib.import_module('cradle.runner.rdr2_runner')
        entry = getattr(runner_module, 'entry')

    elif "stardew" in config.env_short_name.lower():
        runner_module = importlib.import_module('cradle.runner.stardew_runner')
        entry = getattr(runner_module, 'entry')

    assert entry is not None, "Entry function is not defined in the environment module."

    # Run the entry
    entry(args)

def get_args_parser():
    parser = argparse.ArgumentParser("Cradle Agent")
    parser.add_argument("--llmProviderConfig", type=str, default="./conf/openai_config.json", help="The path to the provider config file")
    parser.add_argument("--embedProviderConfig", type=str, default="./conf/openai_config.json", help="The path to the provider config file")
    parser.add_argument("--envConfig", type=str, default="./conf/env_config_dealers.json", help="The path to the environment config file")
    return parser

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    config.load_env_config(args.envConfig)
    config.set_fixed_seed()

    main(args)