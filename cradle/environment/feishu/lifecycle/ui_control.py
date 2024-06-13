from typing import Any

from cradle.config import Config
from cradle.log import Logger
from cradle.gameio import IOEnvironment

config = Config()
logger = Logger()
io_env = IOEnvironment()

PAUSE_SCREEN_WAIT = 1


def pause_game():
    logger.warn("Pause doesn't apply to application!")


def unpause_game():
    logger.warn("Unpause doesn't apply to application!")


def is_env_paused():

    is_paused = False

    logger.warn("Pause doesn't apply to application!")

    return is_paused


__all__ = [
    "pause_game",
    "unpause_game",
]