from cradle.config import Config
from cradle.log import Logger
from cradle.gameio.io_env import IOEnvironment
from cradle.environment import post_skill_wait
from cradle.environment.rdr2.skill_registry import register_skill

config = Config()
logger = Logger()
io_env = IOEnvironment()


@register_skill("open_map")
def open_map():
    """
    Opens the in-game map.
    """

    logger.write("Running open_map()")

    io_env.key_press('m')

    post_skill_wait(config.DEFAULT_POST_ACTION_WAIT_TIME)


@register_skill("add_mark")
def add_mark():
    """
    Marks the current mouse position on the map by pressing "z".
    A red message indicating the mark will appear on the map.
    Clicks the Cancel message if it appears.
    """
    io_env.key_press('z')


@register_skill("add_waypoint")
def add_waypoint():
    """
    Creates a waypoint at the item selected in the opened map index, by pressing "enter".
    Waypoint creation displays the path to the target location.
    """

    logger.write("Running add_waypoint()")

    io_env.key_press('enter')


@register_skill("close_map")
def close_map():
    """
    Closes the in-game map by pressing "esc".
    """
    io_env.key_press('esc')

    post_skill_wait(config.DEFAULT_POST_ACTION_WAIT_TIME)


@register_skill("open_index")
def open_index():
    """
    Opens the map index by pressing the "space" key, after the map is open.
    """

    logger.write("Running open_index()")

    io_env.key_press('space')

    post_skill_wait(config.DEFAULT_POST_ACTION_WAIT_TIME)


@register_skill("close_index")
def close_index():
    """
    Closes the game index by pressing the "space" key.
    """
    io_env.key_press('space')

    post_skill_wait(config.DEFAULT_POST_ACTION_WAIT_TIME)


@register_skill("select_previous_index_object")
def select_previous_index_object():
    """
    When the index is opened, moves to the previous index selection by pressing the "up" arrow key.
    Items of interest may be out of view, so this skill is useful for scrolling through the index.
    """
    io_env.key_press('up')


@register_skill("select_next_index_object")
def select_next_index_object():
    """
    When the index is opened, moves to the next index selection by pressing the "down" arrow key.
    Items of interest may be out of view, so this skill is useful for scrolling through the index.
    """
    io_env.key_press('down')


__all__ = [
    "open_map",
    "add_waypoint",
    "close_map",
    "open_index",
    "close_index",
    "select_previous_index_object",
    "select_next_index_object",
]
