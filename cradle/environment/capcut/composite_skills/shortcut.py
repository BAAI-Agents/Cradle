from cradle.config import Config
from cradle.log import Logger
from cradle.gameio import IOEnvironment
from cradle.environment import post_skill_wait
from cradle.environment.capcut.skill_registry import register_skill
from cradle.environment.capcut.atomic_skills import click_at_position

config = Config()
logger = Logger()
io_env = IOEnvironment()


@register_skill("delete_right")
def delete_right():
    """
    Delete the right of choosen media.
    """

    io_env.key_press("w")
    post_skill_wait(config.DEFAULT_POST_ACTION_WAIT_TIME)


@register_skill("delete_left")
def delete_left():
    """
    Delete the contents before the timestamp/position in the media timeline.
    """

    io_env.key_press("q")
    post_skill_wait(config.DEFAULT_POST_ACTION_WAIT_TIME)


@register_skill("go_to_timestamp")
def go_to_timestamp(seconds = 1):
    """
    Go to the specific timestamp in the timeline (in seconds).

    Parameters:
    - seconds: The timestamp to reach, in seconds. The default value is 1.
    """

    io_env.keys_type(['home'])
    post_skill_wait(config.DEFAULT_POST_ACTION_WAIT_TIME)

    second_in_presses = 3 # 3 shift+left action needed per second timestamp

    for i in range(int(seconds*second_in_presses)):
        io_env.key_press("shift, right")
        post_skill_wait(config.DEFAULT_POST_ACTION_WAIT_TIME/10)


@register_skill("import_media")
def import_media():
    """
    Import media into the timeline.
    """

    io_env.key_press("ctrl, i")
    post_skill_wait(config.DEFAULT_POST_ACTION_WAIT_TIME)


@register_skill("export_project")
def export_project():
    """
    Export the project. Then go to homepage of CapCut.
    """

    io_env.key_press("ctrl, e")
    post_skill_wait(config.DEFAULT_POST_ACTION_WAIT_TIME*2)

    io_env.key_press("enter")
    post_skill_wait(config.DEFAULT_POST_ACTION_WAIT_TIME*2)

    click_at_position(x=0.5, y=0.5, mouse_button="left")

    io_env.key_press("ctrl, w")
    post_skill_wait(config.DEFAULT_POST_ACTION_WAIT_TIME)

    io_env.key_press("ctrl, w")
    post_skill_wait(config.DEFAULT_POST_ACTION_WAIT_TIME)


@register_skill("create_new_project")
def create_new_project():
    """
    Create a new project.
    """

    io_env.key_press("ctrl, n")
    post_skill_wait(config.DEFAULT_POST_ACTION_WAIT_TIME)


@register_skill("switch_material_panel")
def switch_material_panel(times = 1):
    """
    Switch the material panel. The material panel includes the Media, Audio, Text, Stickers, Effects, Transitions, Filters, Adjustments, and Templates tabs. Choose this skill to switch between these tabs one by one.

    Parameters:
    - times: The number of times to switch the material panel. The default value is 1.
    """

    for _ in range(times):
        io_env.key_press("tab")
        post_skill_wait(config.DEFAULT_POST_ACTION_WAIT_TIME)


@register_skill("press_enter")
def press_enter():
    """
    Press the enter key.
    """

    io_env.key_press("enter")
    post_skill_wait(config.DEFAULT_POST_ACTION_WAIT_TIME)


@register_skill("close_window")
def close_window():
    """
    Close the current window.
    """

    io_env.key_press("ctrl, w")
    post_skill_wait(config.DEFAULT_POST_ACTION_WAIT_TIME)


__all__ = [
    "delete_right",
    "delete_left",
    "go_to_timestamp",
    "import_media",
    "export_project",
    "create_new_project",
    "switch_material_panel",
    "press_enter",
    "close_window",
]
