from cradle.config import Config
from cradle.log import Logger
from cradle.gameio import IOEnvironment
from cradle.environment import post_skill_wait
from cradle.environment.outlook.skill_registry import register_skill
from cradle.environment.outlook.atomic_skills.interact import click_at_position, move_mouse_to_position

config = Config()
logger = Logger()
io_env = IOEnvironment()


@register_skill("click_on_label")
def click_on_label(label_id, mouse_button):
    """
    Moves the mouse to the position of the specified box id inside the application window and clicks.

    Parameters:
    - label_id: The numerical label id of the bounding box to click at.
    - mouse_button: The mouse button to be clicked. It should be one of the following values: "left", "right", "middle".
    """
    label_id = str(label_id)
    x, y = 0.5, 0.5
    click_at_position(x, y, mouse_button)


@register_skill("hover_on_label")
def hover_on_label(label_id):
    """
    Moves the mouse to the position of the specified box id inside the application window, to hover over the UI item without clicking on it.

    Parameters:
    - label_id: The numerical label id of the bounding box to click at.
    """
    label_id = str(label_id)
    x, y = 0.5, 0.5
    move_mouse_to_position(x, y)


@register_skill("go_to_mail_view")
def go_to_mail_view():
    """
    Go to the mail view. Useful when you want to navigate to the mail view from another application view.
    """
    view_button_position = (24/config.DEFAULT_ENV_RESOLUTION[0], 68/config.DEFAULT_ENV_RESOLUTION[1])
    click_at_position(view_button_position[0], view_button_position[1], io_env.LEFT_MOUSE_BUTTON)
    post_skill_wait(config.DEFAULT_POST_ACTION_WAIT_TIME)


@register_skill("go_to_calendar_view")
def go_to_calendar_view():
    """
    Go to the calendar view. Useful when you want to navigate to the calendar view from another application view.
    """
    view_button_position = (24/config.DEFAULT_ENV_RESOLUTION[0], 116/config.DEFAULT_ENV_RESOLUTION[1])
    click_at_position(view_button_position[0], view_button_position[1], io_env.LEFT_MOUSE_BUTTON)
    post_skill_wait(config.DEFAULT_POST_ACTION_WAIT_TIME)


__all__ = [
    "click_on_label",
    "hover_on_label",
    "go_to_mail_view",
    "go_to_calendar_view",
]
