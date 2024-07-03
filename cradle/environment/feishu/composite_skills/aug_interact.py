from cradle.config import Config
from cradle.log import Logger
from cradle.gameio import IOEnvironment
from cradle.environment.feishu.skill_registry import register_skill
from cradle.environment.feishu.atomic_skills.interact import click_at_position, move_mouse_to_position

config = Config()
logger = Logger()
io_env = IOEnvironment()


@register_skill("click_on_label")
def click_on_label(label, mouse_button):
    """
    Moves the mouse to the position of the specified box id inside the application window and clicks.

    Parameters:
    - label: The numerical label id of the bounding box to click at.
    - mouse_button: The mouse button to be clicked. It should be one of the following values: "left", "right", "middle".
    """
    label = str(label)
    x, y = 0.5, 0.5
    click_at_position(x, y, mouse_button)


@register_skill("hover_on_label")
def hover_on_label(label):
    """
    Moves the mouse to the position of the specified box id inside the application window, to hover over the UI item without clicking on it.

    Parameters:
    - label: The numerical label id of the bounding box to click at.
    """
    label = str(label)
    x, y = 0.5, 0.5
    move_mouse_to_position(x, y)


__all__ = [
    "click_on_label",
    "hover_on_label"
]
