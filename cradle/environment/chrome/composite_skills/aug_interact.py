from cradle.config import Config
from cradle.log import Logger
from cradle.gameio import IOEnvironment
from cradle.environment.chrome.skill_registry import register_skill
from cradle.environment.chrome.atomic_skills.interact import click_at_position

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


__all__ = [
    "click_on_label",
]
