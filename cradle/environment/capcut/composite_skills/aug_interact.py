from cradle.config import Config
from cradle.log import Logger
from cradle.gameio import IOEnvironment
from cradle.environment.capcut.skill_registry import register_skill
from cradle.environment.capcut.atomic_skills.interact import click_at_position

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


@register_skill("double_click_on_label")
def double_click_on_label(label_id, mouse_button):
    """
    Moves the mouse to the position of the specified box id inside the application window and double clicks. Use case: Add a file to the media.

    Parameters:
    - label_id: The numerical label id of the bounding box to click at.
    - mouse_button: The mouse button to be clicked. It should be one of the following values: "left", "right", "middle".
    """
    label_id = str(label_id)
    x, y = 0.5, 0.5
    click_at_position(x, y, mouse_button)


@register_skill("mouse_drag_with_label")
def mouse_drag_with_label(source_label_id, target_label_id, mouse_button):
    """
    Drag from the source label id of bounding box to the target label id of bounding box.

    Parameters:
    - source_label_id: The numerical label id of the source bounding box to drag from.
    - target_label_id: The numerical label id of the target bounding box to drag to.
    - mouse_button: The mouse button to be clicked. It should be one of the following values: "left", "right", "middle".
    """
    source_label_id = str(source_label_id)
    target_label_id = str(target_label_id)
    source_x, source_y = 0.5, 0.5
    target_x, target_y = 0.5, 0.5
    click_at_position(source_x, source_y, "left")
    click_at_position(target_x, target_y, "left")


__all__ = [
    "click_on_label",
    "double_click_on_label",
    "mouse_drag_with_label"
]
