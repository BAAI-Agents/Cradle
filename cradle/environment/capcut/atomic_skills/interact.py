from cradle.config import Config
from cradle.log import Logger
from cradle.gameio import IOEnvironment
from cradle.environment import post_skill_wait
from cradle.environment.capcut.skill_registry import register_skill

config = Config()
logger = Logger()
io_env = IOEnvironment()


@register_skill("click_at_position")
def click_at_position(x, y, mouse_button):
    """
    Moves the mouse to the specified x, y corrdinates inside the application window and clicks at that position.

    Parameters:
    - x: The normalized x-coordinate of the target position. The value should be between 0 and 1.
    - y: The normalized y-coordinate of the target position. The value should be between 0 and 1.
    - mouse_button: The mouse button to be clicked. It should be one of the following values: "left", "right", "middle".
    """
    io_env.mouse_move_normalized(x, y)
    io_env.mouse_click_button(mouse_button)

    post_skill_wait(config.DEFAULT_POST_ACTION_WAIT_TIME)


@register_skill("double_click_at_position")
def double_click_at_position(x, y, mouse_button):
    """
    Moves the mouse to the specified x, y corrdinates inside the application window and double clicks at that position.

    Parameters:
    - x: The normalized x-coordinate of the target position. The value should be between 0 and 1.
    - y: The normalized y-coordinate of the target position. The value should be between 0 and 1.
    - mouse_button: The mouse button to be clicked. It should be one of the following values: "left", "right", "middle".
    """
    io_env.mouse_move_normalized(x, y)
    io_env.mouse_click_button(mouse_button, duration= 0.01)
    io_env.mouse_click_button(mouse_button, duration= 0.01)

    post_skill_wait(config.DEFAULT_POST_ACTION_WAIT_TIME)


@register_skill("mouse_drag")
def mouse_drag(source_x, source_y, target_x, target_y, mouse_button):
    """
    Use the mouse to drag from a source position to a target position. The type of drag depends on the mouse button used.
    The mouse is moved to the source x, y position, the button is pressed, the mouse is moved to the target x, y position, and the button is released.

    Parameters:
    - source_x: The normalized x-coordinate of the source position. The value should be between 0 and 1.
    - source_y: The normalized y-coordinate of the soruce position. The value should be between 0 and 1.
    - target_x: The normalized x-coordinate of the target position. The value should be between 0 and 1.
    - target_y: The normalized y-coordinate of the target position. The value should be between 0 and 1.
    - mouse_button: The mouse button to be held during drag. It should be one of the following values: "left", "right", "middle".
    """
    delta = 0.002
    io_env.mouse_move_normalized(source_x, source_y)
    io_env.mouse_hold_button(mouse_button)
    io_env.mouse_move_normalized(target_x + delta, target_y + delta)  # Workaround for drag issue in some applications
    io_env.mouse_move_normalized(target_x, target_y)
    io_env.mouse_release_button(mouse_button)

    post_skill_wait(config.DEFAULT_POST_ACTION_WAIT_TIME)


@register_skill("press_key")
def press_key(key):
    """
    Pressing the selected key in the current situation.

    Parameters:
    - key: A keyboard key to be pressed. For example, press the 'enter' key.
    """
    io_env.key_press(key)


@register_skill("press_keys_combined")
def press_keys_combined(keys):
    """
    Presses the keys in the list combined. For example, when pressing the shortcut or hotkey combination 'ctrl, p'.

    Parameters:
    - keys: List of keys to press together at the same time. Either list of key names, or a string of comma-separated key names.
    """
    io_env.key_press(keys)


__all__ = [
    "click_at_position",
    "double_click_at_position",
    "mouse_drag",
    "press_key",
    "press_keys_combined",
]
