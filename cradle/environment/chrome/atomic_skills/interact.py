from cradle.config import Config
from cradle.gameio.lifecycle.ui_control import switch_to_environment
from cradle.log import Logger
from cradle.gameio import IOEnvironment
from cradle.environment import post_skill_wait
from cradle.environment.chrome.skill_registry import register_skill

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
    Presses the keys in the list combined. For example, when pressing the shortcut or hotkey combination 'ctrl' + P.

    Parameters:
    - keys: List of keys to press together at the same time. Either list of key names, or a string of comma-separated key names.
    """
    io_env.key_press(keys)


@register_skill("type_text")
def type_text(text):
    """
    Types the specified text using the keyboard. One character at a time.

    Parameters:
    - text: The text to be typed into the current UI control.
    """
    io_env.keys_type(text)

    post_skill_wait(config.DEFAULT_POST_ACTION_WAIT_TIME)


@register_skill("return_to_previous_page")
def return_to_previous_page():
    """
    Return to the previous page.
    """

    io_env.key_press("alt, left")


@register_skill("go_back_to_target_application")
def go_back_to_target_application():
    """
    This function can be used to return to the target application, if some previous action opened a different application.
    """
    switch_to_environment()

    post_skill_wait(config.DEFAULT_POST_ACTION_WAIT_TIME)
    post_skill_wait(config.DEFAULT_POST_ACTION_WAIT_TIME)


__all__ = [
    "go_back_to_target_application",
    "click_at_position",
    "press_key",
    "press_keys_combined",
    "type_text",
    "return_to_previous_page",
]
