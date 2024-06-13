from cradle.config import Config
from cradle.gameio.io_env import IOEnvironment
from cradle.environment import post_skill_wait
from cradle.environment.rdr2.skill_registry import register_skill

config = Config()
io_env = IOEnvironment()


@register_skill("shopkeeper_interaction")
def shopkeeper_interaction():
    """
    Initiates interaction with the shopkeeper by long-pressing the right mouse button.
    This action opens the transaction menu.
    Note: The transaction type must be determined and the interaction closed afterward.
    """
    io_env.mouse_hold_button(button=io_env.RIGHT_MOUSE_BUTTON)

    post_skill_wait(config.DEFAULT_POST_ACTION_WAIT_TIME)


@register_skill("cancel_shopkeeper_interaction")
def cancel_shopkeeper_interaction():
    """
    Cancels the interaction with the shopkeeper by releasing the right mouse button.
    """
    io_env.mouse_release_button(button=io_env.RIGHT_MOUSE_BUTTON)

    post_skill_wait(config.DEFAULT_POST_ACTION_WAIT_TIME)


@register_skill("mouse_select_item")
def mouse_select_item(x, y):
    """
    Move the mouse to a specific location to select the item in the game.
    Parameters:
    - x: The normalized abscissa of the pixel.
    - y: The normalized ordinate of the pixel.
    """
    io_env.mouse_move_normalized(x, y)


@register_skill("mouse_confirm_item")
def mouse_confirm_item():
    """
    Confirms the selection item by clicking the left mouse button once.
    """
    io_env.mouse_click_button(button=io_env.LEFT_MOUSE_BUTTON)


@register_skill("go_back")
def go_back():
    """
    Returns to the upper level by pressing the "esc" key.
    """
    io_env.key_press('esc')

    post_skill_wait(config.DEFAULT_POST_ACTION_WAIT_TIME)


@register_skill("select_upside_product")
def select_upside_product():
    """
    This function simulates the action of selecting the product on the next upside of the current selected product.
    It uses the pydirectinput library to press the "up" key.
    """
    io_env.key_press('up')


@register_skill("select_downside_product")
def select_downside_product():
    """
    This function simulates the action of selecting the product on the next downside of the current selected product.
    It uses the pydirectinput library to press the "down" key.
    """
    io_env.key_press('down')


@register_skill("select_leftside_product")
def select_leftside_product():
    """
    This function simulates the action of selecting the product on the next leftside of the current selected product.
    It uses the pydirectinput library to press the "left" key.
    """
    io_env.key_press('left')


@register_skill("select_rightside_product")
def select_rightside_product():
    """
    This function simulates the action of selecting the product on the next rightside of the current selected product.
    It uses the pydirectinput library to press the "right" key.
    """
    io_env.key_press('right')


@register_skill("select_next_product")
def select_next_product():
    """
    This function simulates the action of selecting the next product of the current selected product.
    It uses the pydirectinput library to press the "right" key.
    """
    io_env.key_press('right')


__all__ = [
    "shopkeeper_interaction",
    "cancel_shopkeeper_interaction",
    "select_upside_product",
    "select_downside_product",
    "select_leftside_product",
    "select_rightside_product",
]
