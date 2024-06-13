from cradle.config import Config
from cradle.gameio.io_env import IOEnvironment
from cradle.environment.rdr2.skill_registry import register_skill

config = Config()
io_env = IOEnvironment()


@register_skill("aim")
def aim():
    """
    Aim the weapon in the game.
    """
    io_env.mouse_hold_button(button=io_env.RIGHT_MOUSE_BUTTON)


@register_skill("select_weapon")
def select_weapon(x, y):
    """
    Move the mouse to a specific location to select the weapon in the game.
    Parameters:
    - x: The normalized abscissa of the pixel.
    - y: The normalized ordinate of the pixel.
    """
    io_env.mouse_move_normalized(x, y)


@register_skill("select_sidearm")
def select_sidearm(x, y):
    """
    Move the mouse to a specific location to select the sidearm in the game.
    Parameters:
    - x: The normalized abscissa of the pixel.
    - y: The normalized ordinate of the pixel.
    """
    select_weapon(x, y)


@register_skill("shoot")
def shoot(x = None, y = None):
    """
    Shoot the weapon at a specific location in view.
    Parameters:
    - x: The normalized abscissa of the pixel.
    - y: The normalized ordinate of the pixel.
    """
    if x is not None and y is not None:
        io_env.mouse_move_normalized(x=x, y=y, relative=True, from_center = True)
    io_env.mouse_click_button(button=io_env.LEFT_MOUSE_BUTTON, clicks=2, duration=0.1)


@register_skill("view_weapons")
def view_weapons():
    """
    View the weapon wheel.
    """
    io_env.key_hold('tab')


@register_skill("fight")
def fight():
    """
    Fight agains another person.
    """
    io_env.key_press('f,f,f,f,f,f')


# def call_animals():
#     """
#     Call animals in the game.
#     """
#     pyautogui.mouseDown(button="right")
#     pydirectinput.keyDown("r")
#     time.sleep(0.5)
#     pydirectinput.keyUp("r")
#     pyautogui.mouseUp(button="right")


__all__ = [
    "aim",
    "shoot",
    "select_weapon",
    "select_sidearm",
    "fight",
    #"call_animals",
]
