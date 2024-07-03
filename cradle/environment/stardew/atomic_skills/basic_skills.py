import re
import time
from cradle.config import Config
from cradle.log import Logger
from cradle.gameio.io_env import IOEnvironment
from cradle.environment.stardew.skill_registry import register_skill

config = Config()
logger = Logger()
io_env = IOEnvironment()

@register_skill("mouse_check_do_action")
def mouse_check_do_action(x, y, duration=0.1):
    """
    Engages with the game element located at the coordinates (x, y) by executing a click using the right mouse button. This function is essential for interacting with objects within the game world, whether it be to open chests, interact with NPCs, or navigate through game menus. The precise execution of the double right-click mimics player actions, offering an automated yet natural interaction within the game environment.
    Note: x and y must be in the scope of the 8 grids around the player.

    Parameters:
     - x: The X-coordinate on the screen representing the horizontal axis where the double right-click is to be performed, directly correlating to the game's graphical interface.
     - y: The Y-coordinate on the screen representing the vertical axis, pinpointing the exact location for the interaction. This ensures that actions are carried out on the intended game element without misplacement or error.
     - duration: The duration for which the right mouse button is held down (default is 0.1 second).
    """
    io_env.mouse_move(x, y)
    io_env.mouse_click_button("right mouse button", duration = duration)

@register_skill("mouse_use_tool")
def mouse_use_tool(x, y, duration=0.1):
    """
    Adjusts the player's orientation towards the specified position (x, y) on the screen, then simulates the action of using or interacting with tools in hand or objects within the game or application. This is achieved by moving the mouse to the specified location and performing a single left mouse click.
    Notes:
    1. This function is specifically designed for scenarios in games or applications where direct mouse interactions are necessary to use tools or interact with objects.
    2. A brief delay follows the click action to ensure the intended interaction is properly registered by the game or application.
    3. The parameters x and y must represent positions that are within the 8 adjacent grids surrounding the player, ensuring the action is contextually relevant and possible within the game's mechanics.

    Parameters:
     - x: The X coordinate on the screen to which the mouse will be moved, reflecting the target location for interaction. Coordinates should be within the scope of the 8 grids surrounding the player.
     - y: The Y coordinate on the screen to which the mouse will be moved, reflecting the target location for interaction. Coordinates should be within the scope of the 8 grids surrounding the player.
     - duration: The duration for which the left mouse button is held down (default is 0.1 second).
    """
    io_env.mouse_move(x, y)
    io_env.mouse_click_button("left mouse button", duration = duration)

@register_skill("do_action")
def do_action():
    """
    The function is designed to perform a generic action on objects or characters within one body length of the player character. This could include planting a seed if a seed is selected, harvesting a plant, interacting with other characters, entering doors, opening boxes, or picking up objects. The action is context-specific and depends on what the player is close to in the game environment. This function is essential for progressing through the game, completing quests, and engaging with various interactive elements in the game world, but is limited to only affecting things in very close proximity to the player character.
    """
    io_env.key_press("x")

@register_skill("use_tool")
def use_tool():
    """
    Executes an in-game action commonly assigned to using the character's current selected tool. According to the selected tool, this action can range from chopping wood using an axe, digging and til soil using a hoe, watering crops using a watering can, breaking stones using a pickaxe, or cutting grass into hay using a scythe. The use of tools is essential for various activities in the game, such as farming, mining, and combat, making this function a versatile and crucial skill for efficient gameplay.
    """
    io_env.key_press("c")

@register_skill("open_menu")
def open_menu():
    """
    Opening the menu.
    """
    io_env.key_press("esc")

@register_skill("close_menu")
def close_menu():
    """
    Closing the menu.
    """
    io_env.key_press("esc")

@register_skill("open_journal")
def open_journal():
    """
    Opening the journal.
    """
    io_env.key_press("f")

@register_skill("close_journal")
def close_journal():
    """
    Closing the journal.
    """
    io_env.key_press("f")

@register_skill("open_map")
def open_map():
    """
    Opening the map.
    """
    io_env.key_press("m")

@register_skill("close_map")
def close_map():
    """
    Closing the map.
    """
    io_env.key_press("m")

@register_skill("open_chatbox")
def open_chatbox():
    """
    Opening the chatbox.
    """
    io_env.key_press("t")

@register_skill("close_chatbox")
def close_chatbox():
    """
    Closing the chatbox.
    """
    io_env.key_press("t")

@register_skill("move_up")
def move_up(duration=0.1):
    """
    Moves the character upward (north) by pressing the 'w' key for the specified duration. This action simulates the character moving up on the game grid, allowing for precise control over the character's position and orientation.
    Note: The movement distance is influenced by the duration of the key press, with a typical rate of approximately 1 grid space per 0.1 seconds of key press. Understanding this relationship is essential for strategic navigation and precise positioning within the game environment.

    Parameters:
     - duration: The duration in seconds for which the 'w' key should be pressed, determining the distance the character will move forward (default is 0.1 second).
    """
    io_env.key_press("w", duration)

@register_skill("move_down")
def move_down(duration=0.1):
    """
    Moves the character downward (south) by pressing the 's' key for the specified duration. This action simulates the character moving down on the game grid, allowing for precise control over the character's position and orientation.
    Note: The movement distance is influenced by the duration of the key press, with a typical rate of approximately 1 grid space per 0.1 seconds of key press. Understanding this relationship is essential for strategic navigation and precise positioning within the game environment.

    Parameters:
     - duration: The duration in seconds for which the 's' key should be pressed, determining the distance the character will move backward (default is 0.1 second).
    """
    io_env.key_press("s", duration)

@register_skill("move_left")
def move_left(duration=0.1):
    """
    Moves the character to the left (west) by pressing the 'a' key for the specified duration. This action simulates the character moving left on the game grid, allowing for precise control over the character's position and orientation.
    Note: The movement distance is influenced by the duration of the key press, with a typical rate of approximately 1 grid space per 0.1 seconds of key press. Understanding this relationship is essential for strategic navigation and precise positioning within the game environment (default is 0.1 second).

    Parameters:
     - duration: The duration in seconds for which the 'a' key should be pressed, determining the distance the character will move to the left (default is 0.1 second).
    """
    io_env.key_press("a", duration)

@register_skill("move_right")
def move_right(duration=0.1):
    """
    Moves the character to the right (east) by pressing the 'd' key for the specified duration. This action simulates the character moving right on the game grid, allowing for precise control over the character's position and orientation.
    Note: The movement distance is influenced by the duration of the key press, with a typical rate of approximately 1 grid space per 0.1 seconds of key press. Understanding this relationship is essential for strategic navigation and precise positioning within the game environment (default is 0.1 second).

    Parameters:
     - duration: The duration in seconds for which the 'd' key should be pressed, determining the distance the character will move to the right (default is 0.1 second).
    """
    io_env.key_press("d", duration)

@register_skill("select_tool")
def select_tool(key):
    """
    Selects a specific tool from the in-game toolbar based on the given tool number. Each tool serves a distinct purpose essential for managing your farm and exploring the world. This function allows for the quick selection of tools, crucial for efficient gameplay during various in-game activities such as farming, mining, or combat.
    Note: Ensure the tool number is within the valid range to prevent errors. This function is essential for efficiently managing tool use in various game scenarios, enabling the player to swiftly switch between tools as the situation demands.

    Parameters:
     - key: A key representing the position of the tool in the toolbar. The value must be in ["1","2","3","4","5","6","7","8","9","0","-","+"], inclusive.
    """
    regex_pattern = r"[0-9\-+]"
    if re.match(regex_pattern, str(key)):
        io_env.key_press(str(key))
    else:
        raise ValueError("Invalid key in select_tool. Key must be in the range [0-9,-,+]")


@register_skill("shift_toolbar")
def shift_toolbar():
    """
    Cycles through the toolbar slots in the game by pressing the "Tab" key. This action allows the player to switch between different sets of tools or items quickly without navigating the inventory screen. Each press of the "Tab" key moves the selection to the next toolbar slot, making it efficient for changing equipment or items during gameplay.
    Note: The number of toolbar slots and the cycle order depend on the game settings and modifications (if any). Ensure to familiarize yourself with the toolbar configuration for optimal use of this skill.
    """
    io_env.key_press("tab")

__all__ = [
    # "mouse_check_do_action",
    # "mouse_use_tool",
    "do_action",
    "use_tool",
    # "access_setting",
    "open_menu",
    "close_menu",
    "open_journal",
    "close_journal",
    "open_map",
    "close_map",
    # "open_chatbox",
    # "close_chatbox",
    "move_up",
    "move_down",
    "move_left",
    "move_right",
    # "move",
    # "walk",
    "select_tool",
    # "shift_toolbar"
]