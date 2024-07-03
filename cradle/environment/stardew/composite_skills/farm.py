import time
from cradle.environment.stardew.atomic_skills.basic_skills import (
    use_tool,
    do_action,
    move_up,
    move_down,
    move_left,
    move_right,
    select_tool,
    mouse_check_do_action,
)
from cradle.environment.stardew.skill_registry import register_skill

@register_skill("get_out_of_house")
def get_out_of_house():
    """
    Move the character out of the house. This function automates the action of moving the character out of the house by navigating through the door.
    Note: This function only takes effect when the character is inside the house and in bed.
    """
    move_left(duration=1.1)
    time.sleep(0.1)
    move_down(duration=0.5)
    time.sleep(2)
    move_down(duration=0.4)

@register_skill("enter_door_and_sleep")
def enter_door_and_sleep():
    """
    Let the character enter the house and then move the character to the bed and interact with it to go to sleep. This function automates the action of moving the character to the bed and interacting with it to go to sleep.
    """
    move_up(duration=1)
    do_action()
    time.sleep(2)

    move_up(duration=0.2)
    time.sleep(0.1)
    move_right(duration=2)
    time.sleep(0.5)
    mouse_check_do_action(400, 920)
    time.sleep(5)

@register_skill("use_tool_multiple_times")
def use_tool_multiple_times():
    """
    Use the current selected tool with the 3X3 size of area in front of the character. This function automates the action of using selected tool in a grid pattern, starting from the right grid next to the character, which serves as the left upper corner of the specified area size. This fuction will move the charater to each of the grid in the area and use the tool at that grid.
    Note: This function is useful for actions that require using a tool several times, such as hoeing the soil in a 3X3 area, watering plants in a 3X3 area, or harvesting crops in a 3X3 area. Before using this function, make sure to select the tool that you want to use.
    """
    move_and_action_like(action=use_tool, size=(3, 3))

@register_skill("do_action_multiple_times")
def do_action_multiple_times():
    """
    Execute do_action() for each grid in the 3X3 area in front of the character. This function automates the action of executing do_action() in a grid pattern, starting from the right grid next to the character, which serves as the left upper corner of the specified area size. This fuction will move the charater to each of the grid in the area and execute do_action() at that grid.
    Note: This function is useful for actions that require being executed several times, especially for planting seeds in the 3X3 area in front of the character. Before using this function, make sure to select the object that you want to use.
    """
    move_and_action_like(action=do_action, size=(3, 3))

SLEEPTIME = 0.5
MOVE_SINGLE_GRID = 0.116

def move_and_action_like(action, size=(3, 3)):
    """
    Helper function to move the character back to the initial position after completing the action across the specified area. This function calculates the necessary movements based on the area's size.
    Note: This internal function supports the 'move_and_action_like' by encapsulating the logic for returning to the starting position, making the main function cleaner and more focused on its primary task.

    Parameters:
    - action: The action function to be performed, such as 'keyboard_use_tool'.
    - size: The size of the area covered, used to determine the necessary movements to return to start.
    """
    assert len(size) <= 2
    if len(size) == 1:
        size = (size[0], size[0])
    for row in range(size[0]):
        if row % 2 == 0:
            move = move_right
            move_ = move_left
        else:
            move = move_left
            move_ = move_right
        for col in range(size[1]):
            action()
            time.sleep(SLEEPTIME)
            move(MOVE_SINGLE_GRID)
            time.sleep(SLEEPTIME)
        move(0.15)
        move_down(0.1)
        move_(0.005)
        time.sleep(SLEEPTIME)
    # go back to where it started
    if size[0] % 2 == 0:
        for i in range(size[0]):
            move_up(0.1)
            time.sleep(SLEEPTIME)
        move_left(0.1)
        move_right(0.02)
    else:
        for i in range(size[0]):
            move_up(0.1)
            time.sleep(SLEEPTIME)
        for i in range(size[1]):
            move_left(MOVE_SINGLE_GRID)
            time.sleep(SLEEPTIME)
        move_left(0.15)
        move_right(0.01)

__all__ = [
    "use_tool_multiple_times",
    "do_action_multiple_times",
    "get_out_of_house",
    "enter_door_and_sleep",
]