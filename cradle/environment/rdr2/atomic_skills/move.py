from cradle.config import Config
from cradle.log import Logger
from cradle.gameio.io_env import IOEnvironment
from cradle.environment import post_skill_wait
from cradle.environment.rdr2.skill_registry import register_skill

config = Config()
logger = Logger()
io_env = IOEnvironment()


@register_skill("turn_and_move_forward")
def turn_and_move_forward(theta, duration):
    """
    First turns the in-game character left or right based on the specified theta angle and then moves the in-game character forward for the specified duration.

    Parameters:
    - theta: The angle for the turn. Use a negative value to turn left and a positive value to turn right.
    For example, if theta = 30, the character will turn right 30 degrees. If theta = -30, the character will turn left 30 degrees.
    - duration: The duration in seconds for which the character should move forward.
    """
    turn(theta)
    move_forward(duration)


@register_skill("turn")
def turn(theta):
    """
    Turns the in-game character left or right based on the specified theta angle.

    Parameters:
    - theta: The angle for the turn. Use a negative value to turn left and a positive value to turn right.
    For example, if theta = 30, the character will turn right 30 degrees. If theta = -30, the character will turn left 30 degrees.
    """
    io_env.mouse_move_horizontal_angle(theta)


@register_skill("move_forward")
def move_forward(duration):
    """
    Moves the in-game character forward for the specified duration.

    Parameters:
    - duration: The duration in seconds for which the character should move forward.
    """
    io_env.key_hold('w', duration)


@register_skill("mount_horse")
def mount_horse():
    """
    Needs to be close to the horse. Mounts the horse by pressing the "e" key.
    """
    io_env.key_press('e')

    post_skill_wait(config.DEFAULT_POST_ACTION_WAIT_TIME)


@register_skill("dismount_horse")
def dismount_horse():
    """
    Dismounts the horse by pressing the "e" key.
    """
    io_env.key_press('e')

    post_skill_wait(config.DEFAULT_POST_ACTION_WAIT_TIME)


@register_skill("stop_horse")
def stop_horse():
    """
    Stops the horse by pressing the "Ctrl" key.
    """
    io_env.key_press('ctrl', 0.5)


__all__ = [
    "turn",
    "move_forward",
    "turn_and_move_forward",
    "mount_horse",
    "dismount_horse",
    "stop_horse",
]
