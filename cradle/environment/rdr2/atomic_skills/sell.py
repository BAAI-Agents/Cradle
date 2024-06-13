from cradle.gameio.io_env import IOEnvironment
from cradle.environment.rdr2.skill_registry import register_skill

io_env = IOEnvironment()


@register_skill("sell_product")
def sell_product(duration=1):
    """
    Opens the trade bar for selling products.
    Note: it must run the shopkeeper_interaction function before running this function

    Parameters:
     - duration: The duration for which the "r" key is held down (default is 1 second).
    """
    io_env.key_hold('r', duration)


@register_skill("sell_single_product_all_quantity")
def sell_single_product_all_quantity(duration=1.0):
    """
    Presses and holds the "f" key to sell the quantity of one unit of the current product.
    Note: The product quantity must be greater than one.

    Parameters:
     - duration: The duration for which the "f" key is held down in seconds (default is 1.0 second).
    """
    io_env.key_hold('f', duration)


@register_skill("sell_one_product")
def sell_one_product():
    """
    Presses "enter" to sell one unit of the current product.
    """
    io_env.key_press('enter')


@register_skill("switch_to_next_product_type")
def switch_to_next_product_type():
    """
    Presses "e" to switch to the next product type.
    """
    io_env.key_press('e')


@register_skill("switch_to_previous_product_type")
def switch_to_previous_product_type():
    """
    Presses "q" to switch to the previous product type.
    """
    io_env.key_press('q')


__all__ = []
