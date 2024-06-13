from cradle.config import Config
from cradle.log import Logger
from cradle.gameio.io_env import IOEnvironment
from cradle.environment.dealers.skill_registry import register_skill

config = Config()
logger = Logger()
io_env = IOEnvironment()


@register_skill("dialogue")
def dialogue():
    """
    The function to click on to choose the option of the dialog to make the game going on.
    """
    io_env.mouse_move(280, 475)
    io_env.mouse_click_button("left", duration=0.2, clicks=1)


@register_skill("close_description_page")
def close_description_page():
    """
    The function to close a description page showing information about the item details, daily stats, or the traits of the buyer or seller.
    """
    io_env.mouse_move(1745, 340)
    io_env.mouse_click_button("left", duration=0.2, clicks=1)

# @register_skill("close_item_description")
# def close_item_description():
#     """
#     The function to close the item description dialog that is opened.
#     """
#     io_env.mouse_move(1745, 340)
#     io_env.mouse_click_button("left", duration=0.1, clicks=1)
#
#
# @register_skill("close_daily_stats")
# def close_daily_stats():
#     """
#     The function to close the daily stats summary page that is opened.
#     """
#     io_env.mouse_move(1745, 340)
#     io_env.mouse_click_button("left", duration=0.1, clicks=1)
#
#
# @register_skill("close_customer_description")
# def close_customer_description():
#     """
#     The function to close the customer description page that is opened.
#     """
#     io_env.mouse_move(1745, 340)
#     io_env.mouse_click_button("left", duration=0.1, clicks=1)
#

@register_skill("accept_deal")
def accept_deal():
    """
    The function to click on the check mark to accept the deal on the confirmation dialog.
    """
    io_env.mouse_move(1400, 950)
    io_env.mouse_click_button("left", duration=0.2, clicks=1)


@register_skill("reject_deal")
def reject_deal():
    """
    The function to click on the cross mark to reject the deal on the confirmation dialog.
    """
    io_env.mouse_move(1600, 950)
    io_env.mouse_click_button("left", duration=0.2, clicks=1)


@register_skill("finish_deal")
def finish_deal():
    """
    The function to click on the ok button to finish the deal on the confirmation dialog noticed with "Deal!".
    """
    io_env.mouse_move(960, 680)
    io_env.mouse_click_button("left", duration=0.2, clicks=1)


@register_skill("finish_sold")
def finish_sold():
    """
    The function to click on the ok button to finish the selling on the confirmation dialog noticed with "Sold!".
    """
    io_env.mouse_move(960, 740)
    io_env.mouse_click_button("left", duration=0.2, clicks=1)


@register_skill("skip_warning")
def skip_warning():
    """
    The function to skip the popped up warning dialog noticed with "Warning".
    """
    io_env.mouse_move(724, 670)
    io_env.mouse_click_button("left", duration=0.2, clicks=1)

@register_skill("open_shop")
def open_shop():
    """
    The function to open the dealer's shop to start dealing for today. This should only be used at the interface of day start, where there is a piece of newspaper titled "Daily News" on the screen.
    """
    io_env.mouse_move(300, 500)
    io_env.mouse_click_button("left", duration=0.2, clicks=1)


@register_skill("give_price")
def give_price(price):
    """
    The function to give a price for the item in the deal. The price must be an integer number.

    Parameters:
     - price: a price for the item. It must be a number.
    """
    io_env.mouse_move(1447, 790)
    io_env.mouse_click_button("left", duration=0.2, clicks=1)

    # Convert price to string and type each character using the keyboard
    price_str = str(price)
    for char in price_str:
        io_env.key_press(char)

    # Optionally, press Enter if needed to submit the price
    io_env.key_press('enter')


__all__ = [
    # dialog handling
    "dialogue",

    # description handling (desc is on the tablet)
    # "close_item_description",
    # "close_daily_stats",
    # "close_customer_description",
    "close_description_page",

    # deal confirmation and finish
    "accept_deal",
    "reject_deal",
    "finish_deal",
    "finish_sold",
    "skip_warning",

    # open shop
    "open_shop",

    # give price
    "give_price"
]
