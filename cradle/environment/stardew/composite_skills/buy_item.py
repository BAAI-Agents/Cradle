import time
import os
from typing import Tuple

import numpy as np
import cv2
import mss
from PIL import Image

import traceback

from cradle.environment.stardew.skill_registry import register_skill
from cradle.config import Config
from cradle.log import Logger
from cradle.gameio.io_env import IOEnvironment
from cradle import constants
from cradle.utils.template_matching import match_template_image

config = Config()
logger = Logger()
io_env = IOEnvironment()


@register_skill("buy_item")
def buy_item(item_name: str = "Parsnip Seeds",
             item_count: int = 10,
             ):
    """
    Buy items in Pierre's store.
    Note: This function only takes effect when the character is standing in front of the counter.
    
    Parameters:
    - item_name: The name of the item to be bought at the Pierre's General Store.
    - item_count: The number of items to be bought at the Pierre's General Store.
    """
    cv_buy_item(item_name=item_name, item_count=item_count)


"""
 It opens up the shopping interface at Pierre's General Store when the character is standing in front of the cyan counter. It then buys the selected item of the specified quantity at the Pierre's General Store at the store's interface. Four items should be shown at this interface in four columns. This function automates the action of buying the selected item of the specified quantity at the Pierre's General Store by template matching of the item icon in the store interface. If a match of item icon is found at the store interface, the function will click on the item icon to buy the item. The function will repeat the action of buying the item for the specified quantity. After buying the item, the function will exit the store interface by pressing the 'esc' key twice.

 Parameters:
    - item_name: The name of the item to be bought at the Pierre's General Store.
    - item_count: The number of items to be bought at the Pierre's General Store.
"""

def cv_open_shop_interface():
    # press 'x' to open the shop interface when standing in front of the shop counter
    io_env.key_press(constants.KEY_X)


def index_of_item_in_matched_template(objects_list, item_name) -> int:
    """
    Args:
        objects_list: The list of objects detected in the image by template matching
        item_name: The name of the item to be found in the list of objects detected

    Returns: The index of the item in the list of objects detected with the highest confidence
    """
    index = -1
    max_confidence = .0
    for obj in objects_list:
        if obj["name"] == item_name:
            if obj["confidence"] > max_confidence:
                max_confidence = obj["confidence"]
                index = objects_list.index(obj)
    return index


def cv_template_matching(image_file_name: str, item_shop_icon_path: str, item_name: str,
                         debug=False):
    item_index_in_template = dict()

    # match the item in the shop
    # returned is a list of dicts with the following
    # {"type","name","bounding_box","reasoning","value","confidence"}
    objects_list = match_template_image(image_file_name, item_shop_icon_path, debug=debug)

    # guard clause for no objects detected
    if not objects_list:
        logger.write(
            f"No objects detected for buying in shop from template matching. File path: {item_shop_icon_path}")
        return None

    # find index of item in the matched template
    index = index_of_item_in_matched_template(objects_list, item_name)

    # guard clause for no target item found in the detected objects
    if index not in range(len(objects_list)):
        logger.write(
            f"No item \"{item_name}\" found among the detected objects from template matching.")
        return None

    else:
        item_index_in_template[item_name] = objects_list[index]
        logger.write(f"Item \"{item_name}\" found in the shop.")

        # return the bounding box of the item
        return objects_list[index]["bounding_box"]


def take_screenshot():
    with mss.mss() as sct:
        current_time = time.time()
        region = config.env_region
        screen_image = sct.grab(region)
        image = Image.frombytes("RGB", screen_image.size, screen_image.bgra, "raw", "BGRX")
        image_name = f'screen_buy_item_interact_{current_time}.jpg'
        output_dir = config.work_dir
        image_name = os.path.join(output_dir, image_name)
        image.save(image_name)

        config.update_current_game_screenshot_path(image_name)

def cv_buy_item(item_name: str = "Parsnip Seeds",
                item_count: int = 10,
                debug: bool = False,
                *arg, **kwargs):
    try:
        # open the shop interface
        cv_open_shop_interface()
        # replace space in item_name with underscore and convert to lower case
        item_icon_file_name = item_name.replace(" ", "_").lower()
        # append _icon to the end of the item_icon_file_name
        item_icon_file_name += "_icon"
        # match the item
        item_shop_icon_path = f'./res/{config.env_sub_path}/icons/shop_interface/{item_icon_file_name}.png'

        take_screenshot()

        image_path = config.current_game_screenshot_path

        item_bounding_box = cv_template_matching(image_path, item_shop_icon_path, item_icon_file_name)

        if item_bounding_box:
            iterate_buy_item(item_bounding_box, item_name=item_name, item_count=item_count, debug=debug)
    except Exception as e:
        # printing stack trace
        if debug:
            traceback.print_exc()
            raise e


def iterate_buy_item(item_bounding_box: Tuple[int, int, int, int],
                     item_name: str = "Parsnip Seeds",
                     item_count: int = 10,
                     debug: bool = False,
                     *arg, **kwargs):
    # find center of the item bounding box
    item_center = {'x': item_bounding_box[0] + item_bounding_box[2] // 2,
                   'y': item_bounding_box[1] + item_bounding_box[3] // 2}

    # click the item_center to buy
    for i in range(item_count):
        mouse_left_click(**item_center)
        time.sleep(1)

        if debug:
            logger.write(f"Buy {item_name} {i + 1}/{item_count}")

    # press 'esc' twice to exit the shop
    # The first esc to cancel item selection
    io_env.key_press(constants.KEY_ESC)
    time.sleep(1)
    # The second esc to exit the shop interface
    io_env.key_press(constants.KEY_ESC)
    time.sleep(1)


def mouse_left_click(x, y, duration=0.1):
    io_env.mouse_move(x, y)
    io_env.mouse_click_button("left mouse button", duration=duration)


__all__ = [
    "buy_item",
]