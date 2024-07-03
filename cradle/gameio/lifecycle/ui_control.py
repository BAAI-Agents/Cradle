import time
from typing import Any, List, Tuple

from PIL import Image, ImageDraw, ImageFont
import mss
import mss.tools
import cv2
import numpy as np
import mss
import mss.tools

from cradle.config import Config
from cradle.gameio.gui_utils import TargetWindow, _get_active_window, check_window_conditions, is_top_level_window
from cradle.log import Logger
from cradle.gameio import IOEnvironment
from cradle.utils.image_utils import draw_mouse_pointer_file_, crop_grow_image
from cradle.utils.os_utils import getProcessIDByWindowHandle, getProcessNameByPID

config = Config()
logger = Logger()
io_env = IOEnvironment()


def switch_to_environment():

    named_windows = io_env.get_windows_by_config()

    if len(named_windows) == 0:
        error_msg = f"Cannot find the game window {config.env_name}!"
        logger.error(error_msg)
        raise EnvironmentError(error_msg)
    else:
        try:
            env_window = select_window(named_windows)
            env_window.activate()
            config.env_window = env_window
        except Exception as e:
            if "Error code from Windows: 0" in str(e):
                # Handle pygetwindow exception
                pass
            else:
                raise e

    time.sleep(1)


def select_window(named_windows: List[TargetWindow]) -> TargetWindow:

    if len(named_windows) == 1:
        return named_windows[0]
    elif len(named_windows) == 0:
        raise ValueError("No windows to select from.")

    env_window = None

    logger.warn(f'-----------------------------------------------------------------')
    candidate = named_windows[0]

    if candidate.left < 0 or candidate.top < 0:
        candidate = named_windows[-1]

    is_top, parent_handle = is_top_level_window(candidate.window._hWnd)
    if not is_top:
        for idx in range(len(named_windows)):
            if named_windows[idx].window._hWnd == parent_handle:
                logger.warn(f'Chosing parent window: index {idx}.')
                env_window = named_windows[idx]
                break

        if env_window is None:
            env_window = candidate  # Backup

    else:
        logger.warn(f'Cannot find unique env window: {config.env_name}|{config.win_name_pattern}. Using index 0.')
        env_window = candidate

    logger.warn(f'-----------------------------------------------------------------')
    return env_window


def check_active_window():
    result = False

    if config.env_window is not None:
        result = config.env_window.is_active()

    logger.debug(f"Active window check: {result}")

    if result == False:
        named_windows = io_env.get_windows_by_config()

        active_win = _get_active_window()

        # Workaround for dialogs until we can map sub-window to window/process
        dialog_names = ["Open", "Save", "Save As", "Confirm Save As", "Select a media resource"]

        # On Windows, dialogs belong to same process name, but different process ID, no parent window (0).

        if active_win.title in dialog_names:
             logger.debug(f"Dialog {active_win} is open and active.")
             result = True

        # Temporary hardcode due to CapCut behaviour of creating new windows under some actions
        if result == False and ("CapCut" in config.env_name or "XiuXiu" in config.env_name):

            x, y = config.env_window.left, config.env_window.top

            if len(named_windows) > 0:

                if len(named_windows) > 1:
                    for candidate in named_windows:
                        if candidate.window._hWnd == active_win._hWnd:

                            is_top, parent_handle = is_top_level_window(candidate.window._hWnd)
                            if not is_top:
                                for idx in range(len(named_windows)):
                                    if named_windows[idx].window._hWnd == parent_handle:
                                        logger.debug(f"Active window is child: {candidate.window._hWnd} of: {parent_handle}")
                                        candidate = named_windows[idx]
                                        break
                            else:
                                logger.debug(f"Active window is top: {candidate.window._hWnd}")

                            config.env_window = candidate
                            break
                else:
                    config.env_window = named_windows[0]

                check_window_conditions(config.env_window)
                config.env_window = config.env_window.moveTo(x, y)

                switch_to_environment()

                if "CapCut" in config.env_name:
                    from cradle.environment.capcut.atomic_skills import click_at_position
                    click_at_position(x=0.5, y=0.5, mouse_button="left")

                result = True
                logger.debug(f"Active window check after app-specific re-acquiring: {result}")

        # Check if it's a new window beloging to same original process
        if result == False:

            active_handle = active_win._hWnd
            env_handle = config.env_window.window._hWnd

            active_pid = getProcessIDByWindowHandle(active_handle)
            env_pid = getProcessIDByWindowHandle(env_handle)

            if active_pid == env_pid:
                logger.debug(f"Active window also belongs to env PID {env_pid}.")
                result = True
            else:
                active_proc_name = getProcessNameByPID(active_pid)
                env_proc_name = getProcessNameByPID(env_pid)
                if  active_proc_name == env_proc_name:
                    logger.debug(f"Active window also belongs to env proc_name {env_proc_name}.")
                    result = True
                else:
                    logger.warn(f"Active window does not belong to env PID {env_pid}. Check failed.")

    return result


def take_screenshot(tid: float,
                    screen_region: tuple[int, int, int, int] = None,
                    draw_axis: bool = False,
                    crop_border: bool = False) -> Tuple[str, str]:

    if screen_region is None:
        screen_region = config.env_region

    region = screen_region
    region = {
        "left": region[0],
        "top": region[1],
        "width": region[2],
        "height": region[3],
    }

    output_dir = config.work_dir

    # Save screenshots
    screen_image_filename = output_dir + "/screen_" + str(tid) + ".jpg"

    with mss.mss() as sct:
        screen_image = sct.grab(region)
        image = Image.frombytes("RGB", screen_image.size, screen_image.bgra, "raw", "BGRX")
        image.save(screen_image_filename)

    if draw_axis:
        # Draw axis on the screenshot
        draw = ImageDraw.Draw(screen_image)
        width, height = screen_image.size
        cx, cy = width // 2, height // 2

        draw.line((cx, 0, cx, height), fill="blue", width=3)  # Y
        draw.line((0, cy, width, cy), fill="blue", width=3)  # X

        font = ImageFont.truetype("arial.ttf", 30)
        offset_for_text = 30
        interval = 0.1

        for i in range(10):
            if i > 0:
                draw.text((cx + interval * (i ) * width // 2, cy), str(i ), fill="blue", font = font)
                draw.text((cx - interval * (i) * width // 2, cy), str(-i), fill="blue", font = font)
                draw.text((cx - offset_for_text - 10, cy + interval * (i ) * height // 2), str(-i), fill="blue", font = font)
            draw.text((cx - offset_for_text, cy - interval * (i ) * height // 2), str(i), fill="blue", font = font)

        axes_image_filename = output_dir + "/axes_screen_" + str(tid) + ".jpg"
        screen_image.save(axes_image_filename)

    if crop_border:
        screen_image_filename = crop_grow_image(screen_image_filename)

    return screen_image_filename, None


def normalize_coordinates(window_coorindates : tuple[int, int]) -> tuple[float, float]:
    x, y = window_coorindates
    default_resolution = config.DEFAULT_ENV_RESOLUTION
    x, y = x / default_resolution[0], y / default_resolution[1]
    return (x, y)


def draw_mouse_pointer_file(img_path: str, x: int, y: int) -> str:
    return draw_mouse_pointer_file_(img_path, x, y)


__all__ = [
    "switch_to_environment",
    "take_screenshot",
    "normalize_coordinates",
    "draw_mouse_pointer_file"
]
