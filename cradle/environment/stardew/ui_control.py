import time

from PIL import Image
import mss

from cradle.config import Config
from cradle.log import Logger
from cradle.gameio.io_env import IOEnvironment
from cradle import constants
from cradle.environment import UIControl


config = Config()
logger = Logger()
io_env = IOEnvironment()

class StardewUIControl(UIControl):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def pause_game(self, env_name: str, ide_name: str) -> None:
        if ide_name:
            ide_window = io_env.get_windows_by_name(ide_name)[0]
            ide_window.activate()
            ide_window.show()
        time.sleep(0.5)


    def unpause_game(self, env_name: str, ide_name: str) -> None:
        target_window = io_env.get_windows_by_name(config.env_name)[0]
        if self.is_env_paused():
            target_window.activate()  # Activate the game window to unpause the game
        else:
            logger.debug("The environment is not paused!")
        io_env.mouse_move(1890, 950)
        time.sleep(0.5)


    def switch_to_game(self, env_name: str, ide_name: str) -> None:

        target_window = io_env.get_windows_by_name(config.env_name)[0]
        try:
            target_window.activate()
        except Exception as e:
            if "Error code from Windows: 0" in str(e):
                # Handle pygetwindow exception
                pass
            else:
                raise e
        time.sleep(1)


    def exit_back_to_pause(self, env_name: str, ide_name: str) -> None:
        max_steps = 10

        back_steps = 0
        while not self.is_env_paused() and back_steps < max_steps:
            back_steps += 1
            self.pause_game(env_name, ide_name)
            time.sleep(constants.PAUSE_SCREEN_WAIT)

        if back_steps >= max_steps:
            logger.warn("The environment fails to pause!")


    def exit_back_to_game(self, env_name: str, ide_name: str) -> None:

        self.exit_back_to_pause(env_name, ide_name)

        # Unpause the game, to keep the rest of the agent flow consistent
        self.unpause_game(env_name, ide_name)


    def is_env_paused(self) -> bool:
        target_window = io_env.get_windows_by_name(config.env_name)[0]
        is_active = target_window.is_active()
        return not is_active


    def take_screenshot(self,
                        tid: float,
                        screen_region: tuple[int, int, int, int] = None) -> str:

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

        return screen_image_filename
