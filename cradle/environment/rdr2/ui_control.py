import os
import time

from PIL import Image
import mss

from cradle.config import Config
from cradle.log import Logger
from cradle.gameio.io_env import IOEnvironment
from cradle.utils.template_matching import match_template_image
from cradle import constants
from cradle.environment import UIControl

config = Config()
logger = Logger()
io_env = IOEnvironment()


class RDR2UIControl(UIControl):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def pause_game(self, env_name: str, ide_name: str) -> None:
        if not self.is_env_paused():
            io_env.handle_hold_in_pause()

            io_env.key_press('esc')
            time.sleep(constants.PAUSE_SCREEN_WAIT)
        else:
            logger.debug("The environment does not need to be paused!")

        # While game is paused, quickly re-center mouse location on x axis to avoid clipping at game window border with time
        io_env.mouse_move(config.env_resolution[0] // 2, config.env_resolution[1] // 2, relative=False)


    def unpause_game(self, env_name: str, ide_name: str) -> None:
        if self.is_env_paused():
            io_env.key_press('esc', 0)
            time.sleep(constants.PAUSE_SCREEN_WAIT)

            io_env.handle_hold_in_unpause()
        else:
            logger.debug("The environment is not paused!")


    def switch_to_game(self, env_name: str, ide_name: str) -> None:

        named_windows = io_env.get_windows_by_name(env_name)
        if len(named_windows) == 0:
            logger.error(f"Cannot find the game window {env_name}!")
            return
        else:
            try:
                named_windows[0].activate()
            except Exception as e:
                if "Error code from Windows: 0" in str(e):
                    # Handle pygetwindow exception
                    pass
                else:
                    raise e

        time.sleep(1)
        self.unpause_game(env_name, ide_name)
        time.sleep(1)


    def exit_back_to_pause(self, env_name: str, ide_name: str) -> None:

        max_steps = 10

        back_steps = 0
        while not self.is_env_paused() and back_steps < max_steps:
            back_steps += 1
            io_env.key_press('esc')
            time.sleep(constants.PAUSE_SCREEN_WAIT)

        if back_steps >= max_steps:
            logger.warn("The environment fails to pause!")


    def exit_back_to_game(self, env_name: str, ide_name: str) -> None:

        self.exit_back_to_pause(env_name, ide_name)

        # Unpause the game, to keep the rest of the agent flow consistent
        self.unpause_game(env_name, ide_name)


    def is_env_paused(self) -> bool:

        is_paused = False
        confidence_threshold = 0.85

        # Multiple-scale-template-matching example, decide whether the game is paused according to the confidence score
        pause_clock_template_file = f'./res/{config.env_sub_path}/icons/clock.jpg'

        screenshot = self.take_screenshot(time.time())
        match_info = match_template_image(screenshot, pause_clock_template_file, debug=True, output_bb=True,
                                          save_matches=True, scale='full')

        is_paused = match_info[0]['confidence'] >= confidence_threshold

        # Renaming pause candidate screenshot to ease debugging or gameplay scenarios
        os.rename(screenshot, screenshot.replace('screen', 'pause_screen_candidate'))

        return is_paused


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
