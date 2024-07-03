from cradle.config import Config
from cradle.gameio.lifecycle.ui_control import switch_to_environment, take_screenshot
from cradle.log import Logger
from cradle.gameio.io_env import IOEnvironment
from cradle.environment import UIControl

config = Config()
logger = Logger()
io_env = IOEnvironment()


class SoftwareUIControl(UIControl):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def pause_game(self, env_name: str, ide_name: str) -> None:
        pass


    def unpause_game(self, env_name: str, ide_name: str) -> None:
        pass


    def switch_to_game(self, env_name: str, ide_name: str) -> None:
        switch_to_environment()


    def exit_back_to_pause(self, env_name: str, ide_name: str) -> None:
        pass


    def exit_back_to_game(self, env_name: str, ide_name: str) -> None:
        pass


    def is_env_paused(self) -> bool:

        is_paused = False
        return is_paused


    def take_screenshot(self,
                        tid: float,
                        screen_region: tuple[int, int, int, int] = None) -> str:

        screenshot, _ = take_screenshot(tid, screen_region=screen_region)
        return screenshot


    def check_for_close_icon(skill: str, x, y) -> bool:
        """
        Check if trying to click the app close icon
        """

        # The left bottom corner of the close icon
        close_icon_coordinates_x = 1854/config.DEFAULT_ENV_RESOLUTION[0]
        close_icon_coordinates_y = 60/config.DEFAULT_ENV_RESOLUTION[1]

        return ('click_' in skill) and (x >= close_icon_coordinates_x and x <= config.DEFAULT_ENV_RESOLUTION[0]) and (y >= 0 and y <= close_icon_coordinates_y)
