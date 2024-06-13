import time
import argparse

from cradle.environment.capcut.magic_skills.magic_interact import get_infomation_from_video
from cradle.gameio import GameManager
from cradle.provider.openai import OpenAIProvider
from cradle.gameio.lifecycle.ui_control import switch_to_game
from cradle.environment.rdr2.composite_skills.navigation import cv_navigation
from cradle.environment.rdr2.composite_skills.go_to_icon import go_to_horse
from cradle.environment.rdr2.atomic_skills.map import open_map, open_index, close_index
from cradle.environment.rdr2.atomic_skills.move import turn, mount_horse, move_forward
from cradle.environment.rdr2.composite_skills.follow import follow
from cradle.environment.rdr2.composite_skills.auto_shoot import shoot_wolves, shoot_people
from cradle.config import Config
from cradle.log import Logger
from cradle.gameio import IOEnvironment
from cradle.gameio.io_env import IOEnvironment
import cradle.environment.outlook
import cradle.environment.chrome
import cradle.environment.capcut
import cradle.environment.feishu
import cradle.environment.xiuxiu

config = Config()
logger = Logger()
io_env = IOEnvironment()


class SwitchWindow(object):
    def __enter__(self):
        switch_to_game()

    def __exit__(self, exc_type, exc_val, exc_tb):
        # switch_to_code()
        pass


if __name__ == "__main__":

    config.load_env_config("./conf/env_config_outlook.json")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s',"--skill_chosen",type=str,default="following",help="['navigation', 'go_to_horse', 'map_operation', 'shoot_people', 'shoot_wolves', 'following']"
    )
    args = parser.parse_args()
    skill_chosen = args.skill_chosen
    debug = True

    gm = GameManager(config.env_name)

    with SwitchWindow():
        if skill_chosen == 'navigation':  # navigation example, you need to have a red line in the mini-map first
            cv_navigation(1000, debug=debug)

        elif skill_chosen == 'go_to_horse':  # find-and-get-on-horse example
            go_to_horse()
            mount_horse()
            time.sleep(3)

        elif skill_chosen == 'map_operation':  # map operation example
            open_map()
            time.sleep(1)
            open_index()
            time.sleep(1)
            close_index()
            time.sleep(1)

        elif skill_chosen == 'shoot_people':  # shoot example
            # prompt: "Protect Dutch"
            shoot_people()

        elif skill_chosen == 'shoot_wolves':  # shoot example
            # prompt: "Keep the wolves away from Javier and John"
            shoot_wolves()

        elif skill_chosen == 'following':  # follow companion against wolves example
            # prompt: "Follow Javier"
            # prompt: "Catch up to Javier"
            follow()
