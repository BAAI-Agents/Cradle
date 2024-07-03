import os
from typing import Dict, Any
from copy import deepcopy

from cradle.config.config import Config
from cradle.provider import BaseProvider
from cradle.log import Logger
from cradle.memory import LocalMemory
from cradle.provider import VideoRecordProvider
from cradle import constants

config = Config()
logger = Logger()


class SkillExecuteProvider(BaseProvider):

    def __init__(self, *args,
                 gm: Any,
                 use_unpause_game: bool = False,
                 **kwargs):

        super(SkillExecuteProvider, self).__init__()

        self.gm = gm
        self.memory = LocalMemory()
        self.video_recorder = VideoRecordProvider(os.path.join(config.work_dir, 'video.mp4'))


    @BaseProvider.write
    def __call__(self,
                 *args,
                 **kwargs) -> Dict[str, Any]:

        params = deepcopy(self.memory.working_area)

        skill_steps = params.get("skill_steps", [])
        pre_screen_classification = params.get("pre_screen_classification", "")
        screen_classification = params.get("screen_classification", "")
        pre_action = params.get("pre_action", "")

        self.gm.unpause_game()

        # @TODO: Rename GENERAL_GAME_INTERFACE
        if (pre_screen_classification.lower() == constants.GENERAL_GAME_INTERFACE and
                (screen_classification.lower() == constants.MAP_INTERFACE or
                 screen_classification.lower() == constants.SATCHEL_INTERFACE) and pre_action):
            exec_info = self.gm.execute_actions([pre_action])

        start_frame_id = self.video_recorder.get_current_frame_id()
        exec_info = self.gm.execute_actions(skill_steps)
        screenshot_path = self.gm.capture_screen()
        end_frame_id = self.video_recorder.get_current_frame_id()

        try:
            pause_flag = self.gm.pause_game(screen_classification.lower())
            logger.write(f'Pause flag: {pause_flag}')
            if not pause_flag:
                self.gm.pause_game(screen_type=None)
        except Exception as e:
            logger.write(f"Error while pausing the game: {e}")

        # exec_info also has the list of successfully executed skills. skill_steps is the full list, which may differ if there were execution errors.
        pre_action = exec_info["last_skill"]
        pre_screen_classification = screen_classification

        logger.write(f"Execute skill steps by frame id ({start_frame_id}, {end_frame_id}).")

        res_params = {
            "start_frame_id": start_frame_id,
            "end_frame_id": end_frame_id,
            "screenshot_path": screenshot_path,
            "pre_action": pre_action,
            "pre_screen_classification": pre_screen_classification,
            "exec_info": exec_info,
        }

        self.memory.update_info_history(res_params)

        del params

        return res_params
