import os
from typing import Dict, Any
from copy import deepcopy

from cradle.config.config import Config
from cradle.environment.software.skill_registry import SoftwareSkillRegistry
from cradle.provider import BaseProvider
from cradle.log import Logger
from cradle.memory import LocalMemory
from cradle.provider.video import VideoRecordProvider
from cradle import constants

logger = Logger()
config = Config()


class Executor(BaseProvider):

    def __init__(self, *args,
                 env_manager: Any,
                 use_unpause_game: bool = False,
                 **kwargs):

        super(Executor, self).__init__()
        self.gm = env_manager

        self.video_recorder = VideoRecordProvider(os.path.join(config.work_dir, 'video.mp4'))
        self.memory = LocalMemory(memory_path=config.work_dir, max_recent_steps=config.max_recent_steps)


    @BaseProvider.write
    def __call__(self,
                 *args,
                 **kwargs) -> Dict[str, Any]:

        # > Pre-processing
        params = self.memory.working_area.copy()

        skill_steps = params.get(constants.SKILL_STEPS, [])
        som_map = params.get(constants.SOM_MAP, {})
        pre_screen_classification = params.get("pre_screen_classification", "")
        screen_classification = params.get("screen_classification", "")
        pre_action = params.get("pre_action", "")

        if config.is_game == True:

            self.gm.unpause_game()

            # @TODO: Rename GENERAL_GAME_INTERFACE
            if (pre_screen_classification.lower() == constants.GENERAL_GAME_INTERFACE and
                    (screen_classification.lower() == constants.MAP_INTERFACE or
                    screen_classification.lower() == constants.SATCHEL_INTERFACE) and pre_action):
                exec_info = self.gm.execute_actions([pre_action])

        else:
            skill_steps = SoftwareSkillRegistry.pre_process_skill_steps(skill_steps, som_map)

        # >> Calling SKILL EXECUTION
        logger.write(f'>>> Calling SKILL EXECUTION')
        logger.write(f'Skill Steps: {skill_steps}')

        # Execute actions
        start_frame_id = self.video_recorder.get_current_frame_id()

        exec_info = self.gm.execute_actions(skill_steps)

        # > Post-processing
        logger.write(f'>>> Post skill execution sensing...')

        # Sense here to avoid changes in state after action execution completes
        mouse_x, mouse_y = self.gm.get_mouse_position()

        if config.is_game == True:
            cur_screenshot_path = self.gm.capture_screen()
        else:
            # First, check if interaction left the target environment
            if not self.gm.check_active_window():
                logger.warn(f"Target environment window is no longer active!")
                cur_screenshot_path = self.gm.get_out_screen()
            else:
                cur_screenshot_path = self.gm.capture_screen()

        end_frame_id = self.video_recorder.get_current_frame_id()

        logger.write(f'>>> Sensing done.')

        if config.is_game == True:
            pause_flag = self.gm.pause_game(screen_classification.lower())
            logger.write(f'Pause flag: {pause_flag}')
            if not pause_flag:
                self.gm.pause_game(screen_type=None)

        # exec_info also has the list of successfully executed skills. skill_steps is the full list, which may differ if there were execution errors.
        pre_action = exec_info[constants.EXECUTED_SKILLS]
        pre_screen_classification = screen_classification

        self.memory.add_recent_history_kv(constants.ACTION, pre_action)
        if exec_info[constants.ERRORS]:
            self.memory.add_recent_history_kv(constants.ACTION_ERROR, exec_info[constants.ERRORS_INFO])
        else:
            self.memory.add_recent_history_kv(constants.ACTION_ERROR, constants.EMPTY_STRING)

        response = {
            f"{constants.START_FRAME_ID}": start_frame_id,
            f"{constants.END_FRAME_ID}": end_frame_id,
            f"{constants.CUR_SCREENSHOT_PATH}": cur_screenshot_path,
            f"{constants.MOUSE_POSITION}" : (mouse_x, mouse_y),
            f"{constants.PRE_ACTION}": pre_action,
            f"{constants.EXEC_INFO}": exec_info,
            "pre_screen_classification": pre_screen_classification,
        }

        self.memory.update_info_history(response)

        del params

        return response
