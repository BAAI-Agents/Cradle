import time
from typing import Tuple, Dict, Any

from cradle import constants
from cradle.config import Config
from cradle.log import Logger
from cradle.gameio import IOEnvironment
from cradle.gameio.lifecycle.ui_control import check_active_window, take_screenshot, draw_mouse_pointer_file
from cradle.environment import ENVIORNMENT_REGISTRY
from cradle.utils.file_utils import assemble_project_path

config = Config()
logger = Logger()
io_env = IOEnvironment()


class GameManager:

    def __init__(
        self,
        env_name,
        embedding_provider = None
    ):
        self.env_name = env_name
        self.interface = ENVIORNMENT_REGISTRY[config.env_short_name]()
        self.skill_registry = self.interface.SkillRegistry(
            local_path = config.skill_local_path,
            from_local = config.skill_from_local,
            store_path = config.work_dir,
            skill_scope = config.skill_scope,
            embedding_provider = embedding_provider
        )


    def get_interface(self):
        return self.interface


    def pause_game(self, screen_type=constants.GENERAL_GAME_INTERFACE):

        if screen_type==constants.GENERAL_GAME_INTERFACE or screen_type==constants.PAUSE_INTERFACE or screen_type==constants.RADIAL_INTERFACE:
            self.interface.pause_game()


    def unpause_game(self):
        self.interface.unpause_game()


    def switch_to_game(self):
        self.interface.switch_to_game()


    def check_active_window(self):
        return check_active_window()


    def exit_back_to_pause(self):
        self.interface.exit_back_to_pause()


    def get_skill_information(self, skill_list):

        filtered_skill_library = []

        for skill_name in skill_list:
            skill_item = self.skill_registry.get_from_skill_library(skill_name)
            filtered_skill_library.append(skill_item)

        return filtered_skill_library


    def add_new_skill(self, skill_code, overwrite = True):
        return self.skill_registry.register_skill_from_code(skill_code = skill_code, overwrite = overwrite)


    def delete_skill(self, skill_name):
        self.skill_registry.delete_skill(skill_name)


    def retrieve_skills(self, query_task, skill_num, screen_type):
        return self.skill_registry.retrieve_skills(query_task, skill_num, screen_type)


    def register_available_skills(self, candidates):
        self.skill_registry.register_available_skills(candidates)


    def get_skill_library_in_code(self, skill) -> Tuple[str, str]:
        return self.skill_registry.get_skill_library_in_code(skill)


    def execute_navigation(self, action):

        # Execute action
        total_time_step = 500
        if action == "navigate_path":
            time.sleep(2)
            self.interface.navigate_path(total_time_step)


    def execute_actions(self, actions) -> Dict[str, Any]:

        exec_info = {
            constants.EXECUTED_SKILLS: [],
            constants.LAST_SKILL: '',
            constants.ERRORS : False,
            constants.ERRORS_INFO: ""
        }

        io_env.update_timeouts()

        if actions is None or len(actions) == 0 or actions == '' or actions[0] == '':
            logger.warn(f"No actions to execute! Executing nop.")
            self.skill_registry.execute_nop_skill()

            exec_info[constants.ERRORS] = False
            return exec_info

        skill_name = '-'
        skill_params = '-'
        skill_response = None

        try:
            for skill in actions:

                if constants.INVALID_BBOX in skill:
                    exec_info[constants.ERRORS] = True
                    label_id = skill.split(": ")[1]
                    exec_info[constants.ERRORS_INFO] = f"Label ID {label_id} not found in SOM map."
                    return exec_info

                skill_name, skill_params = self.skill_registry.convert_expression_to_skill(skill)

                logger.write(f"Executing skill: {skill_name} with params: {skill_params}")

                # Enable OCR for composite skills, start the ocr check
                if skill_name in config.ocr_check_composite_skill_names:
                    if not config.ocr_fully_ban:
                        config.ocr_different_previous_text = False
                        config.enable_ocr = True
                    else:
                        config.ocr_different_previous_text = False
                        config.enable_ocr = False

                if "navigate" in skill_name:
                    self.execute_navigation(skill_name)
                else:
                    skill_response = self.skill_registry.execute_skill(name=skill_name, params=skill_params)

                skill = skill + " # " + f"""{str(skill_response)}""" if skill_response else skill
                exec_info[constants.EXECUTED_SKILLS].append(skill)
                exec_info[constants.LAST_SKILL] = skill

                self.post_action_wait()
                logger.write(f"Finished executing skill: {skill} and wait.")

        except Exception as e:
            msg = f'Error executing skill {skill_name} with params {skill_params} (from actions: {actions}):\n{e}'
            logger.error(msg)
            exec_info[constants.ERRORS] = True
            exec_info[constants.ERRORS_INFO] = msg

        # @TODO re-add hold timeout check call

        return exec_info


    # Currently all actions have wait in them, if needed
    def post_action_wait(self):
        #time.sleep(config.DEFAULT_POST_ACTION_WAIT_TIME)
        time.sleep(1)


    def get_out_screen(self):
        out_screen_file = "./res/software/samples/out_of_target_screen.jpg"
        full_path = assemble_project_path(out_screen_file)
        return full_path


    def capture_screen(self, include_minimap: bool = False) -> Tuple[str, str]:
        tid = time.time()
        return take_screenshot(tid, include_minimap=include_minimap)


    def extract_minimap(self, screenshot_path):
        return self.interface.segment_minimap(screenshot_path)


    def list_session_screenshots(self, session_dir: str = config.work_dir):
        return io_env.list_session_screenshots(session_dir)


    def store_skills(self):
        self.skill_registry.store_skills()


    def cleanup_io(self):
        io_env.release_held_keys()
        io_env.release_held_buttons()
