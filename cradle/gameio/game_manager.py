import time
from typing import Tuple, Dict, Any

from cradle import constants
from cradle.config import Config
from cradle.environment.ui_control import UIControl
from cradle.log import Logger
from cradle.gameio import IOEnvironment
from cradle.gameio.lifecycle.ui_control import check_active_window
from cradle.utils.file_utils import assemble_project_path

config = Config()
logger = Logger()
io_env = IOEnvironment()


class GameManager:

    def __init__(
        self,
        env_name,
        embedding_provider = None,
        llm_provider = None,
        skill_registry = None,
        ui_control: UIControl = None,
    ):

        self.env_name = env_name
        self.embedding_provider = embedding_provider
        self.llm_provider = llm_provider
        self.skill_registry = skill_registry
        self.ui_control = ui_control
        io_env.llm_provider = self.llm_provider # @TODO needs a better DI


    def pause_game(self,
                   *args,
                   env_name=config.env_name,
                   ide_name=config.ide_name,
                   screen_type=constants.GENERAL_GAME_INTERFACE,
                   **kwargs):

        if screen_type==constants.PAUSE_INTERFACE:
            return False
        else:
            self.ui_control.pause_game(
                env_name=env_name,
                ide_name=ide_name,
                **kwargs
            )
            return True


    def unpause_game(self,
                     *args,
                     env_name=config.env_name,
                     ide_name=config.ide_name,
                     **kwargs):

        self.ui_control.unpause_game(
            env_name=env_name,
            ide_name=ide_name,
            **kwargs
        )
        return True


    def switch_to_game(self,
                       *args,
                       env_name=config.env_name,
                       ide_name=config.ide_name,
                       **kwargs):

        self.ui_control.switch_to_game(
            env_name=env_name,
            ide_name=ide_name,
            **kwargs
        )


    def check_active_window(self):
        return check_active_window()


    def exit_back_to_pause(self,
                           *args,
                           env_name=config.env_name,
                           ide_name=config.ide_name,
                           **kwargs):

        self.ui_control.exit_back_to_pause(
            env_name=env_name,
            ide_name=ide_name,
            **kwargs
        )


    def get_skill_information(self,
                              skill_list,
                              skill_library_with_code = False
                              ):

        filtered_skill_library = []

        for skill_name in skill_list:
            skill_item = self.skill_registry.get_from_skill_library(skill_name, skill_library_with_code = skill_library_with_code)
            filtered_skill_library.append(skill_item)

        return filtered_skill_library


    def add_new_skill(self,
                      skill_code,
                      overwrite = True):
        return self.skill_registry.register_skill_from_code(skill_code = skill_code, overwrite = overwrite)


    def delete_skill(self, skill_name):
        self.skill_registry.delete_skill(skill_name)


    def retrieve_skills(self, query_task, skill_num, screen_type):
        return self.skill_registry.retrieve_skills(query_task, skill_num, screen_type)


    def register_available_skills(self, candidates):
        self.skill_registry.register_available_skills(candidates)


    def get_skill_library_in_code(self, skill) -> Tuple[str, str]:
        return self.skill_registry.get_skill_code(skill)

    def convert_expression_to_skill(self, expression):
        return self.skill_registry.convert_expression_to_skill(expression)


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

                skill_response = self.skill_registry.execute_skill(skill_name=skill_name, skill_params=skill_params)

                if config.is_game is False:
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


    def capture_screen(self):
        tid = time.time()
        return self.ui_control.take_screenshot(tid)


    def get_mouse_position(self, absolute = False) -> Tuple[int, int]:
        return io_env.get_mouse_position(absolute)


    def list_session_screenshots(self, session_dir: str = config.work_dir):
        return io_env.list_session_screenshots(session_dir)


    def store_skills(self, path = None):
        self.skill_registry.store_skills(path)


    def load_skills(self, path = None):
        self.skill_registry.load_skill_library(path)


    def get_all_skills(self):
        return self.skill_registry.get_all_skills()


    def cleanup_io(self):
        io_env.release_held_keys()
        io_env.release_held_buttons()
