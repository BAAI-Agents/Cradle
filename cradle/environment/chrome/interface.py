from cradle.config import Config
from cradle.log import Logger
from cradle.gameio import IOEnvironment
from cradle.environment.chrome.lifecycle.ui_control import pause_game, unpause_game
from cradle.environment.chrome.skill_registry import SkillRegistry
from cradle.environment import register_environment
from cradle.utils.image_utils import draw_mouse_pointer
import cradle.environment.chrome.atomic_skills
import cradle.environment.chrome.composite_skills

config = Config()
logger = Logger()
io_env = IOEnvironment()


@register_environment("chrome")
class Interface():

    def draw_mouse_pointer(self, frame, x, y):
        return draw_mouse_pointer(frame, x, y)


    def __init__(self):

        # load UI control in lifecycle
        self.pause_game = pause_game
        self.unpause_game = unpause_game
        self.augment_image = self.draw_mouse_pointer

        # load skill registry
        self.SkillRegistry = SkillRegistry

        # init planner parms
        self.planner_params = {
            "__check_list__": [
                "decision_making",
                "gather_information",
                "self_reflection",
                "information_summary",
            ],
            "prompt_paths": {
                "inputs": {
                    "decision_making": "./res/chrome/prompts/inputs/decision_making.json",
                    "gather_information": "./res/chrome/prompts/inputs/gather_information.json",
                    "success_detection": "",
                    "self_reflection": "./res/chrome/prompts/inputs/self_reflection.json",
                    "information_summary": "./res/chrome/prompts/inputs/information_summary.json",
                    "gather_text_information": ""
                },
                "templates": {
                    "decision_making": "./res/chrome/prompts/templates/decision_making.prompt",
                    "gather_information": "./res/chrome/prompts/templates/gather_information.prompt",
                    "success_detection": "",
                    "self_reflection": "./res/chrome/prompts/templates/self_reflection.prompt",
                    "information_summary": "./res/chrome/prompts/templates/information_summary.prompt",
                    "gather_text_information": ""
                },
            }
        }

        # init skill library
        self.skill_library = [
            "click_at_position",
            "type_text",
            "press_key",
            "return_to_previous_page",
            "press_keys_combined",
            "click_on_label",
            "go_back_to_target_application",
        ]

        # init task description
        self.task_description = ""
