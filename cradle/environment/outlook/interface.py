from cradle.config import Config
from cradle.log import Logger
from cradle.gameio import IOEnvironment
from cradle.environment.outlook.lifecycle.ui_control import pause_game, unpause_game
from cradle.environment.outlook.skill_registry import SkillRegistry
from cradle.environment import register_environment
from cradle.utils.image_utils import draw_mouse_pointer
import cradle.environment.outlook.atomic_skills
import cradle.environment.outlook.composite_skills

config = Config()
logger = Logger()
io_env = IOEnvironment()


@register_environment("outlook")
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
                    "decision_making": "./res/outlook/prompts/inputs/decision_making.json",
                    "gather_information": "./res/outlook/prompts/inputs/gather_information.json",
                    "success_detection": "",
                    "self_reflection": "./res/outlook/prompts/inputs/self_reflection.json",
                    "information_summary": "./res/outlook/prompts/inputs/information_summary.json",
                    "gather_text_information": ""
                },
                "templates": {
                    "decision_making": "./res/outlook/prompts/templates/decision_making.prompt",
                    "gather_information": "./res/outlook/prompts/templates/gather_information.prompt",
                    "success_detection": "",
                    "self_reflection": "./res/outlook/prompts/templates/self_reflection.prompt",
                    "information_summary": "./res/outlook/prompts/templates/information_summary.prompt",
                    "gather_text_information": ""
                },
            }
        }

        # init skill library
        self.skill_library = [
            "press_keyboard_shortcut",
            "click_at_position",
            "move_mouse_to_position",
            "mouse_drag",
            "type_text",
            "press_key",
            "press_keys_combined",
            "click_on_label",
            "hover_on_label",
            "go_back_to_target_application",
            "go_to_mail_view",
            "go_to_calendar_view",
        ]

        # init task description
        self.task_description = ""
