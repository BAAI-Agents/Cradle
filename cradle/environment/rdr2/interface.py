from cradle.config import Config
from cradle.log import Logger
from cradle.gameio import IOEnvironment
from cradle.environment.rdr2.lifecycle.ui_control import (segment_minimap,
                                                          switch_to_game,
                                                          pause_game,
                                                          unpause_game,
                                                          exit_back_to_pause,
                                                          IconReplacer)
from cradle.environment.rdr2.composite_skills.navigation import navigate_path
from cradle.environment.rdr2.skill_registry import SkillRegistry
from cradle.environment import register_environment

config = Config()
logger = Logger()
io_env = IOEnvironment()


@register_environment("rdr2")
class Interface():
    def __init__(self):

        # load ui control in lifecycle
        self.segment_minimap = segment_minimap
        self.pause_game = pause_game
        self.unpause_game = unpause_game
        self.switch_to_game = switch_to_game
        self.exit_back_to_pause = exit_back_to_pause
        self.IconReplacer = IconReplacer

        # load skills
        self.navigate_path = navigate_path

        # load skill registry
        self.SkillRegistry = SkillRegistry

        # init planner parms
        self.planner_params = {
            "__check_list__": [
                "decision_making",
                "gather_information",
                "success_detection",
                "self_reflection",
                "information_summary",
                "gather_text_information"
            ],
            "prompt_paths": {
                "inputs": {
                    "decision_making": "./res/rdr2/prompts/inputs/decision_making.json",
                    "gather_information": "./res/rdr2/prompts/inputs/gather_information.json",
                    "success_detection": "./res/rdr2/prompts/inputs/success_detection.json",
                    "self_reflection": "./res/rdr2/prompts/inputs/self_reflection.json",
                    "information_summary": "./res/rdr2/prompts/inputs/information_summary.json",
                    "gather_text_information": "./res/rdr2/prompts/inputs/gather_text_information.json"
                },
                "templates": {
                    "decision_making": "./res/rdr2/prompts/templates/decision_making.prompt",
                    "gather_information": "./res/rdr2/prompts/templates/gather_information.prompt",
                    "success_detection": "./res/rdr2/prompts/templates/success_detection.prompt",
                    "self_reflection": "./res/rdr2/prompts/templates/self_reflection.prompt",
                    "information_summary": "./res/rdr2/prompts/templates/information_summary.prompt",
                    "gather_text_information": "./res/rdr2/prompts/templates/gather_text_information.prompt"
                },
            }
        }

        # init skill library
        self.skill_library = [
            'turn',
            'move_forward',
            'turn_and_move_forward',
            'follow',
            'aim',
            'shoot',
            'shoot_wolves',
            'select_weapon',
            'select_sidearm',
            'fight',
            'mount_horse'
        ]

        # init task description
        self.task_description = ""
