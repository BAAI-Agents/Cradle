from cradle.config import Config
from cradle.log import Logger
from cradle.gameio import IOEnvironment

from cradle.environment.stardew.skill_registry import SkillRegistry
from cradle.environment import register_environment

config = Config()
logger = Logger()
io_env = IOEnvironment()


@register_environment("stardew")
class Interface():
    def __init__(self):

        # load ui control in lifecycle


        # load skills

        # load skill registry
        self.SkillRegistry = SkillRegistry
