import inspect
import base64

from cradle.environment import SkillRegistry
from cradle.environment import Skill
from cradle.utils.singleton import Singleton
from cradle.config import Config

config = Config()

SKILLS = {}
def register_skill(name):
    def decorator(skill):

        skill_name = name
        skill_function = skill
        skill_code = inspect.getsource(skill)

        # Remove unnecessary annotation in skill library
        if f"@register_skill(\"{name}\")\n" in skill_code:
            skill_code = skill_code.replace(f"@register_skill(\"{name}\")\n", "")

        skill_code_base64 = base64.b64encode(skill_code.encode('utf-8')).decode('utf-8')

        skill_ins = Skill(skill_name,
                       skill_function,
                       "" , # skill_embedding
                       skill_code,
                       skill_code_base64)
        SKILLS[skill_name] = skill_ins

        return skill_ins

    return decorator

class StardewSkillRegistry(SkillRegistry, metaclass=Singleton):
    def __init__(self,
                 *args,
                 skill_from_default = config.skill_from_default,
                 skill_mode = config.skill_mode,
                 skill_names_basic = config.skill_names_basic,
                 skill_names_movement = config.skill_names_movement,
                 skill_names_map = config.skill_names_map,
                 skill_names_trade = config.skill_names_trade,
                 skill_names_deny = config.skill_names_deny,
                 skill_names_allow = config.skill_names_allow,
                 skill_registried = None,
                 embedding_provider = None,
                 **kwargs):

        if skill_registried is not None:
            self.skill_registried = skill_registried
        else:
            self.skill_registried = SKILLS

        super(StardewSkillRegistry, self).__init__(skill_from_default=skill_from_default,
                         skill_mode=skill_mode,
                         skill_names_basic=skill_names_basic,
                         skill_names_movement=skill_names_movement,
                         skill_names_map=skill_names_map,
                         skill_names_trade=skill_names_trade,
                         skill_names_deny=skill_names_deny,
                         skill_names_allow=skill_names_allow,
                         skill_registried=self.skill_registried,
                         embedding_provider=embedding_provider)