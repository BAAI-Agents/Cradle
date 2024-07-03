import inspect
import base64
from typing import List, Any

from cradle import constants
from cradle.config.config import Config
from cradle.environment import SkillRegistry
from cradle.environment import Skill
from cradle.utils.singleton import Singleton


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


class XiuxiuSkillRegistry(SkillRegistry, metaclass=Singleton):
    def __init__(self,
                 *args,
                 skill_configs: dict[str, Any]  = config.skill_configs,
                 embedding_provider=None,
                 **kwargs):

        if skill_configs[constants.SKILL_CONFIG_REGISTERED_SKILLS] is None:
            skill_configs[constants.SKILL_CONFIG_REGISTERED_SKILLS] = SKILLS

        super(XiuxiuSkillRegistry, self).__init__(skill_configs=skill_configs,
                                                  embedding_provider=embedding_provider)
