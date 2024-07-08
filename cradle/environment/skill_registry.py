"""Base class for Skill Registry."""
import os
import inspect
import base64
import re
import ast
import time
from copy import deepcopy
from typing import Type, AnyStr, List, Any, Dict, Tuple

import numpy as np

from cradle import constants
from cradle.config import Config
from cradle.log import Logger
from cradle.utils.json_utils import load_json, save_json
from cradle.utils.dict_utils import kget
from cradle.environment.skill import Skill
from cradle.environment.utils import serialize_skills, deserialize_skills
from cradle.utils.check import is_valid_value
from cradle.gameio.io_env import IOEnvironment
from cradle.constants import *


config = Config()
logger = Logger()
io_env = IOEnvironment()


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
                       None, # skill_embedding
                       skill_code,
                       skill_code_base64)
        SKILLS[skill_name] = skill_ins

        return skill_ins

    return decorator


class SkillRegistry():
    """Base class for Skill Registry."""

    def __init__(self,
                 *args,
                 skill_configs: dict[str, Any] = config.skill_configs,
                 embedding_provider = None,
                 **kwargs):

        super(SkillRegistry, self).__init__()

        self.skill_from_default = skill_configs[constants.SKILL_CONFIG_FROM_DEFAULT]
        self.skill_mode = skill_configs[constants.SKILL_CONFIG_MODE]
        self.skill_names_basic = skill_configs[constants.SKILL_CONFIG_NAMES_BASIC]
        self.skill_names_allow = skill_configs[constants.SKILL_CONFIG_NAMES_ALLOW]
        self.skill_names_deny = skill_configs[constants.SKILL_CONFIG_NAMES_DENY]
        skill_names_others = kget(skill_configs, constants.SKILL_CONFIG_NAMES_OTHERS, default=dict())
        self.skill_names_movement = kget(skill_names_others, constants.SKILL_CONFIG_NAMES_MOVEMENT, default=[])
        self.skill_names_map = kget(skill_names_others, constants.SKILL_CONFIG_NAMES_MAP, default=[])
        self.skill_names_trade = kget(skill_names_others, constants.SKILL_CONFIG_NAMES_TRADE, default=[])
        self.recent_skills = []

        if skill_configs[constants.SKILL_CONFIG_REGISTERED_SKILLS] is not None:
            self.skill_registered = skill_configs[constants.SKILL_CONFIG_REGISTERED_SKILLS]
        else:
            self.skill_registered = SKILLS

        if self.skill_mode == constants.SKILL_LIB_MODE_BASIC:
            self.skill_library_filename = constants.SKILL_BASIC_LIB_FILE
        elif self.skill_mode == constants.SKILL_LIB_MODE_FULL:
            self.skill_library_filename = constants.SKILL_FULL_LIB_FILE
        else:
            self.skill_from_default = True

        self.embedding_provider = embedding_provider

        self.skills = {}

        os.makedirs(config.skill_local_path, exist_ok=True)
        if self.skill_from_default and os.path.exists(os.path.join(config.skill_local_path, self.skill_library_filename)):
            self.skills = self.load_skills_from_file(os.path.join(config.skill_local_path, self.skill_library_filename))
        else:
            self.skills = self.load_skills_from_scripts()

        self.skills = self.filter_skills(self.skills)


    def set_embedding_provider(self, embedding_provider):
        self.embedding_provider = embedding_provider


    def get_embedding(self, skill_name, skill_doc):
        return np.array(self.embedding_provider.embed_query('{}: {}'.format(skill_name, skill_doc)))


    def load_skills_from_file(self, file_path) -> Dict[str, Skill]:

        logger.write(f"Loading skills from {file_path}")

        skill_local = load_json(file_path)
        skill_local = deserialize_skills(skill_local)

        skills = {}

        for skill_name in skill_local.keys():

            skill_embedding = skill_local[skill_name].skill_embedding
            skill_code_base64 = base64.b64encode(skill_local[skill_name].skill_code.encode('utf-8')).decode('utf-8')

            regenerate_flag = False

            if skill_code_base64 != skill_local[skill_name].skill_code_base64: # The skill_code is modified
                regenerate_flag = True

            if not is_valid_value(skill_embedding): # The skill_embedding is invalid
                regenerate_flag = True

            if skill_name not in self.skill_registered.keys(): # The skill is not in the skill registry
                regenerate_flag = True

            if not regenerate_flag:
                logger.debug(f"No need to regenerate skill {skill_name}")
                skills[skill_name] = Skill(skill_name,
                                           self.skill_registered[skill_name].skill_function,
                                           skill_local[skill_name].skill_embedding,
                                           skill_local[skill_name].skill_code,
                                           skill_code_base64)
            else: # skill_code has been modified, we should recompute embedding
                logger.write(f"Regenerate skill {skill_name}")
                self.register_skill_from_code(skill_local[skill_name].skill_code)

        self.store_skills_to_file(file_path, skills)

        return skills


    def load_skills_from_scripts(self) -> Dict[str, Skill]:

        logger.write("Loading skills from scripts")

        skills = {}
        for skill_name in self.skill_registered.keys():

            skill_embedding = self.skill_registered[skill_name].skill_embedding
            skill_code_base64 = base64.b64encode(self.skill_registered[skill_name].skill_code.encode('utf-8')).decode('utf-8')

            regenerate_flag = False

            if skill_code_base64 != self.skill_registered[skill_name].skill_code_base64: # The skill_code is modified
                regenerate_flag = True

            if not is_valid_value(skill_embedding): # The skill_embedding is invalid
                regenerate_flag = True

            if not regenerate_flag:
                logger.debug(f"No need to regenerate skill {skill_name}")
                skills[skill_name] = Skill(skill_name,
                                           self.skill_registered[skill_name].skill_function,
                                           self.skill_registered[skill_name].skill_embedding,
                                           self.skill_registered[skill_name].skill_code,
                                           skill_code_base64)
            else: # skill_code has been modified, we should recompute embedding
                logger.write(f"Regenerate skill {skill_name}")
                skills[skill_name] = Skill(skill_name,
                                           self.skill_registered[skill_name].skill_function,
                                           self.get_embedding(skill_name, inspect.getdoc(self.skill_registered[skill_name].skill_function)),
                                           self.skill_registered[skill_name].skill_code,
                                           skill_code_base64)

        self.store_skills_to_file(os.path.join(config.skill_local_path, self.skill_library_filename), skills)

        return skills


    def filter_skills(self, skills) -> Dict[str, Skill]:

        filtered_skills = {}

        if self.skill_mode == constants.SKILL_LIB_MODE_BASIC:
            for skill_name in self.skills:
                if skill_name in self.skill_names_basic:
                    filtered_skills[skill_name] = self.skills[skill_name]
        elif self.skill_mode == constants.SKILL_LIB_MODE_FULL:
            filtered_skills = deepcopy(skills)
        else:
            filtered_skills = deepcopy(skills)

        return filtered_skills


    def convert_expression_to_skill(self, expression: str = "open_map()"):
        try:
            parsed = ast.parse(expression, mode='eval')

            if isinstance(parsed.body, ast.Call):
                skill_name, skill_params = self.extract_function_info(expression)
                return skill_name, skill_params
            elif isinstance(parsed.body, ast.List):

                skills_list = []
                for call in parsed.body.elts:
                    if isinstance(call, ast.Call):
                        call_str = ast.unparse(call).strip()
                        skill_name, skill_params = self.extract_function_info(call_str)
                        skills_list.append((skill_name, skill_params))
                    else:
                        raise ValueError("Input must be a list of function calls")
                return skills_list
            else:
                raise ValueError("Input must be a function call or a list of function calls")

        except SyntaxError as e:
            raise ValueError(f"Error parsing input: {e}")


    def extract_function_info(self, input_string: str = "open_map()"):

        pattern = re.compile(r'(\w+)\((.*?)\)')

        match = pattern.match(input_string)

        if match:
            function_name = match.group(1)
            raw_arguments = match.group(2)

            # To avoid simple errors based on faulty model output
            if raw_arguments is not None and len(raw_arguments) > 0:
                raw_arguments = raw_arguments.replace("=false", "=False").replace("=true", "=True")

            try:
                parsed_arguments = ast.parse(f"fake_func({raw_arguments})", mode='eval')
            except SyntaxError:
                raise ValueError("Invalid function call/arg format to parse.")

            arguments = {}
            for node in ast.walk(parsed_arguments):
                if isinstance(node, ast.keyword):
                    arguments[node.arg] = ast.literal_eval(node.value)

            if len(raw_arguments) > 0 and len(arguments.keys()) == 0:
                raise ValueError("Call arguments not properly parsed!")

            return function_name, arguments

        else:
            raise ValueError("Invalid function call format string.")


    def convert_code_to_skill_info(self, skill_code: str):
        tree = ast.parse(skill_code)
        function_name = None
        arguments = {}
        # TODO: This is a very naive way to get the function name. We should improve this.
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_name = node.name
        return function_name, arguments


    def get_from_skill_library(self,
                               skill_name: str,
                               skill_library_with_code: bool = False) -> Dict:

        skill = self.skills[skill_name]
        skill_function = skill.skill_function

        docstring = inspect.getdoc(skill_function)

        skill_code = ""
        for item in self.skills:
            if item == skill_name:
                skill_code = self.skills[item].skill_code
                break

        if docstring:

            params = inspect.signature(skill_function).parameters

            if len(params) > 0:
                param_descriptions = {}

                for param in params.values():
                    name = param.name
                    param_description = re.search(rf"- {name}: (.+).", docstring).group(1)
                    param_descriptions[name] = param_description

                res =  {
                    "function_expression": f"{skill_name}({', '.join(params.keys())})",
                    "description": docstring,
                    "parameters": param_descriptions,
                }
            else:
                res =  {
                    "function_expression": f"{skill_name}()",
                    "description": docstring,
                    "parameters": {},
                }
        else:
            res =  None

        if skill_library_with_code and res is not None:
            res["code"] = skill_code

        return res


    def get_skill_code(self, skill: Any) -> Tuple[str, str]:

        info = None
        try:
            skill_name, _ = self.extract_function_info(skill)
        except:
            skill_name = skill

        skill_code = None
        for item in self.skills:
            if item == skill_name:
                skill_code = self.skills[item].skill_code
                break

        if skill_code is None:
            info = f"Skill '{skill_name}' not found in the registry."

        return skill_code, info


    def execute_skill(self,
                      skill_name: str = "open_map",
                      skill_params: Dict = None):
        try:
            if skill_name in self.skills:
                skill_function = self.skills[skill_name].skill_function
                skill_response = skill_function(**skill_params)
            else:
                raise ValueError(f"Function '{skill_name}' not found in the skill library.")
        except Exception as e:
            logger.error(f"Error executing skill {skill_name}: {e}")
            raise e

        return skill_response


    def execute_nop_skill(self):
        time.sleep(2)


    def register_skill_from_code(self, skill_code: str, overwrite = False) -> Tuple[bool, str]:
        """Register the skill function from the code string.

        Args:
            skill_code: the code of skill.
            overwrite: the flag indicates whether to overwrite the skill with the same name or not.

        Returns:
            bool: the true value means that there is no problem in the skill_code. The false value means that we may need to re-generate it.
            str: the detailed information about the bool.
        """
        def lower_func_name(skill_code):
            skill_name, _ = self.convert_code_to_skill_info(skill_code)
            replaced_name = skill_name.lower()

            # To make sure the skills in .py files will not be overwritten.
            # The skills not in .py files can still be overwritten.
            if replaced_name in self.skills.keys():
                replaced_name = replaced_name+'_generated'

            return skill_code.replace(skill_name, replaced_name)

        def check_param_description(skill) -> bool:
            docstring = inspect.getdoc(skill)
            if docstring:
                params = inspect.signature(skill).parameters
                if len(params) > 0:
                    for param in params.values():
                        if not re.search(rf"- {param.name}: (.+).", docstring):
                            return False
                    return True
                else:
                    return True
            else:
                return True

        def check_protection_conflict(skill):
            for word in self.skill_names_allow:
                if word in skill:
                    return True

            for word in self.skill_names_deny:
                if word in skill:
                    return False

            return True

        info = None

        if skill_code.count('(') < 2:
            info = "Skill code contains no functionality."
            logger.error(info)
            return True, info

        skill_code = lower_func_name(skill_code)
        skill_name, _ = self.convert_code_to_skill_info(skill_code)

        # Always avoid adding skills that are ambiguous with existing pre-defined ones.
        if check_protection_conflict(skill_name) == False:
            info = f"Skill '{skill_name}' conflicts with protected skills."
            for word in self.skill_names_deny:
                if word in skill_name:
                    for protected_skill in self.skill_names_basic:
                        if word in protected_skill:
                            self.recent_skills.append(protected_skill)
            logger.write(info)
            return True, info

        if overwrite:
            if skill_name in self.skills:
                self.delete_skill(skill_name)
                logger.write(f"Skill '{skill_name}' will be overwritten.")

        if skill_name in self.skills:
            info = f"Skill '{skill_name}' already exists."
            logger.write(info)
            return True, info

        try:
            exec(skill_code)
            skill = eval(skill_name)
        except:
            info = "The skill code is invalid."
            logger.error(info)
            return False, info

        if check_param_description(skill) == False:
            info = "The format of parameter description is wrong."
            logger.error(info)
            return False, info

        skill_code_base64 = base64.b64encode(skill_code.encode('utf-8')).decode('utf-8')
        skill_ins = Skill(skill_name,
                          skill,
                          self.get_embedding(skill_name, inspect.getdoc(skill)),
                          skill_code,
                          skill_code_base64)

        self.skills[skill_name] = skill_ins
        self.recent_skills.append(skill_name)

        info = f"Skill '{skill_name}' has been registered."
        logger.write(info)
        return True, info


    def delete_skill(self, skill_name: str) -> None:

        try:
            skill_name, _ = self.extract_function_info(skill_name)
        except:
            skill_name = skill_name

        if skill_name in self.skills:
            del self.skills[skill_name]
        if skill_name in self.recent_skills:
            position = self.recent_skills.index(skill_name)
            self.recent_skills.pop(position)


    def retrieve_skills(self, query_task: str, skill_num: int, screen_type: str) -> List[str]:
        skill_num = min(skill_num, len(self.skills))
        target_skills = [skill for skill in self.recent_skills]

        task_emb = np.array(self.embedding_provider.embed_query(query_task))

        sorted_skills = sorted(self.skills.items(), key=lambda x: -np.dot(x[1].skill_embedding, task_emb))

        for skill in sorted_skills:

            skill_name, skill = skill

            if len(target_skills) >= skill_num:
                break
            else:
                if skill_name not in target_skills:
                    target_skills.append(skill_name)

        self.recent_skills = []

        # Add required skills based on screen type
        if screen_type == constants.GENERAL_GAME_INTERFACE:
            target_skills += [skill for skill in self.skill_names_movement]
        elif screen_type == constants.TRADE_INTERFACE or screen_type == constants.SATCHEL_INTERFACE:
            target_skills += [skill for skill in self.skill_names_trade]
        elif screen_type == constants.MAP_INTERFACE:
            target_skills += [skill for skill in self.skill_names_map]

        return target_skills


    def register_available_skills(self, candidates:List[str]) -> None:
        for skill_key in candidates:
            if skill_key not in self.skills:
                logger.error(f"Skill '{skill_key}' does not exist.")

        for skill_key in list(self.skills.keys()):
            if skill_key not in candidates:
                del self.skills[skill_key]


    def get_all_skills(self) -> List[str]:
        return list(self.skills.keys())


    def convert_str_to_func(self, skill_name, skill_local):
        exec(skill_local[skill_name].skill_code)
        skill = eval(skill_name)
        return skill


    def store_skills_to_file(self,
                             file_path: str,
                             skills: Dict[str, Skill]) -> None:
        serialized_skills = serialize_skills(skills)
        save_json(file_path, serialized_skills, indent=4)
