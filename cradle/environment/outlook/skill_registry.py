import re
import ast
import time
from typing import Dict, Any, List, Tuple
import inspect
import copy
import os
import base64

import numpy as np

from cradle.config import Config
from cradle.log import Logger
from cradle.utils.json_utils import load_json, save_json
from cradle.gameio import IOEnvironment
from cradle import constants

config = Config()
logger = Logger()
io_env = IOEnvironment()

SKILL_REGISTRY = {}
SKILL_INDEX = []

SKILL_NAME_KEY = 'skill_name'
SKILL_EMBEDDING_KEY = 'skill_emb'
SKILL_CODE_KEY = 'skill_code'
SKILL_CODE_HASH_KEY = 'skill_code_base64'
EXPL_SKILL_LIB_FILE='skill_lib.json'
BASIC_SKILL_LIB_FILE='skill_lib_basic.json'
BASIC_SKILLS = [
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
DENY_LIST_TERMS = []
ALLOW_LIST_TERMS = []


def register_skill(name):
    def decorator(skill):
        SKILL_REGISTRY[name] = skill
        skill_code = inspect.getsource(skill)

        # Remove unnecessary annotation in skill library
        if f"@register_skill(\"{name}\")\n" in skill_code:
            skill_code = skill_code.replace(f"@register_skill(\"{name}\")\n", "")

        SKILL_INDEX.append({SKILL_NAME_KEY:          name,
                            SKILL_EMBEDDING_KEY:     None,
                            SKILL_CODE_KEY:          skill_code})
        return skill
    return decorator


def post_skill_wait(wait_time = config.DEFAULT_POST_ACTION_WAIT_TIME):
    """Wait for skill to finish. Like if there is an animation"""
    time.sleep(wait_time)


class SkillRegistry:


    def __init__(
        self,
        local_path = '',
        from_local = False,
        store_path = '',
        skill_scope = 'Full',
        embedding_provider = None
    ):

        self.from_local = from_local
        if skill_scope == 'Basic':
            self.skill_library_filename = BASIC_SKILL_LIB_FILE
        elif skill_scope == 'Full':
            self.skill_library_filename = EXPL_SKILL_LIB_FILE
        elif skill_scope == None:
            self.from_local = False

        self.skill_scope = skill_scope
        self.local_path = local_path
        self.store_path = store_path
        self.embedding_provider = embedding_provider
        self.basic_skills = copy.deepcopy(BASIC_SKILLS)
        self.movement_skills = []
        self.recent_skills = []

        if self.from_local:
            if not os.path.exists(os.path.join(self.local_path, self.skill_library_filename)):
                logger.error(f"Skill library file {os.path.join(self.local_path, self.skill_library_filename)} does not exist.")
                self.filter_skill_library()
                self.store_skills(os.path.join(self.local_path, self.skill_library_filename))
            else:
                self.load_skill_library(os.path.join(self.local_path, self.skill_library_filename))
        else:
            self.filter_skill_library()


    def extract_function_info(self, input_string: str = "drag_email()"):

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


    def execute_skill(self, name: str = "drag_email", params: Dict = None):

        # @TODO return execution error info

        if name in self.skill_registry:
            skill = self.skill_registry[name]
            skill(**params)
        else:
            raise ValueError(f"Function '{name}' not found in the registry.")


    def execute_nop_skill(self):
        time.sleep(2)


    def convert_expression_to_skill(self, expression: str = "drag_email()"):
        skill_name, skill_params = self.extract_function_info(expression)
        return skill_name, skill_params


    def get_from_skill_library(self, skill_name: str) -> Dict:
        skill = self.skill_registry[skill_name]
        docstring = inspect.getdoc(skill)

        if docstring:

            params = inspect.signature(skill).parameters

            if len(params) > 0:
                param_descriptions = {}
                for param in params.values():
                    name = param.name
                    param_description = re.search(rf"- {name}: (.+).", docstring).group(1)
                    param_descriptions[name] = param_description
                return {
                    "function_expression": f"{skill.__name__}({', '.join(params.keys())})",
                    "description": docstring,
                    "parameters": param_descriptions,
                }
            else:
                return {
                    "function_expression": f"{skill.__name__}()",
                    "description": docstring,
                    "parameters": {},
                }
        else:
            return None


    def get_skill_library_in_code(self, skill: Any) -> Tuple[str, str]:

        info = None
        try:
            skill_name, _ = self.extract_function_info(skill)
        except:
            skill_name = skill

        skill_code = None
        for item in self.skill_index:
            if item[SKILL_NAME_KEY] == skill_name:
                skill_code = item[SKILL_CODE_KEY]
                if f"@register_skill(\"{skill_name}\")\n" in skill_code:
                    skill_code = skill_code.replace(f"@register_skill(\"{skill_name}\")\n", "")
                break

        if skill_code is None:
            info = f"Skill '{skill_name}' not found in the registry."

        return skill_code, info


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
            skill_name = get_func_name(skill_code)
            replaced_name = skill_name.lower()

            # To make sure the skills in .py files will not be overwritten.
            # The skills not in .py files can still be overwritten.
            if replaced_name in SKILL_REGISTRY:
                replaced_name = replaced_name+'_generated'

            return skill_code.replace(skill_name, replaced_name)


        def get_func_name(skill_code):
            return skill_code.split('def ')[-1].split('(')[0]


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
            for word in ALLOW_LIST_TERMS:
                if word in skill:
                    return True

            for word in DENY_LIST_TERMS:
                if word in skill:
                    return False

            return True

        info = None

        if skill_code.count('(') < 2:
            info = "Skill code contains no functionality."
            logger.error(info)
            return True, info

        skill_code = lower_func_name(skill_code)
        skill_name = get_func_name(skill_code)

        # Always avoid adding skills that are ambiguous with existing pre-defined ones.
        if check_protection_conflict(skill_name) == False:
            info = f"Skill '{skill_name}' conflicts with protected skills."
            for word in DENY_LIST_TERMS:
                if word in skill_name:
                    for protected_skill in BASIC_SKILLS:
                        if word in protected_skill:
                            self.recent_skills.append(protected_skill)
            logger.write(info)
            return True, info

        if overwrite:
            if skill_name in self.skill_registry:
                self.delete_skill(skill_name)
                logger.write(f"Skill '{skill_name}' will be overwritten.")

        if skill_name in self.skill_registry:
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

        self.skill_registry[skill_name] = skill
        self.skill_index.append({SKILL_NAME_KEY:     skill_name,
                                SKILL_EMBEDDING_KEY: self.get_embedding(skill_name, inspect.getdoc(skill)),
                                SKILL_CODE_KEY:      skill_code})
        self.recent_skills.append(skill_name)

        info = f"Skill '{skill_name}' has been registered."
        logger.write(info)
        return True, info


    def delete_skill(self, skill_name: str) -> None:

        try:
            skill_name, _ = self.extract_function_info(skill_name)
        except:
            skill_name = skill_name

        if skill_name in self.skill_registry:
            del self.skill_registry[skill_name]
            position = next((i for i, skill in enumerate(self.skill_index) if skill[SKILL_NAME_KEY] == skill_name), None)
            self.skill_index.pop(position)

        if skill_name in self.recent_skills:
            position = self.recent_skills.index(skill_name)
            self.recent_skills.pop(position)


    def retrieve_skills(self, query_task: str, skill_num: int, screen_type: str) -> List[str]:
        skill_num = min(skill_num, len(self.skill_index))
        target_skills = [skill for skill in self.recent_skills]
        task_emb = np.array(self.embedding_provider.embed_query(query_task))
        self.skill_index.sort(key = lambda x: -np.dot(x[SKILL_EMBEDDING_KEY],task_emb))
        for skill in self.skill_index:
            if len(target_skills)>=skill_num:
                break
            else:
                if skill[SKILL_NAME_KEY] not in target_skills:
                    target_skills.append(skill[SKILL_NAME_KEY])
        self.recent_skills = []

        # Add required skills based on screen type
        if screen_type == constants.GENERAL_GAME_INTERFACE:
            target_skills += [skill for skill in self.movement_skills]
        elif screen_type == constants.TRADE_INTERFACE or screen_type == constants.SATCHEL_INTERFACE:
            target_skills += [skill for skill in self.trade_skills]
        elif screen_type == constants.MAP_INTERFACE:
            target_skills += [skill for skill in self.map_skills]

        return target_skills


    def register_available_skills(self, candidates:List[str]) -> None:
        for skill_key in candidates:
            if skill_key not in self.skill_registry:
                logger.error(f"Skill '{skill_key}' does not exist.")

        for skill_key in list(self.skill_registry.keys()):
            if skill_key not in candidates:
                del self.skill_registry[skill_key]
        self.skill_index_t = []

        for skill in self.skill_index:
            if skill[SKILL_NAME_KEY] in candidates:
                self.skill_index_t.append(skill)
        self.skill_index = copy.deepcopy(self.skill_index_t)

        del self.skill_index_t


    def get_all_skills(self) -> List[str]:
        return list(self.skill_registry.keys())


    def get_embedding(self, skill_name, skill_doc):
        return np.array(self.embedding_provider.embed_query('{}: {}'.format(skill_name, skill_doc)))


    def convert_str_to_func(self, skill_name, skill_local):
        exec(skill_local[skill_name][SKILL_CODE_KEY])
        skill = eval(skill_name)
        return skill


    def store_skills(self, file_path = None) -> None:

        if file_path == None:
            file_path = os.path.join(self.store_path, self.skill_library_filename)

        store_file = {}
        for skill in self.skill_index:
            store_file[skill[SKILL_NAME_KEY]] = {SKILL_CODE_KEY:skill[SKILL_CODE_KEY],
                                                 SKILL_EMBEDDING_KEY:base64.b64encode(skill[SKILL_EMBEDDING_KEY].tobytes()).decode('utf-8'),
                                                 SKILL_CODE_HASH_KEY:base64.b64encode(skill[SKILL_CODE_KEY].encode('utf-8')).decode('utf-8')}

        save_json(file_path = file_path, json_dict = store_file, indent = 4)


    def load_skill_library(self, file_name) -> None:

        skill_local = load_json(file_name)

        self.skill_index = []
        self.skill_registry = {}

        for skill_name in skill_local.keys():

            if skill_name in SKILL_REGISTRY:

                # the manually-designed skills follow the code in .py files
                self.skill_registry[skill_name] = SKILL_REGISTRY[skill_name]

                skill_code_base64 = base64.b64encode(skill_local[skill_name][SKILL_CODE_KEY].encode('utf-8')).decode('utf-8')

                if skill_code_base64 == skill_local[skill_name][SKILL_CODE_HASH_KEY]: # the skill_code is not modified
                    self.skill_index.append({SKILL_NAME_KEY:skill_name,
                                             SKILL_EMBEDDING_KEY:np.frombuffer(base64.b64decode(skill_local[skill_name][SKILL_EMBEDDING_KEY]), dtype=np.float64),
                                             SKILL_CODE_KEY:inspect.getsource(SKILL_REGISTRY[skill_name])})

                else: # skill_code has been modified, we should recompute embeddings
                    self.skill_index.append({SKILL_NAME_KEY:skill_name,
                                             SKILL_EMBEDDING_KEY:self.get_embedding(skill_name, inspect.getdoc(SKILL_REGISTRY[skill_name])),
                                             SKILL_CODE_KEY:inspect.getsource(SKILL_REGISTRY[skill_name])})

            else:
                # the skills got from gather_information follow the code in .json file
                skill = self.convert_str_to_func(skill_name, skill_local)
                self.skill_registry[skill_name] = skill

                skill_code_base64 = base64.b64encode(skill_local[skill_name][SKILL_CODE_KEY].encode('utf-8')).decode('utf-8')

                if skill_code_base64 == skill_local[skill_name][SKILL_CODE_HASH_KEY]: # the skill_code is not modified
                    self.skill_index.append({SKILL_NAME_KEY:skill_name,
                                             SKILL_EMBEDDING_KEY:np.frombuffer(base64.b64decode(skill_local[skill_name][SKILL_EMBEDDING_KEY]), dtype=np.float64),
                                             SKILL_CODE_KEY:skill_local[skill_name][SKILL_CODE_KEY]})

                else: # skill_code has been modified, we should recompute embedding
                    self.skill_index.append({SKILL_NAME_KEY:skill_name,
                                             SKILL_EMBEDDING_KEY:self.get_embedding(skill_name, inspect.getdoc(skill)),
                                             SKILL_CODE_KEY:skill_local[skill_name][SKILL_CODE_KEY]})


    def filter_skill_library(self) -> None:

        if self.skill_scope == 'Basic':
            self.skill_registry = {}
            self.skill_index = []
            for skill in SKILL_INDEX:
                if skill[SKILL_NAME_KEY] in self.basic_skills:
                    self.skill_registry[skill[SKILL_NAME_KEY]] = SKILL_REGISTRY[skill[SKILL_NAME_KEY]]
                    self.skill_index.append(skill)

        if self.skill_scope == 'Full':
            self.skill_registry = copy.deepcopy(SKILL_REGISTRY)
            self.skill_index = copy.deepcopy(SKILL_INDEX)

        if self.skill_scope == None:
            self.skill_registry = {}
            self.skill_index = []
            for skill in SKILL_INDEX:
                if skill[SKILL_NAME_KEY] in self.necessary_skills:
                    self.skill_registry[skill[SKILL_NAME_KEY]] = SKILL_REGISTRY[skill[SKILL_NAME_KEY]]
                    self.skill_index.append(skill)

        for skill in self.skill_index:
            skill[SKILL_EMBEDDING_KEY] = self.get_embedding(skill[SKILL_NAME_KEY], inspect.getdoc(SKILL_REGISTRY[skill[SKILL_NAME_KEY]]))
