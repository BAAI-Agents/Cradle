from typing import Dict, Any

from cradle.config import Config
from cradle.log import Logger
from cradle.utils.file_utils import assemble_project_path, exists_in_project_path

config = Config()
logger = Logger()


def is_valid_value(value: Any) -> bool:
    if value is None:
        return False
    elif isinstance(value, str) and value.strip() == "":
        return False
    elif isinstance(value, list) and len(value) == 0:
        return False
    elif isinstance(value, dict) and len(value) == 0:
        return False
    return True


def check_planner_params(planner: Dict = None):

    flag = True

    try:
        # check keys
        assert "prompt_paths" in planner, f"prompt_paths is not in planner"
        assert "__check_list__" in planner, f"__check_list__ is not in planner"

        prompt_paths = planner["prompt_paths"]
        assert "inputs" in prompt_paths, f"input_example is not in prompt_paths"
        assert "templates" in prompt_paths, f"templates is not in prompt_paths"

        input_example = prompt_paths["inputs"]
        templates = prompt_paths["templates"]

        """check modules"""
        check_list = planner["__check_list__"]
        for check_item in check_list:
            assert (check_item in input_example and
                    exists_in_project_path(input_example[check_item])), \
                f"{check_item} is not in input_example or {assemble_project_path(input_example[check_item])} does not exist"
            assert (check_item in templates and
                    exists_in_project_path(templates[check_item])), \
                f"{check_item} is not in template or {assemble_project_path(templates[check_item])} does not exist"

    except Exception as e:
        logger.error(f"Error in check_prompt_paths: {e}")
        flag = False

    return flag
