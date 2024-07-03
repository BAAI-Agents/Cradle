import inspect
import base64
import re
from typing import Dict, List, Any

from cradle import constants
from cradle.config.config import Config
from cradle.log import Logger
from cradle.environment import SkillRegistry
from cradle.environment import Skill
from cradle.gameio.lifecycle.ui_control import normalize_coordinates
from cradle.environment.software.ui_control import SoftwareUIControl
from cradle.utils.singleton import Singleton


config = Config()
logger = Logger()

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


class SoftwareSkillRegistry(SkillRegistry, metaclass=Singleton):
    def __init__(self,
                 *args,
                 skill_configs: dict[str, Any]  = config.skill_configs,
                 embedding_provider=None,
                 **kwargs):

        super().__init__(skill_configs=skill_configs,
                         embedding_provider=embedding_provider)

        if skill_configs[constants.SKILL_CONFIG_REGISTERED_SKILLS] is not None:
            self.skill_registered = skill_configs[constants.SKILL_CONFIG_REGISTERED_SKILLS]
        else:
            self.skill_registered = SKILLS


    def pre_process_skill_library(skill_library_list: List[str]) -> List[str]:
        """
        Pre-process the skill library to filter out specific functions.

        Args:
            skill_library_list (List[str]): The list of skills to be processed.

        Returns:
            List[str]: The processed list of skills.
        """

        processed_skill_library = skill_library_list.copy()
        processed_skill_library = [skill for skill in processed_skill_library if 'click_at_position(x, y, mouse_button)' not in skill['function_expression']]
        processed_skill_library = [skill for skill in processed_skill_library if 'double_click_at_position(x, y, mouse_button)' not in skill['function_expression']]
        processed_skill_library = [skill for skill in processed_skill_library if 'mouse_drag(source_x, source_y, target_x, target_y, mouse_button)' not in skill['function_expression']]
        processed_skill_library = [skill for skill in processed_skill_library if 'move_mouse_to_position(x, y)' not in skill['function_expression']]

        return processed_skill_library


    def pre_process_skill_steps(skill_steps: List[str], som_map: Dict) -> List[str]:

        processed_skill_steps = skill_steps.copy()

        for i in range(len(processed_skill_steps)):
            try:
                step = processed_skill_steps[i]

                # Remove leading and trailing ' or " from step
                if len(step) > 1 and ((step[0] == '"' and step[-1] == '"') or (step[0] == "'" and step[-1] == "'")):
                    step = step[1:-1]
                    processed_skill_steps[i] = step

                # Change label_id to x, y coordinates for click_on_label, double_click_on_label and hover_on_label
                if 'on_label(' in step:
                    skill = step
                    tokens = skill.split('(')
                    args_suffix_str = tokens[1]
                    func_str = tokens[0]

                    if 'label_id=' in args_suffix_str or 'label=' in args_suffix_str:
                        try:
                            label_key = 'label_id=' if 'label_id=' in args_suffix_str else 'label='
                            label_id = str(args_suffix_str.split(label_key)[1].split(',')[0].split(')')[0]).replace("'", "").replace('"', "").replace(" ", "")

                            if label_id in som_map:
                                x, y = normalize_coordinates(som_map[label_id])
                                args_suffix_str = args_suffix_str.replace(f'{label_key}{label_id}', f'x={x}, y={y}').replace(",)", ")")
                                if func_str.startswith('click_'):
                                    processed_skill_steps[i] = f'click_at_position({args_suffix_str} # Click on {label_key.strip("=")}: {label_id}'
                                elif func_str.startswith('double_'):
                                    processed_skill_steps[i] = f'double_click_at_position({args_suffix_str} # Double click on {label_key.strip("=")}: {label_id}'
                                else:
                                    processed_skill_steps[i] = f'move_mouse_to_position({args_suffix_str} # Move to {label_key.strip("=")}: {label_id}'

                                if config.disable_close_app_icon and SoftwareUIControl.check_for_close_icon(skill, x, y):
                                    processed_skill_steps[i] = f"# {processed_skill_steps[i]} # {constants.CLOSE_ICON_DETECTED}"

                            else:
                                logger.debug(f"{label_key.strip('=')} {label_id} not found in SOM map.")
                                msg = f" # {constants.INVALID_BBOX} for {label_key.strip('=')}: {label_id}"

                                # HACK to go back
                                if 'click_on_label(' in step and "# invalid_bbox for label_id: 99" in msg:
                                    processed_skill_steps[i] = f"go_back_to_target_application()"
                                else:
                                    processed_skill_steps[i] = f"{processed_skill_steps[i]}{msg}"

                        except:
                            logger.error("Invalid skill format.")
                            processed_skill_steps[i] = processed_skill_steps[i] + f"# {constants.INVALID_BBOX} for invalid skill format."

                    else:
                        # Handle case without label_id or label
                        coords_str = args_suffix_str.split(')')[0]
                        coords_list = [s.strip() for s in re.split(r'[,\s]+', coords_str) if s.isdigit()]
                        if len(coords_list) == 2:
                            x, y = coords_list
                            args_suffix_str = args_suffix_str.replace(coords_str, f'x={x}, y={y}')
                            if func_str.startswith('click_'):
                                processed_skill_steps[i] = f'click_at_position({args_suffix_str}'
                            elif func_str.startswith('double_'):
                                processed_skill_steps[i] = f'double_click_at_position({args_suffix_str}'
                            else:
                                processed_skill_steps[i] = f'move_mouse_to_position({args_suffix_str}'

                            if config.disable_close_app_icon and SoftwareUIControl.check_for_close_icon(skill, x, y):
                                    processed_skill_steps[i] = f"# {processed_skill_steps[i]} # {constants.CLOSE_ICON_DETECTED}"

                        else:
                            logger.error("Invalid coordinate format.")
                            processed_skill_steps[i] = processed_skill_steps[i] + f"# {constants.INVALID_BBOX} for coordinates: {coords_str}"

                elif 'mouse_drag_with_label(' in step:
                    skill = step
                    tokens = skill.split('(')
                    args_suffix_str = tokens[1]
                    func_str = tokens[0]

                    label_ids = args_suffix_str.split('label_id=')[1:]
                    source_label_id = str(label_ids[0].split(',')[0]).replace("'", "").replace('"', "").replace(" ", "")
                    target_label_id = str(label_ids[1].split(',')[0]).replace("'", "").replace('"', "").replace(" ", "")

                    if source_label_id in som_map and target_label_id in som_map:
                        source_x, source_y = normalize_coordinates(som_map[source_label_id])
                        target_x, target_y = normalize_coordinates(som_map[target_label_id])
                        args_suffix_str = args_suffix_str.replace(f'source_label_id={source_label_id}', f'source_x={source_x}, source_y={source_y}')
                        args_suffix_str = args_suffix_str.replace(f'target_label_id={target_label_id}', f'target_x={target_x}, target_y={target_y}').replace(",)", ")")

                        processed_skill_steps[i] = f'mouse_drag({args_suffix_str} # Drag things from  source_label_id={source_label_id} to target_label_id={target_label_id}'
                    else:
                        missing_ids = [label_id for label_id in [source_label_id, target_label_id] if label_id not in som_map]
                        logger.debug(f"Label IDs {missing_ids} not found in SOM map.")
                        msg = f" # {constants.INVALID_BBOX} for label_ids: {', '.join(missing_ids)}"
                        processed_skill_steps[i] = f"{step}{msg}"

                # Change keyboard and mouse combination
                elif '+' in step and 'key' in step:
                    skill = step.replace('+', ",")
                    processed_skill_steps[i] = skill

                if ('Control' in step or 'control' in step) and 'press_key' in step:
                    step = re.sub(r'(?i)control', 'ctrl', step)
                    processed_skill_steps[i] = step

                if 'press_keys_combined(' in step:
                    pattern = re.compile(r'press_keys_combined\((keys=)?(\[.*?\]|\(.*?\)|".*?"|\'.*?\')\)')
                    match = pattern.search(step)

                    if match:
                        keys_str = match.group(2)
                        keys_str = keys_str.strip('[]()"\'')
                        keys_list = [key.strip().replace('"', '').replace("'", '') for key in keys_str.split(',')]
                        keys_processed = ', '.join(keys_list)
                        new_step = f"press_keys_combined(keys='{keys_processed}')"
                        processed_skill_steps[i] = new_step

            except Exception as e:
                logger.error(f"Error processing skill steps: {e}")
                processed_skill_steps[i] = f"{step} # Invalid skill format."

            if processed_skill_steps != skill_steps:
                logger.write(f'>>> {skill_steps} -> {processed_skill_steps} <<<')

        return processed_skill_steps
