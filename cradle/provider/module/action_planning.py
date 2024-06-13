import os
import re
from typing import List, Dict, Any
import json
from copy import deepcopy

from cradle.utils.file_utils import assemble_project_path, read_resource_file
from cradle.utils.json_utils import parse_semi_formatted_text
from cradle.provider import BaseProvider
from cradle.log import Logger
from cradle.config import Config
from cradle.memory import LocalMemory
from cradle.provider import VideoRecordProvider
from cradle import constants

config = Config()
logger = Logger()
memory = LocalMemory()
video_record = VideoRecordProvider()

class ActionPlanningProvider(BaseProvider):
    def __init__(self,
                 *args,
                 template_path: str,
                 llm_provider: Any = None,
                 gm: Any = None,
                 **kwargs):
        super(ActionPlanningProvider, self).__init__(*args, **kwargs)
        self.template_path = template_path
        self.llm_provider = llm_provider
        self.gm = gm

        self.template, self.input_keys, self.output_keys = self._extract_keys_from_template()

    @BaseProvider.write
    def _extract_keys_from_template(self):
        template_path = assemble_project_path(self.template_path)
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template file {template_path} does not exist")

        template = read_resource_file(template_path)

        # <$few_shots$> -> few_shots
        parse_input_keys = re.findall(r'<\$(.*?)\$>', template)
        input_keys = [key.strip() for key in parse_input_keys]
        logger.write(f"Recommended input parameters: {input_keys}")

        # TODO: Extract output text should be general
        start_output_line_index = template.find('You should only respond')
        output_text = template[start_output_line_index+1:]
        output = parse_semi_formatted_text(output_text)
        output_keys = list(output.keys())
        logger.write(f"Recommended output parameters: {output_keys}")

        return template, input_keys, output_keys

    @BaseProvider.write
    def _check_input_keys(self, params: Dict[str, Any]):
        for key in self.input_keys:
            if key not in params:
                logger.write(f"Key {key} is not in the input parameters")
                params[key] = None

    @BaseProvider.error
    def _check_output_keys(self, processed_response: Dict[str, Any]):
        for key in self.output_keys:
            if key not in processed_response:
                logger.error(f"Key {key} is not in the response")
                processed_response[key] = None

    def _preprocess(self, params: Dict[str, Any], use_screen_shot_augmented = False,  **kwargs):

        prompts = [
            "This screenshot is the previous step of the game.",
            "This screenshot is the current step of the game."
        ]

        screen_shot_paths = memory.get_recent_history("screen_shot_path", k=config.action_planning_image_num)
        screen_shot_augmnented_paths = memory.get_recent_history("screen_shot_augmented_path", k=config.action_planning_image_num)

        if not use_screen_shot_augmented:
            image_introduction = []
            for i in range(len(screen_shot_paths), 0, -1):
                image_introduction.append(
                    {
                        "introduction": prompts[-i],
                        "path": screen_shot_paths[-i],
                        "assistant": ""
                    })
        else:
            image_introduction = []
            for i in range(len(screen_shot_augmnented_paths), 0, -1):
                image_introduction.append(
                    {
                        "introduction": prompts[-i],
                        "path": screen_shot_augmnented_paths[-i],
                        "assistant": ""
                    })

        res_params = {
            "image_introduction": image_introduction
        }

        memory.current_info.update(res_params)
        return res_params

    def _postprocess(self, processed_response: Dict[str, Any], **kwargs):

        skill_steps = []
        if 'actions' in processed_response:
            skill_steps = processed_response['actions']

        if skill_steps:
            skill_steps = [i for i in skill_steps if i != '']
        else:
            skill_steps = ['']

        skill_steps = skill_steps[:config.number_of_execute_skills]

        if config.number_of_execute_skills > 1:
            actions = "[" + ",".join(skill_steps) + "]"
        else:
            actions = str(skill_steps[0])

        decision_making_reasoning = processed_response['reasoning']

        self._check_output_keys(processed_response)

        res_params = {
            key: processed_response[key] for key in self.output_keys
        }
        res_params.update({
            "actions": actions,
            "decision_making_reasoning": decision_making_reasoning,
            "skill_steps": skill_steps,
        })

        return res_params

    @BaseProvider.debug
    @BaseProvider.error
    @BaseProvider.write
    def __call__(self,
                 *args,
                 use_screen_shot_augmented = False,
                 **kwargs):

        params = deepcopy(memory.current_info)
        params.update(self._preprocess(params, use_screen_shot_augmented,  **kwargs))

        self._check_input_keys(params)

        message_prompts = self.llm_provider.assemble_prompt(template_str=self.template, params=params)
        logger.debug(f'{logger.UPSTREAM_MASK}{json.dumps(message_prompts, ensure_ascii=False)}\n')

        processed_response = {}
        try:
            response, info = self.llm_provider.create_completion(message_prompts)
            logger.debug(f'{logger.DOWNSTREAM_MASK}{response}\n')

            # Convert the response to dict
            processed_response = parse_semi_formatted_text(response)

        except Exception as e:
            logger.error(f"Response of image description is not in the correct format: {e}, retrying...")

        res_params = self._postprocess(processed_response, **kwargs)
        memory.update_info_history(res_params)

        del params

        return res_params

class RDR2ActionPlanningProvider(BaseProvider):
    def __init__(self,
                 *args,
                 planner,
                 gm,
                 **kwargs):
        super(RDR2ActionPlanningProvider, self).__init__(*args, **kwargs)
        self.planner = planner
        self.gm = gm

    def __call__(self, *args, **kwargs):
        response_keys = memory.get_recent_history("response_keys", k=1)[0]
        response = memory.get_recent_history("response", k=1)[0]
        pre_action = memory.get_recent_history("pre_action", k=1)[0]
        pre_self_reflection_reasoning = memory.get_recent_history("pre_self_reflection_reasoning", k=1)[0]
        pre_screen_classification = memory.get_recent_history("pre_screen_classification", k=1)[0]
        screen_classification = memory.get_recent_history("screen_classification", k=1)[0]
        skill_library = memory.get_recent_history("skill_library", k=1)[0]
        task_description = memory.get_recent_history("task_description", k=1)[0]

        # Decision making preparation
        input = deepcopy(self.planner.action_planning_.input_map)
        img_prompt_decision_making = self.planner.action_planning_.input_map["image_introduction"]

        number_of_execute_skills = input["number_of_execute_skills"]

        if pre_action:
            input["previous_action"] = memory.get_recent_history("action", k=1)[-1]
            input["previous_reasoning"] = memory.get_recent_history("decision_making_reasoning", k=1)[-1]

        if pre_self_reflection_reasoning:
            input["previous_self_reflection_reasoning"] = memory.get_recent_history("self_reflection_reasoning", k=1)[-1]

        input['skill_library'] = skill_library
        input['info_summary'] = memory.get_recent_history("summarization", k=1)[0]

        # @TODO: few shots should be REMOVED in prompt decision making
        input['few_shots'] = []

        # @TODO Temporary solution with fake augmented entries if no bounding box exists. Ideally it should read images, then check for possible augmentation.
        image_memory = memory.get_recent_history("screen_shot_path", k=config.action_planning_image_num)
        augmented_image_memory = memory.get_recent_history(constants.AUGMENTED_IMAGES_MEM_BUCKET, k=config.action_planning_image_num)

        image_introduction = []
        for i in range(len(image_memory), 0, -1):

            if len(augmented_image_memory) >=i and augmented_image_memory[-i] != constants.NO_IMAGE:
                image_introduction.append(
                    {
                        "introduction": img_prompt_decision_making[-i]["introduction"],
                        "path": augmented_image_memory[-i],
                        "assistant": img_prompt_decision_making[-i]["assistant"]
                    })
            else:
                image_introduction.append(
                    {
                        "introduction": img_prompt_decision_making[-i]["introduction"],
                        "path": image_memory[-i],
                        "assistant": img_prompt_decision_making[-i]["assistant"]
                    })

        input["image_introduction"] = image_introduction
        input["task_description"] = task_description

        # Minimap info tracking
        if constants.MINIMAP_INFORMATION in response_keys:
            minimap_information = response[constants.MINIMAP_INFORMATION]
            logger.write(f"{constants.MINIMAP_INFORMATION}: {minimap_information}")

            minimap_info_str = ""
            for key, value in minimap_information.items():
                if value:
                    for index, item in enumerate(value):
                        minimap_info_str = minimap_info_str + key + ' ' + str(index) + ': angle ' + str(
                            int(item['theta'])) + ' degree' + '\n'
            minimap_info_str = minimap_info_str.rstrip('\n')

            logger.write(f'minimap_info_str: {minimap_info_str}')
            input[constants.MINIMAP_INFORMATION] = minimap_info_str

        data = self.planner.action_planning(input=input)

        skill_steps = data['res_dict']['actions']
        if skill_steps is None:
            skill_steps = []

        logger.write(f'R: {skill_steps}')

        # Filter nop actions in list
        skill_steps = [i for i in skill_steps if i != '']
        if len(skill_steps) == 0:
            skill_steps = ['']

        skill_steps = skill_steps[:number_of_execute_skills]
        logger.write(f'Skill Steps: {skill_steps}')

        self.gm.unpause_game()

        # @TODO: Rename GENERAL_GAME_INTERFACE
        if (pre_screen_classification.lower() == constants.GENERAL_GAME_INTERFACE and
                (screen_classification.lower() == constants.MAP_INTERFACE or
                 screen_classification.lower() == constants.SATCHEL_INTERFACE) and pre_action):
            exec_info = self.execute_actions([pre_action])

        start_frame_id = video_record.get_current_frame_id()

        exec_info = self.gm.execute_actions(skill_steps)

        cur_screenshot_path = self.gm.capture_screen()

        end_frame_id = video_record.get_current_frame_id()

        pause_flag = self.gm.pause_game(screen_classification.lower())
        logger.write(f'Pause flag: {pause_flag}')
        if not pause_flag:
            self.gm.pause_game(screen_type=None)

        # exec_info also has the list of successfully executed skills. skill_steps is the full list, which may differ if there were execution errors.
        pre_action = exec_info["last_skill"]

        pre_decision_making_reasoning = ''
        if 'res_dict' in data.keys() and 'reasoning' in data['res_dict'].keys():
            pre_decision_making_reasoning = data['res_dict']['reasoning']

        pre_screen_classification = screen_classification

        memory.update_info_history(
            {
                "action": pre_action,
                "decision_making_reasoning": pre_decision_making_reasoning,
            }
        )

        res_params = {
            "pre_action": pre_action,
            "pre_decision_making_reasoning": pre_decision_making_reasoning,
            "pre_screen_classification": pre_screen_classification,
            "exec_info": exec_info,
            "start_frame_id": start_frame_id,
            "end_frame_id": end_frame_id,
            "screen_shot_path": cur_screenshot_path,
        }

        memory.update_info_history(res_params)


class StardewActionPlanningProvider(BaseProvider):
    def __init__(self,
                 *args,
                 planner,
                 gm,
                 **kwargs):
        super(StardewActionPlanningProvider, self).__init__(*args, **kwargs)
        self.planner = planner
        self.gm = gm

    def __call__(self, *args, **kwargs):

        pre_action = memory.get_recent_history("pre_action", k=1)[0]
        pre_self_reflection_reasoning = memory.get_recent_history("pre_self_reflection_reasoning", k=1)[0]
        toolbar_information = memory.get_recent_history("toolbar_information", k=1)[0]
        selected_position = memory.get_recent_history("selected_position", k=1)[0]
        summarization = memory.get_recent_history("summarization", k=1)[0]
        skill_library = memory.get_recent_history("skill_library", k=1)[0]
        task_description = memory.get_recent_history("task_description", k=1)[0]
        subtask_description = memory.get_recent_history("subtask_description", k=1)[0]
        history_summary = memory.get_recent_history("summarization", k=1)[0]

        # Decision making preparation
        input = deepcopy(self.planner.action_planning_.input_map)
        img_prompt_decision_making = self.planner.action_planning_.input_map["image_introduction"]
        input["toolbar_information"] = toolbar_information if toolbar_information is not None else input[
            "toolbar_information"]
        input["selected_position"] = selected_position if selected_position is not None else input["selected_position"]

        number_of_execute_skills = input["number_of_execute_skills"]

        if pre_action:
            input["previous_action"] = memory.get_recent_history("action", k=1)[0]
            input["previous_reasoning"] = memory.get_recent_history("decision_making_reasoning", k=1)[0]

        if pre_self_reflection_reasoning:
            input["previous_self_reflection_reasoning"] =  memory.get_recent_history("self_reflection_reasoning", k=1)[0]

        input['skill_library'] = skill_library

        # @TODO Temporary solution with fake augmented entries if no bounding box exists. Ideally it should read images, then check for possible augmentation.
        image_memory = memory.get_recent_history("augmented_image", k=config.action_planning_image_num)

        image_introduction = []
        for i in range(len(image_memory), 0, -1):
            image_introduction.append(
                {
                    "introduction": img_prompt_decision_making[-i]["introduction"],
                    "path": image_memory[-i],
                    "assistant": img_prompt_decision_making[-i]["assistant"]
                })

        input["image_introduction"] = image_introduction
        input["task_description"] = task_description
        input["subtask_description"] = subtask_description
        input["history_summary"] = history_summary

        data = self.planner.action_planning(input=input)

        skill_steps = data['res_dict']['actions']
        if skill_steps is None:
            skill_steps = []

        logger.write(f'R: {skill_steps}')

        # Filter nop actions in list
        skill_steps = [i for i in skill_steps if i != '']
        if len(skill_steps) == 0:
            skill_steps = ['']

        skill_steps = skill_steps[:number_of_execute_skills]
        logger.write(f'Skill Steps: {skill_steps}')

        self.gm.unpause_game()

        start_frame_id =video_record.get_current_frame_id()

        exec_info = self.gm.execute_actions(skill_steps)

        cur_screen_shot_path = self.gm.capture_screen()

        end_frame_id = video_record.get_current_frame_id()
        self.gm.pause_game(ide_name=config.IDE_NAME)

        pre_action = "[" + ",".join(skill_steps) + "]"

        pre_decision_making_reasoning = ''
        if 'res_dict' in data.keys() and 'reasoning' in data['res_dict'].keys():
            pre_decision_making_reasoning = data['res_dict']['reasoning']

        memory.update_info_history({
            "action": pre_action,
            "decision_making_reasoning": pre_decision_making_reasoning,
        })

        logger.write(f'Decision reasoning: {pre_decision_making_reasoning}')

        res_params = {
            "pre_action": pre_action,
            "pre_decision_making_reasoning": pre_decision_making_reasoning,
            "exec_info": exec_info,
            "start_frame_id": start_frame_id,
            "end_frame_id": end_frame_id,
            "cur_screen_shot_path": cur_screen_shot_path,
            "summarization": summarization
        }

        memory.update_info_history(res_params)
