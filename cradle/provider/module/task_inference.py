import os
import re
from typing import List, Dict, Any
import json
from copy import deepcopy

from cradle.provider import BaseProvider
from cradle.log import Logger
from cradle.config import Config
from cradle.memory import LocalMemory
from cradle.provider import VideoRecordProvider
from cradle.utils.file_utils import read_resource_file, assemble_project_path
from cradle.utils.json_utils import parse_semi_formatted_text

config = Config()
logger = Logger()
memory = LocalMemory()
video_record = VideoRecordProvider()

class TaskInferenceProvider(BaseProvider):
    def __init__(self,
                 *args,
                 template_path: str,
                 llm_provider: Any = None,
                 gm: Any = None,
                 use_subtask: bool = False,
                 **kwargs):
        super(TaskInferenceProvider, self).__init__(*args, **kwargs)
        self.template_path = template_path
        self.llm_provider = llm_provider
        self.use_subtask = use_subtask
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

    def _preprocess(self, params: Dict[str, Any], use_screen_shot_augmented = False, use_video = False, **kwargs):

        if not use_video:
            screen_shot_path = memory.get_recent_history("screen_shot_path")[-1]
            screen_shot_augmnented_path = memory.get_recent_history("screen_shot_augmented_path")[-1]

            if not use_screen_shot_augmented:
                image_introduction = [
                    {
                        "introduction": "This screenshot is the current step of the game.",
                        "path": screen_shot_path,
                        "assistant": ""
                    }
                ]
            else:
                image_introduction = [
                    {
                        "introduction": "This screenshot is the current step of the game.",
                        "path": screen_shot_augmnented_path,
                        "assistant": ""
                    }
                ]

            res_params = {
                "image_introduction": image_introduction
            }

        else:
            images = memory.get_recent_history('screen_shot_path', config.event_count)
            reasonings = memory.get_recent_history('decision_making_reasoning', config.event_count)

            image_introduction = [
                {
                    "path": images[event_i],
                    "assistant": "",
                    "introduction": 'This is the {} screenshot of recent events. The description of this image: {}'.format(
                        ['first', 'second', 'third', 'fourth', 'fifth'][event_i], reasonings[event_i])
                } for event_i in range(config.event_count)
            ]

            res_params = {
                "image_introduction": image_introduction,
                "event_count": config.event_count
            }

        memory.current_info.update(res_params)
        return res_params

    def _postprocess(self, processed_response: Dict[str, Any], **kwargs):
        # Check output keys
        self._check_output_keys(processed_response)

        subtask_description = processed_response["subtask"]

        res_params = {
            key: processed_response[key] for key in self.output_keys
        }
        res_params.update({
            "subtask_description": subtask_description
        })

        if not self.use_subtask:
            for key in self.output_keys:
                if "subtask" in key:
                    res_params.pop(key)

        return res_params

    @BaseProvider.debug
    @BaseProvider.error
    @BaseProvider.write
    def __call__(self,
                 *args,
                 use_screen_shot_augmented = False,
                 used_video = False,
                 **kwargs):

        params = deepcopy(memory.current_info)
        params.update(self._preprocess(params, use_screen_shot_augmented,  used_video, **kwargs))

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

class RDR2TaskInferenceProvider(BaseProvider):
    def __init__(self,
                 *args,
                 planner,
                 gm,
                 **kwargs):
        super(RDR2TaskInferenceProvider, self).__init__(*args, **kwargs)
        self.planner = planner
        self.gm = gm

    def __call__(self, *args, **kwargs):

        task_description = memory.get_recent_history("task_description", k=1)[0]
        cur_screenshot_path = memory.get_recent_history("screen_shot_path", k=1)[0]

        # Information summary preparation
        if len(memory.get_recent_history("decision_making_reasoning", memory.max_recent_steps)) == memory.max_recent_steps:

            input = self.planner.task_inference_.input_map
            logger.write(f'> Information summary call...')

            images = memory.get_recent_history('screen_shot_path', config.event_count)
            reasonings = memory.get_recent_history('decision_making_reasoning', config.event_count)

            image_introduction = [
                {
                    "path": images[event_i], "assistant": "",
                    "introduction": 'This is the {} screenshot of recent events. The description of this image: {}'.format(
                        ['first', 'second', 'third', 'fourth', 'fifth'][event_i], reasonings[event_i])
                } for event_i in range(config.event_count)
            ]

            input["image_introduction"] = image_introduction
            input["previous_summarization"] = memory.get_summarization()
            input["task_description"] = task_description
            input["event_count"] = str(config.event_count)

            # >> Calling INFORMATION SUMMARY
            logger.write(f'>> Calling INFORMATION SUMMARY')

            data = self.planner.task_inference(input=input)
            if 'info_summary' not in data['res_dict'].keys():
                data['res_dict']['info_summary'] = ''
            info_summary = data['res_dict']['info_summary']
            entities_and_behaviors = data['res_dict']['entities_and_behaviors']
            logger.write(f'R: Summary: {info_summary}')
            logger.write(f'R: entities_and_behaviors: {entities_and_behaviors}')

            memory.update_info_history({
                "summarization": info_summary,
            })

        memory.update_info_history(
            {
                "screen_shot_path": cur_screenshot_path,
            }
        )

class StardewTaskInferenceProvider(BaseProvider):
    def __init__(self,
                 *args,
                 planner,
                 gm,
                 **kwargs):
        super(StardewTaskInferenceProvider, self).__init__(*args, **kwargs)
        self.planner = planner
        self.gm = gm

    def __call__(self, *args, **kwargs):

        task_description = memory.get_recent_history("task_description", k=1)[0]
        cur_screen_shot_path = memory.get_recent_history("screen_shot_path", k=1)[0]
        previous_summarization = memory.get_recent_history("summarization", 1)[0]
        substask_description = memory.get_recent_history("subtask_description", 1)[0]
        substask_reasoning = memory.get_recent_history("subtask_reasoning", 1)[0]
        toolbar_information = memory.get_recent_history("toolbar_information", 1)[0]

        # Information summary preparation
        input = self.planner.task_inference_.input_map
        logger.write(f'> Information summary call...')

        images = memory.get_recent_history('augmented_screen_shot_path', 1)
        decision_making_reasoning = memory.get_recent_history('decision_making_reasoning', 1)
        self_reflection_reasoning = memory.get_recent_history('self_reflection_reasoning', 1)

        image_introduction = []
        image_introduction.append(
            {
                "introduction": "This screenshot is the current step of the game. The blue band represents the left side and the yellow band represents the right side.",
                "path": images,
                "assistant": ""
            })

        input["image_introduction"] = image_introduction
        input["previous_summarization"] = previous_summarization
        input["task_description"] = task_description
        input["subtask_description"] = substask_description
        input["subtask_reasoning"] = substask_reasoning
        input["previous_reasoning"] = decision_making_reasoning
        input["self_reflection_reasoning"] = self_reflection_reasoning
        input["toolbar_information"] = toolbar_information

        # >> Calling INFORMATION SUMMARY
        logger.write(f'>> Calling INFORMATION SUMMARY')

        data = self.planner.task_inference(input=input)
        history_summary = data['res_dict']['history_summary']
        # entities_and_behaviors = data['res_dict']['entities_and_behaviors']
        logger.write(f'R: Summary: {history_summary}')
        # logger.write(f'R: entities_and_behaviors: {entities_and_behaviors}')
        # self.memory.add_summarization(history_summary)

        # self.memory.add_recent_history("image", cur_screen_shot_path)

        subtask_description = data['res_dict']['subtask']
        subtask_reasoning = data['res_dict']['subtask_reasoning']
        logger.write(f'R: Subtask: {subtask_description}')
        logger.write(f'R: Subtask reasoning: {subtask_reasoning}')

        res_params = {
            'summarization': history_summary,
            'subtask_description': subtask_description,
            'subtask_reasoning': subtask_reasoning
        }

        memory.update_info_history(res_params)