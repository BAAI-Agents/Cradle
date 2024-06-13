import os
import re
from typing import List, Dict, Any
import json
from copy import deepcopy

from cradle.provider import BaseProvider
from cradle.utils.file_utils import assemble_project_path, read_resource_file
from cradle.utils.json_utils import parse_semi_formatted_text
from cradle.log import Logger
from cradle.config import Config
from cradle.memory import LocalMemory
from cradle.provider import VideoRecordProvider
from cradle.utils.check import is_valid_value

config = Config()
logger = Logger()
memory = LocalMemory()
video_record = VideoRecordProvider()

class SelfReflectionProvider(BaseProvider):
    def __init__(self,
                 *args,
                 template_path: str,
                 llm_provider: Any = None,
                 gm: Any = None,
                 **kwargs):
        super(SelfReflectionProvider, self).__init__(*args, **kwargs)
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

    def _preprocess(self, params: Dict[str, Any],  use_screen_shot_augmented = False, used_video = False, **kwargs):

        if not used_video:
            prompts = [
                "This screenshot is the previous observation before executing the last action.",
                "This screenshot is the current observation after executing the last action."
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
                            "assistant": "",
                            "resolution": "low"
                        })
            else:
                image_introduction = []
                for i in range(len(screen_shot_augmnented_paths), 0, -1):
                    image_introduction.append(
                        {
                            "introduction": prompts[-i],
                            "path": screen_shot_augmnented_paths[-i],
                            "assistant": "",
                            "resolution": "low"
                        })

            res_params = {
                "image_introduction": image_introduction
            }
        else:

            start_frame_id = memory.get_recent_history("start_frame_id", k=1)
            end_frame_id = memory.get_recent_history("end_frame_id", k=1)

            action_frames = []
            video_frames = video_record.get_frames(start_frame_id, end_frame_id)

            if len(video_frames) <= config.max_images_in_self_reflection * config.duplicate_frames + 1:
                action_frames = [frame[1] for frame in video_frames[1::config.duplicate_frames]]
            else:
                for i in range(config.max_images_in_self_reflection):
                    step = len(video_frames) // config.max_images_in_self_reflection * i + 1
                    action_frames.append(video_frames[step][1])

            image_introduction = [
                {
                    "introduction": "Here are the sequential frames of the character executing the last action.",
                    "path": action_frames,
                    "assistant": "",
                    "resolution": "low"
                }
            ]

            actions = memory.get_recent_history("actions", k=1)
            action_code = ""
            action_str = ""
            if is_valid_value(actions):
                pre_action = actions[0]
                pre_action_name, _ = self.gm.skill_registry.convert_expression_to_skill(pre_action)
                action_str = pre_action_name
                action_code, action_code_info = self.gm.get_skill_library_in_code(pre_action_name)
                action_code = action_code if action_code is not None else action_code_info

            res_params = {
                "image_introduction": image_introduction,
                "actions": action_str,
                "action_code": action_code
            }

        memory.current_info.update(res_params)
        return res_params

    def _postprocess(self, processed_response: Dict[str, Any], **kwargs):
        # Check output keys
        self._check_output_keys(processed_response)

        res_params = {
            key: processed_response[key] for key in self.output_keys
        }
        res_params.update({
            "self_reflection_reasoning": processed_response.get("reasoning", "")
        })

        return res_params

    @BaseProvider.debug
    @BaseProvider.error
    @BaseProvider.write
    def __call__(self,
                 *args,
                 use_screen_shot_augmented = False,
                 use_video = False,
                 **kwargs):

        params = deepcopy(memory.current_info)
        params.update(self._preprocess(params, use_screen_shot_augmented, use_video, **kwargs))

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

class RDR2SelfReflectionProvider(BaseProvider):
    def __init__(self,
                 *args,
                 planner,
                 gm,
                 **kwargs):
        super(RDR2SelfReflectionProvider, self).__init__(*args, **kwargs)
        self.planner = planner
        self.gm = gm

    def __call__(self, *args, **kwargs):

        start_frame_id = memory.get_recent_history("start_frame_id", k=1)[0]
        end_frame_id = memory.get_recent_history("end_frame_id", k=1)[0]
        task_description = memory.get_recent_history("task_description", k=1)[0]
        pre_action = memory.get_recent_history("pre_action", k=1)[0]
        pre_decision_making_reasoning = memory.get_recent_history("pre_decision_making_reasoning", k=1)[0]
        exec_info = memory.get_recent_history("exec_info", k=1)[0]
        skill_library = memory.get_recent_history("skill_library", k=1)[0]

        self_reflection_reasoning = ""

        if start_frame_id > -1:
            input = self.planner.self_reflection_.input_map
            action_frames = []
            video_frames = video_record.get_frames(start_frame_id, end_frame_id)

            if len(video_frames) <= config.max_images_in_self_reflection * config.duplicate_frames + 1:
                action_frames = [frame[1] for frame in video_frames[1::config.duplicate_frames]]
            else:
                for i in range(config.max_images_in_self_reflection):
                    step = len(video_frames) // config.max_images_in_self_reflection * i + 1
                    action_frames.append(video_frames[step][1])

            image_introduction = [
                {
                    "introduction": "Here are the sequential frames of the character executing the last action.",
                    "path": action_frames,
                    "assistant": "",
                    "resolution": "low"
                }]

            input["image_introduction"] = image_introduction
            input["task_description"] = task_description
            input['skill_library'] = skill_library
            input["previous_reasoning"] = pre_decision_making_reasoning

            if pre_action:
                pre_action_name, pre_action_params = self.gm.convert_expression_to_skill(pre_action)

                # only input the pre_action name
                input["previous_action"] = pre_action_name
                action_code, action_code_info = self.gm.get_skill_library_in_code(pre_action_name)
                input['action_code'] = action_code if action_code is not None else action_code_info
            else:
                input["previous_action"] = ""
                input['action_code'] = ""

            if exec_info["errors"]:
                input['executing_action_error'] = exec_info["errors_info"]
            else:
                input['executing_action_error'] = ""

            # >> Calling SELF REFLECTION
            logger.write(f'>> Calling SELF REFLECTION')
            reflection_data = self.planner.self_reflection(input=input)

            if 'reasoning' in reflection_data['res_dict'].keys():
                self_reflection_reasoning = reflection_data['res_dict']['reasoning']
            else:
                self_reflection_reasoning = ""

            memory.update_info_history({
                "self_reflection_reasoning": self_reflection_reasoning
            })

        res_params = {
            "pre_self_reflection_reasoning": self_reflection_reasoning
        }
        memory.update_info_history(res_params)


class StardewSelfReflectionProvider(BaseProvider):
    def __init__(self,
                 *args,
                 planner,
                 gm,
                 augment_methods,
                 **kwargs):
        super(StardewSelfReflectionProvider, self).__init__(*args, **kwargs)
        self.planner = planner
        self.gm = gm
        self.augment_methods = augment_methods

    def augment_image(self, image):
        for augment_method in self.augment_methods:
            image = augment_method(image)
        return image

    def __call__(self, *args, **kwargs):

        start_frame_id = memory.get_recent_history("start_frame_id", k=1)[0]
        end_frame_id = memory.get_recent_history("end_frame_id", k=1)[0]
        task_description = memory.get_recent_history("task_description", k=1)[0]
        pre_action = memory.get_recent_history("pre_action", k=1)[0]
        pre_decision_making_reasoning = memory.get_recent_history("pre_decision_making_reasoning", k=1)[0]
        exec_info = memory.get_recent_history("exec_info", k=1)[0]
        skill_library = memory.get_recent_history("skill_library", k=1)[0]
        datetime = memory.get_recent_history("datetime", k=1)[0]
        toolbar_information = memory.get_recent_history("toolbar_information", k=1)[0]
        previous_toolbar_information = memory.get_recent_history("previous_toolbar_information", k=1)[0]
        history_summary = memory.get_recent_history("history_summary", k=1)[0]
        subtask_description = memory.get_recent_history("subtask_description", k=1)[0]
        subtask_reasoning = memory.get_recent_history("subtask_reasoning", k=1)[0]

        self_reflection_reasoning = ""
        if start_frame_id > -1:
            input = self.planner.self_reflection_.input_map
            action_frames = []
            video_frames = video_record.get_frames(start_frame_id, end_frame_id)

            action_frames.append(self.augment_image(video_frames[0][1]))
            action_frames.append(self.augment_image(video_frames[-1][1]))

            image_introduction = [
                {
                    "introduction": "Here are the sequential frames of the character executing the last action.",
                    "path": action_frames,
                    "assistant": "",
                    "resolution": "low"
                }]

            input["image_introduction"] = image_introduction
            input["task_description"] = task_description
            input['skill_library'] = skill_library
            input["previous_reasoning"] = pre_decision_making_reasoning

            input["date_time"] = datetime
            input["toolbar_information"] = toolbar_information
            input["previous_toolbar_information"] = previous_toolbar_information
            input["history_summary"] = history_summary
            input["subtask_description"] = subtask_description
            input["subtask_reasoning"] = subtask_reasoning

            if pre_action:
                pre_action_name = []
                pre_action_code = []

                for item in self.gm.convert_expression_to_skill(pre_action):
                    name, params = item
                    action_code, action_info = self.gm.get_skill_library_in_code(name)

                    pre_action_name.append(name)
                    pre_action_code.append(action_code if action_code is not None else action_info)
                input["previous_action"] = ",".join(pre_action_name)
                input['action_code'] = "\n".join(list(set(pre_action_code)))
            else:
                input["previous_action"] = ""
                input['action_code'] = ""

            if exec_info["errors"]:
                input['executing_action_error'] = exec_info["errors_info"]
            else:
                input['executing_action_error'] = ""

            # >> Calling SELF REFLECTION
            logger.write(f'>> Calling SELF REFLECTION')
            reflection_data = self.planner.self_reflection(input=input)

            if 'reasoning' in reflection_data['res_dict'].keys():
                self_reflection_reasoning = reflection_data['res_dict']['reasoning']
            else:
                self_reflection_reasoning = ""

            memory.update_info_history({
                "self_reflection_reasoning": self_reflection_reasoning
            })
            logger.write(f'Self-reflection reason: {self_reflection_reasoning}')

        res_params = {
            "pre_self_reflection_reasoning": self_reflection_reasoning
        }
        memory.update_info_history(res_params)