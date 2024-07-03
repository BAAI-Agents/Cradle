import os
from typing import Dict, Any, List
from copy import deepcopy

from cradle.log import Logger
from cradle.config import Config
from cradle.memory import LocalMemory
from cradle.provider import BaseProvider
from cradle.provider import VideoRecordProvider
from cradle.utils.check import is_valid_value
from cradle import constants

config = Config()
logger = Logger()


class SelfReflectionPreprocessProvider(BaseProvider):

    def __init__(self, *args,
                 gm: Any,
                 use_screenshot_augmented = False,
                 use_video = False,
                 **kwargs):

        super().__init__(*args, **kwargs)

        self.gm = gm
        self.memory = LocalMemory()
        self.video_recorder = VideoRecordProvider(os.path.join(config.work_dir, 'video.mp4'))

        self.use_screenshot_augmented = use_screenshot_augmented
        self.use_video = use_video

    def __call__(self):

        if not self.use_video:
            prompts = [
                "This screenshot is the previous observation before executing the last action.",
                "This screenshot is the current observation after executing the last action."
            ]

            screenshot_paths = self.memory.get_recent_history(constants.IMAGES_MEM_BUCKET, k=config.action_planning_image_num)
            screenshot_augmnented_paths = self.memory.get_recent_history(constants.AUGMENTED_IMAGES_MEM_BUCKET, k=config.action_planning_image_num)

            if not self.use_screenshot_augmented:
                image_introduction = []
                for i in range(len(screenshot_paths), 0, -1):
                    image_introduction.append(
                        {
                            "introduction": prompts[-i],
                            "path": screenshot_paths[-i],
                            "assistant": "",
                            "resolution": "low"
                        })
            else:
                image_introduction = []
                for i in range(len(screenshot_augmnented_paths), 0, -1):
                    image_introduction.append(
                        {
                            "introduction": prompts[-i],
                            "path": screenshot_augmnented_paths[-i],
                            "assistant": "",
                            "resolution": "low"
                        })

            processed_params = {
                "image_introduction": image_introduction
            }

        else:

            start_frame_id = self.memory.get_recent_history("start_frame_id", k=1)
            end_frame_id = self.memory.get_recent_history("end_frame_id", k=1)

            action_frames = []
            video_frames = self.video_recorder.get_frames(start_frame_id, end_frame_id)

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

            actions = self.memory.get_recent_history("actions", k=1)
            action_code = ""
            action_str = ""

            if is_valid_value(actions):
                pre_action = actions[0]
                pre_action_name, _ = self.gm.skill_registry.convert_expression_to_skill(pre_action)
                action_str = pre_action_name
                action_code, action_code_info = self.gm.get_skill_library_in_code(pre_action_name)
                action_code = action_code if action_code is not None else action_code_info

            processed_params = {
                "image_introduction": image_introduction,
                "actions": action_str,
                "action_code": action_code
            }

        self.memory.working_area.update(processed_params)

        return processed_params


class RDR2SelfReflectionPreprocessProvider(BaseProvider):

    def __init__(self, *args,
                 gm: Any,
                 **kwargs):

        super().__init__(*args, **kwargs)

        self.gm = gm
        self.memory = LocalMemory()
        self.video_recorder = VideoRecordProvider(os.path.join(config.work_dir, 'video.mp4'))


    def __call__(self):

        logger.write(f'RDR2 Self Reflection Preprocess')

        start_frame_id = self.memory.get_recent_history("start_frame_id", k=1)[0]
        end_frame_id = self.memory.get_recent_history("end_frame_id", k=1)[0]
        task_description = self.memory.get_recent_history("task_description", k=1)[0]
        pre_action = self.memory.get_recent_history("pre_action", k=1)[0]
        pre_decision_making_reasoning = self.memory.get_recent_history("pre_decision_making_reasoning", k=1)[0]
        exec_info = self.memory.get_recent_history("exec_info", k=1)[0]
        skill_library = self.memory.get_recent_history("skill_library", k=1)[0]

        processed_params = {
            "start_frame_id": start_frame_id,
            "end_frame_id": end_frame_id,
            "task_description": task_description,
            "skill_library": skill_library,
            "exec_info": exec_info,
            "pre_action": pre_action,
            "pre_decision_making_reasoning": pre_decision_making_reasoning
        }

        if start_frame_id > -1:
            action_frames = []
            video_frames = self.video_recorder.get_frames(start_frame_id, end_frame_id)

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

            if pre_action:
                pre_action_name, pre_action_params = self.gm.convert_expression_to_skill(pre_action)

                # only input the pre_action name
                previous_action = pre_action_name
                action_code, action_code_info = self.gm.get_skill_library_in_code(pre_action_name)
                action_code = action_code if action_code is not None else action_code_info
            else:
                previous_action = ""
                action_code = ""

            if exec_info["errors"]:
                executing_action_error = exec_info["errors_info"]
            else:
                executing_action_error = ""

            processed_params.update({
                "image_introduction": image_introduction,
                "task_description": task_description,
                "skill_library": skill_library,
                "previous_reasoning": pre_decision_making_reasoning,
                "previous_action": previous_action,
                "action_code": action_code,
                "executing_action_error": executing_action_error
            })

        self.memory.working_area.update(processed_params)

        return processed_params


class StardewSelfReflectionPreprocessProvider(BaseProvider):

    def __init__(self, *args,
                 gm: Any,
                 augment_methods,
                 **kwargs):

        super().__init__(*args, **kwargs)

        self.gm = gm
        self.memory = LocalMemory()
        self.video_recorder = VideoRecordProvider(os.path.join(config.work_dir, 'video.mp4'))

        self.augment_methods = augment_methods


    def augment_image(self, image):
        for augment_method in self.augment_methods:
            image = augment_method(image)
        return image


    def __call__(self):

        logger.write(f'Stardew Self Reflection Preprocess')

        prompts = [
            "Here are the sequential frames of the character executing the last action."
        ]

        start_frame_id = self.memory.get_recent_history("start_frame_id", k=1)[0]
        end_frame_id = self.memory.get_recent_history("end_frame_id", k=1)[0]
        task_description = self.memory.get_recent_history("task_description", k=1)[0]
        pre_action = self.memory.get_recent_history("pre_action", k=1)[0]
        pre_decision_making_reasoning = self.memory.get_recent_history("pre_decision_making_reasoning", k=1)[0]
        exec_info = self.memory.get_recent_history("exec_info", k=1)[0]
        skill_library = self.memory.get_recent_history("skill_library", k=1)[0]
        datetime = self.memory.get_recent_history("datetime", k=1)[0]
        toolbar_information = self.memory.get_recent_history("toolbar_information", k=1)[0]
        previous_toolbar_information = self.memory.get_recent_history("previous_toolbar_information", k=1)[0]
        history_summary = self.memory.get_recent_history("history_summary", k=1)[0]
        subtask_description = self.memory.get_recent_history("subtask_description", k=1)[0]
        subtask_reasoning = self.memory.get_recent_history("subtask_reasoning", k=1)[0]

        processed_params = {
            "start_frame_id": start_frame_id,
            "end_frame_id": end_frame_id,
            "task_description": task_description,
            "skill_library": skill_library,
            "exec_info": exec_info,
            "pre_decision_making_reasoning": pre_decision_making_reasoning,
            "datetime": datetime,
            "toolbar_information": toolbar_information,
            "previous_toolbar_information": previous_toolbar_information,
            "history_summary": history_summary,
            "subtask_description": subtask_description,
            "subtask_reasoning": subtask_reasoning
        }

        if start_frame_id > -1:
            action_frames = []
            video_frames = self.video_recorder.get_frames(start_frame_id, end_frame_id)

            action_frames.append(self.augment_image(video_frames[0][1]))
            action_frames.append(self.augment_image(video_frames[-1][1]))

            image_introduction = [
                {
                    "introduction": prompts[-1],
                    "path": action_frames,
                    "assistant": "",
                    "resolution": "low"
                }]

            if pre_action:
                pre_action_name = []
                pre_action_code = []

                if isinstance(pre_action, str):
                    if "[" not in pre_action:
                        pre_action = "[" + pre_action + "]"
                elif isinstance(pre_action, list):
                    pre_action = "[" + ",".join(pre_action) + "]"

                for item in self.gm.convert_expression_to_skill(pre_action):
                    name, params = item
                    action_code, action_info = self.gm.get_skill_library_in_code(name)

                    pre_action_name.append(name)
                    pre_action_code.append(action_code if action_code is not None else action_info)
                previous_action = ",".join(pre_action_name)
                action_code = "\n".join(list(set(pre_action_code)))
            else:
                previous_action = ""
                action_code = ""

            if exec_info["errors"]:
                executing_action_error = exec_info["errors_info"]
            else:
                executing_action_error = ""

            processed_params.update({
                "image_introduction": image_introduction,
                "previous_action": previous_action,
                "action_code": action_code,
                "executing_action_error": executing_action_error,
                "previous_reasoning": pre_decision_making_reasoning,
            })

        self.memory.working_area.update(processed_params)

        return processed_params


class SelfReflectionPostprocessProvider(BaseProvider):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.memory = LocalMemory()


    def __call__(self, response: Dict):

        processed_response = deepcopy(response)

        processed_response = {
            key: processed_response[key] for key in processed_response
        }
        processed_response.update({
            "self_reflection_reasoning": processed_response.get("reasoning", "")
        })

        self.memory.update_info_history(processed_response)

        return processed_response


class RDR2SelfReflectionPostprocessProvider(BaseProvider):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.memory = LocalMemory()


    def __call__(self, response: Dict):

        logger.write(f'RDR2 Self Reflection Postprocess')

        processed_response = deepcopy(response)

        if 'reasoning' in response:
            self_reflection_reasoning = response['reasoning']
        else:
            self_reflection_reasoning = ""

        processed_response.update({
            "self_reflection_reasoning": self_reflection_reasoning,
            "pre_self_reflection_reasoning": self_reflection_reasoning
        })

        self.memory.update_info_history(processed_response)

        return processed_response


class StardewSelfReflectionPostprocessProvider(BaseProvider):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.memory = LocalMemory()


    def __call__(self, response: Dict):

        logger.write(f'Stardew Self Reflection Postprocess')

        processed_response = deepcopy(response)

        if 'reasoning' in response:
            self_reflection_reasoning = response['reasoning']
        else:
            self_reflection_reasoning = ""

        processed_response.update({
            "self_reflection_reasoning": self_reflection_reasoning,
            "pre_self_reflection_reasoning": self_reflection_reasoning
        })

        self.memory.update_info_history(processed_response)

        return processed_response
