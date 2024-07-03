from typing import Dict, Any
from copy import deepcopy

from cradle.log import Logger
from cradle.config import Config
from cradle.memory import LocalMemory
from cradle.provider import BaseProvider
from cradle import constants

config = Config()
logger = Logger()
memory = LocalMemory()

class ActionPlanningPreprocessProvider(BaseProvider):

    def __init__(self, *args,
                 gm: Any,
                 use_screenshot_augmented = False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.gm = gm
        self.use_screenshot_augmented = use_screenshot_augmented

    def __call__(self):

        prompts = [
            "This screenshot is the previous step of the game.",
            "This screenshot is the current step of the game."
        ]

        screenshot_paths = memory.get_recent_history("screenshot_path", k=config.action_planning_image_num)
        screenshot_augmnented_paths = memory.get_recent_history("screenshot_augmented_path", k=config.action_planning_image_num)

        if not self.use_screenshot_augmented:
            image_introduction = []
            for i in range(len(screenshot_paths), 0, -1):
                image_introduction.append(
                    {
                        "introduction": prompts[-i],
                        "path": screenshot_paths[-i],
                        "assistant": ""
                    })
        else:
            image_introduction = []
            for i in range(len(screenshot_augmnented_paths), 0, -1):
                image_introduction.append(
                    {
                        "introduction": prompts[-i],
                        "path": screenshot_augmnented_paths[-i],
                        "assistant": ""
                    })

        processed_params = {
            "image_introduction": image_introduction
        }

        memory.working_area.update(processed_params)

        return processed_params


class RDR2ActionPlanningPreprocessProvider(BaseProvider):

    def __init__(self, *args,
                 gm: Any,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.gm = gm

    def __call__(self):

        logger.write("RDR2 Action Planning Preprocess")

        prompts = [
            "Now, I will give you five screenshots for decision making.",
            "This screenshot is five steps before the current step of the game",
            "This screenshot is three steps before the current step of the game",
            "This screenshot is two steps before the current step of the game",
            "This screenshot is the previous step of the game",
            "This screenshot is the current step of the game"
        ]

        response_keys = memory.get_recent_history("response_keys", k=1)[0]
        response = memory.get_recent_history("response", k=1)[0]
        pre_action = memory.get_recent_history("pre_action", k=1)[0]
        pre_self_reflection_reasoning = memory.get_recent_history("pre_self_reflection_reasoning", k=1)[0]
        pre_screen_classification = memory.get_recent_history("pre_screen_classification", k=1)[0]
        screen_classification = memory.get_recent_history("screen_classification", k=1)[0]
        skill_library = memory.get_recent_history("skill_library", k=1)[0]
        task_description = memory.get_recent_history("task_description", k=1)[0]

        previous_action = ""
        previous_reasoning = ""
        if pre_action:
            previous_action = memory.get_recent_history("action", k=1)[0]
            previous_reasoning = memory.get_recent_history("decision_making_reasoning", k=1)[0]

        previous_self_reflection_reasoning = ""
        if pre_self_reflection_reasoning:
            previous_self_reflection_reasoning = memory.get_recent_history("self_reflection_reasoning", k=1)[0]

        info_summary = memory.get_recent_history("summarization", k=1)[0]

        # @TODO Temporary solution with fake augmented entries if no bounding box exists. Ideally it should read images, then check for possible augmentation.
        image_memory = memory.get_recent_history("screenshot_path", k=config.action_planning_image_num)
        augmented_image_memory = memory.get_recent_history(constants.AUGMENTED_IMAGES_MEM_BUCKET,
                                                           k=config.action_planning_image_num)

        image_introduction = []
        for i in range(len(image_memory), 0, -1):
            if len(augmented_image_memory) >= i and augmented_image_memory[-i] != constants.NO_IMAGE:
                if i == len(image_memory):
                    image_introduction.append(
                        {
                            "introduction": prompts[-i],
                            "path": augmented_image_memory[-i],
                            "assistant": "",
                            "resolution": "high",
                        })
                else:
                    image_introduction.append(
                        {
                            "introduction": prompts[-i],
                            "path": augmented_image_memory[-i],
                            "assistant": "",
                        })
            else:
                image_introduction.append(
                    {
                        "introduction": prompts[-i],
                        "path": image_memory[-i],
                        "assistant": ""
                    })

        # Minimap info tracking
        minimap_information = ""
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
            minimap_information = minimap_info_str

        processed_params = {
            "pre_screen_classification": pre_screen_classification,
            "screen_classification": screen_classification,
            "previous_action": previous_action,
            "previous_reasoning": previous_reasoning,
            "previous_self_reflection_reasoning": previous_self_reflection_reasoning,
            "skill_library": skill_library,
            "task_description": task_description,
            "minimap_information": minimap_information,
            "info_summary": info_summary,
            "image_introduction": image_introduction
        }

        memory.working_area.update(processed_params)

        return processed_params

class StardewActionPlanningPreprocessProvider(BaseProvider):

    def __init__(self, *args,
                 gm: Any,
                 toolbar_information: str,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.gm = gm
        self.toolbar_information = toolbar_information

    def __call__(self):

        logger.write("Stardew Action Planning Preprocess")

        prompts = [
            "Now, I will give you five screenshots for decision making."
            "This screenshot is five steps before the current step of the game",
            "This screenshot is three steps before the current step of the game",
            "This screenshot is two steps before the current step of the game",
            "This screenshot is the previous step of the game. The blue band represents the left side and the yellow band represents the right side.",
            "This screenshot is the current step of the game. The blue band represents the left side and the yellow band represents the right side."
        ]

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
        toolbar_information = toolbar_information if toolbar_information is not None else self.toolbar_information
        selected_position = selected_position if selected_position is not None else 1

        previous_action = ""
        previous_reasoning = ""
        if pre_action:
            previous_action = memory.get_recent_history("action", k=1)[0]
            previous_reasoning = memory.get_recent_history("decision_making_reasoning", k=1)[0]

        previous_self_reflection_reasoning = ""
        if pre_self_reflection_reasoning:
            previous_self_reflection_reasoning = memory.get_recent_history("self_reflection_reasoning", k=1)[0]

        # @TODO Temporary solution with fake augmented entries if no bounding box exists. Ideally it should read images, then check for possible augmentation.
        image_memory = memory.get_recent_history("augmented_image", k=config.action_planning_image_num)

        image_introduction = []
        for i in range(len(image_memory), 0, -1):
            image_introduction.append(
                {
                    "introduction": prompts[-i],
                    "path": image_memory[-i],
                    "assistant": ""
                })

        processed_params = {
            "pre_self_reflection_reasoning": pre_self_reflection_reasoning,
            "toolbar_information": toolbar_information,
            "selected_position": selected_position,
            "summarization": summarization,
            "skill_library": skill_library,
            "task_description": task_description,
            "subtask_description": subtask_description,
            "history_summary": history_summary,
            "previous_action": previous_action,
            "previous_reasoning": previous_reasoning,
            "previous_self_reflection_reasoning": previous_self_reflection_reasoning,
            "image_introduction": image_introduction
        }

        memory.working_area.update(processed_params)

        return processed_params

class ActionPlanningPostprocessProvider(BaseProvider):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, response: Dict):

        processed_response = deepcopy(response)

        skill_steps = []
        if 'actions' in response:
            skill_steps = response['actions']

        if skill_steps:
            skill_steps = [i for i in skill_steps if i != '']
        else:
            skill_steps = ['']

        skill_steps = skill_steps[:config.number_of_execute_skills]

        if config.number_of_execute_skills > 1:
            actions = "[" + ",".join(skill_steps) + "]"
        else:
            actions = str(skill_steps[0])

        decision_making_reasoning = response['reasoning']

        processed_response.update({
            "actions": actions,
            "decision_making_reasoning": decision_making_reasoning,
            "skill_steps": skill_steps,
        })
        memory.update_info_history(processed_response)

        return processed_response

class RDR2ActionPlanningPostprocessProvider(BaseProvider):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, response: Dict):

        logger.write("RDR2 Action Planning Postprocess")

        processed_response = deepcopy(response)

        skill_steps = []
        if 'actions' in response:
            skill_steps = response['actions']

        if skill_steps:
            skill_steps = [i for i in skill_steps if i != '']
        else:
            skill_steps = ['']

        skill_steps = skill_steps[:config.number_of_execute_skills]

        if config.number_of_execute_skills > 1:
            actions = "[" + ",".join(skill_steps) + "]"
        else:
            actions = str(skill_steps[0])

        decision_making_reasoning = response['reasoning']
        pre_decision_making_reasoning = decision_making_reasoning

        processed_response.update({
            "action": actions,
            "pre_decision_making_reasoning": pre_decision_making_reasoning,
            "decision_making_reasoning": decision_making_reasoning,
            "skill_steps": skill_steps,
        })
        memory.update_info_history(processed_response)

        return processed_response


class StardewActionPlanningPostprocessProvider(BaseProvider):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, response: Dict):

        logger.write("Stardew Action Planning Postprocess")

        processed_response = deepcopy(response)

        skill_steps = []
        if 'actions' in response:
            skill_steps = response['actions']

        if skill_steps:
            skill_steps = [i for i in skill_steps if i != '']
        else:
            skill_steps = ['']

        skill_steps = skill_steps[:config.number_of_execute_skills]
        pre_action = "[" + ",".join(skill_steps) + "]"

        if config.number_of_execute_skills > 1:
            actions = "[" + ",".join(skill_steps) + "]"
        else:
            actions = str(skill_steps[0])

        decision_making_reasoning = response['reasoning']
        pre_decision_making_reasoning = decision_making_reasoning

        processed_response.update({
            "pre_action": pre_action,
            "action": actions,
            "pre_decision_making_reasoning": pre_decision_making_reasoning,
            "decision_making_reasoning": decision_making_reasoning,
            "skill_steps": skill_steps,
        })
        memory.update_info_history(processed_response)

        return processed_response