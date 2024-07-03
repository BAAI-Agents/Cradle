import os
from typing import Dict, Any, List
from copy import deepcopy

try:
    from groundingdino.util.inference import load_image
except:
    pass

from cradle.log import Logger
from cradle.config import Config
from cradle.memory import LocalMemory
from cradle.provider import BaseProvider
from cradle.provider.others.task_guidance import TaskGuidanceProvider
from cradle import constants
from cradle.utils.check import is_valid_value
from cradle.provider import VideoRecordProvider
from cradle.utils.image_utils import save_annotate_frame
from cradle.utils.image_utils import segment_toolbar, segment_new_icon, segement_inventory


config = Config()
logger = Logger()

class InformationGatheringPreprocessProvider(BaseProvider):

    def __init__(self, *args,
                 gm = Any,
                 use_screenshot_augmented = False,
                 use_task_guidance = False,
                 task_description = "",
                 **kwargs):

        super().__init__(*args, **kwargs)

        self.gm = gm
        self.memory = LocalMemory()
        self.task_guidance = TaskGuidanceProvider(task_description=task_description)

        self.use_screenshot_augmented = use_screenshot_augmented
        self.use_task_guidance = use_task_guidance


    def __call__(self):

        screenshot_path = self.memory.get_recent_history(constants.IMAGES_MEM_BUCKET)[-1]
        screenshot_augmnented_path = self.memory.get_recent_history(constants.AUGMENTED_IMAGES_MEM_BUCKET)[-1]

        if not self.use_screenshot_augmented:
            image_introduction = [
                {
                    "introduction": "This is a screenshot of the current moment in the game.",
                    "path": screenshot_path,
                    "assistant": ""
                }
            ]
        else:
            image_introduction = [
                {
                    "introduction": "This is a screenshot of the current moment in the game with multiple augmentation to help you understand it better. The screenshot is organized into a grid layout with 15 segments, arranged in 3 rows and 5 columns. Each segment in the grid is uniquely identified by coordinates, which are displayed at the center of each segment in white text. The layout also features color-coded bands for orientation: a blue band on the left side and a yellow band on the right side of the screenshot.",
                    "path": screenshot_augmnented_path,
                    "assistant": ""
                }
            ]

        processed_params = {
            "image_introduction": image_introduction
        }

        if self.use_task_guidance:
            task_description = self.task_guidance.get_task_guidance(use_last=False)
            processed_params["task_description"] = task_description

        self.memory.working_area.update(processed_params)

        return processed_params


class InformationGatheringPostprocessProvider(BaseProvider):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.memory = LocalMemory()


    def __call__(self, response: Dict):

        processed_response = deepcopy(response)

        if (constants.LAST_TASK_HORIZON in response and is_valid_value(response[constants.LAST_TASK_HORIZON])):
            long_horizon = True
        else:
            long_horizon = False

        if (constants.SCREEN_CLASSIFICATION in response and is_valid_value(response[constants.SCREEN_CLASSIFICATION])):
            screen_classification = response[constants.SCREEN_CLASSIFICATION]
        else:
            screen_classification = "None"

        processed_response.update({
            "long_horizon": long_horizon,
            "screen_classification": screen_classification
        })

        self.memory.update_info_history(processed_response)

        return processed_response


class RDR2InformationGatheringPreprocessProvider(BaseProvider):

    def __init__(self, *args,
                 gm = Any,
                 **kwargs):

        super().__init__(*args, **kwargs)

        self.gm = gm
        self.memory = LocalMemory()
        self.video_recorder = VideoRecordProvider(os.path.join(config.work_dir, 'video.mp4'))
        self.task_guidance = TaskGuidanceProvider()

    def __call__(self):

        logger.write("RDR2 Information Gathering Preprocess")

        prompts = [
            "This is a screenshot of the current moment in the game"
        ]

        start_frame_id = self.memory.get_recent_history("start_frame_id", k=1)[0]
        end_frame_id = self.memory.get_recent_history("end_frame_id", k=1)[0]
        screenshot_path = self.memory.get_recent_history(constants.IMAGES_MEM_BUCKET, k=1)[0]

        # Gather information preparation
        logger.write(f'Gather Information Start Frame ID: {start_frame_id}, End Frame ID: {end_frame_id}')

        text_input = {
            "image_introduction": [
                {
                    "introduction": "This is a screenshot of the current moment in the game",
                    "path": "",
                    "assistant": ""
                }
            ],
            "information_type": ["Item_status",
                                 "Notification",
                                 "Environment_information",
                                 "Action_guidance",
                                 "Task_guidance",
                                 "Dialogue",
                                 "Others"]
        }

        video_clip_path = self.video_recorder.get_video(start_frame_id, end_frame_id)
        task_description = self.task_guidance.get_task_guidance(use_last=False)

        get_text_image_introduction = [
            {
                "introduction": prompts[-1],
                "path": screenshot_path,
                "assistant": ""
            }
        ]

        # Configure the gather_information module
        gather_information_configurations = {
            "frame_extractor": True,
            "icon_replacer": True,
            "llm_description": True,
            "object_detector": True
        }

        # Modify the general input for gather_information here
        image_introduction = [get_text_image_introduction[-1]]
        task_description = task_description

        # Modify the input for get_text module in gather_information here
        text_input["image_introduction"] = get_text_image_introduction

        processed_params = {
            "image_introduction": image_introduction,
            "task_description": task_description,
            "text_input": text_input,
            "video_clip_path": video_clip_path,
            "gather_information_configurations": gather_information_configurations
        }

        self.memory.working_area.update(processed_params)

        return processed_params


class StardewInformationGatheringPreprocessProvider(BaseProvider):

    def __init__(self, *args,
                 gm = Any,
                 **kwargs):

        super().__init__(*args, **kwargs)

        self.gm = gm
        self.memory = LocalMemory()
        self.video_recorder = VideoRecordProvider(os.path.join(config.work_dir, 'video.mp4'))


    def __call__(self):

        logger.write("Stardew Information Gathering Preprocess")

        prompts = [
            "This is a screenshot of the current moment in the game with multiple augmentation to help you understand it better. The screenshot is organized into a grid layout with 15 segments, arranged in 3 rows and 5 columns. Each segment in the grid is uniquely identified by coordinates, which are displayed at the center of each segment in white text. The layout also features color-coded bands for orientation: a blue band on the left side and a yellow band on the right side of the screenshot."
        ]

        start_frame_id = self.memory.get_recent_history("start_frame_id", k=1)[0]
        end_frame_id = self.memory.get_recent_history("end_frame_id", k=1)[0]
        screenshot_path = self.memory.get_recent_history(constants.IMAGES_MEM_BUCKET, k=1)[0]
        augmented_screenshot_path = self.memory.get_recent_history(constants.AUGMENTED_IMAGES_MEM_BUCKET, k=1)[0]
        task_description = self.memory.get_recent_history("task_description", k=1)[0]

        # Gather information preparation
        logger.write(f'Gather Information Start Frame ID: {start_frame_id}, End Frame ID: {end_frame_id}')
        video_clip_path = self.video_recorder.get_video(start_frame_id, end_frame_id)

        # Configure the test
        # if you want to test with a pre-defined screenshot, you can replace the cur_screenshot_path with the path to the screenshot
        pre_defined_sreenshot = None
        pre_defined_sreenshot_augmented = None
        if pre_defined_sreenshot is not None:
            cur_screenshot_path = pre_defined_sreenshot
            cur_screenshot_path_augmented = pre_defined_sreenshot_augmented
        else:
            cur_screenshot_path = screenshot_path
            cur_screenshot_path_augmented = augmented_screenshot_path

        cur_toolbar_shot_path = segment_toolbar(cur_screenshot_path)
        cur_new_icon_image_shot_path, cur_new_icon_name_image_shot_path = segment_new_icon(cur_screenshot_path)
        cur_inventories_shot_paths = segement_inventory(r"{}".format(cur_toolbar_shot_path))

        image_introduction = [
            {
                "introduction": prompts[-1],
                "path": cur_screenshot_path_augmented,
                "assistant": ""
            }
        ]

        # Configure the gather_information module
        gather_information_configurations = {
            "frame_extractor": False,  # extract text from the video clip
            "icon_replacer": False,
            "llm_description": True,  # get the description of the current screenshot
            "object_detector": False,
            "get_item_number": True  # use llm to get item number in the toolbox
        }

        processed_params = {
            "image_introduction": image_introduction,
            "task_description": task_description,
            constants.IMAGES_MEM_BUCKET: cur_screenshot_path,
            constants.AUGMENTED_IMAGES_MEM_BUCKET: cur_screenshot_path_augmented,
            "cur_inventories_shot_paths": cur_inventories_shot_paths,
            "cur_new_icon_image_shot_path": cur_new_icon_image_shot_path,
            "cur_new_icon_name_image_shot_path": cur_new_icon_name_image_shot_path,
            "video_clip_path": video_clip_path,
            "gather_information_configurations": gather_information_configurations
        }

        self.memory.working_area.update(processed_params)

        return processed_params


class RDR2InformationGatheringPostprocessProvider(BaseProvider):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.memory = LocalMemory()


    def __call__(self, response: Dict):

        logger.write("RDR2 Information Gathering Postprocess")

        processed_response = deepcopy(response)
        task_description = self.memory.working_area["task_description"]

        gathered_information_JSON = response['gathered_information_JSON']
        screenshot_path = self.memory.get_recent_history(constants.IMAGES_MEM_BUCKET, k=1)[0]

        all_generated_actions = gathered_information_JSON.search_type_across_all_indices(constants.ACTION_GUIDANCE)

        if constants.LAST_TASK_GUIDANCE in response:
            last_task_guidance = response[constants.LAST_TASK_GUIDANCE]
            if constants.LAST_TASK_HORIZON in response:
                long_horizon = bool(
                    int(response[constants.LAST_TASK_HORIZON][0]))  # Only first character is relevant
            else:
                long_horizon = False
        else:
            logger.warn(f"No {constants.LAST_TASK_GUIDANCE} in response.")
            last_task_guidance = ""
            long_horizon = False

        if constants.IMAGE_DESCRIPTION in response:
            if constants.SCREEN_CLASSIFICATION in response:
                screen_classification = response[constants.SCREEN_CLASSIFICATION]
            else:
                screen_classification = "None"
        else:
            logger.warn(f"No {constants.IMAGE_DESCRIPTION} in response.")
            screen_classification = "None"

        if constants.TARGET_OBJECT_NAME in response:
            target_object_name = response[constants.TARGET_OBJECT_NAME]
        else:
            logger.write("> No target object")
            target_object_name = ""

        if "boxes" in response:
            image_source, image = load_image(screenshot_path)
            boxes = response["boxes"]
            logits = response["logits"]
            phrases = response["phrases"]
            directory, filename = os.path.split(screenshot_path)
            bb_image_path = os.path.join(directory, "bb_" + filename)
            save_annotate_frame(image_source,
                                boxes,
                                logits,
                                phrases,
                                target_object_name.title(),
                                bb_image_path)

            if boxes is not None and boxes.numel() != 0:
                # Add the screenshot with bounding boxes into working memory

                self.memory.update_info_history({
                        constants.AUGMENTED_IMAGES_MEM_BUCKET: bb_image_path
                    }
                )
            else:
                self.memory.update_info_history({
                        constants.AUGMENTED_IMAGES_MEM_BUCKET: constants.NO_IMAGE
                    }
                )
        else:

            self.memory.update_info_history({
                    constants.AUGMENTED_IMAGES_MEM_BUCKET: constants.NO_IMAGE
                }
            )

        processed_response.update({
            "long_horizon": long_horizon,
            "last_task_guidance": last_task_guidance,
            "all_generated_actions": all_generated_actions,
            "screen_classification": screen_classification,
            "task_description": task_description,
            "response_keys": list(response.keys()),
            "response": response,
        })

        self.memory.update_info_history(processed_response)

        return processed_response


class StardewInformationGatheringPostprocessProvider(BaseProvider):

    def __init__(self, *args, base_toolbar_objects, **kwargs):

        super().__init__(*args, **kwargs)

        self.base_toolbar_objects = base_toolbar_objects

        self.memory = LocalMemory()


    def prepare_toolbar_information(self,
                                    tool_dict_list: List[Dict[str, Any]],
                                    selected_position: int):

        toolbar_information = "The items in the toolbar are arranged from left to right in the following order.\n"
        selected_item = None

        for item in tool_dict_list:

            name = item["name"]
            number = item["number"]
            position = item["position"]

            if name in self.base_toolbar_objects:
                toolbar_object = self.base_toolbar_objects[name]
                true_name = toolbar_object["name"]
                type = toolbar_object["type"]
                description = toolbar_object["description"]

                if type == "Tool":
                    toolbar_information += f"{position}. {true_name}: {type}. {description}\n"
                elif type == "Blank":
                    toolbar_information += f"{position}. {true_name}: {description}\n"
                else:
                    toolbar_information += f"{position}. {true_name}: {type}. {description} Quality: {number}.\n"

                if selected_position is not None and selected_position == position:
                    selected_item = true_name
            else:
                toolbar_object = self.base_toolbar_objects["unknown"]
                true_name = toolbar_object["name"]
                type = toolbar_object["type"]
                description = toolbar_object["description"]
                toolbar_information += f"{position}. {true_name}: {description}\n"

        # selected item
        if selected_item is not None:
            toolbar_information += f"Now the item you selected is: {selected_position}. {selected_item}\n"
        else:
            toolbar_information += f"Now you are not selecting any item.\n"

        return toolbar_information


    def __call__(self, response: Dict):

        logger.write("Stardew Information Gathering Postprocess")

        processed_response = deepcopy(response)

        response['toolbar_information'] = self.prepare_toolbar_information(
            response['toolbar_dict_list'],
            response['selected_position'])
        response['image_description'] = response['description']

        previous_toolbar_information = None
        toolbar_information = None
        selected_position = None

        energy = None
        dialog = None
        date_time = None

        if constants.IMAGE_DESCRIPTION in response:
            if 'toolbar_information' in response:
                previous_toolbar_information = toolbar_information
                toolbar_information = response['toolbar_information']
            if 'selected_position' in response:
                selected_position = response['selected_position']
            if 'energy' in response:
                energy = response['energy']
            if 'dialog' in response:
                dialog = response['dialog']
            if 'date_time' in response:
                date_time = response['date_time']
        else:
            logger.warn(f"No {constants.IMAGE_DESCRIPTION} in response.")

        processed_response.update({
            "response_keys": list(response.keys()),
            "response": response,
            "toolbar_information": toolbar_information,
            "previous_toolbar_information": previous_toolbar_information,
            "selected_position": selected_position,
            "energy": energy,
            "dialog": dialog,
            "date_time": date_time,
        })

        self.memory.update_info_history(processed_response)

        return processed_response
