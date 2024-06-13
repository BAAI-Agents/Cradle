import os
import json
from groundingdino.util.inference import load_image
from typing import Any, Dict, List
import re
from copy import deepcopy

from cradle.provider import BaseProvider
from cradle.log import Logger
from cradle.config import Config
from cradle.memory import LocalMemory
from cradle.provider import VideoRecordProvider
from cradle.provider.others.task_guidance import TaskGuidanceProvider
from cradle import constants
from cradle.utils.image_utils import save_annotate_frame
from cradle.utils.file_utils import assemble_project_path, read_resource_file
from cradle.utils.json_utils import parse_semi_formatted_text
from cradle.utils.check import is_valid_value
from cradle.utils.image_utils import segment_toolbar, segment_new_icon, segement_inventory

config = Config()
logger = Logger()
memory = LocalMemory()
video_record = VideoRecordProvider()
task_guidance = TaskGuidanceProvider()

class InformationGatheringProvider(BaseProvider):
    def __init__(self,
                 *args,
                 template_path: str,
                 llm_provider: Any = None,
                 gm: Any = None,
                 use_task_guidance: bool = False,
                 **kwargs):
        super(InformationGatheringProvider, self).__init__(*args, **kwargs)
        self.template_path = template_path
        self.llm_provider = llm_provider
        self.use_task_guidance = use_task_guidance
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

    def _preprocess(self, params: Dict[str, Any], use_screen_shot_augmented = False, **kwargs):

        screen_shot_path = memory.get_recent_history("screen_shot_path")[-1]
        screen_shot_augmnented_path = memory.get_recent_history("screen_shot_augmented_path")[-1]

        if not use_screen_shot_augmented:
            image_introduction = [
                {
                    "introduction": "This is a screenshot of the current moment in the game.",
                    "path": screen_shot_path,
                    "assistant": ""
                }
            ]
        else:
            image_introduction = [
                {
                    "introduction": "This is a screenshot of the current moment in the game with multiple augmentation to help you understand it better. The screenshot is organized into a grid layout with 15 segments, arranged in 3 rows and 5 columns. Each segment in the grid is uniquely identified by coordinates, which are displayed at the center of each segment in white text. The layout also features color-coded bands for orientation: a blue band on the left side and a yellow band on the right side of the screenshot.",
                    "path": screen_shot_augmnented_path,
                    "assistant": ""
                }
            ]

        res_params = {
            "image_introduction": image_introduction
        }

        if self.use_task_guidance:
            task_description = task_guidance.get_task_guidance(use_last=False)
            res_params["task_description"] = task_description

        memory.current_info.update(res_params)
        return res_params

    def _postprocess(self, processed_response: Dict[str, Any], **kwargs):
        # Check output keys
        self._check_output_keys(processed_response)

        res_params = {
            key: processed_response[key] for key in self.output_keys
        }

        return res_params

    @BaseProvider.debug
    @BaseProvider.error
    @BaseProvider.write
    def __call__(self,
                 *args,
                 use_screen_shot_augmented: bool = False,
                 **kwargs):

        params = deepcopy(memory.current_info)
        params.update(self._preprocess(params, use_screen_shot_augmented, **kwargs))

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

        if (constants.LAST_TASK_HORIZON in res_params and
                is_valid_value(res_params[constants.LAST_TASK_HORIZON])):
            long_horizon = True
        else:
            long_horizon = False

        if (constants.SCREEN_CLASSIFICATION in res_params and
                is_valid_value(res_params[constants.SCREEN_CLASSIFICATION])):
                screen_classification = res_params[constants.SCREEN_CLASSIFICATION]
        else:
            screen_classification = "None"

        res_params.update({
            "long_horizon": long_horizon,
            "screen_classification": screen_classification
        })

        memory.update_info_history(res_params)

        del params

        return res_params

class RDR2InformationGatheringProvider(BaseProvider):
    def __init__(self,
                 *args,
                 planner = None,
                 gm = None,
                 **kwargs):
        super(RDR2InformationGatheringProvider, self).__init__(*args, **kwargs)
        self.planner = planner
        self.gm = gm

    def __call__(self, *args, **kwargs):

        start_frame_id = memory.get_recent_history("start_frame_id", k=1)[0]
        end_frame_id = memory.get_recent_history("end_frame_id", k=1)[0]
        cur_screenshot_path = memory.get_recent_history("screen_shot_path", k=1)[0]

        # Gather information preparation
        logger.write(f'Gather Information Start Frame ID: {start_frame_id}, End Frame ID: {end_frame_id}')
        input = self.planner.information_gathering_.input_map
        text_input = self.planner.information_gathering_.text_input_map
        video_clip_path = video_record.get_video(start_frame_id, end_frame_id)
        task_description = task_guidance.get_task_guidance(use_last=False)

        get_text_image_introduction = [
            {
                "introduction": input["image_introduction"][-1]["introduction"],
                "path": memory.get_recent_history("screen_shot_path", k=1)[0],
                "assistant": input["image_introduction"][-1]["assistant"]
            }
        ]

        # Configure the gather_information module
        gather_information_configurations = {
            "frame_extractor": True,
            "icon_replacer": True,
            "llm_description": True,
            "object_detector": True
        }
        input["gather_information_configurations"] = gather_information_configurations

        # Modify the general input for gather_information here
        image_introduction = [get_text_image_introduction[-1]]
        input["task_description"] = task_description
        input["video_clip_path"] = video_clip_path
        input["image_introduction"] = image_introduction

        # Modify the input for get_text module in gather_information here
        text_input["image_introduction"] = get_text_image_introduction
        input["text_input"] = text_input

        # >> Calling INFORMATION GATHERING
        logger.write(f'>> Calling INFORMATION GATHERING')
        data = self.planner.information_gathering(input=input)

        # Any information from the gathered_information_JSON
        gathered_information_JSON = data['res_dict']['gathered_information_JSON']

        if gathered_information_JSON is not None:
            gathered_information = gathered_information_JSON.data_structure
        else:
            logger.warn("NO data_structure in gathered_information_JSON")
            gathered_information = dict()

        # Sort the gathered_information by timestamp
        gathered_information = dict(sorted(gathered_information.items(), key=lambda item: item[0]))
        all_dialogue = gathered_information_JSON.search_type_across_all_indices(constants.DIALOGUE)
        all_task_guidance = gathered_information_JSON.search_type_across_all_indices(constants.TASK_GUIDANCE)
        all_generated_actions = gathered_information_JSON.search_type_across_all_indices(constants.ACTION_GUIDANCE)
        classification_reasons = gathered_information_JSON.search_type_across_all_indices(
            constants.GATHER_TEXT_REASONING)

        response_keys = data['res_dict'].keys()

        if constants.LAST_TASK_GUIDANCE in response_keys:
            last_task_guidance = data['res_dict'][constants.LAST_TASK_GUIDANCE]
            if constants.LAST_TASK_HORIZON in response_keys:
                long_horizon = bool(
                    int(data['res_dict'][constants.LAST_TASK_HORIZON][0]))  # Only first character is relevant
            else:
                long_horizon = False
        else:
            logger.warn(f"No {constants.LAST_TASK_GUIDANCE} in response.")
            last_task_guidance = ""
            long_horizon = False

        if constants.IMAGE_DESCRIPTION in response_keys:
            image_description = data['res_dict'][constants.IMAGE_DESCRIPTION]
            if constants.SCREEN_CLASSIFICATION in response_keys:
                screen_classification = data['res_dict'][constants.SCREEN_CLASSIFICATION]
            else:
                screen_classification = "None"
        else:
            logger.warn(f"No {constants.IMAGE_DESCRIPTION} in response.")
            image_description = "No description"
            screen_classification = "None"

        if constants.TARGET_OBJECT_NAME in response_keys:
            target_object_name = data['res_dict'][constants.TARGET_OBJECT_NAME]
            object_name_reasoning = data['res_dict'][constants.GATHER_INFO_REASONING]
        else:
            logger.write("> No target object")
            target_object_name = ""
            object_name_reasoning = ""

        if "boxes" in response_keys:
            image_source, image = load_image(cur_screenshot_path)
            boxes = data['res_dict']["boxes"]
            logits = data['res_dict']["logits"]
            phrases = data['res_dict']["phrases"]
            directory, filename = os.path.split(cur_screenshot_path)
            bb_image_path = os.path.join(directory, "bb_" + filename)
            save_annotate_frame(image_source, boxes, logits, phrases, target_object_name.title(),
                                               bb_image_path)

            if boxes is not None and boxes.numel() != 0:
                # Add the screenshot with bounding boxes into working memory

                memory.update_info_history(
                    {
                        constants.AUGMENTED_IMAGES_MEM_BUCKET: bb_image_path
                    }
                )
            else:

                memory.update_info_history(
                    {
                        constants.AUGMENTED_IMAGES_MEM_BUCKET: constants.NO_IMAGE
                    }
                )
        else:

            memory.update_info_history(
                {
                    constants.AUGMENTED_IMAGES_MEM_BUCKET: constants.NO_IMAGE
                }
            )

        res_params = {
            "long_horizon": long_horizon,
            "last_task_guidance": last_task_guidance,
            "all_generated_actions": all_generated_actions,
            "screen_classification": screen_classification,
            "task_description": task_description,
            "response_keys": response_keys,
            "response": data['res_dict'],
        }

        memory.update_info_history(res_params)


class StardewInformationGatheringProvider(BaseProvider):
    def __init__(self,
                 *args,
                 planner,
                 gm,
                 base_toolbar_objects,
                 **kwargs):
        super(StardewInformationGatheringProvider, self).__init__(*args, **kwargs)
        self.planner = planner
        self.gm = gm
        self.base_toolbar_objects = base_toolbar_objects

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

    def __call__(self, *args, **kwargs):
        # Get params

        start_frame_id = memory.get_recent_history("start_frame_id", k=1)[0]
        end_frame_id = memory.get_recent_history("end_frame_id", k=1)[0]
        screen_shot_path = memory.get_recent_history("screen_shot_path", k=1)[0]
        augmented_screen_shot_path = memory.get_recent_history("augmented_screen_shot_path", k=1)[0]
        task_description = memory.get_recent_history("task_description", k=1)[0]

        # Gather information preparation
        logger.write(f'Gather Information Start Frame ID: {start_frame_id}, End Frame ID: {end_frame_id}')
        input = self.planner.informationgathering_.input_map
        video_clip_path = video_record.get_video(start_frame_id, end_frame_id)

        # Configure the test
        # if you want to test with a pre-defined screenshot, you can replace the cur_screen_shot_path with the path to the screenshot
        pre_defined_sreenshot = None
        pre_defined_sreenshot_augmented = None
        if pre_defined_sreenshot is not None:
            cur_screen_shot_path = pre_defined_sreenshot
            cur_screen_shot_path_augmented = pre_defined_sreenshot_augmented
        else:
            cur_screen_shot_path = screen_shot_path
            cur_screen_shot_path_augmented = augmented_screen_shot_path

        cur_toolbar_shot_path = segment_toolbar(cur_screen_shot_path)
        cur_new_icon_image_shot_path, cur_new_icon_name_image_shot_path = segment_new_icon(cur_screen_shot_path)
        cur_inventories_shot_paths = segement_inventory(r"{}".format(cur_toolbar_shot_path))

        # # Modify the general input for gather_information here
        input["image_introduction"][0]["path"] = cur_screen_shot_path_augmented
        input['cur_inventories_shot_paths'] = cur_inventories_shot_paths
        input['cur_new_icon_image_shot_path'] = cur_new_icon_image_shot_path
        input['cur_new_icon_name_image_shot_path'] = cur_new_icon_name_image_shot_path

        # Configure the gather_information module
        gather_information_configurations = {
            "frame_extractor": False,  # extract text from the video clip
            "icon_replacer": False,
            "llm_description": True,  # get the description of the current screenshot
            "object_detector": False,
            "get_item_number": True  # use llm to get item number in the toolbox
        }
        input["gather_information_configurations"] = gather_information_configurations

        input["video_clip_path"] = video_clip_path

        # >> Calling INFORMATION GATHERING
        logger.write(f'>> Calling INFORMATION GATHERING')


        data = self.planner.information_gathering(input=input)
        data['res_dict']['toolbar_information'] = self.prepare_toolbar_information(
            data['res_dict']['toolbar_dict_list'],
            data['res_dict']['selected_position'])
        data['res_dict']['image_description'] = data['res_dict']['description']

        response_keys = data['res_dict'].keys()

        previous_toolbar_information = None
        image_description = "No description"
        # screen_classification = "None"
        toolbar_information = None
        selected_position = None

        energy = None
        dialog = None
        date_time = None

        if constants.IMAGE_DESCRIPTION in response_keys:
            image_description = data['res_dict'][constants.IMAGE_DESCRIPTION]
            # if constants.SCREEN_CLASSIFICATION in response_keys:
            #     screen_classification = data['res_dict'][constants.SCREEN_CLASSIFICATION]
            if 'toolbar_information' in response_keys:
                previous_toolbar_information = toolbar_information
                toolbar_information = data['res_dict']['toolbar_information']
            if 'selected_position' in response_keys:
                selected_position = data['res_dict']['selected_position']
            if 'energy' in response_keys:
                energy = data['res_dict']['energy']
            if 'dialog' in response_keys:
                dialog = data['res_dict']['dialog']
            if 'date_time' in response_keys:
                date_time = data['res_dict']['date_time']
        else:
            logger.warn(f"No {constants.IMAGE_DESCRIPTION} in response.")

        memory.update_info_history({
            "screen_shot_path": cur_screen_shot_path,
            "augmented_screen_shot_path": cur_screen_shot_path_augmented,
        })

        res_params = {
            "response_keys": response_keys,
            "response": data['res_dict'],
            "toolbar_information": toolbar_information,
            "previous_toolbar_information": previous_toolbar_information,
            "selected_position": selected_position,
            "energy": energy,
            "dialog": dialog,
            "date_time": date_time,
        }

        memory.update_info_history(res_params)

