import time
import json
import os
from typing import Dict, Any, List
import time
import asyncio

from cradle.config import Config
from cradle.gameio.video.VideoFrameExtractor import JSONStructure
from cradle.log import Logger
from cradle.planner.base import BasePlanner
from cradle.provider.base_llm import LLMProvider
from cradle.utils.check import check_planner_params
from cradle.utils.file_utils import assemble_project_path, read_resource_file
from cradle.utils.json_utils import load_json, parse_semi_formatted_text
from cradle import constants

config = Config()
logger = Logger()

PROMPT_EXT = ".prompt"
JSON_EXT = ".json"


async def gather_information_get_completion_parallel(llm_provider, text_input_map, current_frame_path, time_stamp,
                                                     text_input, get_text_template, i,video_prefix,gathered_information_JSON):

    logger.write(f"Start gathering text information from the {i + 1}th frame")

    text_input = text_input_map if text_input is None else text_input
    image_introduction = text_input["image_introduction"]

    # Set the last frame path as the current frame path
    image_introduction[-1] = {
        "introduction": image_introduction[-1]["introduction"],
        "path": f"{current_frame_path}",
        "assistant": image_introduction[-1]["assistant"]
    }
    text_input["image_introduction"] = image_introduction
    message_prompts = llm_provider.assemble_prompt(template_str=get_text_template, params=text_input)

    logger.debug(f'{logger.UPSTREAM_MASK}{json.dumps(message_prompts, ensure_ascii=False)}\n')

    success_flag = False
    while not success_flag:
        try:
            response, info = await llm_provider.create_completion_async(message_prompts)
            logger.debug(f'{logger.DOWNSTREAM_MASK}{response}\n')

            # Convert the response to dict
            processed_response = parse_semi_formatted_text(response)
            success_flag = True
        except Exception as e:
            logger.error(f"Response is not in the correct format: {e}, retrying...")
            success_flag = False

            # wait 2 seconds for the next request and retry
            await asyncio.sleep(2)

    # Convert the response to dict
    if processed_response is None or len(response) == 0:
        logger.warn('Empty response in gather text information call')
        logger.debug("response", response, "processed_response", processed_response)

    objects = processed_response
    objects_index = str(video_prefix) + '_' + time_stamp
    gathered_information_JSON.add_instance(objects_index, objects)
    logger.write(f"Finish gathering text information from the {i + 1}th frame")

    return True


def gather_information_get_completion_sequence(llm_provider, text_input_map, current_frame_path, time_stamp,
                                               text_input, get_text_template, i, video_prefix, gathered_information_JSON):

    logger.write(f"Start gathering text information from the {i + 1}th frame")
    text_input = text_input_map if text_input is None else text_input

    image_introduction = text_input["image_introduction"]

    # Set the last frame path as the current frame path
    image_introduction[-1] = {
        "introduction": image_introduction[-1]["introduction"],
        "path": f"{current_frame_path}",
        "assistant": image_introduction[-1]["assistant"]
    }
    text_input["image_introduction"] = image_introduction

    message_prompts = llm_provider.assemble_prompt(template_str=get_text_template, params=text_input)

    logger.debug(f'{logger.UPSTREAM_MASK}{json.dumps(message_prompts, ensure_ascii=False)}\n')

    response, info = llm_provider.create_completion(message_prompts)

    logger.debug(f'{logger.DOWNSTREAM_MASK}{response}\n')
    success_flag = False
    while not success_flag:
        try:
            # Convert the response to dict
            processed_response = parse_semi_formatted_text(response)
            success_flag = True
        except Exception as e:
            logger.error(f"Response is not in the correct format: {e}, retrying...")
            success_flag = False

            time.sleep(2)

    # Convert the response to dict
    if processed_response is None or len(response) == 0:
        logger.warn('Empty response in gather text information call')
        logger.debug("response", response, "processed_response", processed_response)

    objects = processed_response
    objects_index = str(video_prefix) + '_' + time_stamp
    gathered_information_JSON.add_instance(objects_index, objects)

    logger.write(f"Finish gathering text information from the {i + 1}th frame")

    return True


async def get_completion_in_parallel(llm_provider, text_input_map, extracted_frame_paths, text_input,get_text_template,video_prefix,gathered_information_JSON):
    tasks =[]

    for i, (current_frame_path, time_stamp) in enumerate(extracted_frame_paths):

        task=gather_information_get_completion_parallel(llm_provider, text_input_map, current_frame_path, time_stamp,
                                                   text_input, get_text_template, i,video_prefix,gathered_information_JSON)

        tasks.append(task)

        # wait 2 seconds for the next request
        time.sleep(2)

    return await asyncio.gather(*tasks)


def get_completion_in_sequence(llm_provider, text_input_map, extracted_frame_paths, text_input,get_text_template,video_prefix,gathered_information_JSON):

    for i, (current_frame_path, time_stamp) in enumerate(extracted_frame_paths):
        gather_information_get_completion_sequence(llm_provider, text_input_map, current_frame_path, time_stamp,
                                                   text_input, get_text_template, i,video_prefix,gathered_information_JSON)

    return True


class ScreenClassification():
    def __init__(self,
                 input_example: Dict = None,
                 template: Dict = None,
                 llm_provider: LLMProvider = None,
                 ):
        self.input_example = input_example
        self.template = template
        self.llm_provider = llm_provider

    def _pre(self, *args, input=None, screenshot_file=None, **kwargs):
        return input, screenshot_file

    def __call__(self, *args, input=None, screenshot_file=None, **kwargs):
        raise NotImplementedError('ScreenClassification is not implemented independently yet')

    def _post(self, *args, data=None, **kwargs):
        return data


class GatherInformation():

    def __init__(self,
                 input_map: Dict = None,
                 template: str = None,
                 icon_replacer: Any = None,
                 object_detector: Any = None,
                 llm_provider: LLMProvider = None,
                 text_input_map: Dict = None,
                 get_text_template: str = None,
                 frame_extractor: Any = None
                 ):

        self.input_map = input_map
        self.template = template
        self.icon_replacer = icon_replacer
        self.object_detector = object_detector
        self.llm_provider = llm_provider
        self.text_input_map = text_input_map
        self.get_text_template = get_text_template
        self.frame_extractor = frame_extractor


    def _pre(self, *args, input: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        return input


    def __call__(self, *args, input: Dict[str, Any] = None, class_=None, **kwargs) -> Dict[str, Any]:
        gather_infromation_configurations = input["gather_information_configurations"]

        frame_extractor_gathered_information = None
        icon_replacer_gathered_information = None
        object_detector_gathered_information = None
        llm_description_gathered_information = None


        input = self.input_map if input is None else input
        input = self._pre(input=input)

        image_files = []
        if "image_introduction" in input.keys():
            for image_info in input["image_introduction"]:
                image_files.append(image_info["path"])

        flag = True
        processed_response = {}

        # Gather information by frame extractor
        if gather_infromation_configurations["frame_extractor"] is True:

            logger.write(f"Using frame extractor to gather information")

            if self.frame_extractor is not None:

                text_input = input["text_input"]
                video_path = input["video_clip_path"]

                if "test_text_image" in input.keys() and input["test_text_image"]:  # offline test
                    extracted_frame_paths = input["test_text_image"]

                else:  # online run
                    # extract the text information of the whole video
                    # run the frame_extractor to get the key frames
                    extracted_frame_paths = self.frame_extractor.extract(video_path=video_path)

                # Gather information by Icon replacer
                if gather_infromation_configurations["icon_replacer"] is True:
                    logger.write(f"Using icon replacer to gather information")
                    if self.icon_replacer is not None:
                        try:
                            extracted_frame_paths = self._replace_icon(extracted_frame_paths)
                        except Exception as e:
                            logger.error(f"Error in gather information by Icon replacer: {e}")
                            flag = False
                    else:
                        logger.warn('Icon replacer is not set, skipping gather information by Icon replacer')

                # For each keyframe, use llm to get the text information
                video_prefix = os.path.basename(video_path).split('.')[0].split('_')[-1]  # Different video should have differen prefix for avoiding the same time stamp
                frame_extractor_gathered_information = JSONStructure()

                if config.parallel_request_gather_information:
                    # Create completions in parallel
                    logger.write(f"Start gathering text information from the whole video in parallel")

                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    try:
                        loop.run_until_complete(
                            get_completion_in_parallel(self.llm_provider, self.text_input_map, extracted_frame_paths,
                                                       text_input,self.get_text_template,video_prefix,frame_extractor_gathered_information))

                    except KeyboardInterrupt:

                        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
                        for task in tasks:
                            task.cancel()

                        loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
                    finally:
                        loop.close()

                else:
                    logger.write(f"Start gathering text information from the whole video in sequence")
                    get_completion_in_sequence(self.llm_provider, self.text_input_map, extracted_frame_paths,
                                               text_input,self.get_text_template,video_prefix,frame_extractor_gathered_information)

                frame_extractor_gathered_information.sort_index_by_timestamp()
                logger.write(f"Finish gathering text information from the whole video")

            else:
                logger.warn('Frame extractor is not set, skipping gather information by frame extractor')
                frame_extractor_gathered_information = None

            # Get dialogue information from the gathered_information_JSON at the subfounder find the dialogue frames
            if frame_extractor_gathered_information is not None:
                dialogues = [item["values"] for item in frame_extractor_gathered_information.search_type_across_all_indices("dialogue")]
            else:
                if self.frame_extractor is not None:
                    msg = "No gathered_information_JSON received, so no dialogue information is provided."
                else:
                    msg = "No gathered_information_JSON available, no Frame Extractor in use."

                logger.warn(msg)
                dialogues = []

            # Update the <$task_description$> in the gather_information template with the latest task_description
            all_task_guidance = frame_extractor_gathered_information.search_type_across_all_indices(constants.TASK_GUIDANCE)

            # Remove the content of "task is none"
            all_task_guidance = [task_guidance for task_guidance in all_task_guidance if constants.NONE_TASK_OUTPUT not in task_guidance["values"].lower()]

            if len(all_task_guidance) != 0:
                # New task guidance is found, use the latest one
                last_task_guidance = max(all_task_guidance, key=lambda x: x['index'])['values']
                input[constants.TASK_DESCRIPTION] = last_task_guidance # this is for the input of the gather_information

            # @TODO: summary the dialogue and use it

        # Gather information by LLM provider
        if gather_infromation_configurations["llm_description"] is True:
            logger.write(f"Using llm description to gather information")
            try:
                # Call the LLM provider for gather information json
                message_prompts = self.llm_provider.assemble_prompt(template_str=self.template, params=input)

                logger.debug(f'{logger.UPSTREAM_MASK}{json.dumps(message_prompts, ensure_ascii=False)}\n')

                gather_information_success_flag = False
                while gather_information_success_flag is False:
                    try:
                        response, info = self.llm_provider.create_completion(message_prompts)
                        logger.debug(f'{logger.DOWNSTREAM_MASK}{response}\n')

                        # Convert the response to dict
                        processed_response = parse_semi_formatted_text(response)
                        gather_information_success_flag = True

                    except Exception as e:
                        logger.error(f"Response of image description is not in the correct format: {e}, retrying...")
                        gather_information_success_flag = False

                        # Wait 2 seconds for the next request and retry
                        time.sleep(2)

                llm_description_gathered_information = processed_response

            except Exception as e:
                logger.error(f"Error in gather image description information: {e}")
                flag = False

        # Assemble the gathered_information_JSON

        if flag:
            objects = []

            if icon_replacer_gathered_information is not None and "objects" in icon_replacer_gathered_information:
                objects.extend(icon_replacer_gathered_information["objects"])
            if object_detector_gathered_information is not None and "objects" in object_detector_gathered_information:
                objects.extend(object_detector_gathered_information["objects"])
            if llm_description_gathered_information is not None and "objects" in llm_description_gathered_information:
                objects.extend(llm_description_gathered_information["objects"])

            objects = list(set(objects))

            processed_response["objects"] = objects

            # Merge the gathered_information_JSON to the processed_response
            processed_response["gathered_information_JSON"] = frame_extractor_gathered_information

            if len(all_task_guidance) == 0:
                processed_response[constants.LAST_TASK_GUIDANCE] = ""
            else:
                processed_response[constants.LAST_TASK_GUIDANCE] = last_task_guidance

        # Gather information by object detector, which is grounding dino.
        if gather_infromation_configurations["object_detector"] is True:
            logger.write(f"Using object detector to gather information")
            if self.object_detector is not None:
                try:
                    target_object_name = processed_response[constants.TARGET_OBJECT_NAME].lower() \
                        if constants.NONE_TARGET_OBJECT_OUTPUT not in processed_response[constants.TARGET_OBJECT_NAME].lower() else ""

                    image_source, boxes, logits, phrases = self.object_detector.detect(image_path=image_files[0],
                                                                                       text_prompt= target_object_name,
                                                                                       box_threshold=0.4, device='cuda')
                    processed_response["boxes"] = boxes
                    processed_response["logits"] = logits
                    processed_response["phrases"] = phrases
                except Exception as e:
                    logger.error(f"Error in gather information by object detector: {e}")
                    flag = False

                try:
                    minimap_detection_objects = self.object_detector.process_minimap_targets(image_files[0])

                    processed_response.update({constants.MINIMAP_INFORMATION:minimap_detection_objects})

                except Exception as e:
                    logger.error(f"Error in gather information by object detector for minimap: {e}")
                    flag = False

        success = self._check_success(data=processed_response)

        data = dict(
            flag=flag,
            success=success,
            input=input,
            res_dict=processed_response,
        )

        data = self._post(data=data)

        return data


    def _post(self, *args, data: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        return data


    def _check_success(self, *args, data, **kwargs):

        success = False

        prop_name = "description"

        if prop_name in data.keys():
            desc = data[prop_name]
            success = desc is not None and len(desc) > 0
        return success


    def _replace_icon(self, extracted_frame_paths):
        extracted_frames = [frame[0] for frame in extracted_frame_paths]
        extracted_timesteps = [frame[1] for frame in extracted_frame_paths]
        extracted_frames = self.icon_replacer(image_paths=extracted_frames)
        extracted_frame_paths = list(zip(extracted_frames, extracted_timesteps))
        return extracted_frame_paths


class DecisionMaking():
    def __init__(self,
                 input_map: Dict = None,
                 template: Dict = None,
                 llm_provider: LLMProvider = None,
                 ):

        self.input_map = input_map
        self.template = template
        self.llm_provider = llm_provider


    def _pre(self, *args, input: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        return input


    def __call__(self, *args, input: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:

        input = self.input_map if input is None else input
        input = self._pre(input=input)

        flag = True
        processed_response = {}

        try:
            message_prompts = self.llm_provider.assemble_prompt(template_str=self.template, params=input)

            logger.debug(f'{logger.UPSTREAM_MASK}{json.dumps(message_prompts, ensure_ascii=False)}\n')

            # Call the LLM provider for decision making
            response, info = self.llm_provider.create_completion(message_prompts)

            logger.debug(f'{logger.DOWNSTREAM_MASK}{response}\n')

            if response is None or len(response) == 0:
                logger.warn('No response in decision making call')
                logger.debug(input)

            # Convert the response to dict
            processed_response = parse_semi_formatted_text(response)

        except Exception as e:
            logger.error(f"Error in decision_making: {e}")
            logger.error_ex(e)
            flag = False

        data = dict(
            flag=flag,
            input=input,
            res_dict=processed_response,
        )

        data = self._post(data=data)
        return data


    def _post(self, *args, data: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        return data


class SuccessDetection():
    def __init__(self,
                 input_map: Dict = None,
                 template: Dict = None,
                 llm_provider: LLMProvider = None,
                 ):
        self.input_map = input_map
        self.template = template
        self.llm_provider = llm_provider


    def _pre(self, *args, input: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        return input


    def __call__(self, *args, input: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:

        input = self.input_map if input is None else input
        input = self._pre(input=input)

        flag = True
        processed_response = {}

        try:

            # Call the LLM provider for success detection
            message_prompts = self.llm_provider.assemble_prompt(template_str=self.template, params=input)

            logger.debug(f'{logger.UPSTREAM_MASK}{json.dumps(message_prompts, ensure_ascii=False)}\n')

            response, info = self.llm_provider.create_completion(message_prompts)

            logger.debug(f'{logger.DOWNSTREAM_MASK}{response}\n')

            # Convert the response to dict
            processed_response = parse_semi_formatted_text(response)

        except Exception as e:
            logger.error(f"Error in success_detection: {e}")
            flag = False

        data = dict(
            flag=flag,
            input=input,
            res_dict=processed_response,
        )

        data = self._post(data=data)
        return data


    def _post(self, *args, data: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        return data


class SelfReflection():

    def __init__(self,
                 input_map: Dict = None,
                 template: Dict = None,
                 llm_provider: LLMProvider = None,
                 ):
        self.input_map = input_map
        self.template = template
        self.llm_provider = llm_provider


    def _pre(self, *args, input: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        return input


    def __call__(self, *args, input: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:

        input = self.input_map if input is None else input
        input = self._pre(input=input)

        flag = True
        processed_response = {}

        try:

            # Call the LLM provider for self reflection
            message_prompts = self.llm_provider.assemble_prompt(template_str=self.template, params=input)

            logger.debug(f'{logger.UPSTREAM_MASK}{json.dumps(message_prompts, ensure_ascii=False)}\n')

            response, info = self.llm_provider.create_completion(message_prompts)

            logger.debug(f'{logger.DOWNSTREAM_MASK}{response}\n')

            # Convert the response to dict
            processed_response = parse_semi_formatted_text(response)

        except Exception as e:
            logger.error(f"Error in self reflection: {e}")
            flag = False

        data = dict(
            flag=flag,
            input=input,
            res_dict=processed_response,
        )

        data = self._post(data=data)
        return data


    def _post(self, *args, data: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        return data


class InformationSummary():

    def __init__(self,
                 input_map: Dict = None,
                 template: Dict = None,
                 llm_provider: LLMProvider = None,
                 ):

        self.input_map = input_map
        self.template = template
        self.llm_provider = llm_provider


    def _pre(self, *args, input: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        return input


    def __call__(self, *args, input: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:

        input = self.input_map if input is None else input
        input = self._pre(input=input)

        flag = True
        processed_response = {}
        res_json = None

        try:

            # Call the LLM provider for information summary
            message_prompts = self.llm_provider.assemble_prompt(template_str=self.template, params=input)

            logger.debug(f'{logger.UPSTREAM_MASK}{json.dumps(message_prompts, ensure_ascii=False)}\n')

            response, info = self.llm_provider.create_completion(message_prompts)

            logger.debug(f'{logger.DOWNSTREAM_MASK}{response}\n')

            # Convert the response to dict
            processed_response = parse_semi_formatted_text(response)

        except Exception as e:
            logger.error(f"Error in information_summary: {e}")
            flag = False

        data = dict(
            flag=flag,
            input=input,
            res_dict=processed_response,
            # res_json=res_json,
        )

        data = self._post(data=data)
        return data


    def _post(self, *args, data: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        return data


class Planner(BasePlanner):

    def __init__(self,
                 llm_provider: Any = None,
                 planner_params: Dict = None,
                 use_screen_classification: bool = False,
                 use_information_summary: bool = False,
                 use_self_reflection: bool = False,
                 gather_information_max_steps: int = 1,  # 5,
                 icon_replacer: Any = None,
                 object_detector: Any = None,
                 frame_extractor: Any = None,
                 ):
        """
        inputs: input key-value pairs
        templates: template for composing the prompt

        planner_params = {
            "__check_list__":[
              "screen_classification",
              "gather_information",
              "decision_making",
              "information_summary",
              "self_reflection"
            ],
            "prompt_paths": {
              "inputs": {
                "screen_classification": "./res/prompts/inputs/screen_classification.json",
                "gather_information": "./res/prompts/inputs/gather_information.json",
                "decision_making": "./res/prompts/inputs/decision_making.json",
                "success_detection": "./res/prompts/inputs/success_detection.json",
                "information_summary": "./res/prompts/inputs/information_summary.json",
                "self_reflection": "./res/prompts/inputs/self_reflection.json",
              },
              "templates": {
                "screen_classification": "./res/prompts/templates/screen_classification.prompt",
                "gather_information": "./res/prompts/templates/gather_information.prompt",
                "decision_making": "./res/prompts/templates/decision_making.prompt",
                "success_detection": "./res/prompts/templates/success_detection.prompt",
                "information_summary": "./res/prompts/templates/information_summary.prompt",
                "self_reflection": "./res/prompts/templates/self_reflection.prompt",
              }
            }
          }
        """

        super(BasePlanner, self).__init__()

        self.llm_provider = llm_provider

        self.use_screen_classification = use_screen_classification
        self.use_information_summary = use_information_summary
        self.use_self_reflection = use_self_reflection
        self.gather_information_max_steps = gather_information_max_steps

        self.icon_replacer = icon_replacer
        self.object_detector = object_detector
        self.frame_extractor = frame_extractor
        self.set_internal_params(planner_params=planner_params,
                                 use_screen_classification=use_screen_classification,
                                 use_information_summary=use_information_summary)


    # Allow re-configuring planner
    def set_internal_params(self,
                            planner_params: Dict = None,
                            use_screen_classification: bool = False,
                            use_information_summary: bool = False):

        self.planner_params = planner_params
        if not check_planner_params(self.planner_params):
            raise ValueError(f"Error in planner_params: {self.planner_params}")

        self.inputs = self._init_inputs()
        self.templates = self._init_templates()

        if use_screen_classification:
            self.screen_classification_ = ScreenClassification(input_example=self.inputs["screen_classification"],
                                                               template=self.templates["screen_classification"],
                                                               llm_provider=self.llm_provider)
        else:
            self.screen_classification_ = None

        self.gather_information_ = GatherInformation(input_map=self.inputs["gather_information"],
                                                     template=self.templates["gather_information"],
                                                     text_input_map=self.inputs["gather_text_information"],
                                                     get_text_template=self.templates["gather_text_information"],
                                                     frame_extractor=self.frame_extractor,
                                                     icon_replacer=self.icon_replacer,
                                                     object_detector=self.object_detector,
                                                     llm_provider=self.llm_provider)

        self.decision_making_ = DecisionMaking(input_map=self.inputs["decision_making"],
                                               template=self.templates["decision_making"],
                                               llm_provider=self.llm_provider)

        self.success_detection_ = SuccessDetection(input_map=self.inputs["success_detection"],
                                                   template=self.templates["success_detection"],
                                                   llm_provider=self.llm_provider)

        if self.use_self_reflection:
            self.self_reflection_ = SelfReflection(input_map=self.inputs["self_reflection"],
                                                   template=self.templates["self_reflection"],
                                                   llm_provider=self.llm_provider)
        else:
            self.self_reflection_ = None

        if use_information_summary:
            self.information_summary_ = InformationSummary(input_map=self.inputs["information_summary"],
                                                           template=self.templates["information_summary"],
                                                           llm_provider=self.llm_provider)
        else:
            self.information_summary_ = None


    def _init_inputs(self):

        input_examples = dict()
        prompt_paths = self.planner_params["prompt_paths"]
        input_example_paths = prompt_paths["inputs"]

        for key, value in input_example_paths.items():
            path = assemble_project_path(value)
            if path.endswith(PROMPT_EXT):
                input_examples[key] = read_resource_file(path)
            else:
                input_examples[key] = load_json(path)

        return input_examples


    def _init_templates(self):

        templates = dict()
        prompt_paths = self.planner_params["prompt_paths"]
        template_paths = prompt_paths["templates"]

        for key, value in template_paths.items():
            path = assemble_project_path(value)
            if path.endswith(PROMPT_EXT):
                templates[key] = read_resource_file(path)
            else:
                templates[key] = load_json(path)

        return templates


    def gather_information(self, *args, input: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:

        if input is None:
            input = self.inputs["gather_information"]

        image_file = input["image_introduction"][0]["path"]

        if self.use_screen_classification:
            class_ = self.screen_classification_(screenshot_file=image_file)["class_"]
        else:
            class_ = None

        for i in range(self.gather_information_max_steps):
            data = self.gather_information_(input=input, class_=class_)

            success = data["success"]

            if success:
                break

        return data


    def decision_making(self, *args, input: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:

        if input is None:
            input = self.inputs["decision_making"]

        data = self.decision_making_(input=input)

        return data


    def success_detection(self, *args, input: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:

        if input is None:
            input = self.inputs["success_detection"]

        data = self.success_detection_(input=input)

        return data


    def self_reflection(self, *args, input: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:

        if input is None:
            input = self.inputs["self_reflection"]

        data = self.self_reflection_(input=input)

        return data


    def information_summary(self, *args, input: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:

        if input is None:
            input = self.inputs["information_summary"]

        data = self.information_summary_(input=input)

        return data
