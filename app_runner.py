import os
import time
from copy import deepcopy
import argparse
from typing import Dict, Any, List
import atexit
import re

from cradle import constants
from cradle.log import Logger
from cradle.planner.planner import Planner
from cradle.config import Config
from cradle.memory import LocalMemory
from cradle.provider.openai import OpenAIProvider
from cradle.provider.sam_provider import SamProvider
from cradle.gameio.io_env import IOEnvironment
from cradle.gameio.game_manager import GameManager
from cradle.gameio.video.VideoRecorder import VideoRecorder
from cradle.gameio.lifecycle.ui_control import switch_to_game, normalize_coordinates, draw_mouse_pointer_file
import cradle.environment.outlook
import cradle.environment.chrome
import cradle.environment.capcut
import cradle.environment.feishu
import cradle.environment.xiuxiu
from cradle.utils.dict_utils import kget
from cradle.utils.image_utils import calculate_pixel_diff
from cradle.utils.file_utils import copy_file

config = Config()
logger = Logger()
io_env = IOEnvironment()


class PipelineRunner():

    def __init__(self,
                 llm_provider_config_path: str,
                 use_success_detection: bool = False,
                 use_self_reflection: bool = False,
                 use_information_summary: bool = False):

        self.llm_provider_config_path = llm_provider_config_path

        self.use_success_detection = use_success_detection
        self.use_self_reflection = use_self_reflection
        self.use_information_summary = use_information_summary

        # Success flag
        self.stop_flag = False

        # Count the number of turns
        self.count_turns = 0
        # Count the consecutive times the action is empty
        self.count_empty_action = 0
        self.count_empty_action_threshold = 2

        # Init internal params
        self.set_internal_params()


    def set_internal_params(self, *args, **kwargs):

        # Init LLM provider
        self.llm_provider = OpenAIProvider()
        self.llm_provider.init_provider(self.llm_provider_config_path)
        io_env.llm_provider = self.llm_provider # @TODO needs a better DI

        # Init GD provider (not used in this example)
        self.gd_detector = None

        # Init Sam provider
        self.sam_provider = SamProvider()

        # Init video frame extractor (not used in this example)
        self.frame_extractor = None

        # Init icon replacer
        self.icon_replacer = None

        # Init memory
        self.memory = LocalMemory(memory_path=config.work_dir,
                                  max_recent_steps=config.max_recent_steps)
        self.memory.load(config.memory_load_path)

        # Init game manager
        self.gm = GameManager(env_name=config.env_name,
                              embedding_provider=self.llm_provider)

        self.interface = self.gm.get_interface()
        self.planner_params = self.interface.planner_params
        self.skill_library = self.interface.skill_library
        self.task_description = self.interface.task_description

        # Init planner
        self.planner = Planner(llm_provider=self.llm_provider,
                               planner_params=self.planner_params,
                               frame_extractor=self.frame_extractor,
                               icon_replacer=self.icon_replacer,
                               object_detector=self.gd_detector,
                               use_self_reflection=self.use_self_reflection,
                               use_information_summary=self.use_information_summary)

        # Init skill library
        if config.skill_retrieval:
            self.gm.register_available_skills(self.skill_library)
            self.skill_library = self.gm.retrieve_skills(query_task=self.task_description,
                                                         skill_num=config.skill_num,
                                                         screen_type=constants.GENERAL_GAME_INTERFACE)
        self.skill_library = self.gm.get_skill_information(self.skill_library)
        self.processed_skill_library = pre_process_skill_library(self.skill_library)

        # Init video recorder
        self.videocapture = VideoRecorder(os.path.join(config.work_dir, 'video.mp4'))


    def run(self):

        self.task_description = "No Task"

        task_id, subtask_id = 1, 0
        try:
            # Read end to end task description from config file
            self.task_description = kget(config.env_config, constants.TASK_DESCRIPTION_LIST, default='')[task_id-1][constants.TASK_DESCRIPTION]
            # To focus on a subtask description, you can use the following code:
            # self.task_description = kget(config.env_config, constants.TASK_DESCRIPTION_LIST, default='')[task_id-1][constants.SUB_TASK_DESCRIPTION_LIST][subtask_id-1]
        except:
            logger.error(f"Task description is not found for task_id: {task_id} and/or subtask_id: {subtask_id}")

        self.memory.add_recent_history(constants.TASK_DESCRIPTION, self.task_description)

        params = {}

        # Switch to target environment
        switch_to_game()

        # Prepare
        if config.enable_videocapture:
            self.videocapture.start_capture()
        start_frame_id = self.videocapture.get_current_frame_id()

        # First sense
        cur_screenshot_path, _ = self.gm.capture_screen()
        mouse_x, mouse_y = io_env.get_mouse_position()

        time.sleep(2)
        end_frame_id = self.videocapture.get_current_frame_id()

        params.update({
            "start_frame_id": start_frame_id,
            "end_frame_id": end_frame_id,
            "cur_screenshot_path": cur_screenshot_path,
            "mouse_position" : (mouse_x, mouse_y),
            "exec_info": {
                "errors": False,
                "errors_info": ""
            },
            "pre_action": "",
            "pre_decision_making_reasoning": "",
            "pre_self_reflection_reasoning": "",
            "task_description": self.task_description,
            "summarization": "",
            "subtask_description": "",
            "subtask_reasoning": "",
            f"{constants.PREVIOUS_AUGMENTATION_INFO}": None,
        })

        while not self.stop_flag:

            try:

                logger.write(f'>>> Overall Task Description: {self.task_description}')

                # Gather information
                gather_information_params = self.gather_information(params, debug=False)
                params.update(gather_information_params)

                # Self reflection
                self_reflection_params = self.self_reflection(params)
                params.update(self_reflection_params)

                # # Skill retrieval
                # skill_retrieval_params = self.skill_retrieval(params)
                # params.update(skill_retrieval_params)

                # Information summary
                information_summary_params = self.information_summary(params)
                params.update(information_summary_params)

                # Decision making
                decision_making_params = self.decision_making(params)
                params.update(decision_making_params)

                # # Success detection
                # success_detection_params = self.success_detection(params)
                # params.update(success_detection_params)

                # success = success_detection_params["success"]

                self.gm.store_skills()
                self.memory.save()

            except KeyboardInterrupt:
                logger.write('KeyboardInterrupt Ctrl+C detected, exiting.')
                self.pipeline_shutdown()
                break

        self.pipeline_shutdown()


    def self_reflection(self, params: Dict[str, Any]):

        start_frame_id = params["start_frame_id"]
        end_frame_id = params["end_frame_id"]

        task_description = params["task_description"]
        pre_action = params["pre_action"]

        previous_augmentation = params[constants.PREVIOUS_AUGMENTATION_INFO]
        current_augmentation = params[constants.CURRENT_AUGMENTATION_INFO]

        pre_decision_making_reasoning = params["pre_decision_making_reasoning"]
        exec_info = params["exec_info"]

        image_same_flag = current_augmentation[constants.IMAGE_SAME_FLAG]
        mouse_position_same_flag = current_augmentation[constants.MOUSE_POSITION_SAME_FLAG]

        image_memory = self.memory.get_recent_history("image", k=config.self_reflection_image_num)

        self_reflection_reasoning = ""
        success_detection = False
        if self.use_self_reflection and self.count_turns > 0:
            input = self.planner.self_reflection_.input_map
            action_frames = []
            video_frames = self.videocapture.get_frames(start_frame_id, end_frame_id)

            # if len(video_frames) <= config.max_images_in_self_reflection * config.duplicate_frames + 1:
            #     action_frames = [frame[1] for frame in video_frames[1::config.duplicate_frames]]
            # else:
            #     for i in range(config.max_images_in_self_reflection):
            #         step = len(video_frames) // config.max_images_in_self_reflection * i + 1
            #         action_frames.append(video_frames[step][1])

            # only use the first and last frame for self-reflection
            # add grid and color band to the frames
            # action_frames.append(self.gm.interface.augment_image(video_frames[0][1], x = response[constants.MOUSE_X_POSITION], y = response[constants.MOUSE_Y_POSITION], encoding = cv2.COLOR_BGRA2RGB))
            # action_frames.append(self.gm.interface.augment_image(video_frames[-1][1], x = response[constants.MOUSE_X_POSITION], y = response[constants.MOUSE_Y_POSITION], encoding = cv2.COLOR_BGRA2RGB))
            if config.show_mouse_in_screenshot:
                action_frames.append(previous_augmentation[constants.AUG_MOUSE_IMG_PATH])
                action_frames.append(current_augmentation[constants.AUG_MOUSE_IMG_PATH])
            else:
                action_frames.append(previous_augmentation[constants.AUG_BASE_IMAGE_PATH])
                action_frames.append(current_augmentation[constants.AUG_BASE_IMAGE_PATH])

            image_introduction = [
                {
                    "introduction": "Here are the sequential frames of the character executing the last action.",
                    "path": action_frames,
                    "assistant": "",
                    "resolution": "low"
                }]

            input["image_introduction"] = image_introduction
            input["current_image_description"] = current_augmentation[constants.IMAGE_DESCRIPTION]

            input["task_description"] = task_description
            input['skill_library'] = self.processed_skill_library
            input["previous_reasoning"] = pre_decision_making_reasoning
            input["history_summary"] = self.memory.get_summarization()
            input["subtask_description"] = params["subtask_description"]

            input[constants.IMAGE_SAME_FLAG] = str(image_same_flag)
            input[constants.MOUSE_POSITION_SAME_FLAG] = str(mouse_position_same_flag)

            if pre_action:

                pre_action_name = []
                pre_action_code = []

                for action in pre_action:
                    skill = self.gm.skill_registry.convert_expression_to_skill(action)
                    name, params = skill
                    action_code, action_info = self.gm.get_skill_library_in_code(name)
                    pre_action_name.append(name)
                    pre_action_code.append(action_code if action_code is not None else action_info)

                    input["previous_action"] = ",".join(pre_action_name)
                    input["previous_action_call"] = pre_action
                    input['action_code'] = "\n".join(list(set(pre_action_code)))
                    input[constants.KEY_REASON_OF_LAST_ACTION] = self.memory.get_recent_history(constants.KEY_REASON_OF_LAST_ACTION, k=1)[-1]
                    input[constants.SUCCESS_DETECTION] = self.memory.get_recent_history(constants.SUCCESS_DETECTION, k=1)[-1]
            else:
                input["previous_action"] = ""
                input["previous_action_call"] = ""
                input['action_code'] = ""

            if exec_info["errors"]:
                input['executing_action_error'] = exec_info["errors_info"]
            else:
                input['executing_action_error'] = ""

            # >> Calling SELF REFLECTION
            logger.write(f'>> Calling SELF REFLECTION')
            reflection_data = self.planner.self_reflection(input=input)

            self_reflection_reasoning = extract_response_key(reflection_data['res_dict'], constants.SELF_REFLECTION_REASONING)
            success_detection = extract_response_key(reflection_data['res_dict'], constants.SUCCESS_DETECTION)

            self.memory.add_recent_history(constants.SELF_REFLECTION_REASONING, self_reflection_reasoning)
            self.memory.add_recent_history(constants.SUCCESS_DETECTION, success_detection)
            logger.write(f'Self-reflection reason: {self_reflection_reasoning}')
            logger.write(f'Success detection: {success_detection}')

        res_params = {
            "pre_self_reflection_reasoning": self_reflection_reasoning,
            f"{constants.SUCCESS_DETECTION}": success_detection,
            f"{constants.CURRENT_AUGMENTATION_INFO}": current_augmentation,
            f"{constants.PREVIOUS_AUGMENTATION_INFO}": previous_augmentation
        }

        return res_params


    def pipeline_shutdown(self):
        self.gm.cleanup_io()
        self.videocapture.finish_capture()
        logger.write('>>> Bye.')


    # def skill_retrieval(self, params: Dict[str, Any]):

    #     last_task_guidance = params["last_task_guidance"]
    #     long_horizon = params["long_horizon"]
    #     all_generated_actions = params["all_generated_actions"]
    #     screen_classification = params["screen_classification"]
    #     task_description = params["task_description"]

    #     if last_task_guidance:
    #         task_description = last_task_guidance
    #         self.memory.add_task_guidance(last_task_guidance, long_horizon)

    #     logger.write(f'Current Task Guidance: {task_description}')

    #     if config.skill_retrieval:
    #         for extracted_skills in all_generated_actions:
    #             extracted_skills = extracted_skills['values']
    #             for extracted_skill in extracted_skills:
    #                 self.gm.add_new_skill(skill_code=extracted_skill['code'])

    #         skill_library = self.gm.retrieve_skills(query_task=task_description, skill_num=config.skill_num,
    #                                                 screen_type=screen_classification.lower())
    #         logger.write(f'skill_library: {skill_library}')
    #         skill_library = self.gm.get_skill_information(skill_library)

    #     self.videocapture.clear_frame_buffer()

    #     res_params = {}
    #     return res_params


    def gather_information(self, params: Dict[str, Any], debug=False):

        # Get params
        start_frame_id = params["start_frame_id"]
        end_frame_id = params["end_frame_id"]
        cur_screenshot_path: List[str] = params["cur_screenshot_path"]
        self.memory.add_recent_history(key=constants.IMAGES_MEM_BUCKET, info=cur_screenshot_path)

        # Gather information preparation
        logger.write(f'Gather Information Start Frame ID: {start_frame_id}, End Frame ID: {end_frame_id}')
        input = self.planner.gather_information_.input_map
        input["task_description"] = self.task_description

        # Configure the test
        # if you want to test with a pre-defined screenshot, you can replace the cur_screenshot_path with the path to the screenshot
        pre_defined_sreenshot = None
        if pre_defined_sreenshot is not None:
            cur_screenshot_path = pre_defined_sreenshot
        else:
            cur_screenshot_path = params['cur_screenshot_path']

        input["image_introduction"][0]["path"] = cur_screenshot_path

        previous_augmentation = params[constants.PREVIOUS_AUGMENTATION_INFO]
        current_augmentation = {}
        current_augmentation[constants.AUG_BASE_IMAGE_PATH] = cur_screenshot_path

        # record the last collected mouse position
        mouse_position = kget(params, 'mouse_position')
        if mouse_position:
            mouse_x, mouse_y = mouse_position
            current_augmentation[constants.AUG_MOUSE_X] = mouse_x
            current_augmentation[constants.AUG_MOUSE_Y] = mouse_y
        else:
            logger.warn("No mouse position to draw in info gathering augmentation!")

        # Compare current screenshot with the previous to determine changes
        if mouse_position and previous_augmentation:
            previous_screenshot_path = previous_augmentation[constants.AUG_BASE_IMAGE_PATH]
            count_same_of_pic = calculate_pixel_diff(previous_screenshot_path, cur_screenshot_path)
            image_same_flag = count_same_of_pic <= config.pixel_diff_threshold
            mouse_position_same_flag = (
                current_augmentation[constants.AUG_MOUSE_X] == previous_augmentation[constants.AUG_MOUSE_X] and
                current_augmentation[constants.AUG_MOUSE_Y] == previous_augmentation[constants.AUG_MOUSE_Y]
            )
        else:
            image_same_flag = mouse_position_same_flag = False # Assume false for the first run to generate augmentation image
        logger.write(f'Image Same Flag: {image_same_flag}, Mouse Position Same Flag: {mouse_position_same_flag}')

        current_augmentation[constants.IMAGE_SAME_FLAG] = image_same_flag
        current_augmentation[constants.MOUSE_POSITION_SAME_FLAG] = mouse_position_same_flag

        if mouse_position and config.show_mouse_in_screenshot:
            if mouse_position_same_flag and image_same_flag:
                aug_mouse_img_path = cur_screenshot_path.replace(".jpg", f"_with_mouse.jpg")
                current_augmentation[constants.AUG_MOUSE_IMG_PATH] = aug_mouse_img_path
                copy_file(previous_augmentation[constants.AUG_MOUSE_IMG_PATH], current_augmentation[constants.AUG_MOUSE_IMG_PATH])
            else:
                current_augmentation[constants.AUG_MOUSE_IMG_PATH] = draw_mouse_pointer_file(cur_screenshot_path, mouse_x, mouse_y)
            input["image_introduction"][0]["path"] = current_augmentation[constants.AUG_MOUSE_IMG_PATH]

        if config.use_sam_flag:

            logger.write(f'Starting SOM augmentation.')

            if image_same_flag:
                aug_som_img_path = cur_screenshot_path.replace(".jpg", f"_som.jpg")
                current_augmentation[constants.AUG_SOM_IMAGE_PATH] = aug_som_img_path
                copy_file(previous_augmentation[constants.AUG_SOM_IMAGE_PATH], current_augmentation[constants.AUG_SOM_IMAGE_PATH])
                current_augmentation[constants.AUG_SOM_MAP] = previous_augmentation[constants.AUG_SOM_MAP].copy()
                current_augmentation[constants.LENGTH_OF_SOM_MAP] = previous_augmentation[constants.LENGTH_OF_SOM_MAP]

            else:

                som_img_path, som_map = self.sam_provider.calc_and_plot_som_results(cur_screenshot_path)

                current_augmentation[constants.AUG_SOM_IMAGE_PATH] = som_img_path
                current_augmentation[constants.AUG_SOM_MAP] = som_map.copy()
                current_augmentation[constants.LENGTH_OF_SOM_MAP] = len(som_map.keys())

            logger.write(f'SOM augmentation finished.')

            input["image_introduction"][0]["path"] = current_augmentation[constants.AUG_SOM_IMAGE_PATH]
            input[constants.LENGTH_OF_SOM_MAP] = str(current_augmentation[constants.LENGTH_OF_SOM_MAP])

            if mouse_position and config.show_mouse_in_screenshot:
                if mouse_position_same_flag and image_same_flag:
                    aug_som_mouse_img_path = cur_screenshot_path.replace(".jpg", f"_som_with_mouse.jpg")
                    current_augmentation[constants.AUG_SOM_MOUSE_IMG_PATH] = aug_som_mouse_img_path
                    copy_file(previous_augmentation[constants.AUG_SOM_MOUSE_IMG_PATH], current_augmentation[constants.AUG_SOM_MOUSE_IMG_PATH])
                else:
                    current_augmentation[constants.AUG_SOM_MOUSE_IMG_PATH] = draw_mouse_pointer_file(current_augmentation[constants.AUG_SOM_IMAGE_PATH], mouse_x, mouse_y)
                input["image_introduction"][0]["path"] = current_augmentation[constants.AUG_SOM_MOUSE_IMG_PATH]

        if previous_augmentation is None:
            previous_augmentation = current_augmentation.copy()

        self.memory.add_recent_history(constants.AUGMENTED_IMAGES_MEM_BUCKET, current_augmentation)

        subtask_description = ""
        if "subtask_description" in params.keys():
            input["subtask_description"] = params["subtask_description"]
            subtask_description = params["subtask_description"]

        # Configure the gather_information module
        gather_information_configurations = {
            "frame_extractor": False,  # extract text from the video clip
            "icon_replacer": False,
            "llm_description": True,  # get the description of the current screenshot
            "object_detector": False
        }
        input["gather_information_configurations"] = gather_information_configurations

        # >> Calling INFORMATION GATHERING
        logger.write(f'>> Calling INFORMATION GATHERING')

        if debug:
            # Do not call GPT-4V, just take the screenshot
            data = {
                "res_dict": {
                    f"{constants.IMAGE_DESCRIPTION}": "No description",
                }
            }
        else:
            data = self.planner.gather_information(input=input)

        current_augmentation[constants.IMAGE_DESCRIPTION] = extract_response_key(data['res_dict'], constants.IMAGE_DESCRIPTION)
        target_object = extract_response_key(data['res_dict'], constants.TARGET_OBJECT_NAME)

        if constants.DESCRIPTION_OF_BOUNDING_BOXES in data['res_dict'].keys() and config.use_sam_flag:
            description_of_bounding_boxes = data['res_dict'][constants.DESCRIPTION_OF_BOUNDING_BOXES]
            current_augmentation[constants.DESCRIPTION_OF_BOUNDING_BOXES] = description_of_bounding_boxes.replace("\n", " ")
        else:
            logger.warn(f"No {constants.DESCRIPTION_OF_BOUNDING_BOXES} in response.")
            current_augmentation[constants.DESCRIPTION_OF_BOUNDING_BOXES] = "No descriptions"

        logger.write('Gather Information response: ', data[constants.PA_PROCESSED_RESPONSE_TAG])
        logger.write(f'Image Description: {current_augmentation[constants.IMAGE_DESCRIPTION]}')
        logger.write(f'Target object: {target_object}')
        logger.write(f'Image Description of Bounding Boxes: {current_augmentation[constants.DESCRIPTION_OF_BOUNDING_BOXES]}')

        res_params = {
            f"{constants.PA_RESPONSE_TAG}": data[constants.PA_PROCESSED_RESPONSE_TAG].keys(),
            f"{constants.PA_RESPONSE_KEYS_TAG}": data[constants.PA_PROCESSED_RESPONSE_TAG],
            f"{constants.SUB_TASK_DESCRIPTION}": subtask_description,
            f"{constants.TARGET_OBJECT_NAME}": target_object,
            f"{constants.CURRENT_AUGMENTATION_INFO}": current_augmentation,
            f"{constants.PREVIOUS_AUGMENTATION_INFO}": previous_augmentation
        }

        return res_params


    def decision_making(self, params: Dict[str, Any]):

        response_keys = params["response_keys"]
        response = params["response"]
        current_augmentation = params[constants.CURRENT_AUGMENTATION_INFO]
        previous_augmentation = params[constants.PREVIOUS_AUGMENTATION_INFO]
        pre_action = params["pre_action"]
        pre_self_reflection_reasoning = params["pre_self_reflection_reasoning"]
        success_detection = params[constants.SUCCESS_DETECTION]

        # Decision making preparation
        input = deepcopy(self.planner.decision_making_.input_map)
        img_prompt_decision_making = self.planner.decision_making_.input_map["image_introduction"]

        number_of_execute_skills = input["number_of_execute_skills"]

        if pre_action:
            input["previous_action"] = self.memory.get_recent_history("action", k=1)[-1]
            input["previous_reasoning"] = self.memory.get_recent_history(constants.DECISION_MAKING_REASONING, k=1)[-1]
            input[constants.KEY_REASON_OF_LAST_ACTION] = self.memory.get_recent_history(constants.KEY_REASON_OF_LAST_ACTION, k=1)[-1]
            input[constants.SUCCESS_DETECTION] = self.memory.get_recent_history(constants.SUCCESS_DETECTION, k=1)[-1]

        if pre_self_reflection_reasoning:
            input["previous_self_reflection_reasoning"] = self.memory.get_recent_history(constants.SELF_REFLECTION_REASONING, k=1)[-1]

        if success_detection:
            input[constants.SUCCESS_DETECTION] = success_detection

        input['skill_library'] = self.processed_skill_library
        input['info_summary'] = self.memory.get_summarization()

        # @TODO: few shots should be REMOVED in prompt decision making if not used
        # input['few_shots'] = []

        # @TODO Temporary solution with fake augmented entries if no bounding box exists. Ideally it should read images, then check for possible augmentation.
        image_memory = self.memory.get_recent_history("image", k=config.decision_making_image_num)

        image_introduction = []
        for i in range(len(image_memory), 0, -1):
            image_introduction.append(
                {
                    "introduction": img_prompt_decision_making[-i]["introduction"],
                    "path":image_memory[-i],
                    "assistant": img_prompt_decision_making[-i]["assistant"]
                })

        input["image_introduction"] = image_introduction
        input[constants.IMAGE_DESCRIPTION] = current_augmentation[constants.IMAGE_DESCRIPTION]
        input["task_description"] = self.task_description
        input["previous_summarization"] = self.memory.get_summarization()

        input[constants.TARGET_OBJECT_NAME] = extract_response_key(params, constants.TARGET_OBJECT_NAME)
        input["subtask_description"] = extract_response_key(params, 'subtask_description')
        subtask_description = extract_response_key(params, 'subtask_description')

        input["image_introduction"][0]["path"] = previous_augmentation[constants.AUG_BASE_IMAGE_PATH]
        input["image_introduction"][-1]["path"] = current_augmentation[constants.AUG_BASE_IMAGE_PATH]

        if config.show_mouse_in_screenshot:
            input["image_introduction"][0]["path"] = previous_augmentation[constants.AUG_MOUSE_IMG_PATH]
            input["image_introduction"][-1]["path"] = current_augmentation[constants.AUG_MOUSE_IMG_PATH]

        if config.use_sam_flag:
            input["image_introduction"][0]["path"] = previous_augmentation[constants.AUG_SOM_IMAGE_PATH]
            input["image_introduction"][-1]["path"] = current_augmentation[constants.AUG_SOM_IMAGE_PATH]
            input[constants.DESCRIPTION_OF_BOUNDING_BOXES] = current_augmentation[constants.DESCRIPTION_OF_BOUNDING_BOXES]

            if config.show_mouse_in_screenshot:
                input["image_introduction"][0]["path"] = previous_augmentation[constants.AUG_SOM_MOUSE_IMG_PATH]
                input["image_introduction"][-1]["path"] = current_augmentation[constants.AUG_SOM_MOUSE_IMG_PATH]

        input[constants.IMAGE_SAME_FLAG] = str(current_augmentation[constants.IMAGE_SAME_FLAG])
        input[constants.MOUSE_POSITION_SAME_FLAG] = str(current_augmentation[constants.MOUSE_POSITION_SAME_FLAG])

        # >> Calling DECISION MAKING
        logger.write(f'>> Calling DECISION MAKING')
        data = self.planner.decision_making(input=input)

        pre_decision_making_reasoning = extract_response_key(data['res_dict'], constants.DECISION_MAKING_REASONING)
        key_reason_of_last_action = extract_response_key(data['res_dict'], constants.KEY_REASON_OF_LAST_ACTION)

        # For such cases with no expected response, we should define a retry limit
        logger.write(f'Decision reasoning: {pre_decision_making_reasoning}')
        logger.write(f'Decision key_reason: {key_reason_of_last_action}')

        # Try to execute selected skills
        try:
            skill_steps = data['res_dict']['actions']
        except KeyError:
            logger.error(f'No actions found in decision making response.')
            skill_steps = None

        if skill_steps is None:
            skill_steps = []

        logger.write(f'Response steps: {skill_steps}')

        # >> Calling SKILL EXECUTION
        logger.write(f'>>> Calling SKILL EXECUTION')

        # Filter nop actions in list
        skill_steps = [ i for i in skill_steps if i != '']
        if len(skill_steps) == 0:
            skill_steps = ['']

        skill_steps = skill_steps[:number_of_execute_skills]

        skill_steps = pre_process_skill_steps(skill_steps, current_augmentation[constants.AUG_SOM_MAP])

        logger.write(f'Skill Steps: {skill_steps}')

        if skill_steps == ['']:
            self.count_empty_action += 1
        else:
            self.count_empty_action = 0

        if self.count_empty_action >= self.count_empty_action_threshold:
            self.stop_flag = True
            logger.write(f'Empty action count reached {self.count_empty_action_threshold} times. Task is considered successful.')

        start_frame_id = self.videocapture.get_current_frame_id()

        if self.count_turns >= config.max_turn_count:
            self.stop_flag = True
            logger.write(f'Turn count reached {config.max_turn_count} times. Task is considered successful.')

        exec_info = self.gm.execute_actions(skill_steps)

        # exec_info also has the list of successfully executed skills. skill_steps is the full list, which may differ if there were execution errors.
        pre_action = exec_info[constants.EXECUTED_SKILLS]
        logger.write(f'>>> Post skill execution sensing...')

        # Sense here to avoid changes in state after action execution completes
        mouse_x, mouse_y = io_env.get_mouse_position()

        # First, check if interaction left the target environment
        if not self.gm.check_active_window():
            logger.warn(f"Target environment window is no longer active!")
            cur_screenshot_path = self.gm.get_out_screen()
        else:
            cur_screenshot_path, _ = self.gm.capture_screen()

        end_frame_id = self.videocapture.get_current_frame_id()

        logger.write(f'>>> Done.')

        self.memory.add_recent_history("action", pre_action)
        if exec_info["errors"]:
            self.memory.add_recent_history("action_error", exec_info["errors_info"])
        else:
            self.memory.add_recent_history("action_error", "")

        self.memory.add_recent_history("decision_making_reasoning", pre_decision_making_reasoning)
        self.memory.add_recent_history(constants.KEY_REASON_OF_LAST_ACTION, key_reason_of_last_action)
        previous_augmentation = current_augmentation.copy()

        res_params = {
            "pre_action": pre_action,
            "pre_decision_making_reasoning": pre_decision_making_reasoning,
            'subtask_description': subtask_description,
            "exec_info": exec_info,
            "start_frame_id": start_frame_id,
            "end_frame_id": end_frame_id,
            "cur_screenshot_path": cur_screenshot_path,
            "mouse_position" : (mouse_x, mouse_y),
            f"{constants.PREVIOUS_AUGMENTATION_INFO}" : previous_augmentation,
        }

        self.count_turns += 1
        logger.write(f'-----------------------------Count turns: {self.count_turns} ----------------------------------')

        return res_params


    def information_summary(self, params: Dict[str, Any]):

        task_description = params["task_description"]
        success_detection = params[constants.SUCCESS_DETECTION]

        # Information summary preparation
        input = self.planner.information_summary_.input_map
        logger.write(f'> Information summary call...')

        mem_entries = self.memory.get_recent_history(constants.AUGMENTED_IMAGES_MEM_BUCKET, 1)
        image = mem_entries[0][constants.AUG_MOUSE_IMG_PATH]

        decision_making_reasoning = self.memory.get_recent_history(constants.KEY_REASON_OF_LAST_ACTION, k=1)[-1]
        self_reflection_reasoning = self.memory.get_recent_history('self_reflection_reasoning', 1)

        # image_introduction = [
        #     {
        #         "path": images[event_i], "assistant": "",
        #         "introduction": 'This is the {} screenshot of recent events. The description of this image: {}'.format(
        #             ['first', 'second', 'third', 'fourth', 'fifth'][event_i], reasonings[event_i])
        #     } for event_i in range(config.event_count)
        # ]
        image_introduction = []
        image_introduction.append(
        {
            "introduction": "This screenshot is the current step if the application.",
            "path":image,
            "assistant": ""
        })

        input["image_introduction"] = image_introduction
        input["previous_summarization"] = self.memory.get_summarization()
        input["task_description"] = task_description
        # input["event_count"] = str(config.event_count)

        input["subtask_description"] = params["subtask_description"]
        input["subtask_reasoning"] = params["subtask_reasoning"]
        input["previous_reasoning"] = decision_making_reasoning
        input["self_reflection_reasoning"] = self_reflection_reasoning
        input["previous_action"] = self.memory.get_recent_history("action", k=1)[-1]
        input["executing_action_error"] = self.memory.get_recent_history("action_error", k=1)[-1]

        if success_detection:
            input[constants.SUCCESS_DETECTION] = success_detection

        # input["error_message"] = params["error_message"]

        # >> Calling INFORMATION SUMMARY
        logger.write(f'>> Calling INFORMATION SUMMARY')

        data = self.planner.information_summary(input=input)

        history_summary = extract_response_key(data['res_dict'], 'history_summary')

        # entities_and_behaviors = data['res_dict']['entities_and_behaviors']
        logger.write(f'R: Summary: {history_summary}')
        # logger.write(f'R: entities_and_behaviors: {entities_and_behaviors}')
        self.memory.add_summarization(history_summary)

        # self.memory.add_recent_history("image", cur_screen_shot_path)

        subtask_description = extract_response_key(data['res_dict'], 'subtask_description')
        subtask_reasoning = extract_response_key(data['res_dict'], 'subtask_reasoning')
        # task_decomposition = pre_process_keys(data['res_dict'], 'task_decomposition')

        logger.write(f'R: Subtask: {subtask_description}')
        logger.write(f'R: Subtask reasoning: {subtask_reasoning}')
        # logger.write(f'R: Task Decomposition: {task_decomposition}')

        res_params = {
            'summarization': history_summary,
            'subtask_description': subtask_description,
            'subtask_reasoning': subtask_reasoning
        }

        return res_params


    # def success_detection(self, params: Dict[str, Any]):

    #     task_description = params["task_description"]

    #     success = False
    #     success_reasoning = ""
    #     success_criteria = ""

    #     # Success detection preparation
    #     if self.use_success_detection:
    #         input = self.planner.success_detection_.input_map
    #         image_introduction = [
    #             {
    #                 "introduction": input["image_introduction"][-2]["introduction"],
    #                 "path": self.memory.get_recent_history("image", k=2)[0],
    #                 "assistant": input["image_introduction"][-2]["assistant"]
    #             },
    #             {
    #                 "introduction": input["image_introduction"][-1]["introduction"],
    #                 "path": self.memory.get_recent_history("image", k=1)[0],
    #                 "assistant": input["image_introduction"][-1]["assistant"]
    #             }
    #         ]
    #         input["image_introduction"] = image_introduction

    #         input["task_description"] = task_description
    #         input["previous_action"] = self.memory.get_recent_history("action", k=1)[-1]
    #         input["previous_reasoning"] = self.memory.get_recent_history("decision_making_reasoning", k=1)[-1]

    #         # >> Calling SUCCESS DETECTION
    #         logger.write(f'>> Calling SUCCESS DETECTION')
    #         data = self.planner.success_detection(input=input)

    #         success = data['res_dict']['success']
    #         success_reasoning = data['res_dict']['reasoning']
    #         success_criteria = data['res_dict']['criteria']

    #         self.memory.add_recent_history("success_detection_reasoning", success_reasoning)

    #         logger.write(f'Success: {success}')
    #         logger.write(f'Success criteria: {success_criteria}')
    #         logger.write(f'Success reason: {success_reasoning}')

    #     res_params = {
    #         "success": success,
    #         "success_reasoning": success_reasoning,
    #         "success_criteria": success_criteria
    #     }
    #     return res_params


# @TODO Should be in env (software) interface
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

                            if config.disable_close_app_icon and check_for_close_icon(skill, x, y):
                                processed_skill_steps[i] = f"{processed_skill_steps[i]} # {constants.CLOSE_ICON_DETECTED}"

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

                        if config.disable_close_app_icon and check_for_close_icon(skill, x, y):
                                processed_skill_steps[i] = f"{processed_skill_steps[i]} # {constants.CLOSE_ICON_DETECTED}"

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


def check_for_close_icon(skill: str, x, y) -> bool:
    """
    Check if trying to click the app close icon
    """

    # The left bottom corner of the close icon
    close_icon_coordinates_x = 1854/config.DEFAULT_ENV_RESOLUTION[0]
    close_icon_coordinates_y = 60/config.DEFAULT_ENV_RESOLUTION[1]

    return ('click_' in skill) and (x >= close_icon_coordinates_x and x <= config.DEFAULT_ENV_RESOLUTION[0]) and (y >= 0 and y <= close_icon_coordinates_y)


def pre_process_skill_library(skill_library_list: List[str]) -> List[str]:

    process_skill_library = skill_library_list.copy()
    process_skill_library = [skill for skill in process_skill_library if 'click_at_position(x, y, mouse_button)' not in skill['function_expression']]
    process_skill_library = [skill for skill in process_skill_library if 'double_click_at_position(x, y, mouse_button)' not in skill['function_expression']]
    process_skill_library = [skill for skill in process_skill_library if 'mouse_drag(source_x, source_y, target_x, target_y, mouse_button)' not in skill['function_expression']]
    process_skill_library = [skill for skill in process_skill_library if 'move_mouse_to_position(x, y)' not in skill['function_expression']]

    return process_skill_library


def extract_response_key(data: Dict, key: str) -> str:

        if key in data.keys():
            return data[key]
        else:
            logger.warn(f"No {key} in response.")
            return f"No {key}."


def exit_cleanup(runner: PipelineRunner):
    logger.write("Exiting pipeline.")
    runner.pipeline_shutdown()


def get_args_parser():

    parser = argparse.ArgumentParser("Cradle Prototype Runner")
    parser.add_argument("--providerConfig", type=str, default="./conf/openai_config.json", help="The path to the provider config file")
    parser.add_argument("--envConfig", type=str, default="./conf/env_config_outlook.json", help="The path to the environment config file")
    return parser


def main(args):

    config.load_env_config(args.envConfig)
    config.set_fixed_seed()

    config.ocr_fully_ban = True # not use local OCR-checks
    config.ocr_enabled = False
    config.skill_retrieval = True
    config.skill_from_local = True

    config.show_mouse_in_screenshot = True

    # Set the number of images to be used in self-reflection
    config.self_reflection_image_num = 2

    pipelineRunner = PipelineRunner(args.providerConfig,
                                    use_success_detection = False,
                                    use_self_reflection = True,
                                    use_information_summary = True)

    atexit.register(exit_cleanup, pipelineRunner)

    pipelineRunner.run()


if __name__ == '__main__':

    args = get_args_parser()
    args = args.parse_args()
    main(args)
