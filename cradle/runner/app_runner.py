import os
import time
from copy import deepcopy
from typing import Dict, Any, List
import atexit

from cradle import constants
from cradle.environment.skill_registry_factory import SkillRegistryFactory
from cradle.environment.software.skill_registry import SoftwareSkillRegistry
from cradle.environment.ui_control_factory import UIControlFactory
from cradle.log import Logger
from cradle.module.executor import Executor
from cradle.planner.planner import Planner
from cradle.config import Config
from cradle.memory import LocalMemory
from cradle.provider.llm.llm_factory import LLMFactory
from cradle.provider.sam_provider import SamProvider
from cradle.gameio.io_env import IOEnvironment
from cradle.gameio.game_manager import GameManager
from cradle.provider.video.video_recorder import VideoRecordProvider
from cradle.gameio.lifecycle.ui_control import switch_to_environment, draw_mouse_pointer_file
import cradle.environment.capcut
import cradle.environment.chrome
import cradle.environment.feishu
import cradle.environment.outlook
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
                 embed_provider_config_path: str,
                 task_description: str,
                 use_self_reflection: bool = False,
                 use_task_inference: bool = False):

        self.llm_provider_config_path = llm_provider_config_path
        self.embed_provider_config_path = embed_provider_config_path

        self.task_description = task_description
        self.use_self_reflection = use_self_reflection
        self.use_task_inference = use_task_inference

        # Success flag
        self.stop_flag = False

        # Count the number of loop iterations
        self.count_turns = 0

        # Count the consecutive turns where 'action' is empty
        self.count_empty_action = 0
        self.count_empty_action_threshold = 2

        # Init internal params
        self.set_internal_params()


    def set_internal_params(self, *args, **kwargs):

        # Init LLM and embedding provider(s)
        lf = LLMFactory()
        self.llm_provider, self.embed_provider = lf.create(self.llm_provider_config_path, self.embed_provider_config_path)

        # Init memory
        self.memory = LocalMemory(memory_path=config.work_dir,
                                  max_recent_steps=config.max_recent_steps)
        self.memory.load(config.memory_load_path)

        srf = SkillRegistryFactory()
        srf.register_builder(config.env_short_name, config.skill_registry_name)
        self.skill_registry = srf.create(config.env_short_name, skill_configs=config.skill_configs, embedding_provider=self.embed_provider)

        ucf = UIControlFactory()
        ucf.register_builder(config.env_short_name, config.ui_control_name)
        self.env_ui_control = ucf.create(config.env_short_name)

        # Init game manager
        self.gm = GameManager(env_name=config.env_name,
                              embedding_provider=self.embed_provider,
                              llm_provider=self.llm_provider,
                              skill_registry=self.skill_registry,
                              ui_control=self.env_ui_control,
                             )

        self.planner_params = config.planner_params

        # Init GD provider (not used for software applications)
        self.gd_detector = None

        # Init Sam provider
        self.sam_provider = SamProvider()

        # Init video frame extractor (not currently used for software)
        self.frame_extractor = None

        # Init icon replacer
        self.icon_replacer = None

        # Init planner
        self.planner = Planner(llm_provider=self.llm_provider,
                               planner_params=self.planner_params,
                               frame_extractor=self.frame_extractor,
                               icon_replacer=self.icon_replacer,
                               object_detector=self.gd_detector,
                               use_self_reflection=self.use_self_reflection,
                               use_task_inference=self.use_task_inference)

        # Init skill library
        skills = self.gm.retrieve_skills(query_task=self.task_description,
                                         skill_num=config.skill_configs[constants.SKILL_CONFIG_MAX_COUNT],
                                         screen_type=constants.GENERAL_GAME_INTERFACE)

        self.skill_library = self.gm.get_skill_information(skills, config.skill_library_with_code)
        self.source_skill_library = self.skill_library
        self.skill_library = SoftwareSkillRegistry.pre_process_skill_library(self.skill_library)

        self.memory.update_info_history({constants.SKILL_LIBRARY: self.skill_library})

        # Init video recorder
        self.video_recorder = VideoRecordProvider(os.path.join(config.work_dir, 'video.mp4'))

        # Init skill execute provider
        self.skill_execute = Executor(env_manager=self.gm)

        # Init checkpoint path
        self.checkpoint_path = os.path.join(config.work_dir, './checkpoints')
        os.makedirs(self.checkpoint_path, exist_ok=True)


    def run(self):

        self.memory.add_recent_history_kv(constants.TASK_DESCRIPTION, self.task_description)

        working_memory = {}

        # Switch to target environment
        switch_to_environment()

        # Prepare
        if config.enable_videocapture:
            self.video_recorder.start_capture()
        start_frame_id = self.video_recorder.get_current_frame_id()

        # First sense
        cur_screenshot_path = self.gm.capture_screen()
        mouse_x, mouse_y = io_env.get_mouse_position()

        time.sleep(2)
        end_frame_id = self.video_recorder.get_current_frame_id()

        # Initiate loop parameters
        working_memory.update({
            f"{constants.START_FRAME_ID}": start_frame_id,
            f"{constants.END_FRAME_ID}": end_frame_id,
            f"{constants.CUR_SCREENSHOT_PATH}": cur_screenshot_path,
            f"{constants.MOUSE_POSITION}" : (mouse_x, mouse_y),
            f"{constants.EXEC_INFO}": {
                f"{constants.ERRORS}": False,
                f"{constants.ERRORS_INFO}": constants.EMPTY_STRING
            },
            f"{constants.PRE_ACTION}": constants.EMPTY_STRING,
            f"{constants.PRE_DECISION_MAKING_REASONING}": constants.EMPTY_STRING,
            f"{constants.PRE_SELF_REFLECTION_REASONING}": constants.EMPTY_STRING,
            f"{constants.SKILL_LIBRARY}": self.skill_library,
            f"{constants.TASK_DESCRIPTION}": self.task_description,
            f"{constants.SUMMARIZATION}": constants.EMPTY_STRING,
            f"{constants.SUBTASK_DESCRIPTION}": constants.EMPTY_STRING,
            f"{constants.SUBTASK_REASONING}": constants.EMPTY_STRING,
            f"{constants.PREVIOUS_AUGMENTATION_INFO}": None,
        })

        self.memory.update_info_history(working_memory)

        while not self.stop_flag:

            try:

                logger.write(f'>>> Overall Task Description: {self.task_description}')
                logger.write(f'>>> Agent loop #{self.count_turns}')

                # Information gathering
                self.run_information_gathering(debug=False)

                # Self-reflection
                self.run_self_reflection()

                # Task inference
                self.run_task_inference()

                # Skill Curation
                self.run_skill_curation()

                # Action planning
                self.run_action_planning()

                # Skill execution
                self.execute_actions()

                # self.gm.store_skills()
                self.memory.save()

                self.count_turns += 1
                logger.write(f'-----------------------------Count turns: {self.count_turns} ----------------------------------')

                if self.count_turns % config.checkpoint_interval == 0:
                    checkpoint_path = os.path.join(self.checkpoint_path, 'checkpoint_{:06d}.json'.format(self.count_turns))
                    self.memory.save(checkpoint_path)

                if self.count_turns >= config.max_turn_count:
                    self.stop_flag = True
                    logger.error(f'Turn count reached: {config.max_turn_count}. Task has failed!')

            except KeyboardInterrupt:
                logger.write('KeyboardInterrupt Ctrl+C detected, exiting.')
                self.pipeline_shutdown()
                break

        self.pipeline_shutdown()


    def pipeline_shutdown(self):
        self.gm.cleanup_io()
        self.video_recorder.finish_capture()
        logger.write('>>> Bye.')


    def run_information_gathering(self, debug=False):

        # 1. Prepare the parameters to call llm api
        self.information_gathering_preprocess()

        # 2. Call llm api for information gathering
        response = self.information_gathering(debug)

        # 3. Postprocess the response
        self.information_gathering_postprocess(response)


    def run_self_reflection(self):

        # 1. Prepare the parameters to call llm api
        self.self_reflection_preprocess()

        # 2. Call llm api for self reflection
        response = self.self_reflection()

        # 3. Postprocess the response
        self.self_reflection_postprocess(response)


    def run_task_inference(self):

        # 1. Prepare the parameters to call llm api
        self.task_inference_preprocess()

        # 2. Call llm api for task inference
        response = self.task_inference()

        # 3. Postprocess the response
        self.task_inference_postprocess(response)


    def run_action_planning(self):

        # 1. Prepare the parameters to call llm api
        self.action_planning_preprocess()

        # 2. Call llm api for action planning
        response = self.action_planning()

        # 3. Postprocess the response
        self.action_planning_postprocess(response)


    def run_skill_curation(self):
        pass


    def information_gathering_preprocess(self):

        # > Pre-processing
        params = self.memory.working_area.copy()

        # Update module parameters for information gathering
        start_frame_id = params[constants.START_FRAME_ID]
        end_frame_id = params[constants.END_FRAME_ID]
        cur_screenshot_path: List[str] = params[constants.CUR_SCREENSHOT_PATH]
        self.memory.add_recent_history_kv(key=constants.IMAGES_MEM_BUCKET, info=cur_screenshot_path)

        # Information gathering preparation
        logger.write(f'Information Gathering - Start Frame ID: {start_frame_id}, End Frame ID: {end_frame_id}')
        input = self.planner.information_gathering_.input_map
        input[constants.TASK_DESCRIPTION] = self.task_description

        # Configure the test
        # if you want to test with a pre-defined screenshot, you can replace the cur_screenshot_path with the path to the screenshot
        pre_defined_sreenshot = None
        if pre_defined_sreenshot is not None:
            cur_screenshot_path = pre_defined_sreenshot
        else:
            cur_screenshot_path = params[constants.CUR_SCREENSHOT_PATH]

        input[constants.IMAGES_INPUT_TAG_NAME][0][constants.IMAGE_PATH_TAG_NAME] = cur_screenshot_path

        previous_augmentation = params[constants.PREVIOUS_AUGMENTATION_INFO]
        current_augmentation = {}
        current_augmentation[constants.AUG_BASE_IMAGE_PATH] = cur_screenshot_path

        # record the last collected mouse position
        mouse_position = kget(params, constants.MOUSE_POSITION)
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
            input[constants.IMAGES_INPUT_TAG_NAME][0][constants.IMAGE_PATH_TAG_NAME] = current_augmentation[constants.AUG_MOUSE_IMG_PATH]

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

            input[constants.IMAGES_INPUT_TAG_NAME][0][constants.IMAGE_PATH_TAG_NAME] = current_augmentation[constants.AUG_SOM_IMAGE_PATH]
            input[constants.LENGTH_OF_SOM_MAP] = str(current_augmentation[constants.LENGTH_OF_SOM_MAP])

            if mouse_position and config.show_mouse_in_screenshot:
                if mouse_position_same_flag and image_same_flag:
                    aug_som_mouse_img_path = cur_screenshot_path.replace(".jpg", f"_som_with_mouse.jpg")
                    current_augmentation[constants.AUG_SOM_MOUSE_IMG_PATH] = aug_som_mouse_img_path
                    copy_file(previous_augmentation[constants.AUG_SOM_MOUSE_IMG_PATH], current_augmentation[constants.AUG_SOM_MOUSE_IMG_PATH])
                else:
                    current_augmentation[constants.AUG_SOM_MOUSE_IMG_PATH] = draw_mouse_pointer_file(current_augmentation[constants.AUG_SOM_IMAGE_PATH], mouse_x, mouse_y)
                input[constants.IMAGES_INPUT_TAG_NAME][0][constants.IMAGE_PATH_TAG_NAME] = current_augmentation[constants.AUG_SOM_MOUSE_IMG_PATH]

        if previous_augmentation is None:
            previous_augmentation = current_augmentation.copy()

        self.memory.add_recent_history_kv(constants.AUGMENTED_IMAGES_MEM_BUCKET, current_augmentation)

        subtask_description = constants.EMPTY_STRING
        if constants.SUBTASK_DESCRIPTION in params.keys():
            subtask_description = params[constants.SUBTASK_DESCRIPTION]
        input[constants.SUBTASK_DESCRIPTION] = subtask_description
        self.memory.add_recent_history_kv(constants.SUBTASK_DESCRIPTION, subtask_description)

        # Configure the gather_information module
        gather_information_configurations = {
            f"{constants.FRAME_EXTRACTOR}": False,  # extract text from the video clip
            f"{constants.ICON_REPLACER}": False,
            f"{constants.LLM_DESCRIPTION}": True,  # get the description of the current screenshot
            f"{constants.OBJECT_DETECTOR}": False
        }
        input[constants.GATHER_INFORMATION_CONFIGURATIONS] = gather_information_configurations

        self.memory.working_area.update(input)


    def information_gathering(self, debug):
        logger.write(f'>> Calling INFORMATION GATHERING')

        if debug:
            # Do not call GPT-4V, just take the screenshot
            response = {
                f"{constants.PA_PROCESSED_RESPONSE_TAG}": {
                    f"{constants.IMAGE_DESCRIPTION}": "No description",
                }
            }
        else:
            response = self.planner.information_gathering(input=self.memory.working_area)

        return response


    def information_gathering_postprocess(self, response):

        processed_response = deepcopy(response)

        current_augmentation = self.memory.get_recent_history(constants.AUGMENTED_IMAGES_MEM_BUCKET, k=1)[-1]
        subtask_description = self.memory.get_recent_history(constants.SUBTASK_DESCRIPTION, k=1)[-1]

        if self.count_turns == 0:
            previous_augmentation = current_augmentation.copy()
        else:
            previous_augmentation = self.memory.get_recent_history(constants.AUGMENTED_IMAGES_MEM_BUCKET, k=2)[0]

        current_augmentation[constants.IMAGE_DESCRIPTION] = extract_response_key(response[constants.PA_PROCESSED_RESPONSE_TAG], constants.IMAGE_DESCRIPTION)
        target_object = extract_response_key(response[constants.PA_PROCESSED_RESPONSE_TAG], constants.TARGET_OBJECT_NAME)

        if constants.DESCRIPTION_OF_BOUNDING_BOXES in response[constants.PA_PROCESSED_RESPONSE_TAG].keys() and config.use_sam_flag:
            description_of_bounding_boxes = response[constants.PA_PROCESSED_RESPONSE_TAG][constants.DESCRIPTION_OF_BOUNDING_BOXES]
            current_augmentation[constants.DESCRIPTION_OF_BOUNDING_BOXES] = description_of_bounding_boxes.replace("\n", " ")
        else:
            logger.warn(f"No {constants.DESCRIPTION_OF_BOUNDING_BOXES} in response.")
            current_augmentation[constants.DESCRIPTION_OF_BOUNDING_BOXES] = "No descriptions"

        logger.write('Information Gathering response: ', response[constants.PA_PROCESSED_RESPONSE_TAG])
        logger.write(f'Image Description: {current_augmentation[constants.IMAGE_DESCRIPTION]}')
        logger.write(f'Target object: {target_object}')
        logger.write(f'Image Description of Bounding Boxes: {current_augmentation[constants.DESCRIPTION_OF_BOUNDING_BOXES]}')

        processed_response.update({
            f"{constants.SUB_TASK_DESCRIPTION}": subtask_description,
            f"{constants.TARGET_OBJECT_NAME}": target_object,
            f"{constants.CURRENT_AUGMENTATION_INFO}": current_augmentation,
            f"{constants.PREVIOUS_AUGMENTATION_INFO}": previous_augmentation
        })

        self.memory.update_info_history(processed_response)

        return processed_response


    def self_reflection_preprocess(self):
        # > Pre-processing
        params = self.memory.working_area.copy()

        start_frame_id = params[constants.START_FRAME_ID]
        end_frame_id = params[constants.END_FRAME_ID]

        task_description = params[constants.TASK_DESCRIPTION]
        pre_action = params[constants.PRE_ACTION]

        previous_augmentation = params[constants.PREVIOUS_AUGMENTATION_INFO]
        current_augmentation = params[constants.CURRENT_AUGMENTATION_INFO]

        pre_decision_making_reasoning = params[constants.PRE_DECISION_MAKING_REASONING]
        exec_info = params[constants.EXEC_INFO]

        image_same_flag = current_augmentation[constants.IMAGE_SAME_FLAG]
        mouse_position_same_flag = current_augmentation[constants.MOUSE_POSITION_SAME_FLAG]

        image_memory = self.memory.get_recent_history(constants.IMAGES_MEM_BUCKET, k=config.max_images_in_self_reflection)

        if self.use_self_reflection and self.count_turns > 0:
            input = self.planner.self_reflection_.input_map
            action_frames = []
            # video_frames = self.video_recorder.get_frames(start_frame_id, end_frame_id)

            if config.show_mouse_in_screenshot:
                action_frames.append(previous_augmentation[constants.AUG_MOUSE_IMG_PATH])
                action_frames.append(current_augmentation[constants.AUG_MOUSE_IMG_PATH])
            else:
                action_frames.append(previous_augmentation[constants.AUG_BASE_IMAGE_PATH])
                action_frames.append(current_augmentation[constants.AUG_BASE_IMAGE_PATH])

            image_introduction = [
                {
                    f"{constants.IMAGE_INTRO_TAG_NAME}": "Here are the sequential frames of the character executing the last action.",
                    f"{constants.IMAGE_PATH_TAG_NAME}": action_frames,
                    f"{constants.IMAGE_ASSISTANT_TAG_NAME}": constants.EMPTY_STRING,
                    f"{constants.IMAGE_RESOLUTION_TAG_NAME}": "low"
                }]

            input[constants.IMAGES_INPUT_TAG_NAME] = image_introduction
            input[constants.CURRENT_IMAGE_DESCRIPTION] = current_augmentation[constants.IMAGE_DESCRIPTION]

            input[constants.TASK_DESCRIPTION] = task_description
            input[constants.SKILL_LIBRARY] = self.skill_library
            input[constants.PREVIOUS_REASONING] = pre_decision_making_reasoning
            input[constants.HISTORY_SUMMARY] = self.memory.get_summarization()
            input[constants.SUBTASK_DESCRIPTION] = params[constants.SUBTASK_DESCRIPTION]

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

                    input[constants.PREVIOUS_ACTION] = ",".join(pre_action_name)
                    input[constants.PREVIOUS_ACTION_CALL] = pre_action
                    input[constants.ACTION_CODE] = "\n".join(list(set(pre_action_code)))
                    input[constants.KEY_REASON_OF_LAST_ACTION] = self.memory.get_recent_history(constants.KEY_REASON_OF_LAST_ACTION, k=1)[-1]
                    input[constants.SUCCESS_DETECTION] = self.memory.get_recent_history(constants.SUCCESS_DETECTION, k=1)[-1]
            else:
                input[constants.PREVIOUS_ACTION] = constants.EMPTY_STRING
                input[constants.PREVIOUS_ACTION_CALL] = constants.EMPTY_STRING
                input[constants.ACTION_CODE] = constants.EMPTY_STRING

            if exec_info[constants.ERRORS]:
                input[constants.EXECUTING_ACTION_ERROR] = exec_info[constants.ERRORS_INFO]
            else:
                input[constants.EXECUTING_ACTION_ERROR] = constants.EMPTY_STRING

            self.memory.working_area.update(input)


    def self_reflection(self):

        if self.use_self_reflection and self.count_turns > 0:
            # >> Calling SELF REFLECTION
            logger.write(f'>> Calling SELF REFLECTION')
            response = self.planner.self_reflection(input=self.memory.working_area)

        else:
            logger.write(f'No self-reflection in turn #{self.count_turns}')
            response = {}

        return response


    def self_reflection_postprocess(self, response):

        # > Post-processing
        processed_response = deepcopy(response)

        current_augmentation = self.memory.get_recent_history(constants.AUGMENTED_IMAGES_MEM_BUCKET, k=1)[-1]
        if self.count_turns == 0:
            previous_augmentation = current_augmentation.copy()
        else:
            previous_augmentation = self.memory.get_recent_history(constants.AUGMENTED_IMAGES_MEM_BUCKET, k=2)[0]

        if self.use_self_reflection and self.count_turns > 0:

            self_reflection_reasoning = extract_response_key(response[constants.PA_PROCESSED_RESPONSE_TAG], constants.SELF_REFLECTION_REASONING)
            success_detection = extract_response_key(response[constants.PA_PROCESSED_RESPONSE_TAG], constants.SUCCESS_DETECTION)

            self.memory.add_recent_history_kv(constants.SELF_REFLECTION_REASONING, self_reflection_reasoning)
            self.memory.add_recent_history_kv(constants.SUCCESS_DETECTION, success_detection)
            logger.write(f'Self-reflection reason: {self_reflection_reasoning}')
            logger.write(f'Success detection: {success_detection}')

            processed_response.update({
                f"{constants.PRE_SELF_REFLECTION_REASONING}": self_reflection_reasoning,
                f"{constants.SUCCESS_DETECTION}": success_detection,
                f"{constants.CURRENT_AUGMENTATION_INFO}": current_augmentation,
                f"{constants.PREVIOUS_AUGMENTATION_INFO}": previous_augmentation,
            })

        else:
            processed_response.update({
                f"{constants.PRE_SELF_REFLECTION_REASONING}": constants.EMPTY_STRING,
                f"{constants.SUCCESS_DETECTION}": False,
                f"{constants.CURRENT_AUGMENTATION_INFO}": current_augmentation,
                f"{constants.PREVIOUS_AUGMENTATION_INFO}": previous_augmentation,
            })

        self.memory.update_info_history(processed_response)

        return processed_response


    def task_inference_preprocess(self):

        params = self.memory.working_area.copy()

        task_description = params[constants.TASK_DESCRIPTION]
        success_detection = params[constants.SUCCESS_DETECTION]

        # Information summary preparation
        input = self.planner.task_inference_.input_map
        logger.write(f'> Memory reading for summary call...')

        mem_entries = self.memory.get_recent_history(constants.AUGMENTED_IMAGES_MEM_BUCKET, 1)
        image = mem_entries[0][constants.AUG_MOUSE_IMG_PATH]

        decision_making_reasoning = self.memory.get_recent_history(constants.KEY_REASON_OF_LAST_ACTION, k=1)[-1]
        self_reflection_reasoning = self.memory.get_recent_history(constants.SELF_REFLECTION_REASONING, 1)

        image_introduction = []
        image_introduction.append(
        {
            f"{constants.IMAGE_INTRO_TAG_NAME}": "This screenshot is the current step if the application.",
            f"{constants.IMAGE_PATH_TAG_NAME}":image,
            f"{constants.IMAGE_ASSISTANT_TAG_NAME}": constants.EMPTY_STRING
        })

        input[constants.IMAGES_INPUT_TAG_NAME] = image_introduction
        input[constants.PREVIOUS_SUMMARIZATION] = self.memory.get_summarization()
        input[constants.TASK_DESCRIPTION] = task_description
        # input["event_count"] = str(config.event_count)

        input[constants.SUBTASK_DESCRIPTION] = params[constants.SUBTASK_DESCRIPTION]
        input[constants.SUBTASK_REASONING] = params[constants.SUBTASK_REASONING]
        input[constants.PREVIOUS_REASONING] = decision_making_reasoning
        input[constants.SELF_REFLECTION_REASONING] = self_reflection_reasoning
        input[constants.PREVIOUS_ACTION] = self.memory.get_recent_history(constants.ACTION, k=1)[-1]
        input[constants.EXECUTING_ACTION_ERROR] = self.memory.get_recent_history(constants.ACTION_ERROR, k=1)[-1]

        if success_detection:
            input[constants.SUCCESS_DETECTION] = success_detection

        # input["error_message"] = params["error_message"]

        self.memory.working_area.update(input)


    def task_inference(self):
        # >> Calling TASK INFERENCE
        logger.write(f'>> Calling TASK INFERENCE')

        response = self.planner.task_inference(input=self.memory.working_area)

        return response


    def task_inference_postprocess(self, response):

        # > Post-processing
        processed_response = deepcopy(response)

        history_summary = extract_response_key(response[constants.PA_PROCESSED_RESPONSE_TAG], constants.HISTORY_SUMMARY)

        # entities_and_behaviors = data[constants.PA_PROCESSED_RESPONSE_TAG]['entities_and_behaviors']
        # logger.write(f'R: entities_and_behaviors: {entities_and_behaviors}')

        logger.write(f'R: Summary: {history_summary}')
        self.memory.add_summarization(history_summary)

        # self.memory.add_recent_history("image", cur_screenshot_path)

        subtask_description = extract_response_key(response[constants.PA_PROCESSED_RESPONSE_TAG], constants.SUBTASK_DESCRIPTION)
        subtask_reasoning = extract_response_key(response[constants.PA_PROCESSED_RESPONSE_TAG], constants.SUBTASK_REASONING)
        # task_decomposition = pre_process_keys(data[constants.PA_PROCESSED_RESPONSE_TAG], 'task_decomposition')

        logger.write(f'R: Subtask: {subtask_description}')
        logger.write(f'R: Subtask reasoning: {subtask_reasoning}')
        # logger.write(f'R: Task Decomposition: {task_decomposition}')

        processed_response.update({
            f"{constants.SUMMARIZATION}": history_summary,
            f"{constants.SUBTASK_DESCRIPTION}": subtask_description,
            f"{constants.SUBTASK_REASONING}": subtask_reasoning,
        })

        self.memory.update_info_history(processed_response)

        return processed_response


    def action_planning_preprocess(self):

        params = self.memory.working_area.copy()

        current_augmentation = params[constants.CURRENT_AUGMENTATION_INFO]
        previous_augmentation = params[constants.PREVIOUS_AUGMENTATION_INFO]
        pre_action = params[constants.PRE_ACTION]
        pre_self_reflection_reasoning = params[constants.PRE_SELF_REFLECTION_REASONING]
        success_detection = params[constants.SUCCESS_DETECTION]

        # Decision making preparation
        input = deepcopy(self.planner.action_planning_.input_map)
        img_prompt_decision_making = self.planner.action_planning_.input_map[constants.IMAGES_INPUT_TAG_NAME]

        self.memory.add_recent_history_kv(constants.PRE_ACTION, pre_action)
        self.memory.add_recent_history_kv(constants.NUMBER_OF_EXECUTE_SKILLS, input[constants.NUMBER_OF_EXECUTE_SKILLS])

        if pre_action:
            input[constants.PREVIOUS_ACTION] = self.memory.get_recent_history(constants.ACTION, k=1)[-1]
            input[constants.PREVIOUS_REASONING] = self.memory.get_recent_history(constants.DECISION_MAKING_REASONING, k=1)[-1]
            input[constants.KEY_REASON_OF_LAST_ACTION] = self.memory.get_recent_history(constants.KEY_REASON_OF_LAST_ACTION, k=1)[-1]
            input[constants.SUCCESS_DETECTION] = self.memory.get_recent_history(constants.SUCCESS_DETECTION, k=1)[-1]

        if pre_self_reflection_reasoning:
            input[constants.PREVIOUS_SELF_REFLECTION_REASONING] = self.memory.get_recent_history(constants.SELF_REFLECTION_REASONING, k=1)[-1]

        if success_detection:
            input[constants.SUCCESS_DETECTION] = success_detection

        input[constants.SKILL_LIBRARY] = self.skill_library
        input[constants.INFO_SUMMARY] = self.memory.get_summarization()

        image_memory = self.memory.get_recent_history(constants.IMAGES_MEM_BUCKET, k=config.action_planning_image_num)

        image_introduction = []
        for i in range(len(image_memory), 0, -1):
            image_introduction.append(
                {
                    f"{constants.IMAGE_INTRO_TAG_NAME}": img_prompt_decision_making[-i][constants.IMAGE_INTRO_TAG_NAME],
                    f"{constants.IMAGE_PATH_TAG_NAME}":image_memory[-i],
                    f"{constants.IMAGE_ASSISTANT_TAG_NAME}": img_prompt_decision_making[-i][constants.IMAGE_ASSISTANT_TAG_NAME]
                })

        input[constants.IMAGES_INPUT_TAG_NAME] = image_introduction
        input[constants.IMAGE_DESCRIPTION] = current_augmentation[constants.IMAGE_DESCRIPTION]
        input[constants.TASK_DESCRIPTION] = self.task_description
        input[constants.PREVIOUS_SUMMARIZATION] = self.memory.get_summarization()

        input[constants.TARGET_OBJECT_NAME] = extract_response_key(params, constants.TARGET_OBJECT_NAME)
        input[constants.SUBTASK_DESCRIPTION] = extract_response_key(params, constants.SUBTASK_DESCRIPTION)
        subtask_description = extract_response_key(params, constants.SUBTASK_DESCRIPTION)

        input[constants.IMAGES_INPUT_TAG_NAME][0][constants.IMAGE_PATH_TAG_NAME] = previous_augmentation[constants.AUG_BASE_IMAGE_PATH]
        input[constants.IMAGES_INPUT_TAG_NAME][-1][constants.IMAGE_PATH_TAG_NAME] = current_augmentation[constants.AUG_BASE_IMAGE_PATH]

        if config.show_mouse_in_screenshot:
            input[constants.IMAGES_INPUT_TAG_NAME][0][constants.IMAGE_PATH_TAG_NAME] = previous_augmentation[constants.AUG_MOUSE_IMG_PATH]
            input[constants.IMAGES_INPUT_TAG_NAME][-1][constants.IMAGE_PATH_TAG_NAME] = current_augmentation[constants.AUG_MOUSE_IMG_PATH]

        if config.use_sam_flag:
            input[constants.IMAGES_INPUT_TAG_NAME][0][constants.IMAGE_PATH_TAG_NAME] = previous_augmentation[constants.AUG_SOM_IMAGE_PATH]
            input[constants.IMAGES_INPUT_TAG_NAME][-1][constants.IMAGE_PATH_TAG_NAME] = current_augmentation[constants.AUG_SOM_IMAGE_PATH]
            input[constants.DESCRIPTION_OF_BOUNDING_BOXES] = current_augmentation[constants.DESCRIPTION_OF_BOUNDING_BOXES]

            if config.show_mouse_in_screenshot:
                input[constants.IMAGES_INPUT_TAG_NAME][0][constants.IMAGE_PATH_TAG_NAME] = previous_augmentation[constants.AUG_SOM_MOUSE_IMG_PATH]
                input[constants.IMAGES_INPUT_TAG_NAME][-1][constants.IMAGE_PATH_TAG_NAME] = current_augmentation[constants.AUG_SOM_MOUSE_IMG_PATH]

        input[constants.IMAGE_SAME_FLAG] = str(current_augmentation[constants.IMAGE_SAME_FLAG])
        input[constants.MOUSE_POSITION_SAME_FLAG] = str(current_augmentation[constants.MOUSE_POSITION_SAME_FLAG])

        self.memory.working_area.update(input)


    def action_planning(self):
        # >> Calling ACTION PLANNING
        logger.write(f'>> Calling ACTION PLANNING')
        response = self.planner.action_planning(input=self.memory.working_area)

        return response


    def action_planning_postprocess(self, response):

        # > Post-processing
        processed_response = deepcopy(response)
        current_augmentation = self.memory.get_recent_history(constants.AUGMENTED_IMAGES_MEM_BUCKET, k=1)[-1]

        pre_decision_making_reasoning = extract_response_key(response[constants.PA_PROCESSED_RESPONSE_TAG], constants.DECISION_MAKING_REASONING)
        key_reason_of_last_action = extract_response_key(response[constants.PA_PROCESSED_RESPONSE_TAG], constants.KEY_REASON_OF_LAST_ACTION)

        # For such cases with no expected response, we should define a retry limit
        logger.write(f'Decision reasoning: {pre_decision_making_reasoning}')
        logger.write(f'Decision key_reason: {key_reason_of_last_action}')

        # Retrieve selected skills
        try:
            skill_steps = response[constants.PA_PROCESSED_RESPONSE_TAG][constants.ACTIONS]
        except KeyError:
            logger.error(f'No actions found in action planning decision response.')
            skill_steps = None

        if skill_steps is None:
            skill_steps = []

        logger.write(f'Response steps: {skill_steps}')

        # Filter nop actions in list
        skill_steps = [ i for i in skill_steps if i != '']
        if len(skill_steps) == 0:
            skill_steps = ['']

        number_of_execute_skills = self.memory.get_recent_history(constants.NUMBER_OF_EXECUTE_SKILLS, k=1)[-1]
        skill_steps = skill_steps[:number_of_execute_skills]

        # Check software stop condition
        if config.is_game == False:
            if skill_steps == ['']:
                self.count_empty_action += 1
            else:
                self.count_empty_action = 0

            if self.count_empty_action >= self.count_empty_action_threshold:
                self.stop_flag = True
                logger.write(f'Empty action count reached {self.count_empty_action_threshold} times. Task is considered successful.')

        self.memory.add_recent_history_kv(constants.DECISION_MAKING_REASONING, pre_decision_making_reasoning)
        self.memory.add_recent_history_kv(constants.KEY_REASON_OF_LAST_ACTION, key_reason_of_last_action)

        processed_response.update({
            f"{constants.SKILL_STEPS}": skill_steps,
            f"{constants.SOM_MAP}": current_augmentation[constants.AUG_SOM_MAP],
            f"{constants.PRE_ACTION}": self.memory.get_recent_history(constants.PRE_ACTION, k=1)[-1],
            f"{constants.PRE_DECISION_MAKING_REASONING}": pre_decision_making_reasoning,
            f"{constants.SUBTASK_DESCRIPTION}": self.memory.get_recent_history(constants.SUBTASK_DESCRIPTION, k=1)[-1],
            f"{constants.PREVIOUS_AUGMENTATION_INFO}" : current_augmentation.copy(),
        })

        self.memory.update_info_history(processed_response)

        return processed_response


    def execute_actions(self):

        self.skill_execute()


def extract_response_key(data: Dict, key: str) -> str:

        if key in data.keys():
            return data[key]
        else:
            logger.warn(f"No {key} in response.")
            return f"No {key}."


def exit_cleanup(runner: PipelineRunner):
    logger.write("Exiting pipeline.")
    runner.pipeline_shutdown()


def entry(args):

    config.ocr_fully_ban = True # not use local OCR-checks
    config.ocr_enabled = False

    config.show_mouse_in_screenshot = True

    task_description = "No Task"

    task_id, subtask_id = 1, 0
    try:
        # Read end to end task description from config file
        task_description = kget(config.env_config, constants.TASK_DESCRIPTION_LIST, default='')[task_id-1][constants.TASK_DESCRIPTION]
        if subtask_id > 0:
            task_description = kget(config.env_config, constants.TASK_DESCRIPTION_LIST, default='')[task_id-1][constants.SUB_TASK_DESCRIPTION_LIST][subtask_id-1]
    except:
        logger.warn(f"Task description is not found for task_id: {task_id} and/or subtask_id: {subtask_id}")
        logger.warn(f"Using default input value: {task_description}")

    pipelineRunner = PipelineRunner(llm_provider_config_path=args.llmProviderConfig,
                                    embed_provider_config_path=args.embedProviderConfig,
                                    task_description=task_description,
                                    use_self_reflection = True,
                                    use_task_inference = True)

    atexit.register(exit_cleanup, pipelineRunner)

    pipelineRunner.run()
