import os
import time
import copy
import argparse

from groundingdino.util.inference import load_image

from cradle.config import Config
from cradle.gameio.game_manager import GameManager
from cradle.log import Logger
from cradle.agent import Agent
from cradle.planner.planner import Planner
from cradle.memory import LocalMemory
from cradle.provider.openai import OpenAIProvider
from cradle.provider import GdProvider
from cradle.gameio.io_env import IOEnvironment
from cradle.gameio.lifecycle.ui_control import switch_to_game, IconReplacer
from cradle.gameio.video.VideoRecorder import VideoRecorder
from cradle.gameio.video.VideoFrameExtractor import VideoFrameExtractor
from cradle.gameio.atomic_skills.trade_utils import __all__ as trade_skills
from cradle.gameio.atomic_skills.buy import __all__ as buy_skills
from cradle.gameio.atomic_skills.map import __all__ as map_skills
from cradle.gameio.atomic_skills.move import __all__ as move_skills
from cradle.gameio.atomic_skills.combat import __all__ as combat_skills
from cradle.gameio.composite_skills.auto_shoot import __all__ as auto_shoot_skills
from cradle.gameio.composite_skills.follow import __all__ as follow_skills
from cradle import constants

config = Config()
logger = Logger()
io_env = IOEnvironment()


def trigger_pipeline_loop(llm_provider_config_path, planner_params, task_description, skill_library, use_success_detection = False, use_self_reflection = False, use_information_summary = False):

    llm_provider = OpenAIProvider()
    llm_provider.init_provider(llm_provider_config_path)

    gd_detector = GdProvider()

    frame_extractor = VideoFrameExtractor()
    icon_replacer = IconReplacer()

    planner = Planner(llm_provider=llm_provider,
                      planner_params=planner_params,
                      frame_extractor=frame_extractor,
                      icon_replacer=icon_replacer,
                      object_detector=gd_detector,
                      use_self_reflection=use_self_reflection,
                      use_information_summary=use_information_summary)

    memory = LocalMemory(memory_path=config.work_dir,
                         max_recent_steps=config.max_recent_steps)
    memory.load(config.memory_load_path)

    gm = GameManager(env_name = config.env_name,
                     embedding_provider = llm_provider)

    img_prompt_decision_making = planner.decision_making_.input_map["image_introduction"]

    if config.skill_retrieval:
        gm.register_available_skills(skill_library)
        skill_library = gm.retrieve_skills(query_task = task_description, skill_num = config.skill_num, screen_type = constants.GENERAL_GAME_INTERFACE)
    skill_library = gm.get_skill_information(skill_library)

    switch_to_game()

    videocapture=VideoRecorder(os.path.join(config.work_dir, 'video.mp4'))
    videocapture.start_capture()
    start_frame_id = videocapture.get_current_frame_id()

    cur_screen_shot_path, _ = gm.capture_screen()
    memory.add_recent_history("image", cur_screen_shot_path)

    success = False
    pre_action = ""
    pre_screen_classification = ""
    pre_decision_making_reasoning = ""
    pre_self_reflection_reasoning = ""

    time.sleep(2)
    end_frame_id = videocapture.get_current_frame_id()
    gm.pause_game()

    while not success:

        try:
            # Gather information preparation
            logger.write(f'Gather Information Start Frame ID: {start_frame_id}, End Frame ID: {end_frame_id}')
            input = planner.gather_information_.input_map
            text_input = planner.gather_information_.text_input_map
            video_clip_path = videocapture.get_video(start_frame_id,end_frame_id)
            task_description = memory.get_task_guidance(use_last=False)

            get_text_image_introduction = [
                {
                    "introduction": input["image_introduction"][-1]["introduction"],
                    "path": memory.get_recent_history("image", k=1)[0],
                    "assistant": input["image_introduction"][-1]["assistant"]
                }
            ]

            # Configure the gather_information module
            gather_information_configurations = {
                "frame_extractor": True, # extract text from the video clip
                "icon_replacer": True,
                "llm_description": True, # get the description of the current screenshot
                "object_detector": True
            }
            input["gather_information_configurations"] = gather_information_configurations

            # Modify the general input for gather_information here
            image_introduction=[get_text_image_introduction[-1]]
            input["task_description"] = task_description
            input["video_clip_path"] = video_clip_path
            input["image_introduction"] = image_introduction

            # Modify the input for get_text module in gather_information here
            text_input["image_introduction"] = get_text_image_introduction
            input["text_input"] = text_input

            # >> Calling INFORMATION GATHERING
            logger.write(f'>> Calling INFORMATION GATHERING')
            data = planner.gather_information(input=input)

            # Any information from the gathered_information_JSON
            gathered_information_JSON=data['res_dict']['gathered_information_JSON']

            if gathered_information_JSON is not None:
                gathered_information=gathered_information_JSON.data_structure
            else:
                logger.warn("NO data_structure in gathered_information_JSON")
                gathered_information = dict()

            # Sort the gathered_information by timestamp
            gathered_information = dict(sorted(gathered_information.items(), key=lambda item: item[0]))
            all_dialogue = gathered_information_JSON.search_type_across_all_indices(constants.DIALOGUE)
            all_task_guidance = gathered_information_JSON.search_type_across_all_indices(constants.TASK_GUIDANCE)
            all_generated_actions = gathered_information_JSON.search_type_across_all_indices(constants.ACTION_GUIDANCE)
            classification_reasons = gathered_information_JSON.search_type_across_all_indices(constants.GATHER_TEXT_REASONING)

            response_keys = data['res_dict'].keys()

            if constants.LAST_TASK_GUIDANCE in response_keys:
                last_task_guidance = data['res_dict'][constants.LAST_TASK_GUIDANCE]
                if constants.LAST_TASK_HORIZON in response_keys:
                    long_horizon = bool(int(data['res_dict'][constants.LAST_TASK_HORIZON][0])) # Only first character is relevant
                else:
                    long_horizon = False
            else:
                logger.warn(f"No {constants.LAST_TASK_GUIDANCE} in response.")
                last_task_guidance = ""
                long_horizon = False

            if constants.IMAGE_DESCRIPTION in response_keys:
                image_description=data['res_dict'][constants.IMAGE_DESCRIPTION]
                if constants.SCREEN_CLASSIFICATION in response_keys:
                    screen_classification=data['res_dict'][constants.SCREEN_CLASSIFICATION]
                else:
                    screen_classification="None"
            else:
                logger.warn(f"No {constants.IMAGE_DESCRIPTION} in response.")
                image_description="No description"
                screen_classification="None"

            # Return to pause if screen type changed
            if screen_classification.lower() == constants.GENERAL_GAME_INTERFACE:
                gm.pause_game(screen_classification.lower())

            if constants.TARGET_OBJECT_NAME in response_keys:
                target_object_name=data['res_dict'][constants.TARGET_OBJECT_NAME]
                object_name_reasoning=data['res_dict'][constants.GATHER_INFO_REASONING]
            else:
                logger.write("> No target object")
                target_object_name = ""
                object_name_reasoning=""

            if "boxes" in response_keys:
                image_source, image = load_image(cur_screen_shot_path)
                boxes = data['res_dict']["boxes"]
                logits = data['res_dict']["logits"]
                phrases = data['res_dict']["phrases"]
                directory, filename = os.path.split(cur_screen_shot_path)
                bb_image_path = os.path.join(directory, "bb_"+filename)
                gd_detector.save_annotate_frame(image_source, boxes, logits, phrases, target_object_name.title(), bb_image_path)

                if boxes is not None and boxes.numel() != 0:
                    # Add the screenshot with bounding boxes into working memory
                    memory.add_recent_history(key=constants.AUGMENTED_IMAGES_MEM_BUCKET, info=bb_image_path)
                else:
                    memory.add_recent_history(key=constants.AUGMENTED_IMAGES_MEM_BUCKET, info=constants.NO_IMAGE)
            else:
                memory.add_recent_history(key=constants.AUGMENTED_IMAGES_MEM_BUCKET, info=constants.NO_IMAGE)

            logger.write(f'Image Description: {image_description}')
            logger.write(f'Object Name: {target_object_name}')
            logger.write(f'Reasoning: {object_name_reasoning}')
            logger.write(f'Screen Classification: {screen_classification}')

            logger.write(f'Dialogue: {all_dialogue}')
            logger.write(f'Gathered Information: {gathered_information}')
            logger.write(f'Classification Reasons: {classification_reasons}')
            logger.write(f'All Task Guidance: {all_task_guidance}')
            logger.write(f'Last Task Guidance: {last_task_guidance}')
            logger.write(f'Long Horizon: {long_horizon}')
            logger.write(f'Generated Actions: {all_generated_actions}')

            if use_self_reflection and start_frame_id > -1:
                input = planner.self_reflection_.input_map
                action_frames = []
                video_frames = videocapture.get_frames(start_frame_id,end_frame_id)

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
                    pre_action_name, pre_action_params = gm.skill_registry.convert_expression_to_skill(pre_action)

                    # only input the pre_action name
                    input["previous_action"] = pre_action_name
                    action_code, action_code_info = gm.get_skill_library_in_code(pre_action_name)
                    input['action_code'] = action_code if action_code is not None else action_code_info
                else:
                    input["previous_action"] = ""
                    input['action_code'] = ""

                if exec_info["errors"]:
                    input['executing_action_error']  = exec_info["errors_info"]
                else:
                    input['executing_action_error']  = ""

                # >> Calling SELF REFLECTION
                logger.write(f'>> Calling SELF REFLECTION')
                reflection_data = planner.self_reflection(input = input)

                if 'reasoning' in reflection_data['res_dict'].keys():
                    self_reflection_reasoning = reflection_data['res_dict']['reasoning']
                else:
                    self_reflection_reasoning = ""
                pre_self_reflection_reasoning = self_reflection_reasoning

                memory.add_recent_history("self_reflection_reasoning", self_reflection_reasoning)
                logger.write(f'Self-reflection reason: {self_reflection_reasoning}')

            if last_task_guidance:
                task_description = last_task_guidance
                memory.add_task_guidance(last_task_guidance, long_horizon)

            logger.write(f'Current Task Guidance: {task_description}')

            if config.skill_retrieval:
                for extracted_skills in all_generated_actions:
                    extracted_skills=extracted_skills['values']
                    for extracted_skill in extracted_skills:
                        gm.add_new_skill(skill_code=extracted_skill['code'])

                skill_library = gm.retrieve_skills(query_task = task_description, skill_num = config.skill_num, screen_type = screen_classification.lower())
                logger.write(f'skill_library: {skill_library}')
                skill_library = gm.get_skill_information(skill_library)

            videocapture.clear_frame_buffer()

            # Decision making preparation
            input = copy.deepcopy(planner.decision_making_.input_map)

            number_of_execute_skills = input["number_of_execute_skills"]

            if pre_action:
                input["previous_action"] = memory.get_recent_history("action", k=1)[-1]
                input["previous_reasoning"] = memory.get_recent_history("decision_making_reasoning", k=1)[-1]

            if pre_self_reflection_reasoning:
                input["previous_self_reflection_reasoning"] = memory.get_recent_history("self_reflection_reasoning", k=1)[-1]

            input['skill_library'] = skill_library
            input['info_summary'] = memory.get_summarization()

            if not "boxes" in response_keys:
                input['few_shots'] = []
            else:
                if boxes is None or boxes.numel() == 0:
                    input['few_shots'] = []

            # @TODO Temporary solution with fake augmented entries if no bounding box exists. Ideally it should read images, then check for possible augmentation.
            image_memory = memory.get_recent_history("image", k=config.decision_making_image_num)
            augmented_image_memory = memory.get_recent_history(constants.AUGMENTED_IMAGES_MEM_BUCKET, k=config.decision_making_image_num)

            image_introduction = []
            for i in range(len(image_memory), 0, -1):
                if augmented_image_memory[-i] != constants.NO_IMAGE:
                    image_introduction.append(
                        {
                            "introduction": img_prompt_decision_making[-i]["introduction"],
                            "path":augmented_image_memory[-i],
                            "assistant": img_prompt_decision_making[-i]["assistant"]
                        })
                else:
                    image_introduction.append(
                        {
                            "introduction": img_prompt_decision_making[-i]["introduction"],
                            "path":image_memory[-i],
                            "assistant": img_prompt_decision_making[-i]["assistant"]
                        })

            input["image_introduction"] = image_introduction
            input["task_description"] = task_description

            # Minimap info tracking
            if constants.MINIMAP_INFORMATION in response_keys:
                minimap_information = data["res_dict"][constants.MINIMAP_INFORMATION]
                logger.write(f"{constants.MINIMAP_INFORMATION}: {minimap_information}")

                minimap_info_str = ""
                for key, value in minimap_information.items():
                    if value:
                        for index, item in enumerate(value):
                            minimap_info_str = minimap_info_str + key + ' ' + str(index) + ': angle '  + str(int(item['theta'])) + ' degree' + '\n'
                minimap_info_str = minimap_info_str.rstrip('\n')

                logger.write(f'minimap_info_str: {minimap_info_str}')
                input[constants.MINIMAP_INFORMATION] = minimap_info_str

            data = planner.decision_making(input = input)

            skill_steps = data['res_dict']['actions']
            if skill_steps is None:
                skill_steps = []

            logger.write(f'R: {skill_steps}')

            # Filter nop actions in list
            skill_steps = [ i for i in skill_steps if i != '']
            if len(skill_steps) == 0:
                skill_steps = ['']

            skill_steps = skill_steps[:number_of_execute_skills]
            logger.write(f'Skill Steps: {skill_steps}')

            gm.unpause_game()

            # @TODO: Rename GENERAL_GAME_INTERFACE
            if pre_screen_classification.lower() == constants.GENERAL_GAME_INTERFACE and (screen_classification.lower() == constants.MAP_INTERFACE or screen_classification.lower() == constants.SATCHEL_INTERFACE) and pre_action:
                exec_info = gm.execute_actions([pre_action])

            start_frame_id = videocapture.get_current_frame_id()

            exec_info = gm.execute_actions(skill_steps)

            cur_screen_shot_path, _ = gm.capture_screen()

            end_frame_id = videocapture.get_current_frame_id()
            gm.pause_game(screen_classification.lower())

            # exec_info also has the list of successfully executed skills. skill_steps is the full list, which may differ if there were execution errors.
            pre_action = exec_info["last_skill"]

            pre_decision_making_reasoning = ''
            if 'res_dict' in data.keys() and 'reasoning' in data['res_dict'].keys():
                pre_decision_making_reasoning = data['res_dict']['reasoning']

            pre_screen_classification = screen_classification
            memory.add_recent_history("action", pre_action)
            memory.add_recent_history("decision_making_reasoning", pre_decision_making_reasoning)

            # For such cases with no expected response, we should define a retry limit
            logger.write(f'Decision reasoning: {pre_decision_making_reasoning}')

            # Information summary preparation
            if use_information_summary and len(memory.get_recent_history("decision_making_reasoning", memory.max_recent_steps)) == memory.max_recent_steps:
                input = planner.information_summary_.input_map
                logger.write(f'> Information summary call...')
                images = memory.get_recent_history('image', config.event_count)
                reasonings = memory.get_recent_history('decision_making_reasoning', config.event_count)
                image_introduction = [{"path": images[event_i],"assistant": "","introduction": 'This is the {} screenshot of recent events. The description of this image: {}'.format(['first','second','third','fourth','fifth'][event_i], reasonings[event_i])} for event_i in range(config.event_count)]

                input["image_introduction"] = image_introduction
                input["previous_summarization"] = memory.get_summarization()
                input["task_description"] = task_description
                input["event_count"] = str(config.event_count)

                # >> Calling INFORMATION SUMMARY
                logger.write(f'>> Calling INFORMATION SUMMARY')

                data = planner.information_summary(input = input)
                info_summary = data['res_dict']['info_summary']
                entities_and_behaviors = data['res_dict']['entities_and_behaviors']
                logger.write(f'R: Summary: {info_summary}')
                logger.write(f'R: entities_and_behaviors: {entities_and_behaviors}')
                memory.add_summarization(info_summary)

            memory.add_recent_history("image", cur_screen_shot_path)

            # Success detection preparation
            if use_success_detection:
                input = planner.success_detection_.input_map
                image_introduction = [
                    {
                        "introduction": input["image_introduction"][-2]["introduction"],
                        "path": memory.get_recent_history("image", k=2)[0],
                        "assistant": input["image_introduction"][-2]["assistant"]
                    },
                    {
                        "introduction": input["image_introduction"][-1]["introduction"],
                        "path": memory.get_recent_history("image", k=1)[0],
                        "assistant": input["image_introduction"][-1]["assistant"]
                    }
                ]
                input["image_introduction"] = image_introduction

                input["task_description"] = task_description
                input["previous_action"] = memory.get_recent_history("action", k=1)[-1]
                input["previous_reasoning"] = memory.get_recent_history("decision_making_reasoning", k=1)[-1]

                # >> Calling SUCCESS DETECTION
                logger.write(f'>> Calling SUCCESS DETECTION')
                data = planner.success_detection(input = input)

                success = data['res_dict']['success']
                success_reasoning = data['res_dict']['reasoning']
                success_criteria = data['res_dict']['criteria']

                memory.add_recent_history("success_detection_reasoning", success_reasoning)

                logger.write(f'Success: {success}')
                logger.write(f'Success criteria: {success_criteria}')
                logger.write(f'Success reason: {success_reasoning}')

            gm.store_skills()
            memory.save()

        except KeyboardInterrupt:
            logger.write('KeyboardInterrupt Ctrl+C detected, exiting.')
            gm.cleanup_io()
            videocapture.finish_capture()
            break

    gm.cleanup_io()
    videocapture.finish_capture()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--providerConfig",
        type=str,
        default="./conf/openai_config.json",
    )

    args = parser.parse_args()

    config.set_fixed_seed()

    # only change the input for different sub-modules
    # the tempaltes are now fixed

    planner_params = {
        "__check_list__": [
            "decision_making",
            "gather_information",
            "success_detection",
            "information_summary",
            "gather_text_information"
        ],
        "prompt_paths": {
            "inputs": {
                "decision_making": "./res/prompts/inputs/decision_making.json",
                "gather_information": "./res/prompts/inputs/gather_information.json",
                "success_detection": "./res/prompts/inputs/success_detection.json",
                "self_reflection": "./res/prompts/inputs/self_reflection.json",
                "information_summary": "./res/prompts/inputs/information_summary.json",
                "gather_text_information": "./res/prompts/inputs/gather_text_information.json"
            },
            "templates": {
                "decision_making": "./res/prompts/templates/decision_making.prompt",
                "gather_information": "./res/prompts/templates/gather_information.prompt",
                "success_detection": "./res/prompts/templates/success_detection.prompt",
                "self_reflection": "./res/prompts/templates/self_reflection.prompt",
                "information_summary": "./res/prompts/templates/information_summary.prompt",
                "gather_text_information": "./res/prompts/templates/gather_text_information.prompt"
            },
        }
    }

    skill_library = ['turn', 'move_forward', 'turn_and_move_forward', 'follow', 'aim', 'shoot', 'shoot_wolves', 'select_weapon', 'select_sidearm', 'fight', 'mount_horse']
    task_description =  ""

    config.ocr_fully_ban = True # not use local OCR-checks
    config.ocr_enabled = False
    config.skill_retrieval = True

    trigger_pipeline_loop(args.providerConfig, planner_params, task_description, skill_library, use_success_detection = False, use_self_reflection = True, use_information_summary = True)
