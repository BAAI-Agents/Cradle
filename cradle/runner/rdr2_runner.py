import os
import atexit
from typing import Any

from cradle.provider.video.video_frame_extractor import VideoFrameExtractorProvider
from cradle.utils.dict_utils import kget
from cradle.utils.string_utils import replace_unsupported_chars
from cradle import constants
from cradle.log import Logger
from cradle.config import Config
from cradle.memory import LocalMemory
from cradle.provider.llm.llm_factory import LLMFactory
from cradle.environment.skill_registry_factory import SkillRegistryFactory
from cradle.environment.ui_control_factory import UIControlFactory
from cradle.gameio.io_env import IOEnvironment
from cradle.gameio.game_manager import GameManager
from cradle.provider import VideoRecordProvider
from cradle.provider import VideoClipProvider
from cradle.provider import RDR2InformationGatheringPreprocessProvider
from cradle.provider import RDR2InformationGatheringPostprocessProvider
from cradle.provider import RDR2SelfReflectionPreprocessProvider
from cradle.provider import RDR2SelfReflectionPostprocessProvider
from cradle.provider import RDR2TaskInferencePreprocessProvider
from cradle.provider import RDR2TaskInferencePostprocessProvider
from cradle.provider import RDR2ActionPlanningPreprocessProvider
from cradle.provider import RDR2ActionPlanningPostprocessProvider
from cradle.provider import RDR2InformationGatheringProvider
from cradle.provider import RDR2SelfReflectionProvider
from cradle.provider import RDR2ActionPlanningProvider
from cradle.provider import RDR2TaskInferenceProvider
from cradle.provider import RDR2SkillCurationProvider
from cradle.provider import SkillExecuteProvider
from cradle.planner.rdr2_planner import RDR2Planner
from cradle.utils.object_utils import GroundingDINO
from cradle.utils.icon_utils import IconReplacer
from log_processor import process_log_messages


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

        # Init internal params
        self.set_internal_params()


    def set_internal_params(self, *args, **kwargs):
        self.provider_configs = config.provider_configs

        # Init LLM and embedding provider(s)
        lf = LLMFactory()
        self.llm_provider, self.embed_provider = lf.create(self.llm_provider_config_path,
                                                           self.embed_provider_config_path)

        srf = SkillRegistryFactory()
        srf.register_builder(config.env_short_name, config.skill_registry_name)
        self.skill_registry = srf.create(config.env_short_name, skill_configs=config.skill_configs,
                                         embedding_provider=self.embed_provider)

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

        self.memory = LocalMemory()
        self.video_recorder = VideoRecordProvider()

        self.frame_extractor = VideoFrameExtractorProvider()
        self.icon_replacer = IconReplacer()
        self.gd_detector = GroundingDINO()

        # Init planner
        self.planner = RDR2Planner(llm_provider=self.llm_provider,
                                   planner_params=config.planner_params,
                                   frame_extractor=self.frame_extractor,
                                   icon_replacer=self.icon_replacer,
                                   object_detector=self.gd_detector,
                                   use_self_reflection=True,
                                   use_task_inference=True)

        # Init skill library
        skills = self.gm.retrieve_skills(query_task=self.task_description,
                                    skill_num=config.skill_configs[constants.SKILL_CONFIG_MAX_COUNT],
                                    screen_type=constants.GENERAL_GAME_INTERFACE)

        self.skill_library = self.gm.get_skill_information(skills, config.skill_library_with_code)
        self.memory.update_info_history({"skill_library": self.skill_library})

        # Init video provider
        self.video_clip = VideoClipProvider(gm=self.gm)

        # Init module providers
        self.information_gathering_preprocess = RDR2InformationGatheringPreprocessProvider(gm = self.gm)
        self.information_gathering = RDR2InformationGatheringProvider(planner = self.planner,gm = self.gm)
        self.information_gathering_postprocess = RDR2InformationGatheringPostprocessProvider(gm = self.gm)

        self.self_reflection_preprocess = RDR2SelfReflectionPreprocessProvider(gm = self.gm)
        self.self_reflection = RDR2SelfReflectionProvider(planner = self.planner,gm = self.gm)
        self.self_reflection_postprocess = RDR2SelfReflectionPostprocessProvider(gm = self.gm)

        self.task_inference_preprocess = RDR2TaskInferencePreprocessProvider(gm = self.gm)
        self.task_inference = RDR2TaskInferenceProvider(planner = self.planner, gm = self.gm)
        self.task_inference_postprocess = RDR2TaskInferencePostprocessProvider(gm = self.gm)

        self.action_planning_preprocess = RDR2ActionPlanningPreprocessProvider(gm = self.gm)
        self.action_planning = RDR2ActionPlanningProvider(planner = self.planner, gm = self.gm)
        self.action_planning_postprocess = RDR2ActionPlanningPostprocessProvider(gm = self.gm)

        self.skill_curation = RDR2SkillCurationProvider(planner = self.planner, gm = self.gm, task_description=self.task_description)

        # Init skill execute provider
        self.skill_execute = SkillExecuteProvider(gm=self.gm)

        # Init checkpoint path
        self.checkpoint_path = os.path.join(config.work_dir, 'checkpoints')
        os.makedirs(self.checkpoint_path, exist_ok=True)


    def pipeline_shutdown(self):

        self.gm.cleanup_io()
        self.video_recorder.finish_capture()

        log = process_log_messages(config.work_dir)

        with open(config.work_dir + '/logs/log.md', 'w') as f:
            log = replace_unsupported_chars(log)
            f.write(log)

        logger.write('>>> Markdown generated.')
        logger.write('>>> Bye.')


    def run(self):

        # 1. Initiate the parameters
        success = False
        init_params = {
            "task_description": self.task_description,
            "skill_library": self.skill_library,
            "exec_info": {
                "errors": False,
                "errors_info": ""
            },
            "pre_action": "",
            "pre_screen_classification": "",
            "pre_decision_making_reasoning": "",
            "pre_self_reflection_reasoning": ""
        }

        self.memory.update_info_history(init_params)

        # 2. Switch to game
        self.gm.switch_to_game()

        # 3. Start video recording
        self.video_recorder.start_capture()

        # 4. Initiate screen shot path and video clip path
        self.video_clip(init = True)

        self.gm.pause_game()

        # 7. Start the pipeline
        step = 0
        while not success:
            try:
                # 7.1. Information gathering
                self.run_information_gathering()

                # 7.2. Self reflection
                self.run_self_reflection()

                # 7.3. Task inference
                self.run_task_inference()

                # 7.4. Skill curation
                self.run_skill_curation()

                # 7.5. Action planning
                self.run_action_planning()

                step += 1

                if step % config.checkpoint_interval == 0:
                    checkpoint_path = os.path.join(self.checkpoint_path, 'checkpoint_{:06d}.json'.format(step))
                    self.memory.save(checkpoint_path)

                if step > config.max_turn_count:
                    logger.write('Max steps reached, exiting.')
                    break

            except KeyboardInterrupt:
                logger.write('KeyboardInterrupt Ctrl+C detected, exiting.')
                self.pipeline_shutdown()
                break

        self.pipeline_shutdown()


    def run_information_gathering(self):

        # 1. Prepare the parameters to call llm api
        self.information_gathering_preprocess()

        # 2. Call llm api for information gathering
        response = self.information_gathering()

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

        # 4. Execute the actions
        self.skill_execute()

    def run_skill_curation(self):

        # 1. Call skill curation
        self.skill_curation()


def exit_cleanup(runner):
    logger.write("Exiting pipeline.")
    runner.pipeline_shutdown()


def entry(args):

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
