import atexit
from typing import Any

from cradle.utils.string_utils import replace_unsupported_chars
from cradle import constants
from cradle.log import Logger
from cradle.config import Config
from cradle.memory import LocalMemory
from cradle.environment import RDR2SkillRegistry
from cradle.environment import RDR2UIControl
from cradle.gameio.io_env import IOEnvironment
from cradle.gameio.game_manager import GameManager
from cradle.log.logger import process_log_messages
from cradle.provider import RestfulClaudeProvider
from cradle.provider import OpenAIProvider
from cradle.provider import VideoRecordProvider
from cradle.provider import VideoClipProvider
from cradle.provider import RDR2InformationGatheringProvider
from cradle.provider import RDR2SelfReflectionProvider
from cradle.provider import RDR2ActionPlanningProvider
from cradle.provider import RDR2TaskInferenceProvider
from cradle.provider import RDR2SkillCurationProvider
from cradle.planner.rdr2_planner import RDR2Planner
from cradle.utils.object_utils import GroundingDINO
from cradle.utils.icon_utils import IconReplacer
from cradle.utils.video_utils import VideoFrameExtractor


config = Config()
logger = Logger()
memory = LocalMemory()
io_env = IOEnvironment()
video_record = VideoRecordProvider()

class PipelineRunner():

    def __init__(self,
                 llm_provider: Any,
                 embed_provider: Any,
                 task_description: str,
                 use_self_reflection: bool = False,
                 use_task_inference: bool = False):

        self.llm_provider = llm_provider
        self.embed_provider = embed_provider

        self.task_description = task_description
        self.use_self_reflection = use_self_reflection
        self.use_task_inference = use_task_inference

        # Init internal params
        self.set_internal_params()

    def set_internal_params(self, *args, **kwargs):
        self.provider_configs = config.provider_configs

        self.skill_registry = RDR2SkillRegistry(
            embedding_provider=self.embed_provider,
        )
        self.ui_control = RDR2UIControl()

        self.gm = GameManager(skill_registry=self.skill_registry,ui_control=self.ui_control)

        self.frame_extractor = VideoFrameExtractor()
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
                                    skill_num=config.skill_num,
                                    screen_type=constants.GENERAL_GAME_INTERFACE)

        self.skill_library = self.gm.get_skill_information(skills, config.skill_library_with_code)
        memory.update_info_history({"skill_library": self.skill_library})

        # Init video provider
        self.video_clip = VideoClipProvider(gm=self.gm)

        # Init module providers
        self.information_gathering = RDR2InformationGatheringProvider(
            planner = self.planner,
            gm = self.gm,
            **self.provider_configs.information_gathering_provider
        )
        self.self_reflection = RDR2SelfReflectionProvider(
            planner = self.planner,
            gm = self.gm,
            **self.provider_configs.self_reflection_provider
        )
        self.task_inference = RDR2TaskInferenceProvider(
            planner = self.planner,
            gm = self.gm,
            **self.provider_configs.task_inference_provider
        )
        self.action_planning = RDR2ActionPlanningProvider(
            planner = self.planner,
            gm = self.gm,
            **self.provider_configs.action_planning_provider
        )
        self.skill_curation = RDR2SkillCurationProvider(
            planner = self.planner,
            gm = self.gm,
            **self.provider_configs.skill_curation_provider
        )

    def pipeline_shutdown(self):
        self.gm.cleanup_io()
        video_record.finish_capture()
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

        memory.update_info_history(init_params)

        # 2. Switch to game
        self.gm.switch_to_game()

        # 3. Start video recording
        video_record.start_capture()

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
                # 7.3. Skill curation
                self.run_skill_curation()
                # 7.4. Task inference
                self.run_task_inference()
                # 7.5. Action planning
                self.run_action_planning()

                step += 1

                if step > config.max_steps:
                    logger.write('Max steps reached, exiting.')
                    break

            except KeyboardInterrupt:
                logger.write('KeyboardInterrupt Ctrl+C detected, exiting.')
                self.pipeline_shutdown()
                break

        self.pipeline_shutdown()


    def run_information_gathering(self):
        self.information_gathering()


    def run_self_reflection(self):
        self.self_reflection()

    def run_task_inference(self):
        self.task_inference()


    def run_action_planning(self):
        self.action_planning()

    def run_skill_curation(self):
        self.skill_curation()


def exit_cleanup(runner):
    logger.write("Exiting pipeline.")
    runner.pipeline_shutdown()


def entry(args):
    task_description = config.task_description

    # Init LLM provider and embedding provider
    if "claude" in args.llmProviderConfig:
        llm_provider = RestfulClaudeProvider()
        llm_provider.init_provider(args.llmProviderConfig)
        logger.write(f"Claude do not support embedding, use OpenAI instead.")
        embed_provider = OpenAIProvider()
        embed_provider.init_provider(args.embedProviderConfig)
    else: # OpenAI
        llm_provider = OpenAIProvider()
        llm_provider.init_provider(args.llmProviderConfig)
        embed_provider = llm_provider

    pipelineRunner = PipelineRunner(llm_provider,
                                    embed_provider,
                                    task_description=task_description,
                                    use_self_reflection = True,
                                    use_task_inference = True)
    
    atexit.register(exit_cleanup, pipelineRunner)
    
    pipelineRunner.run()
