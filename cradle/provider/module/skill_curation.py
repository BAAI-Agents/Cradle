from copy import deepcopy
import os

from cradle.provider import BaseProvider
from cradle.log import Logger
from cradle.config import Config
from cradle.memory import LocalMemory
from cradle.provider.others.task_guidance import TaskGuidanceProvider
from cradle.provider.video.video_recorder import VideoRecordProvider
from cradle import constants

config = Config()
logger = Logger()


class SkillCurationProvider(BaseProvider):

    def __init__(self,
                 *args,
                 gm,
                 task_description="",
                 **kwargs):

        super(SkillCurationProvider, self).__init__()

        self.gm = gm
        self.video_recorder = VideoRecordProvider(os.path.join(config.work_dir, 'video.mp4'))
        self.memory = LocalMemory()
        self.task_guidance = TaskGuidanceProvider(task_description=task_description)


    @BaseProvider.debug
    @BaseProvider.error
    @BaseProvider.write
    def __call__(self,
                 *args,
                 **kwargs):

        params = deepcopy(self.memory.working_area)

        last_task_guidance = params.get('last_task_guidance', '')
        long_horizon = params.get('long_horizon', False)
        all_generated_actions = params.get('all_generated_actions', [])
        screen_classification = params.get('screen_classification', '')
        task_description = params.get('task_description', '')

        if last_task_guidance:
            task_description = last_task_guidance
            self.task_guidance.add_task_guidance(last_task_guidance, long_horizon)

        for extracted_skills in all_generated_actions:
            extracted_skills = extracted_skills['values']
            for extracted_skill in extracted_skills:
                self.gm.add_new_skill(skill_code=extracted_skill['code'])

        skill_names = self.gm.retrieve_skills(query_task=task_description,
                                              skill_num=config.skill_configs[constants.SKILL_CONFIG_MAX_COUNT],
                                              screen_type=screen_classification.lower())
        skill_library = self.gm.get_skill_information(skill_names)

        self.video_recorder.clear_frame_buffer()

        res_params = {
            'skill_library': skill_library
        }

        self.memory.update_info_history(res_params)

        del params

        return res_params


class RDR2SkillCurationProvider(BaseProvider):

    def __init__(self,
                 *args,
                 planner,
                 gm,
                 task_description="",
                 **kwargs):

        super(RDR2SkillCurationProvider, self).__init__()

        self.planner = planner
        self.gm = gm
        self.video_recorder = VideoRecordProvider(os.path.join(config.work_dir, 'video.mp4'))
        self.memory = LocalMemory(memory_path=config.work_dir, max_recent_steps=config.max_recent_steps)
        self.task_guidance = TaskGuidanceProvider(task_description=task_description)


    def __call__(self,
                 *args,
                 **kwargs):

        last_task_guidance = self.memory.get_recent_history("last_task_guidance", k=1)[0]
        long_horizon = self.memory.get_recent_history("long_horizon", k=1)[0]
        all_generated_actions = self.memory.get_recent_history("all_generated_actions", k=1)[0]
        screen_classification = self.memory.get_recent_history("screen_classification", k=1)[0]
        task_description = self.memory.get_recent_history("task_description", k=1)[0]

        if last_task_guidance:
            task_description = last_task_guidance
            self.task_guidance.add_task_guidance(last_task_guidance, long_horizon)

        logger.write(f'Current Task Guidance: {task_description}')

        for extracted_skills in all_generated_actions:
            extracted_skills = extracted_skills['values']
            for extracted_skill in extracted_skills:
                self.gm.add_new_skill(skill_code=extracted_skill['code'])

        skills = self.gm.retrieve_skills(query_task=task_description,
                                                skill_num=config.skill_configs[constants.SKILL_CONFIG_MAX_COUNT],
                                                screen_type=screen_classification.lower())
        skill_library = self.gm.get_skill_information(skills)

        self.memory.update_info_history({"skill_library": skill_library})

        self.video_recorder.clear_frame_buffer()
