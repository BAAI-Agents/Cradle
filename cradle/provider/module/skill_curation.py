from copy import deepcopy

from cradle.provider import BaseProvider
from cradle.log import Logger
from cradle.config import Config
from cradle.memory import LocalMemory
from cradle.provider import VideoRecordProvider
from cradle.provider.others.task_guidance import TaskGuidanceProvider

config = Config()
logger = Logger()
memory = LocalMemory()
video_record = VideoRecordProvider()
task_guidance = TaskGuidanceProvider()

class SkillCurationProvider(BaseProvider):
    def __init__(self,
                 *args,
                 gm,
                 **kwargs):
        super(SkillCurationProvider, self).__init__(*args, **kwargs)
        self.gm = gm

    @BaseProvider.debug
    @BaseProvider.error
    @BaseProvider.write
    def __call__(self,
                 *args,
                 **kwargs):

        params = deepcopy(memory.current_info)

        last_task_guidance = params.get('last_task_guidance', '')
        long_horizon = params.get('long_horizon', False)
        all_generated_actions = params.get('all_generated_actions', [])
        screen_classification = params.get('screen_classification', '')
        task_description = params.get('task_description', '')

        if last_task_guidance:
            task_description = last_task_guidance
            task_guidance.add_task_guidance(last_task_guidance, long_horizon)


        for extracted_skills in all_generated_actions:
            extracted_skills = extracted_skills['values']
            for extracted_skill in extracted_skills:
                self.gm.add_new_skill(skill_code=extracted_skill['code'])

        skill_names = self.gm.retrieve_skills(query_task=task_description,
                                           skill_num=config.skill_num,
                                           screen_type=screen_classification.lower())
        skill_library = self.gm.get_skill_information(skill_names)

        video_record.clear_frame_buffer()

        res_params = {
            'skill_library': skill_library
        }

        memory.update_info_history(res_params)

        del params

        return res_params

class RDR2SkillCurationProvider(BaseProvider):
    def __init__(self,
                 *args,
                 planner,
                 gm,
                 **kwargs):
        super(RDR2SkillCurationProvider, self).__init__(*args, **kwargs)
        self.planner = planner
        self.gm = gm

    def __call__(self,
                 *args,
                 **kwargs):
        last_task_guidance = memory.get_recent_history("last_task_guidance", k=1)[0]
        long_horizon = memory.get_recent_history("long_horizon", k=1)[0]
        all_generated_actions = memory.get_recent_history("all_generated_actions", k=1)[0]
        screen_classification = memory.get_recent_history("screen_classification", k=1)[0]
        task_description = memory.get_recent_history("task_description", k=1)[0]

        if last_task_guidance:
            task_description = last_task_guidance
            task_guidance.add_task_guidance(last_task_guidance, long_horizon)

        logger.write(f'Current Task Guidance: {task_description}')

        for extracted_skills in all_generated_actions:
            extracted_skills = extracted_skills['values']
            for extracted_skill in extracted_skills:
                self.gm.add_new_skill(skill_code=extracted_skill['code'])

        skills = self.gm.retrieve_skills(query_task=task_description,
                                                skill_num=config.skill_num,
                                                screen_type=screen_classification.lower())
        skill_library = self.gm.get_skill_information(skills)

        memory.update_info_history({"skill_library": skill_library})

        video_record.clear_frame_buffer()