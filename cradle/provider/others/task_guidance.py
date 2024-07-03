from cradle.provider import BaseProvider
from cradle.log import Logger
from cradle.config import Config
from cradle.memory import LocalMemory

config = Config()
logger = Logger()


class TaskGuidanceProvider(BaseProvider):

    def __init__(self, task_description = ""):

        super(TaskGuidanceProvider, self).__init__()

        self.task_duration = 3
        self.memory = LocalMemory()

        init_params = {
            "long_horizon_task": task_description,
            "last_task_guidance": task_description,
            "last_task_duration": self.task_duration
        }

        self.memory.update_info_history(init_params)


    def add_task_guidance(self, task_description: str, long_horizon: bool) -> None:

        res_params = {
            "last_task_guidance": task_description,
            "last_task_duration": self.task_duration,
        }

        if long_horizon:
            res_params['long_horizon_task'] = task_description

        self.memory.update_info_history(res_params)


    def get_task_guidance(self, use_last = True) -> str:

        if use_last:
            return self.memory.recent_history['last_task_guidance'][-1]
        else:

            last_task_duration = self.memory.recent_history['last_task_duration'][-1]
            last_task_duration -= 1

            res_params = {
                "last_task_duration": last_task_duration,
            }

            self.memory.update_info_history(res_params)

            if last_task_duration >= 0:
                return self.memory.recent_history['last_task_guidance'][-1]
            else:
                return self.memory.recent_history['long_horizon_task'][-1]
