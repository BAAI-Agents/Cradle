from typing import (
    Any,
    List,
    Dict,
    Union,
    Tuple,
)
import os

from cradle.config import Config
from cradle import constants
from cradle.log import Logger
from cradle.memory.base import BaseMemory, Image
from cradle.utils.json_utils import load_json, save_json
from cradle.utils.singleton import Singleton

config = Config()
logger = Logger()


class LocalMemory(BaseMemory, metaclass=Singleton):

    storage_filename = "memory.json"

    def __init__(
        self,
        memory_path: str = config.work_dir,
        max_recent_steps: int = config.max_recent_steps,
    ) -> None:

        self.max_recent_steps = max_recent_steps
        self.memory_path = memory_path

        # Public working space for the agent to store information during loop
        self.working_area: Dict[str, Any] = {}

        self.task_duration = 3

        # @TODO First memory summary should be based on environment spec
        self.recent_history = {
            constants.IMAGES_MEM_BUCKET: [],
            constants.AUGMENTED_IMAGES_MEM_BUCKET: [],
            "action": [],
            "action_error": [],
            "decision_making_reasoning": [],
            "success_detection_reasoning": [],
            "self_reflection_reasoning": [],
            "image_description": [],
            "task_guidance": [],
            "dialogue": [],
            "task_description": [],
            constants.SKIIL_LIB_MEM_BUCKET: [],
            constants.SUMMARIZATION_MEM_BUCKET: ["The user is using the target application on the PC."],
            constants.LAST_TASK_GUIDANCE: [],
            "long_horizon_task": [],
            "": [self.task_duration],
            constants.KEY_REASON_OF_LAST_ACTION: [],
            constants.SUCCESS_DETECTION: [],
            }


    def add_recent_history_kv(
        self,
        key: str,
        info: Any,
    ) -> None:

        """Add recent info (skill/image/reasoning) to memory."""
        if key not in self.recent_history:
            self.recent_history[key] = []

        self.recent_history[key].append(info)

        if len(self.recent_history[key]) > self.max_recent_steps:
            self.recent_history[key].pop(0)


    def add_recent_history(
        self,
        information
    ) -> None:

        """Add recent info to memory."""
        for key, value in information.items():
            if key not in self.recent_history:
                self.recent_history[key] = []
            self.recent_history[key].append(value)

            if len(self.recent_history[key]) > self.max_recent_steps:
                self.recent_history[key].pop(0)


    def get_recent_history(
        self,
        key: str,
        k: int = 1,
    ) -> List[Any]:

        """Query recent info (skill/image/reasoning) from memory."""

        if key not in self.recent_history or len(self.recent_history[key]) == 0:
            return [""]

        if k is None:
            k = 1

        return self.recent_history[key][-k:] if len(self.recent_history[key]) >= k else self.recent_history[key]


    def update_info_history(self, data: Dict[str, Any]):
        self.working_area.update(data)
        self.add_recent_history(data)


    def add_summarization(self, summary: str) -> None:
        self.recent_history[constants.SUMMARIZATION_MEM_BUCKET] = [summary]


    def get_summarization(self) -> str:
        return self.recent_history[constants.SUMMARIZATION_MEM_BUCKET][-1]


    def add_task_guidance(self, task_description: str, long_horizon: bool) -> None:
        self.recent_history[constants.LAST_TASK_GUIDANCE] = task_description
        self.recent_history[constants.LAST_TASK_DURATION] = self.task_duration
        if long_horizon:
            self.recent_history['long_horizon_task'] = task_description


    def get_task_guidance(self, use_last = True) -> str:
        if use_last:
            return self.recent_history[constants.LAST_TASK_GUIDANCE]
        else:
            self.recent_history[constants.LAST_TASK_DURATION] -= 1
            if self.recent_history[constants.LAST_TASK_DURATION] >= 0:
                return self.recent_history[constants.LAST_TASK_GUIDANCE]
            else:
                return self.recent_history['long_horizon_task']


    def load(self, load_path=None) -> None:
        """Load the memory from the local file."""
        # @TODO load and store whole memory
        if load_path != None:
            if os.path.exists(os.path.join(load_path)):
                self.recent_history = load_json(load_path)
                logger.write(f"{load_path} has been loaded.")
            else:
                logger.error(f"{load_path} does not exist.")


    def save(self, local_path=None) -> None:
        """Save the memory to the local file."""
        # @TODO load and store whole memory
        if local_path:
            save_json(file_path=local_path, json_dict=self.recent_history, indent=4)
        else:
            save_json(file_path=os.path.join(self.memory_path, self.storage_filename), json_dict=self.recent_history,
                      indent=4)
