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

config = Config()
logger = Logger()


class LocalMemory(BaseMemory):

    storage_filename = "memory.json"

    def __init__(
        self,
        memory_path: str = '',
        max_recent_steps: int = 5
    ) -> None:

        self.max_recent_steps = max_recent_steps
        self.memory_path = memory_path
        self.task_duration = 3

        # @TODO First memory summary should be based on environment spec

        self.recent_history = {constants.IMAGES_MEM_BUCKET: [],
                               constants.AUGMENTED_IMAGES_MEM_BUCKET:[],
                               "action": [],
                               "action_error": [],
                               "decision_making_reasoning": [],
                               "success_detection_reasoning": [],
                               "self_reflection_reasoning": [],
                               "image_description":[],
                               "task_guidance":[],
                               "dialogue":[],
                               "task_description":[],
                               "skill_library":[],
                               "summarization":"The user is using the target application on the PC.",
                               "long_horizon_task":"",
                               constants.LAST_TASK_GUIDANCE:"",
                               "last_task_duration": self.task_duration,
                               constants.KEY_REASON_OF_LAST_ACTION:[],
                               constants.SUCCESS_DETECTION:[],}


    def add_recent_history(
        self,
        key: str,
        info: Any,
    ) -> None:

        """Add recent info (skill/image/reasoning) to memory."""
        self.recent_history[key].append(info)

        if len(self.recent_history[key]) > self.max_recent_steps:
            self.recent_history[key].pop(0)


    def get_recent_history(
        self,
        key: str,
        k: int = 1,
    ) -> List[Any]:

        """Query recent info (skill/image/reasoning) from memory."""

        if len(self.recent_history[key]) == 0:
            return [""]

        if k is None:
            k = 1

        return self.recent_history[key][-k:] if len(self.recent_history[key]) >= k else self.recent_history[key]


    def add_summarization(self, summary: str) -> None:
        self.recent_history["summarization"] = summary


    def get_summarization(self) -> str:
        return self.recent_history["summarization"]


    def add_task_guidance(self, task_description: str, long_horizon: bool) -> None:
        self.recent_history['last_task_guidance'] = task_description
        self.recent_history['last_task_duration'] = self.task_duration
        if long_horizon:
            self.recent_history['long_horizon_task'] = task_description


    def get_task_guidance(self, use_last = True) -> str:
        if use_last:
            return self.recent_history['last_task_guidance']
        else:
            self.recent_history['last_task_duration'] -= 1
            if self.recent_history['last_task_duration']>=0:
                return self.recent_history['last_task_guidance']
            else:
                return self.recent_history['long_horizon_task']


    def load(self, load_path = None) -> None:
        """Load the memory from the local file."""
        # @TODO load and store whole memory
        if load_path != None:
            if os.path.exists(os.path.join(load_path, self.storage_filename)):
                self.recent_history = load_json(os.path.join(load_path, self.storage_filename))
                logger.write(f"{os.path.join(load_path, self.storage_filename)} has been loaded.")
            else:
                logger.error(f"{os.path.join(load_path, self.storage_filename)} does not exist.")


    def save(self) -> None:
        """Save the memory to the local file."""
        # @TODO load and store whole memory
        save_json(file_path = os.path.join(self.memory_path, self.storage_filename), json_dict = self.recent_history, indent = 4)
