from typing import (
    Any,
    List,
    Dict,
    Union,
    Tuple,
)
import os

from cradle.config import Config
from cradle.log import Logger
from cradle.memory.base import BaseMemory, Image
from cradle.utils.json_utils import load_json, save_json
from cradle.utils.singleton import Singleton
from cradle import constants

config = Config()
logger = Logger()

class LocalMemory(BaseMemory, metaclass=Singleton):

    storage_filename = "memory.json"

    def __init__(
        self,
        memory_path: str = config.work_dir,
        max_recent_steps: int = config.max_recent_steps
    ) -> None:

        self.max_recent_steps = max_recent_steps
        self.memory_path = memory_path
        self.current_info = {}
        self.task_duration = 3

        self.recent_history = {"screen_shot_path": [],
                               constants.AUGMENTED_IMAGES_MEM_BUCKET:[],
                               "action": [],
                               "decision_making_reasoning": [],
                               "success_detection_reasoning": [],
                               "self_reflection_reasoning": [],
                               "image_description":[],
                               "task_guidance":[],
                               "dialogue":[],
                               "task_description":[],
                               "skill_library":[],
                               "summarization":[],
                               "long_horizon_task":"",
                               "last_task_guidance":"",
                               "last_task_duration": self.task_duration}

        self.recent_history = {}

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

        return self.recent_history[key][-k:] if len(self.recent_history[key]) >= k else self.recent_history[key]


    def update_info_history(self, data: Dict[str, Any]):
        self.current_info.update(data)
        self.add_recent_history(data)

    def load(self, load_path = None) -> None:
        """Load the memory from the local file."""
        # @TODO load and store whole memory
        if load_path != None:
            if os.path.exists(os.path.join(load_path)):
                self.recent_history = load_json(load_path)
                logger.write(f"{load_path} has been loaded.")
            else:
                logger.error(f"{load_path} does not exist.")


    def save(self, loacl_path = None) -> None:
        """Save the memory to the local file."""
        # @TODO load and store whole memory
        if loacl_path:
            save_json(file_path = loacl_path, json_dict = self.recent_history, indent = 4)
        else:
            save_json(file_path = os.path.join(self.memory_path, self.storage_filename), json_dict = self.recent_history, indent = 4)