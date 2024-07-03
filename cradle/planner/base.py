import abc
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)
import json

from cradle.config import Config
from cradle.log import Logger

config = Config()
logger = Logger()


class BasePlanner():
    def __init__(self,
                 ):
        pass

    @abc.abstractmethod
    def information_gathering(self, *args, **kwargs) -> Dict[str, Any]:
        """
        gather information for the task
        :param args:
        :param kwargs:
        :return:
        """
        pass

    @abc.abstractmethod
    def action_planning(self, *args, **kwargs) -> Dict[str, Any]:
        """
        generate the next skill
        :param args:
        :param kwargs:
        :return:
        """
        pass

    @abc.abstractmethod
    def success_detection(self, *args, **kwargs) -> Dict[str, Any]:
        """
        detect whether the task is success
        :param args:
        :param kwargs:
        :return:
        """
        pass
