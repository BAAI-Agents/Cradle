import base64
import time
import json
from typing import Type, AnyStr, Any

import numpy as np
import dill
from dataclasses import dataclass
from dataclass_wizard import JSONWizard
from dataclass_wizard.abstractions import W
from dataclass_wizard.type_def import JSONObject, Encoder

from cradle.config import Config

config = Config()


@dataclass
class Skill(JSONWizard):

    skill_name: str
    skill_function: Any
    skill_embedding: np.ndarray
    skill_code: str
    skill_code_base64: str


    def __call__(self, *args, **kwargs):
        return self.skill_function(*args, **kwargs)


    @classmethod
    def from_dict(cls: Type[W], o: JSONObject) -> W:

        skill_function = dill.loads(bytes.fromhex(o['skill_function'])) # Load skill function from hex string
        skill_embedding = np.frombuffer(base64.b64decode(o['skill_embedding']), dtype=np.float64)

        return cls(
            skill_name=o['skill_name'],
            skill_function=skill_function,
            skill_embedding=skill_embedding,
            skill_code=o['skill_code'],
            skill_code_base64=o['skill_code_base64']
        )


    def to_dict(self) -> JSONObject:
        skill_function_hex = dill.dumps(self.skill_function).hex() # Convert skill function to hex string
        skill_embedding_base64 = base64.b64encode(self.skill_embedding).decode('utf-8')

        return {
            'skill_name': self.skill_name,
            'skill_function': skill_function_hex,
            'skill_embedding': skill_embedding_base64,
            'skill_code': self.skill_code,
            'skill_code_base64': self.skill_code_base64
        }


    def to_json(self: W, *,
                encoder: Encoder = json.dumps,
                **encoder_kwargs) -> AnyStr:
        return json.dumps(self.to_dict(), **encoder_kwargs)


    @classmethod
    def from_json(cls: Type[W], s: AnyStr, *,
                  decoder: Any = json.loads,
                  **decoder_kwargs) -> W:
        return cls.from_dict(json.loads(s, **decoder_kwargs))


def post_skill_wait(wait_time = config.DEFAULT_POST_ACTION_WAIT_TIME):
    """Wait for skill to finish. Like if there is an animation"""
    time.sleep(wait_time)
