import time
from typing import Dict

from cradle.config import Config
from cradle.environment.skill import Skill

config = Config()


def serialize_skills(skills: Dict[str, Skill]) -> Dict[str, Dict]:
    serialized_skills = {name: skill.to_dict() for name, skill in skills.items()}
    return serialized_skills


def deserialize_skills(serialized_skills: Dict[str, Dict]) -> Dict[str, Skill]:
    return {name: Skill.from_dict(skill) for name, skill in serialized_skills.items()}
