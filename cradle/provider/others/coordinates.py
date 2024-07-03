from typing import Dict, Any
from copy import deepcopy

from cradle.provider import BaseProvider
from cradle.utils.check import is_valid_value
from cradle.log import Logger
from cradle.config import Config
from cradle.memory import LocalMemory

config = Config()
logger = Logger()


class CoordinatesProvider(BaseProvider):

    def __init__(self, *args, gm, **kwargs):

        super(CoordinatesProvider, self).__init__()

        self.gm = gm
        self.memory = LocalMemory()


    def convert_coordinates_to_text(self, coordinates: Dict[str, Any]) -> str:

        line_type = coordinates["line_type"]
        point_type = coordinates["point_type"]

        line_text = []
        point_text = []

        for line in line_type:
            if "zone" in line:
                continue
            text = f"{line}: Start from {line_type[line][0]}. End at {line_type[line][1]}."
            line_text.append(text)

        for point in point_type:
            text = f"{point}: Locate at {point_type[point]}."
            point_text.append(text)

        text = ""
        if len(line_text) > 0:
            line_text = "\n".join(line_text)
            text += f"The name, start and end coordinates of the placed building for the line type are as follows:\n{line_text}\n"
        if len(point_text) > 0:
            point_text = "\n".join(point_text)
            text += f"The name and coordinate point location of the placed building of the point type are as follows:\n{point_text}\n"
        return text


    def add_coordinates(self,
                        raw_coordinates: Dict[str, Any],
                        add_name: str = "",
                        add_params: Dict[str, Any] = None) -> Dict[str, Any]:

        line_type = {}
        point_type = {}

        line_id = raw_coordinates["max_line_id"]
        point_id = raw_coordinates["max_point_id"]

        try:
            if is_valid_value(add_name) and is_valid_value(add_params):
                number_list = [int(value) for key, value in add_params.items()]
                if len(number_list) >= 4:
                    line_id += 1
                    line_type[add_name + "-" + "{:03d}".format(line_id)] = [(number_list[0], number_list[1]),
                                                                        (number_list[2], number_list[3])]
                elif len(number_list) >= 2:
                    point_id += 1
                    point_type[add_name + "-" + "{:03d}".format(point_id)] = (number_list[0], number_list[1])

                raw_coordinates["line_type"].update(line_type)
                raw_coordinates["point_type"].update(point_type)
                raw_coordinates["max_line_id"] = line_id
                raw_coordinates["max_point_id"] = point_id

        except Exception as e:
            logger.error(f"Failed to add coordinates: {e}")

        return raw_coordinates


    def _preprocess(self, params: Dict[str, Any], gm: Any = None, init = False, **kwargs):

        action = self.memory.get_recent_history('actions')[-1]
        success = self.memory.get_recent_history('success')[-1]

        if init:
            raw_coordinates = params.get('raw_coordinates', {})
        else:
            raw_coordinates = self.memory.get_recent_history('raw_coordinates')[-1]

        if init or not is_valid_value(action):
            coordinates = self.convert_coordinates_to_text(raw_coordinates)
            res_params = {
                "raw_coordinates": raw_coordinates,
                "coordinates": coordinates,
            }
        else:
            skill_name, skill_params = gm.convert_expression_to_skill(action)
            last_success_try_place_action = self.memory.get_recent_history('last_success_try_place_action')[-1]

            if skill_name is not None and "confirm_placement" in skill_name and success:
                actions = self.memory.get_recent_history('actions', config.max_recent_steps)

                if len(actions) >= 2 and "try_place_" in actions[-2]:
                    try_name, try_params = gm.skill_registry.convert_expression_to_skill(actions[-2])
                    try_name = try_name.replace("try_place_", "")
                    raw_coordinates = self.add_coordinates(raw_coordinates,
                                                           try_name,
                                                           try_params)

                    last_success_try_place_action = actions[-2]

            coordinates = self.convert_coordinates_to_text(raw_coordinates)

            res_params = {
                "raw_coordinates": raw_coordinates,
                "coordinates": coordinates,
                "last_success_try_place_action": last_success_try_place_action
            }

        return res_params


    def _postprocess(self, processed_response: Dict[str, Any], **kwargs):
        return processed_response


    def __call__(self,
                 *args,
                 init = False,
                 **kwargs):

        params = deepcopy(self.memory.working_area)
        params.update(self._preprocess(params, gm = self.gm, init = init,  **kwargs))
        res_params = self._postprocess(params, **kwargs)
        self.memory.update_info_history(res_params)

        del params

        return res_params
