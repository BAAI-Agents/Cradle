from copy import deepcopy
import time

from cradle.utils.image_utils import process_minimap_targets
from cradle.utils.image_utils import exec_clip_minimap
from cradle import constants
from cradle.provider import BaseProvider
from cradle.config import Config
from cradle.provider import GdProvider
from cradle.memory import LocalMemory

config = Config()
gd = GdProvider()
memory = LocalMemory()

class ClipMinimapProvider(BaseProvider):
    def __init__(self):
        super(ClipMinimapProvider, self).__init__()




    def __call__(self,
                 *args,
                 **kwargs):

        params = deepcopy(memory.current_info)

        screen_shot_path = memory.get_recent_history("screen_shot_path")[-1]

        tid = params.get('tid', time.time())
        screen_region = config.env_region
        minimap_region = config.base_minimap_region

        _, minimap_shot_path = exec_clip_minimap(tid, screen_region, minimap_region)

        res_params = {
            "minimap_shot_path": minimap_shot_path
        }

        minimap_detection_objects = process_minimap_targets(screen_shot_path)
        res_params.update({
            constants.MINIMAP_INFORMATION: minimap_detection_objects
        })

        minimap_info_str = ""
        for key, value in minimap_detection_objects.items():
            if value:
                for index, item in enumerate(value):
                    minimap_info_str = minimap_info_str + key + ' ' + str(index) + ': angle ' + str(
                        int(item['theta'])) + ' degree' + '\n'
        minimap_info_str = minimap_info_str.rstrip('\n')

        res_params.update(
            {
                constants.MINIMAP_INFORMATION: minimap_info_str
            }
        )

        memory.update_info_history(res_params)
        del params

        return res_params
