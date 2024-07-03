from copy import deepcopy

from cradle.provider import BaseProvider
from cradle import constants
from cradle.log import Logger
from cradle.memory import LocalMemory
from cradle.utils.object_utils import groundingdino_detect

logger = Logger()
memory = LocalMemory()


class GdProvider(BaseProvider):

    def __init__(self):

        super(GdProvider, self).__init__()

    @BaseProvider.debug
    @BaseProvider.error
    @BaseProvider.write
    def __call__(self,
                 *args,
                 **kwargs):

        params = deepcopy(memory.working_area)

        screenshot_path = memory.get_recent_history("screenshot_path")[-1]

        target_object_name = params[constants.TARGET_OBJECT_NAME].lower() \
            if constants.NONE_TARGET_OBJECT_OUTPUT not in params[
            constants.TARGET_OBJECT_NAME].lower() else ""

        memory.working_area.update({
            "target_object_name": target_object_name,
        })

        image_source, boxes, logits, phrases = groundingdino_detect(image_path=screenshot_path,
                                                         text_prompt=target_object_name,
                                                         box_threshold=0.4,
                                                         device='cuda')

        res_params = {
            "boxes": boxes,
            "logits": logits,
            "phrases": phrases,
        }

        memory.update_info_history(res_params)

        del params
        return res_params
