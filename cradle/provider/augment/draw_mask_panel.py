from typing import Dict, Any
from copy import deepcopy
import os
from PIL import Image

from cradle.provider import BaseProvider
from cradle.log import Logger
from cradle.memory import LocalMemory
from cradle.utils.image_utils import draw_mask_panel

logger = Logger()
memory = LocalMemory()

class DrawMaskPanelProvider(BaseProvider):
    def __init__(self, config: Dict[str, Any] = None):
        super(DrawMaskPanelProvider, self).__init__()
        self.config = config

    def run(self, *args, image, **kwargs):
        image = draw_mask_panel(image)
        return image

    @BaseProvider.write
    def __call__(self,
                 *args,
                 gm: Any = None,
                 video_record: Any = None,
                 **kwargs) -> Dict[str, Any]:

        logger.write(f"Draw mask panel on the screen shot.")

        params = deepcopy(memory.current_info)

        screen_shot_path = params.get('screen_shot_path', None)
        augmented_screen_shot_path = screen_shot_path.replace(".jpg", "_augmented.jpg")

        if not os.path.exists(screen_shot_path):
            logger.error(f"screen_shot_path {screen_shot_path} not exists")
        else:

            if os.path.exists(augmented_screen_shot_path):
                image = Image.open(augmented_screen_shot_path)
            else:
                image = Image.open(screen_shot_path)

            image = draw_mask_panel(image)

            image.save(augmented_screen_shot_path)

        res_params = {
            "augmented_screen_shot_path": augmented_screen_shot_path
        }

        memory.update_info_history(res_params)
        del params

        return res_params