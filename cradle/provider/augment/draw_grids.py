from typing import Dict, Any
from copy import deepcopy
import os
from PIL import ImageDraw, ImageFont, Image

from cradle.provider import BaseProvider
from cradle.log import Logger
from cradle.constants import COLORS
from cradle.memory import LocalMemory
from cradle.utils.image_utils import draw_grids

logger = Logger()
memory = LocalMemory()

class DrawGridsProvider(BaseProvider):
    def __init__(self, config: Dict[str, Any] = None):
        super(DrawGridsProvider, self).__init__()
        self.config = config
        self.crop_region = self.config.get('crop_region', [0, 0, 1920, 960]) # left, top, right, bottom
        self.axis_division = self.config.get('axis_division', [3, 5])
        self.axis_color = COLORS[self.config.get('axis_color', "red")]
        self.axis_linewidth = self.config.get('axis_linewidth', 3)
        self.font_size = self.config.get('font_size', 50)
        self.font_color = COLORS[self.config.get('font_color', "white")]

    def run(self, *args, image, **kwargs):
        image = draw_grids(image,
                          self.crop_region,
                          self.axis_division,
                          self.axis_color,
                          self.axis_linewidth,
                          self.font_color,
                          self.font_size)
        return image

    @BaseProvider.write
    def __call__(self,
                 *args,
                 gm: Any = None,
                 video_record: Any = None,
                 **kwargs) -> Dict[str, Any]:

        logger.write(f"Draw grids on the screen shot.")

        params = deepcopy(memory.current_info)

        screen_shot_path = params.get('screen_shot_path', None)
        augmented_screen_shot_path =  screen_shot_path.replace(".jpg", "_augmented.jpg")

        if not os.path.exists(screen_shot_path):
            logger.error(f"screen_shot_path {screen_shot_path} not exists")
        else:

            if os.path.exists(augmented_screen_shot_path):
                image = Image.open(augmented_screen_shot_path)
            else:
                image = Image.open(screen_shot_path)

            image = draw_grids(image,
                             self.crop_region,
                             self.axis_division,
                             self.axis_color,
                             self.axis_linewidth,
                             self.font_color,
                             self.font_size)

            image.save(augmented_screen_shot_path)

        res_params = {
            "augmented_screen_shot_path": augmented_screen_shot_path,
        }

        memory.update_info_history(res_params)
        del params

        return res_params