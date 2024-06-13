from typing import Dict, Any
from copy import deepcopy
import os
from PIL import Image

from cradle.provider import BaseProvider
from cradle.log import Logger
from cradle.constants import COLORS
from cradle.memory import LocalMemory
from cradle.utils.image_utils import draw_color_band

logger = Logger()
memory = LocalMemory()

class DrawColorBandProvider(BaseProvider):
    def __init__(self, config: Dict[str, Any] = None):
        super(DrawColorBandProvider, self).__init__()
        self.config = config

        self.left_band_width = self.config.get('left_band_width', 200)
        self.left_band_height = self.config.get('left_band_height', 1080)
        self.left_band_color = self.config.get('left_band_color', COLORS["blue"])
        self.right_band_width = self.config.get('right_band_width', 200)
        self.right_band_height = self.config.get('right_band_height', 1080)
        self.right_band_color = self.config.get('right_band_color', COLORS["yellow"])

    def run(self, *args, image, **kwargs):
        image = draw_color_band(image,
                                self.left_band_width,
                                self.left_band_height,
                                self.left_band_color,
                                self.right_band_width,
                                self.right_band_height,
                                self.right_band_color)
        return image

    @BaseProvider.write
    def __call__(self,
                 *args,
                 gm: Any = None,
                 video_record: Any = None,
                 **kwargs) -> Dict[str, Any]:

        logger.write(f"Draw color band on the screen shot.")

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

            image = draw_color_band(image,
                                    self.left_band_width,
                                    self.left_band_height,
                                    self.left_band_color,
                                    self.right_band_width,
                                    self.right_band_height,
                                    self.right_band_color)

            image.save(augmented_screen_shot_path)

        res_params = {
            "augmented_screen_shot_path": augmented_screen_shot_path
        }
        memory.update_info_history(res_params)

        del params

        return res_params