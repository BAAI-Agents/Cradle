from typing import Dict, Any
from copy import deepcopy
import os
from PIL import Image

from cradle import constants
from cradle.provider import BaseProvider
from cradle.log import Logger
from cradle.memory import LocalMemory
from cradle.utils.image_utils import draw_axis, draw_mask_panel, draw_color_band, draw_grids

logger = Logger()
memory = LocalMemory()


class AugmentProvider(BaseProvider):

    def __init__(self,
                 draw_axis: bool = False,
                 draw_grid: bool = False,
                 draw_mask_panel: bool = False,
                 draw_color_band: bool = False,
                 axis_config: Dict[str, Any] = None,
                 grid_config: Dict[str, Any] = None,
                 mask_panel_config: Dict[str, Any] = None,
                 color_band_config: Dict[str, Any] = None,
                 ):

        super(AugmentProvider, self).__init__()

        self.draw_axis = draw_axis
        self.draw_grid = draw_grid
        self.draw_mask_panel = draw_mask_panel
        self.draw_color_band = draw_color_band
        self.axis_config = axis_config
        self.grid_config = grid_config
        self.mask_panel_config = mask_panel_config
        self.color_band_config = color_band_config


    def run(self, *args, image, **kwargs):

        if self.draw_mask_panel:
            image = draw_mask_panel(image, **self.mask_panel_config)

        if self.draw_axis:
            image = draw_axis(image, **self.axis_config)

        if self.draw_grid:
            image = draw_grids(image, **self.grid_config)

        if self.draw_color_band:
            image = draw_color_band(image, **self.color_band_config)

        return image


    @BaseProvider.write
    def __call__(self,
                 *args,
                 **kwargs) -> Dict[str, Any]:

        logger.write(f"Draw axis on the screen shot.")

        params = deepcopy(memory.working_area)

        screenshot_path = params.get(constants.IMAGES_MEM_BUCKET, None)
        augmented_screenshot_path = screenshot_path.replace(".jpg", "_augmented.jpg")

        if not os.path.exists(screenshot_path):
            logger.error(f"screenshot_path {screenshot_path} not exists")
        else:

            if os.path.exists(augmented_screenshot_path):
                image = Image.open(augmented_screenshot_path)
            else:
                image = Image.open(screenshot_path)

            image = self.run(image=image)

            image.save(augmented_screenshot_path)

        res_params = {
            constants.AUGMENTED_IMAGES_MEM_BUCKET: augmented_screenshot_path
        }

        memory.update_info_history(res_params)

        del params

        return res_params
