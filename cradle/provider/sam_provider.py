import glob
from typing import Dict, List, Tuple
import os
import gc

import numpy as np
from segment_anything import (
    SamAutomaticMaskGenerator,
    SamPredictor,
    sam_model_registry,
)
from PIL import Image, ImageEnhance

from cradle.utils import Singleton
from cradle.config import Config
from cradle.log import Logger
from cradle.utils.image_utils import (
    resize_image,
    overlay_image_on_background,
    process_image_for_masks,
    refine_masks,
    calculate_bounding_boxes,
    plot_som,
    calculate_centroid,
    remove_redundant_bboxes,
    filter_inner_bounding_boxes,
    filter_out_watermarks,
    convert_ocr_bbox_format,
    filter_intersecting_rectangles,
)
from cradle.utils.template_matching import icons_match
from cradle.provider.video.video_ocr_extractor import VideoOCRExtractorProvider
from cradle import constants

config = Config()
logger = Logger()


class SamProvider(metaclass=Singleton):

    def __init__(self):

        self.sam_model = None
        self.sam_predictor = None
        self.sam_mask_generator = None
        self.ocr_extractor = None

        try:
            self.sam_model = sam_model_registry[config.sam_model_name](checkpoint="./cache/sam_vit_h_4b8939.pth").to("cuda")
            self.sam_predictor = SamPredictor(self.sam_model)
            self.sam_mask_generator = SamAutomaticMaskGenerator(self.sam_model, pred_iou_thresh=config.sam_pred_iou_thresh)
        except Exception as e:
            logger.error(f"Failed to load the SAM model. Make sure you follow the instructions on README to download the necessary files.\n{e}")


    def calculate_som(self, screenshot_path: str) -> List:
        """
        Generate masks and bounding boxes for the given screenshot.

        Args:
            screenshot_path (str): Path and filename to the screenshot.

        Returns:
            List[dict]: List of bounding boxes for each mask.
        """
        org_img = Image.open(screenshot_path)
        image_area = org_img.size[0] * org_img.size[1]

        def recalculate_som_subarea(screenshot_path: str, bbox: dict, offset_top: int, offset_left: int) -> List:
            """Recursive SOM process for large bounding boxes.

            Args:
                screenshot_path (str): The filename of the screenshot.
                bbox (dict): The bounding box that exceeds the size threshold.
                offset_top (int): The top offset to adjust the bounding box coordinates.
                offset_left (int): The left offset to adjust the bounding box coordinates.

            Returns:
                List[dict]: List of bounding boxes for the refined masks.
            """
            org_img = Image.open(screenshot_path)
            top, left, height, width = bbox['top'], bbox['left'], bbox['height'], bbox['width']
            cropped_img = org_img.crop((left, top, left + width, top + height))
            screenshot_dir = os.path.dirname(screenshot_path)
            temp_crop_path = os.path.join(screenshot_dir, "temp_crop.jpg")
            cropped_img.save(temp_crop_path)

            refined_bboxes = self.calculate_som(temp_crop_path)

            for refined_bbox in refined_bboxes:
                refined_bbox['top'] += offset_top
                refined_bbox['left'] += offset_left

            del org_img, cropped_img
            gc.collect()

            return refined_bboxes

        try:
            image_resized = resize_image(org_img, resize_ratio=config.sam_resize_ratio)
        except ValueError as e:
            logger.warn(f"Failed to resize the image. Error: {str(e)}")
            del org_img
            gc.collect()
            return []

        enhancer = ImageEnhance.Contrast(image_resized)
        contrasted_image = enhancer.enhance(config.sam_contrast_level)
        array = np.array(contrasted_image)
        masks = self.sam_mask_generator.generate(array)

        del image_resized, enhancer, contrasted_image
        gc.collect()

        mask_img = overlay_image_on_background(masks, array.shape)
        if mask_img is None:
            return []

        masks = process_image_for_masks(mask_img)

        del mask_img
        gc.collect()

        # Process masks in batches to reduce memory usage
        def process_masks_in_batches(masks, batch_size=10):
            for i in range(0, len(masks), batch_size):
                yield masks[i:i + batch_size]

        refined_masks = []
        for batch in process_masks_in_batches(masks):
            batch_masks = [np.array(resize_image(Image.fromarray(mask.astype(np.uint8) * 255), resize_ratio=1/config.sam_resize_ratio)) for mask in batch]
            batch_refined_masks = refine_masks(batch_masks)
            refined_masks.extend(batch_refined_masks)

            del batch, batch_masks, batch_refined_masks
            gc.collect()

        bounding_boxes = calculate_bounding_boxes(refined_masks)

        del refined_masks
        gc.collect()

        refined_bounding_boxes = []
        large_bboxes = []

        # Initialize all_bboxes with bounding_boxes
        all_bboxes = bounding_boxes

        if bounding_boxes and bounding_boxes[0]:
            large_bboxes = [bbox for bbox in bounding_boxes if (bbox['height'] * bbox['width'] > config.sam_max_area * image_area and bbox['height'] * bbox['width'] >= config.min_resom_area)]

            for bbox in large_bboxes:
                refined_bounding_boxes.extend(recalculate_som_subarea(screenshot_path, bbox, bbox['top'], bbox['left']))

            all_bboxes = bounding_boxes + refined_bounding_boxes

            del bounding_boxes, refined_bounding_boxes, large_bboxes
            gc.collect()

        # Sort by top first, then left
        all_bboxes.sort(key=lambda bb: (bb['top'], bb['left']))

        return all_bboxes


    # @TODO Move to appropriate location
    def calc_and_plot_som_results(self, screenshot_path: str) -> Tuple[str, Dict[str, any]]:

        som_img_path = screenshot_path.replace(".jpg", f"_som.jpg")

        if "/res/" in screenshot_path:
            return som_img_path, dict()

        som_bbs = self.calculate_som(screenshot_path)

        if config.sam2som_mode == constants.SAM2SOM_OCR_MODE:

            if self.ocr_extractor is None:
                self.ocr_extractor = VideoOCRExtractorProvider()

            ocr_format_bbox = self.ocr_extractor.extract_text(screenshot_path)
            ocr_bbs = convert_ocr_bbox_format(ocr_format_bbox)
            som_bbs = filter_intersecting_rectangles(som_bbs, ocr_bbs)

        # Remove redundant bounding boxes.
        som_bbs = remove_redundant_bboxes(som_bbs)

        # Check for target icons for environment
        target_icons = self.check_for_target_icons()

        if len(target_icons) > 0:
            icon_bbs = icons_match(target_icons, screenshot_path)

            som_bbs += icon_bbs

        # Re-sort bounding boxes after adding icons
        som_bbs.sort(key=lambda bb: (bb['top'], bb['left']))

        som_bbs = filter_inner_bounding_boxes(som_bbs)

        if config.env_name == 'Feishu':
            base_image = Image.open(screenshot_path)
            som_bbs = filter_out_watermarks(base_image, som_bbs)

        # Calculate centroids for all bounding boxes
        centroids = [calculate_centroid(bbox) for bbox in som_bbs]

        som_img = plot_som(screenshot_path, som_bbs)
        centroids_map = {str(i + 1): centroid for i, centroid in enumerate(centroids)}

        som_img.save(som_img_path)
        logger.debug(f"Saved the SOM screenshot to {som_img_path}")

        return som_img_path, centroids_map


    # @TODO Move to appropriate location
    def check_for_target_icons(self):

        directory = os.path.join(config.root_dir, "res", config.env_sub_path, "icons")
        png_files = glob.glob(os.path.join(directory, "*.png"))
        file_names = [os.path.splitext(os.path.basename(file))[0] for file in png_files]

        return file_names
