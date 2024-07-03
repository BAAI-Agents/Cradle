import math
import os
import time
from typing import List, Union

import cv2
from MTM import matchTemplates, drawBoxesOnRGB
import numpy as np
from PIL import Image

from cradle.config import Config
from cradle.gameio.lifecycle.ui_control import take_screenshot
from cradle.log import Logger
from cradle.utils.file_utils import assemble_project_path
from cradle.utils.json_utils import save_json

config = Config()
logger = Logger()


def render(overlay, template_image, output_file_name='', view=False):

    canvas_width = overlay.shape[1] + template_image.shape[1]
    canvas_height = max(overlay.shape[0], template_image.shape[0])
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    canvas[:overlay.shape[0], :overlay.shape[1]] = overlay
    canvas[:template_image.shape[0], overlay.shape[1]:overlay.shape[1] + template_image.shape[1]] = template_image

    if output_file_name:
        cv2.imwrite(output_file_name, canvas)

    if view:
        cv2.namedWindow('match result', 0)
        cv2.imshow('match result', canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def timing(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.debug(f"{func.__name__} consumes {elapsed_time:.4f}s")
        return result

    return wrapper


# @timing
def get_mtm_match(image: np.ndarray, template: np.ndarray, scales: list):
    detection = matchTemplates([('', cv2.resize(template, (0, 0), fx=s, fy=s)) for s in scales], image, N_object=1, method=cv2.TM_CCOEFF_NORMED, maxOverlap=0.1)
    detection['TemplateName'] = [str(round(i, 3)) for i in detection['Score']]  # confidence as name for display
    return detection


def match_template_image(src_file: str, template_file: str, debug = False, output_bb = False, save_matches = False, scale = "normal", rotate_angle : float = 0) -> List[dict]:
    '''
    Multi-scale template matching
    :param src_file: source image file
    :param template_file: template image file
    :param debug: output debug log messages
    :param output_bb: output bounding boxes in json
    :param save_matches: save the matched image
    :param scale: scale for template, default is 'normal', chosen from 'small', 'mid', 'normal', 'full', or you can also specify a list of float numbers
    :param rotate_angle: angle for source image rotation, at the center of image, clockwise

    :return:
    objects_list, a list of dicts, including template name, bounding box and confidence.
    '''

    output_dir = config.work_dir
    tid = time.time()

    scales = scale
    if scales == 'small':
        scales = [0.1, 0.2, 0.3, 0.4, 0.5]
    elif scales == 'mid':
        scales = [0.3, 0.4, 0.5, 0.6, 0.7]
    elif scales == 'normal':
        scales = [0.8, 0.9, 1.0, 1.1, 1.2]
    elif scales == 'full':
        scales = [0.5,0.75,1.0,1.5,2]
    elif not isinstance(scales, list):
        raise ValueError('scales must be a list of float numbers or one of "small", "mid", "normal", "full"')

    image = cv2.imread(assemble_project_path(src_file))
    template = cv2.imread(assemble_project_path(template_file))

    # Resize template according to resolution ratio
    if config.env_name == 'Stardew Valley': # @TODO move to env
        fx=1
        fy=1
    else:
        fx=config.resolution_ratio
        fy=config.resolution_ratio

    template = cv2.resize(template, (0, 0), fx=fx, fy=fy)

    if rotate_angle != 0:
        h, w, c = image.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), rotate_angle, 1)
        image = cv2.warpAffine(image, M, (w, h))  # np.rot90

    begin_detect_time = time.time()
    detection = get_mtm_match(image, template, scales)
    end_detect_time = time.time()

    template_name = os.path.splitext(os.path.basename(template_file))[0]
    source__name = os.path.splitext(os.path.basename(src_file))[0]

    output_prefix = f'match_{str(tid)}_{template_name}_in_{source__name}'

    if save_matches:
        overlay = drawBoxesOnRGB(image, detection,
                                 showLabel=True,
                                 boxThickness=4,
                                 boxColor=(0, 255, 0),
                                 labelColor=(0, 255, 0),
                                 labelScale=1)

        overlay_file_path = os.path.join(output_dir, f'{output_prefix}_overlay.jpg')
        render(overlay, template, overlay_file_path)

    # DataFrame to list of dicts
    objects_list = []
    for bounding_box, confidence in zip(detection['BBox'], detection['Score']):
        object_dict = {
            "type":template_name,
            "name": template_name,
            "bounding_box": bounding_box,
            "reasoning": "",
            "value": 0,
            "confidence": confidence,
        }
        objects_list.append(object_dict)

        if debug:
            logger.debug(f'{src_file}\t{template_file}\t{bounding_box}\t{confidence}\t{end_detect_time - begin_detect_time}',)

    if output_bb:
        bb_output_file = os.path.join(output_dir, f'{output_prefix}_bb.json')
        save_json(bb_output_file, objects_list, 4)

    return objects_list


def icons_match(icon_list: List[str], image_path = None) -> bool:

    confidence_threshold = 0.90

    matches = []

    if image_path is None:
        screenshot = take_screenshot(time.time(), include_minimap=False)[0]
    else:
        screenshot = image_path

    for icon in icon_list:

        # Multiple-scale-template-matching icon
        icon_template_file = f'./res/{config.env_sub_path}/icons/{icon}.png'

        match_info = match_template_image(screenshot, icon_template_file, save_matches=False, scale='full')
        match_info.sort(key=lambda bb: (bb['confidence']))

        if match_info[0]['confidence'] >= confidence_threshold:

            bb = match_info[0]['bounding_box']
            bb_dict = {
                "left": math.ceil(bb[0]),
                "top": math.ceil(bb[1]),
                "width": math.ceil(bb[2]),
                "height": math.ceil(bb[3]),
            }
            matches.append(bb_dict)

    if image_path is None:
        os.remove(screenshot)

    return matches


def match_templates_images(
    src_file_list: List[str],
    base_template_file_list: List[str],
    work_template_file_list: List[str],
    debug=False,
    output_bb=False,
    save_matches=False,
    scale="normal",
    rotate_angle: float = 0,
) -> List[dict]:
    """
    Multi-scale template matching
    :param src_file_list: source image files
    :param base_template_file_list: template image files
    :param debug: output debug log messages
    :param output_bb: output bounding boxes in json
    :param save_matches: save the matched image
    :param scale: scale for template, default is 'normal', chosen from 'small', 'mid', 'normal', 'full', or you can also specify a list of float numbers
    :param rotate_angle: angle for source image rotation, at the center of image, clockwise

    :return:
    corresponding template object for each source image
    """
    coresponding_dict = {}

    all_templates =[assemble_project_path(template) for template in base_template_file_list]+\
                   [assemble_project_path(template) for template in work_template_file_list]

    for original_file in src_file_list:
        original_score = []
        for template in all_templates:
            original_score.append(
                match_template_image(
                    src_file=original_file,
                    template_file=template,
                    debug=debug,
                    output_bb=output_bb,
                    save_matches=save_matches,
                    scale=scale,
                    rotate_angle=rotate_angle,
                )[0]["confidence"]
            )

        best_index = original_score.index(max(original_score))
        coresponding_dict[original_file] = base_template_file_list[best_index]
    return coresponding_dict


def selection_box_identifier(image_path, red_box_region):
    image = Image.open(image_path)

    red_min = 100
    green_max = 50
    blue_max = 50

    cropped_image = image.crop(red_box_region)

    rgb_image = cropped_image.convert('RGB')

    red_pixels = 0
    total_edge_pixels = 0

    def is_edge(x, y, width, height, range=5):
        return abs(x) < range or abs(x - width - 1) < range or abs(y) < range or abs(y - height - 1) < range

    for x in range(rgb_image.width):
        for y in range(rgb_image.height):
            if is_edge(x, y, rgb_image.width, rgb_image.height):
                total_edge_pixels += 1
                r, g, b = rgb_image.getpixel((x, y))
                # Check if the pixel is red
                if r > red_min and g < green_max and b < blue_max:
                    red_pixels += 1

    red_pixels_threshold = total_edge_pixels * 0.2

    return red_pixels >= red_pixels_threshold
