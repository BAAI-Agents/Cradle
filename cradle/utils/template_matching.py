import os
import time
from typing import List, Union

import cv2
from MTM import matchTemplates, drawBoxesOnRGB
import numpy as np

from cradle.config import Config
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
    template = cv2.resize(template, (0, 0), fx=config.resolution_ratio, fy=config.resolution_ratio)


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
