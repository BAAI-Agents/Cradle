import os, math
import time
from typing import Any

from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import torch
from torchvision.ops import box_convert
import supervision as sv
from MTM import matchTemplates

from cradle.config import Config
from cradle.log import Logger
from cradle.gameio import IOEnvironment
from cradle.gameio.lifecycle.ui_control import take_screenshot
from cradle.utils.template_matching import match_template_image

config = Config()
logger = Logger()
io_env = IOEnvironment()

PAUSE_SCREEN_WAIT = 1


def pause_game():

    if not is_env_paused():
        io_env.handle_hold_in_pause()

        io_env.key_press('esc')
        time.sleep(PAUSE_SCREEN_WAIT)
    else:
        logger.debug("The environment does not need to be paused!")

    # While game is paused, quickly re-center mouse location on x axis to avoid clipping at game window border with time
    io_env.mouse_move(config.env_resolution[0] // 2, config.env_resolution[1] // 2, relative=False)


def unpause_game():
    if is_env_paused():
        io_env.key_press('esc', 0)
        time.sleep(PAUSE_SCREEN_WAIT)

        io_env.handle_hold_in_unpause()
    else:
        logger.debug("The environment is not paused!")


def exit_back_to_pause():

    max_steps = 10

    back_steps = 0
    while not is_env_paused() and back_steps < max_steps:
        back_steps += 1
        io_env.key_press('esc')
        time.sleep(PAUSE_SCREEN_WAIT)

    if back_steps >= max_steps:
        logger.warn("The environment fails to pause!")


def exit_back_to_game():

    exit_back_to_pause()

    # Unpause the game, to keep the rest of the agent flow consistent
    unpause_game()


def switch_to_game():
    named_windows = io_env.get_windows_by_name(config.env_name)
    if len(named_windows) == 0:
        logger.error(f"Cannot find the game window {config.env_name}!")
        return
    else:
        try:
            named_windows[0].activate()
        except Exception as e:
            if "Error code from Windows: 0" in str(e):
                # Handle pygetwindow exception
                pass
            else:
                raise e

    time.sleep(1)
    unpause_game()
    time.sleep(1)


def segment_minimap(screenshot_path):

    tid = time.time()
    output_dir = config.work_dir
    minimap_image_filename = output_dir + "/minimap_" + str(tid) + ".jpg"

    minimap_region = config.base_minimap_region
    minimap_region = [int(x * (config.env_resolution[0] / config.base_resolution[0]) ) for x in minimap_region] # (56, 725, 56 + 320, 725 + 320)
    minimap_region[2] += minimap_region[0]
    minimap_region[3] += minimap_region[1]

    # Open the source image file
    with Image.open(screenshot_path) as source_image:

        # Crop the image using the crop_rectangle
        cropped_minimap = source_image.crop(minimap_region)

        # Save the cropped image to a new file
        cropped_minimap.save(minimap_image_filename)

    clip_minimap(minimap_image_filename)

    return minimap_image_filename


def is_env_paused():

    is_paused = False
    confidence_threshold = 0.85

    # Multiple-scale-template-matching example, decide whether the game is paused according to the confidence score
    pause_clock_template_file = f'./res/{config.env_sub_path}/icons/clock.jpg'

    screenshot = take_screenshot(time.time(), include_minimap=False)[0]
    match_info = match_template_image(screenshot, pause_clock_template_file, debug=True, output_bb=True, save_matches=True, scale='full')

    is_paused = match_info[0]['confidence'] >= confidence_threshold

    # Renaming pause candidate screenshot to ease debugging or gameplay scenarios
    os.rename(screenshot, screenshot.replace('screen', 'pause_screen_candidate'))

    return is_paused


def clip_minimap(minimap_image_filename):

    image = cv2.imread(minimap_image_filename)

    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Create a mask of the same size as the image, initialized to white
    mask = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Define the size of the triangular mask at each corner
    triangle_size = int(180 * config.resolution_ratio)

    # Draw black triangles on the four corners of the mask
    # Top-left corner
    cv2.fillConvexPoly(mask, np.array([[0, 0], [triangle_size, 0], [0, triangle_size]]), 0)

    # Top-right corner
    cv2.fillConvexPoly(mask, np.array([[width, 0], [width - triangle_size, 0], [width, triangle_size]]), 0)

    # Bottom-left corner
    cv2.fillConvexPoly(mask, np.array([[0, height], [0, height - triangle_size], [triangle_size, height]]), 0)

    # Bottom-right corner
    cv2.fillConvexPoly(mask, np.array([[width, height], [width, height - triangle_size], [width - triangle_size, height]]), 0)

    # Apply the mask to the image
    masked_image = cv2.bitwise_and(image, mask)

    # Save the result
    cv2.imwrite(minimap_image_filename, masked_image)


def annotate_with_coordinates(image_source, boxes, logits, phrases):
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    logger.debug(f"boxes: {boxes}, xyxy: {xyxy}")

    detections = sv.Detections(xyxy=xyxy)

    # Without coordinates normalization
    labels = [
        f"{phrase} {' '.join(map(str, ['x=', round((xyxy_s[0]+xyxy_s[2])/(2*w), 2), ', y=', round((xyxy_s[1]+xyxy_s[3])/(2*h), 2)]))}"
        for phrase, xyxy_s
        in zip(phrases, xyxy)
    ]

    box_annotator = sv.BoxAnnotator()
    annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    return annotated_frame


class CircleDetector:
    def __init__(self,resolution_ratio):
        if resolution_ratio <= .67:  # need super resolution
            self.sr_model = cv2.dnn_superres.DnnSuperResImpl_create()
            self.k = 2 if resolution_ratio <=.5 else 3
            self.sr_model.readModel(f'./res/models/ESPCN_x{self.k}.pb')
            self.sr_model.setModel('espcn', self.k)
        else:
            self.sr_model = None


    def get_theta(self, origin_x, origin_y, center_x, center_y):
        '''
        The origin of the image coordinate system is usually located in the upper left corner of the image, with the x-axis to the right indicating a positive direction and the y-axis to the down indicating a positive direction. Using vertical upward as the reference line, i.e. the angle between it and the negative direction of the y-axis
        '''
        theta = math.atan2(center_x - origin_x, origin_y - center_y)
        theta = math.degrees(theta)
        return theta


    def detect(self, img_file,
        yellow_range=np.array([[140, 230, 230], [170, 255, 255]]),
        gray_range=np.array([[165, 165, 165], [175, 175, 175]]),
        red_range=np.array([[0, 0, 170], [30, 30, 240]]),
        detect_mode='yellow & gray',
        debug=False
    ):

        image = cv2.imread(img_file)

        # super resolution according to resolution ratio
        if self.sr_model is not None:
            image = self.sr_model.upsample(image)
            if self.k == 3:
                image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

        origin = (image.shape[0] // 2, image.shape[1] // 2)
        circles = cv2.HoughCircles(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.HOUGH_GRADIENT, 1, 10, param1=200,param2=10, minRadius=5 * 2, maxRadius=8 * 2)
        theta = 0x3f3f3f3f
        measure = {'theta': theta, 'distance': theta, 'color': np.array([0, 0, 0]), 'confidence': 0, 'vis': image,
                   'center': origin}

        circles_info = []
        if circles is not None:

            circles = np.round(circles[0, :]).astype("int")

            for (x, y, r) in circles:

                # Crop the circle from the original image
                circle_img = np.zeros_like(image)
                cv2.circle(circle_img, (x, y), r, (255, 255, 255), thickness=-1)
                circle = cv2.bitwise_and(image, circle_img)

                # Define range for red color and create a mask
                red_mask = cv2.inRange(circle, red_range[0], red_range[1])
                gray_mask = cv2.inRange(circle, gray_range[0], gray_range[1])
                yellow_mask = cv2.inRange(circle, yellow_range[0], yellow_range[1])

                # Count red pixels in the circle
                red_count = cv2.countNonZero(red_mask)
                gray_count = cv2.countNonZero(gray_mask)
                yellow_count = cv2.countNonZero(yellow_mask)

                # Add circle information and color counts to the list
                circles_info.append({
                    "center": (x, y),
                    "radius": r,
                    "red_count": red_count,
                    "gray_count": gray_count,
                    "yellow_count": yellow_count
                })

            # Sort the circles based on yellow_count, gray_count, and red_count
            if 'red' in detect_mode:
                circles_info.sort(key=lambda c: (c['red_count'], c['yellow_count'], c['gray_count']), reverse=True)
                detect_criterion = lambda circle: circle["red_count"] >= 5
            else:
                circles_info.sort(key=lambda c: (c['yellow_count'], c['gray_count'], c['red_count']), reverse=True)
                detect_criterion = lambda circle: circle["gray_count"] >= 5 or circle["yellow_count"] >= 5

            for circle in circles_info:

                center_x, center_y, radius = circle["center"][0], circle["center"][1], circle["radius"]

                if detect_criterion(circle):
                    theta = self.get_theta(*origin, center_x, center_y)
                    dis = np.sqrt((center_x - origin[0]) ** 2 + (center_y - origin[1]) ** 2)
                    measure = {'theta': theta, 'distance': dis,
                               'color': "yellow" if circle["yellow_count"] >= 5 else "gray", 'confidence': 1,
                               'center': (center_x, center_y),
                               'bounding_box': (center_x - radius, center_y - radius, 2 * radius, 2 * radius)}
                    break

            if debug:
                for i, circle in enumerate(circles_info):
                    cv2.circle(image, circle["center"], circle["radius"], (0, 255, 0), 2)
                    cv2.putText(image, str(i + 1), (circle["center"][0] - 5, circle["center"][1] + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                measure['vis'] = image

        return theta, measure


class IconReplacer:

    def __init__(self, template_path = f'./res/{config.env_sub_path}/icons/keys'):

        if '/-/' in template_path:
            template_path = f'./res/{config.env_sub_path}/icons/keys'

        self.template_paths = [os.path.join(template_path, filename) for filename in os.listdir(template_path)]


    def __call__(self, image_paths):
        return self.replace_icon(image_paths)


    def _drawBoxesOnRGB(self, image, tableHit, boxThickness=2, boxColor=(255, 255, 00), showLabel=False, labelColor=(255, 255, 0), labelScale=0.5):
        """
        Return a copy of the image with predicted template locations as bounding boxes overlaid on the image
        The name of the template can also be displayed on top of the bounding box with showLabel=True

        Parameters
        ----------
        - image  : image in which the search was performed

        - tableHit: list of hit as returned by matchTemplates or findMatches

        - boxThickness: int
                        thickness of bounding box contour in pixels
        - boxColor: (int, int, int)
                    RGB color for the bounding box

        - showLabel: Boolean
                    Display label of the bounding box (field TemplateName)

        - labelColor: (int, int, int)
                    RGB color for the label

        Returns
        -------
        outImage: RGB image
                original image with predicted template locations depicted as bounding boxes
        """
        # Convert Grayscale to RGB to be able to see the color bboxes
        if image.ndim == 2:
            outImage = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) # convert to RGB to be able to show detections as color box on grayscale image
        else:
            outImage = image.copy()

        for _, row in tableHit.iterrows():

            x,y,w,h = row['BBox']
            text = row['TemplateName']

            if showLabel:
                text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, labelScale, 1)
                text_width, text_height = text_size

                rectangle_pos = [(int(x - 0.2 * w), int(y - 0.15 * h)), (int(x + 1.2 * w), int(y + 1.05 * h))]
                cv2.rectangle(outImage, rectangle_pos[0], rectangle_pos[1], color=boxColor, thickness=-1)

                text_x = int((rectangle_pos[0][0] + rectangle_pos[1][0]) / 2 - text_width / 2)
                text_y = int((rectangle_pos[0][1] + rectangle_pos[1][1]) / 2 + text_height / 2)
                cv2.putText(outImage, text=text, org=(text_x, text_y), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=labelScale, color=labelColor, lineType=cv2.LINE_AA,thickness=1)

        return outImage


    def _get_mtm_match(self, image: np.ndarray, template: np.ndarray, template_name):
        detection = matchTemplates([(template_name, cv2.resize(template, (round(template.shape[1] * s), round(template.shape[0] * s)))) for s in [0.9, 1, 1.1]],
                                image,
                                N_object=1,
                                method=cv2.TM_CCOEFF_NORMED,
                                maxOverlap=0.1)

        if detection['Score'].iloc[0] > 0.75:
            image = self._drawBoxesOnRGB(image, detection, boxThickness=-1, showLabel=True, boxColor=(255, 255, 255), labelColor=(0, 0, 0), labelScale=.62)

        return {'info': detection, 'vis': image}


    def _show(self, image, window_name='screen',show=True,save=''):
        if save:
            cv2.imwrite(save, image)
        if show:
            cv2.namedWindow(window_name, 0)
            cv2.imshow(window_name, image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


    # Image augmentation to mitigate VLM issues
    def replace_icon(self, image_paths):

        replaced_image_paths = []

        for image_path in image_paths:
            image = cv2.imread(image_path)

            for template_file in self.template_paths:
                template = cv2.imread(template_file)
                template_name = os.path.splitext(os.path.basename(template_file))[0]

                if 'left_mouse' in template_name:
                    template_name = 'LM'
                elif 'right_mouse' in template_name:
                    template_name = 'RM'
                elif 'mouse' in template_name:
                    template_name = 'MS'
                elif 'enter' in template_name:
                    template_name = 'Ent'

                detection = self._get_mtm_match(image, template, template_name)
                image = detection['vis']

            directory, filename = os.path.split(image_path)
            save_path = os.path.join(directory, "icon_replace_"+filename)

            self._show(image, save=save_path, show=False)

            replaced_image_paths.append(save_path)

        return replaced_image_paths


__all__ = [
    "pause_game",
    "unpause_game",
    "exit_back_to_game",
    "exit_back_to_pause",
    "take_screenshot",
    "segment_minimap",
    "switch_to_game",
    "IconReplacer"
]