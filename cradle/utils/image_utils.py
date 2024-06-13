import numpy as np
import cv2
import time
from PIL import Image, ImageFont, ImageDraw
import math
import torch
import os
import re
from torchvision.ops import box_convert
import mss
from typing import Tuple
import supervision as sv

from cradle import constants
from cradle.config import Config
from cradle.gameio.io_env import IOEnvironment
from cradle.log import Logger
from cradle.utils.object_utils import groundingdino_detect
from cradle.constants import COLORS

config = Config()
io_env = IOEnvironment()
logger = Logger()

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

def save_annotate_frame(image_source, boxes, logits, phrases, text_prompt, cur_screenshot_path):

    # Remove the main character itself from boxes
    if "person" in text_prompt.lower():
        if len(boxes) > 1:
            index = 0
            dis = 1.5

            for i in range(len(boxes)):
                down_mid = (boxes[i, 0], boxes[i, 1] + boxes[i, 3] / 2)
                distance = torch.sum(torch.abs(torch.tensor(down_mid) - torch.tensor((0.5, 1.0))))

                if distance < dis:
                    dis = distance
                    index = i

            boxes_ = torch.cat([boxes[:index], boxes[index + 1:]])
            logits_ = torch.cat([logits[:index], logits[index + 1:]])

            phrases.pop(index)

            annotated_frame = annotate_with_coordinates(image_source=image_source, boxes=boxes_[:,:], logits=logits_[:], phrases=phrases)
            cv2.imwrite(cur_screenshot_path, annotated_frame)

        elif len(boxes)==1:

            phrases.pop(0)
            boxes_ = torch.tensor(boxes[1:])
            logits_ = torch.tensor(logits[1:])

            annotated_frame = annotate_with_coordinates(image_source=image_source, boxes=boxes_[:,:], logits=logits_[:], phrases=phrases)
            cv2.imwrite(cur_screenshot_path, annotated_frame)
        else:
            annotated_frame = annotate_with_coordinates(image_source=image_source, boxes=boxes[:,:], logits=logits[:], phrases=phrases)
            cv2.imwrite(cur_screenshot_path, annotated_frame)

    else:
        annotated_frame = annotate_with_coordinates(image_source=image_source, boxes=boxes[:,:], logits=logits[:], phrases=phrases)
        cv2.imwrite(cur_screenshot_path, annotated_frame)

def show_image(img):
    if isinstance(img, str):
        img = cv2.imread(img)
    cv2.namedWindow("display", cv2.WINDOW_NORMAL)
    cv2.imshow("display", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def minimap_movement_detection(image_path1, image_path2, threshold = 30):
    '''
    Detect whether two minimaps are the same to determine whether the character moves successfully.
    Args:
        image_path1, image_path2: 2 minimap images to be detected.
        threshold: pixel-level threshold for minimap movement detection, default 30.

    Returns:
        change_detected: True if the movements is above the threshold,
        img_matches: Draws the found matches of keypoints from two images. Can be visualized by plt.imshow(img_matches)
    '''
    img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    if type(descriptors1) != type(None) and type(descriptors2) != type(None):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
    else:
        return True, None, None

    matches = sorted(matches, key = lambda x:x.distance)

    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:20], None, flags=2)
    best_matches = matches[:20]

    average_distance = np.mean([m.distance for m in best_matches])

    change_detected = average_distance > (threshold * config.resolution_ratio) or np.allclose(average_distance, 0, atol=1e-3)
    return change_detected, img_matches, average_distance

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

def process_minimap_targets(image_path):

    minimap_image, boxes, logits, phrases = groundingdino_detect(image_path=segment_minimap(image_path),
                                                            text_prompt=constants.GD_PROMPT,
                                                            box_threshold=0.29, device='cuda')

    get_theta = lambda x0, y0, x, y:math.degrees(math.atan2(x - x0, y0 - y))
    h, w, _ = minimap_image.shape
    xyxy = box_convert(boxes=boxes.detach().cpu() * torch.Tensor([w, h, w, h]), in_fmt="cxcywh", out_fmt="xyxy").numpy().astype(int)

    minimap_detection_objects = {constants.RED_POINTS: [], constants.YELLOW_POINTS: [], constants.YELLOW_REGION: []}

    for detect_xyxy, detect_object, detect_confidence in zip(xyxy, phrases, logits):

        # Exclude too large detections
        if detect_xyxy[2] - detect_xyxy[0] > 0.8 * w and detect_xyxy[3] - detect_xyxy[1] > 0.8 * h:
            continue

        if detect_object == constants.YELLOW_POINTS and (detect_xyxy[2] - detect_xyxy[0] > 0.1 * w or detect_xyxy[3] - detect_xyxy[1] > 0.1 * h):
            detect_object = constants.YELLOW_REGION

        tgt_x = int((detect_xyxy[0] + detect_xyxy[2]) / 2)  # center of the box
        tgt_y = int((detect_xyxy[1] + detect_xyxy[3]) / 2)

        theta = get_theta(h // 2, w // 2, tgt_x, tgt_y)

        minimap_detection_objects[detect_object].append(dict(
            theta=theta,
        ))

    return minimap_detection_objects


def exec_clip_minimap(
        tid : float,
        screen_region : tuple[int, int, int, int] = None,
        minimap_region: tuple[int, int, int, int] = None,) -> Tuple[str, str]:

    if screen_region is None:
        screen_region = config.env_region

    if minimap_region is None:
        minimap_region = config.base_minimap_region

    region = screen_region
    region = {
        "left": region[0],
        "top": region[1],
        "width": region[2],
        "height": region[3],
    }

    output_dir = config.work_dir

    # Save screenshots
    screen_image_filename = output_dir + "/screen_" + str(tid) + ".jpg"

    with mss.mss() as sct:
        screen_image = sct.grab(region)
        image = Image.frombytes("RGB", screen_image.size, screen_image.bgra, "raw", "BGRX")
        image.save(screen_image_filename)

    minimap_image_filename = output_dir + "/minimap_" + str(tid) + ".jpg"

    mm_region = minimap_region
    mm_region = {
        "left": mm_region[0],
        "top": mm_region[1],
        "width": mm_region[2],
        "height": mm_region[3],
    }

    with mss.mss() as sct:
        minimap_image = sct.grab(mm_region)
        mm_image = Image.frombytes("RGB", minimap_image.size, minimap_image.bgra, "raw", "BGRX")
        mm_image.save(minimap_image_filename)

    clip_minimap(minimap_image_filename)

    return screen_image_filename, minimap_image_filename

def segment_toolbar(screenshot_path):
    tid = time.time()
    output_dir = config.work_dir
    toobar_image_filename = output_dir + "/toolbar_" + str(tid) + ".jpg"
    toobar_region = config._cal_toolbar_region()
    toobar_region[2] += toobar_region[0]
    toobar_region[3] += toobar_region[1]
    with Image.open(screenshot_path) as source_image:

        # Crop the image using the crop_rectangle
        cropped_minimap = source_image.crop(toobar_region)

        # Save the cropped image to a new file
        cropped_minimap.save(toobar_image_filename)
    return toobar_image_filename

def segment_new_icon(screenshot_path):
    tid = time.time()
    output_dir = config.work_dir
    # create a new directory to store the new icon
    if not os.path.exists(output_dir + "/icons"):
        os.makedirs(output_dir + "/icons")
    new_icon_image_filename = output_dir + "/icons/new_icon_" + str(tid) + ".jpg"
    new_icon_name_image_filename = output_dir + "/icons/new_icon_name_" + str(tid) + ".jpg"
    new_icon_region = config._cal_new_icon_region()
    new_icon_name_region = config._cal_new_icon_name_region()
    new_icon_region[2] += new_icon_region[0]
    new_icon_region[3] += new_icon_region[1]
    new_icon_name_region[2] += new_icon_name_region[0]
    new_icon_name_region[3] += new_icon_name_region[1]
    with Image.open(screenshot_path) as source_image:

        # Crop the image using the crop_rectangle
        cropped_new_icon = source_image.crop(new_icon_region)

        # Save the cropped image to a new file
        cropped_new_icon.save(new_icon_image_filename)

        cropped_new_icon_name = source_image.crop(new_icon_name_region)
        cropped_new_icon_name.save(new_icon_name_image_filename)
    return new_icon_image_filename, new_icon_name_image_filename


def segement_inventory(toolbarshot_path):
    match = re.search(r"toolbar_(\d+)", toolbarshot_path.replace("\\", "/"))
    tid = match.group(1)
    inventory_dict = config.inventory_dict
    tool_region = {}
    output_dir = config.work_dir
    for i in range(1, 13):

        tool_region[str(i)] = [
            # tool_left + (i - 1) * tool_span + (i - 1) * tool_width,
            inventory_dict["tool_left"]
            + (i - 1)
            * (inventory_dict["tool_width"] - 4 * inventory_dict["tool_span_single"]),
            inventory_dict["tool_top"],
            inventory_dict["tool_width"],
            inventory_dict["tool_height"],
        ]
    filenames = []
    if not os.path.exists(output_dir + "/toolbar_" + str(tid)):
        os.makedirs(output_dir + "/toolbar_" + str(tid))

    for key in tool_region:
        inventory_image_filename = (
            output_dir + "/toolbar_" + str(tid) + "/{}.jpg".format(key)
        )
        inventory_region = tool_region[key]
        inventory_region[2] += inventory_region[0]
        inventory_region[3] += inventory_region[1]
        with Image.open(toolbarshot_path) as source_image:

            # Crop the image using the crop_rectangle
            cropped_minimap = source_image.crop(inventory_region)

            # Save the cropped image to a new file
            cropped_minimap.save(inventory_image_filename)
            filenames.append(inventory_image_filename)
    return filenames


def draw_mask_panel(image: Image):
    # Define the rectangle coordinates
    top_left_blue = (10, 200)
    bottom_right_blue = (475, 255)

    top_left_green = (10, 260)
    bottom_right_green = (475, 315)

    # Define the blue and green color in RGB
    blue_color = (0, 107, 152)
    green_color = (0, 149, 132)
    black_color = (60, 66, 66)

    # Convert the PIL Image to a numpy array for masking
    image_array = np.array(image)

    # Function to check if the pixel color is within the specified range of the target color
    def is_pixel_color_in_range(pixel, target_color, threshold):
        return np.all(np.abs(pixel - np.array(target_color)) <= threshold)

    black_fix_point = (450, 150)
    blue_fix_point = (450, 230)
    green_fix_point = (450, 300)

    if is_pixel_color_in_range(image_array[black_fix_point[1], black_fix_point[0]], black_color, 50) \
            and is_pixel_color_in_range(image_array[blue_fix_point[1], blue_fix_point[0]], blue_color, 50) \
            and is_pixel_color_in_range(image_array[green_fix_point[1], green_fix_point[0]], green_color, 50):
        # Mask the specified rectangle area with green color
        image_array[top_left_blue[1]:bottom_right_blue[1], top_left_blue[0]:bottom_right_blue[0]] = blue_color
        image_array[top_left_green[1]:bottom_right_green[1], top_left_green[0]:bottom_right_green[0]] = green_color

    # Convert the numpy array back to a PIL Image
    masked_image = Image.fromarray(image_array)

    return masked_image


def draw_grids(image: Image,
               crop_region=None,
               axis_color=COLORS["red"],
               axis_division=(3, 5),
               axis_linewidth=3,
               font_color=COLORS["yellow"],
               font_size=50):
    # Prepare to draw on the image
    draw = ImageDraw.Draw(image)
    # Calculate the positions for the 3x3 grid lines

    if crop_region is None:
        left, top, right, bottom = 0, 0, image.width, image.height
    else:
        left, top, right, bottom = crop_region

    width = right - left
    height = bottom - top

    assert len(axis_division) == 2, f"Invalid axis grid division {axis_division}!"
    num_y_intervals, num_x_intervals = axis_division
    assert width % num_x_intervals == 0, f"The width of the image {width} is not divisible by the number of x intervals {num_x_intervals}!"
    assert height % num_y_intervals == 0, f"The height of the image {height} is not divisible by the number of y intervals {num_y_intervals}!"
    x_interval = width / num_x_intervals
    y_interval = height / num_y_intervals
    # Draw the vertical lines
    for i in range(1, num_x_intervals):
        draw.line([(x_interval * i, top), (x_interval * i, bottom)],
                  fill=axis_color, width=axis_linewidth)
    # Draw the horizontal lines
    for i in range(1, num_y_intervals):
        draw.line([(left, y_interval * i), (right, y_interval * i)],
                  fill=axis_color, width=axis_linewidth)
    # Draw the index numbers at the center of each grid cell
    try:
        font = ImageFont.truetype("arial.ttf", size=font_size)  # Adjust path and size as needed
    except IOError:
        font = ImageFont.truetype("Arial.ttf", size=font_size)
    index = 1
    for i in range(num_y_intervals):
        for j in range(num_x_intervals):
            label = f"({j + 1},{i + 1})"
            # Calculate the center position for the current cell
            x_center = (j * x_interval) + (x_interval / 2) + left
            y_center = (i * y_interval) + (y_interval / 2) + top
            # Get the text size to adjust positioning
            text_width, text_height = 0, 0
            # Adjust the start position based on text size
            position = (x_center - text_width / 2, y_center - text_height / 2)

            ### CURRENT: Draw the coordinate label
            draw.text(position, label, fill=font_color, font=font, linewidth=10)

            ### DEPRECATED: Draw the index number
            # Draw the index number
            # draw.text(position, str(index), fill=font_color, font=font, linewidth=10)
            # index += 1

    return image


def draw_color_band(image,
                     left_band_width=200,
                     left_band_height=1080,
                     right_band_width=200,
                     right_band_height=1080,
                     left_band_color=COLORS["blue"],
                     right_band_color=COLORS["yellow"]):

    # Create the left and right bands
    left_band = Image.new('RGB', (left_band_width, left_band_height), left_band_color)
    right_band = Image.new('RGB', (right_band_width, right_band_height), right_band_color)
    # Create a new image with the bands and the original image
    image_with_bands = Image.new('RGB', (image.width + left_band.width + right_band.width, image.height))
    image_with_bands.paste(left_band, (0, 0))
    image_with_bands.paste(image, (left_band.width, 0))
    image_with_bands.paste(right_band, (left_band.width + image.width, 0))
    return image_with_bands

def textsize(draw, text, font=None):
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    return text_width, text_height
def draw_axis(image,
               crop_region=None,
               axis_color=COLORS["black"],
               axis_division=(3, 5),
               axis_linewidth=3,
               font_color=COLORS["black"],
               font_size=50,
               scale_length=20,
               x_y_order=False):

    draw = ImageDraw.Draw(image)

    if crop_region is None:
        left, top, right, bottom = 0, 0, image.width, image.height
    else:
        left, top, right, bottom = crop_region

    try:
        font = ImageFont.truetype("arial.ttf", size=font_size)
    except IOError:
        font = ImageFont.truetype("Arial.ttf", size=font_size)

    # draw the coordinate axis
    draw.line((left, top, left, bottom),
              fill=axis_color,
              width=axis_linewidth)

    draw.line((left, top, right, top),
              fill=axis_color,
              width=axis_linewidth)

    for i in range(left, right - left + 1, (right - left) // axis_division[1]):

        draw.line((i, top, i, top + scale_length),
                  fill=axis_color,
                  width=axis_linewidth)

        text_width, text_height = textsize(draw, str(i), font=font)

        if i == left:
            draw.text((i + scale_length, top + scale_length), str(i),
                      fill=axis_color,
                      width=axis_linewidth,
                      font=font)
        else:
            draw.text(((i if i < right - left else right - text_width) - text_width // 2,
                       top + scale_length), str(i),
                      fill=axis_color,
                      width=axis_linewidth,
                      font=font)

    for i in range(top, bottom - top + 1, (bottom - top) // axis_division[0]):
        if i == 0:
            continue
        draw.line((left, i, left + scale_length, i),
                  fill=axis_color,
                  width=axis_linewidth)

        text_width, text_height = textsize(draw, str(i), font=font)
        draw.text(
            (left + scale_length, (i if i < bottom - top else bottom - text_height) - text_height // 2),
            str(i),
            fill=axis_color,
            width=axis_linewidth,
            font=font)

    return image
