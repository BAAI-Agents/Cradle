import os
import datetime
import time
import re
import random
import math
from typing import List, Dict, Tuple

import cv2
import matplotlib.pyplot as plt
import mss
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageChops
from scipy.ndimage import binary_fill_holes
import supervision as sv
import torch
from torchvision.ops import box_convert

from cradle.constants import COLOURS
from cradle.config import Config
from cradle.gameio import IOEnvironment
from cradle.log import Logger
from cradle import constants

config = Config()
io_env = IOEnvironment()
logger = Logger()

if config.is_game == True:
    from cradle.utils.object_utils import groundingdino_detect


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


def _draw_rectangle(draw, coords, outline="red", width=50):
    x1, y1, x2, y2 = coords
    draw.line([x1, y1, x2, y1], fill=outline, width=width)
    draw.line([x1, y2, x2, y2], fill=outline, width=width)
    draw.line([x1, y1, x1, y2], fill=outline, width=width)
    draw.line([x2, y1, x2, y2], fill=outline, width=width)


def draw_region_on_image(image_path, coordinates, pic_name):

    coords = eval(coordinates)

    image = Image.open(image_path)
    canvas = ImageDraw.Draw(image)
    width, height = image.size

    if len(coords) == 2:
        x, y = coords[0] * width, coords[1] * height
        _draw_rectangle(canvas, [x-1, y-1, x+1, y+1])
    elif len(coords) == 4:
        x1, y1, x2, y2 = coords[0] * width, coords[1] * height, coords[2] * width, coords[3] * height
        _draw_rectangle(canvas, [x1, y1, x2, y2], width=5)
    else:
        msg = "Coordinates must be two- or four-digit tuples"
        logger.error(msg)
        raise ValueError(msg)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # Store image where the original image is, add the pic_name and time to the new image name
    tokens = image_path.rsplit('.', 1)
    base_path_to_filename = tokens[0]
    file_extension = tokens[1]

    output_path = f'{base_path_to_filename}_{pic_name}_{timestamp}.{file_extension}'
    image.save(output_path)
    logger.debug(f"Picture saved: {output_path}")


def draw_mouse_pointer(image: cv2.typing.MatLike, x, y):

    mouse_cursor = cv2.imread('./res/icons/pink_mouse.png', cv2.IMREAD_UNCHANGED)

    if mouse_cursor is None:
        logger.error("Failed to read mouse cursor image file.")
    elif x is None or y is None:
        logger.warn("Mouse coordinates are missing.")
    else:
        new_size = (mouse_cursor.shape[1] // 4, mouse_cursor.shape[0] // 4)
        mouse_cursor = cv2.resize(mouse_cursor, new_size, interpolation=cv2.INTER_AREA)
        image_array = image

        if x + mouse_cursor.shape[1] > 0 and y + mouse_cursor.shape[0] > 0 and x < image_array.shape[1] and y < image_array.shape[0]:
            x_start = max(x, 0)
            y_start = max(y, 0)
            x_end = min(x + mouse_cursor.shape[1], image_array.shape[1])
            y_end = min(y + mouse_cursor.shape[0], image_array.shape[0])
            mouse_cursor_part = mouse_cursor[max(0, -y):y_end-y, max(0, -x):x_end-x]

            for c in range(3):
                alpha_channel = mouse_cursor_part[:, :, 3] / 255.0
                image_array[y_start:y_end, x_start:x_end, c] = \
                    alpha_channel * mouse_cursor_part[:, :, c] + \
                    (1 - alpha_channel) * image_array[y_start:y_end, x_start:x_end, c]

            image = cv2.cvtColor(image_array, cv2.COLOR_BGRA2BGR)

    return image


def _draw_rectangle(draw, coords, outline="red", width=50):
    x1, y1, x2, y2 = coords
    draw.line([x1, y1, x2, y1], fill=outline, width=width)
    draw.line([x1, y2, x2, y2], fill=outline, width=width)
    draw.line([x1, y1, x1, y2], fill=outline, width=width)
    draw.line([x2, y1, x2, y2], fill=outline, width=width)


def draw_region_on_image(image_path, coordinates, pic_name):

    coords = eval(coordinates)

    image = Image.open(image_path)
    canvas = ImageDraw.Draw(image)
    width, height = image.size

    if len(coords) == 2:
        x, y = coords[0] * width, coords[1] * height
        _draw_rectangle(canvas, [x-1, y-1, x+1, y+1])
    elif len(coords) == 4:
        x1, y1, x2, y2 = coords[0] * width, coords[1] * height, coords[2] * width, coords[3] * height
        _draw_rectangle(canvas, [x1, y1, x2, y2], width=5)
    else:
        msg = "Coordinates must be two- or four-digit tuples"
        logger.error(msg)
        raise ValueError(msg)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # Store image where the original image is, add the pic_name and time to the new image name
    tokens = image_path.rsplit('.', 1)
    base_path_to_filename = tokens[0]
    file_extension = tokens[1]

    output_path = f'{base_path_to_filename}_{pic_name}_{timestamp}.{file_extension}'
    image.save(output_path)
    logger.debug(f"Picture saved: {output_path}")


def draw_mouse_pointer_file_(image_path: str, x, y) -> str:

    image = cv2.imread(image_path)

    # get the size of the image to see whether it is a padding
    height, width, _ = image.shape
    if height > config.DEFAULT_ENV_RESOLUTION[1] or width > config.DEFAULT_ENV_RESOLUTION[0]:
        x, y = x + config.som_padding_size, y + config.som_padding_size

    image_with_mouse = draw_mouse_pointer(image, x, y)

    draw_mouse_img_path = image_path.replace(".jpg", f"_with_mouse.jpg")
    cv2.imwrite(draw_mouse_img_path, image_with_mouse)
    logger.debug(f"The image with mouse pointer is saved at {draw_mouse_img_path}")

    return draw_mouse_img_path


def calculate_image_diff(path_1, path_2):

    if not os.path.exists(path_1):
        logger.error(f"The file at {path_1} does not exist.")
        raise FileNotFoundError(f"The file at {path_1} does not exist.")

    if not os.path.exists(path_2):
        logger.error(f"The file at {path_2} does not exist.")
        raise FileNotFoundError(f"The file at {path_2} does not exist.")

    img1 = Image.open(path_1)
    img2 = Image.open(path_2)

    if img1.size != img2.size:
        msg = "Images do not have the same size."
        logger.error(msg)
        raise ValueError(msg)

    diff = ImageChops.difference(img1, img2)
    diff = diff.convert("RGBA")

    pixels = diff.load()
    width, height = diff.size
    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            if r == g == b == 0:
                pixels[x, y] = (0, 0, 0, 0)
            else:
                pixels[x, y] = (r, g, b, 255)

    return diff


def save_image_diff(path_1, path_2):
    diff = calculate_image_diff(path_1, path_2)

    img1_name = os.path.splitext(os.path.basename(path_1))[0]
    img2_name = os.path.splitext(os.path.basename(path_2))[0]

    output_filename = f"diff_{img1_name}_{img2_name}.png"
    output_path = os.path.join(os.path.dirname(path_1), output_filename)
    diff.save(output_path)
    logger.debug(f"Picture saved: {output_path}")

    return output_path


def calculate_pixel_diff_with_diffimage_path(output_path):
    img = Image.open(output_path).convert("RGBA")
    pixels = img.load()
    width, height = img.size

    non_transparent_count = 0
    for y in range(height):
        for x in range(width):
            _, _, _, a = pixels[x, y]
            if a != 0:
                non_transparent_count += 1

    return non_transparent_count


def calculate_pixel_diff(path_1, path_2):
    diff_image_path = save_image_diff(path_1, path_2)
    return calculate_pixel_diff_with_diffimage_path(diff_image_path)


def resize_image(image: Image.Image | str | np.ndarray, resize_ratio: float) -> Image.Image:
    """Resize the given image.

    Args:
        image (Image.Image | str | np.ndarray): The image to be resized.
        resize_ratio (float): The ratio to resize the image.

    Returns:
        Image.Image: The resized image.
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif isinstance(image, str):
        image = Image.open(image)

    # Assume it is already an Image.Image object if neither of the above
    new_size = [int(dim * resize_ratio) for dim in image.size]
    image_resized = image.resize(new_size)
    return image_resized


def overlay_image_on_background(annotations: list, shape) -> Image:
    if len(annotations) == 0:
        return None

    img_data = np.ones((shape[0], shape[1], 4), dtype=np.uint8)

    sorted_anns = sorted(annotations, key=lambda x: x["area"], reverse=True)

    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.rand(3) * 255, [89]])
        img_data[m] = color_mask

    # Convert the NumPy array to a PIL Image and save
    img = Image.fromarray(img_data.astype('uint8'), 'RGBA')

    return img


def process_image_for_masks(original_image: Image) -> list[np.ndarray]:
    """
    Process the image to find unique masks based on color channels.

    Args:
    original_image: A Image object of the original image.

    Returns:
    A list of numpy.ndarrays, each representing a unique mask.
    """
    # logger.debug("starting...")
    original_image_np = np.array(original_image)

    # Assume the last channel is the alpha channel if the image has 4 channels
    if original_image_np.shape[2] == 4:
        original_image_np = original_image_np[:, :, :3]

    # Find unique colors in the image, each unique color corresponds to a unique mask
    unique_colors = np.unique(original_image_np.reshape(-1, original_image_np.shape[2]), axis=0)

    masks = []
    for color in unique_colors:
        # Create a mask for each unique color
        mask = np.all(original_image_np == color, axis=-1)
        masks.append(mask)

    return masks


def display_binary_images_grid(images: list[np.ndarray], grid_size = None, margin: int = 10, cell_size = None):
    """
    Display binary ndarrays as images on a grid with clear separation between grid cells,
    scaling images down as necessary.

    Args:
    images: A list of binary numpy.ndarrays.
    grid_size: Optional tuple (rows, cols) indicating the grid size.
               If not provided, a square grid size will be calculated.
    margin: The margin size between images in the grid.
    cell_size: Optional tuple (width, height) indicating the maximum size for any image cell in the grid.
               Images will be scaled down to fit within this size while maintaining aspect ratio.
    """
    if grid_size is None:
        grid_size = (int(np.ceil(np.sqrt(len(images)))),) * 2

    if cell_size is None:
        # Determine max dimensions of images in the list and use that as cell size
        cell_width = max(image.shape[1] for image in images) + margin
        cell_height = max(image.shape[0] for image in images) + margin
    else:
        cell_width, cell_height = cell_size

    # Create a new image with a white background
    total_width = cell_width * grid_size[1]
    total_height = cell_height * grid_size[0]
    grid_image = Image.new('1', (total_width, total_height), 1)

    for index, binary_image in enumerate(images):
        img = Image.fromarray(binary_image.astype(np.uint8) * 255, 'L').convert('1')

        # Scale down the image to fit within the cell size, maintaining aspect ratio
        img.thumbnail((cell_width - margin, cell_height - margin))

        # Create a new image with margins
        img_with_margin = Image.new('1', (img.width + margin, img.height + margin), 1)
        img_with_margin.paste(img, (margin // 2, margin // 2))

        # Calculate the position on the grid
        row, col = divmod(index, grid_size[1])
        x = col * cell_width
        y = row * cell_height

        # Paste the image into the grid
        grid_image.paste(img_with_margin, (x, y))

    # Display the grid image
    return grid_image


def remove_border_masks(masks: list[np.ndarray], threshold_percent: float = 5.0) -> list[np.ndarray]:
    """
    Removes masks whose "on" pixels are close to the mask borders on all four sides.

    Parameters:
    - masks: A list of ndarrays, where each ndarray is a binary mask.
    - threshold_percent: A float indicating how close the "on" pixels can be to the border,
                         represented as a percentage of the mask's dimensions.

    Returns:
    - A list of ndarrays with the border masks removed.
    """
    def is_close_to_all_borders(mask: np.ndarray, threshold: float) -> bool:

        # Determine actual threshold in pixels based on the percentage
        threshold_rows = int(mask.shape[0] * (threshold_percent / 100))
        threshold_cols = int(mask.shape[1] * (threshold_percent / 100))

        # Check for "on" pixels close to each border
        top = np.any(mask[:threshold_rows, :])
        bottom = np.any(mask[-threshold_rows:, :])
        left = np.any(mask[:, :threshold_cols])
        right = np.any(mask[:, -threshold_cols:])

        # If "on" pixels are close to all borders, return True
        return top and bottom and left and right

    filtered_masks = []
    for mask in masks:
        # Only add mask if it is not close to all borders
        if not is_close_to_all_borders(mask, threshold_percent):
            filtered_masks.append(mask)

    return filtered_masks


def filter_thin_ragged_masks(masks: list[np.ndarray], kernel_size: int = 3, iterations: int = 5) -> list[np.ndarray]:
    """
    Applies morphological operations to filter out thin and ragged masks.

    Parameters:
    - masks: A list of ndarrays, where each ndarray is a binary mask.
    - kernel_size: Size of the structuring element.
    - iterations: Number of times the operation is applied.

    Returns:
    - A list of ndarrays with thin and ragged masks filtered out.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    filtered_masks = []

    for mask in masks:
        # Convert boolean mask to uint8
        try:
            mask_uint8 = mask.astype(np.uint8) * 255
        except MemoryError:
            logger.error("MemoryError: Mask is too large to convert to uint8.")
            continue

        # Perform erosion
        eroded_mask = cv2.erode(mask_uint8, kernel, iterations=iterations)

        # Perform dilation
        dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=iterations)

        # Convert back to boolean mask and add to the filtered list
        filtered_masks.append(dilated_mask > 0)

    return filtered_masks


def refine_masks(masks: list[np.ndarray]) -> list[np.ndarray]:
    """
    Refine the list of masks:
    - Fill holes of any size
    - Remove masks completely contained within other masks.

    Args:
        masks: A list of numpy.ndarrays, each representing a mask.

    Returns:
        A list of numpy.ndarrays, each representing a refined mask.
    """

    masks = remove_border_masks(masks)
    masks = filter_thin_ragged_masks(masks)

    # Fill holes in each mask
    filled_masks = [binary_fill_holes(mask).astype(np.uint8) for mask in masks]

    # Remove masks completely contained within other masks
    refined_masks = []
    for i, mask_i in enumerate(filled_masks):
        contained = False
        for j, mask_j in enumerate(filled_masks):
            if i != j:
                # Check if mask_i is completely contained in mask_j
                if np.array_equal(mask_i & mask_j, mask_i):
                    contained = True
                    break

        if not contained:
            refined_masks.append(mask_i)

    return refined_masks


def extract_masked_images(original_image: Image, masks: list[np.ndarray]):
    """
    Apply each mask to the original image and resize the image to fit the mask's bounding box,
    discarding pixels outside the mask.

    Args:
    original_image: A Image object of the original image.
    masks: A list of numpy.ndarrays, each representing a refined mask.

    Returns:
    A list of Image objects, each cropped to the mask's bounding box and containing the content of the original image within that mask.
    """
    original_image_np = np.array(original_image)
    masked_images = []

    for mask in masks:
        # Find the bounding box of the mask
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Crop the mask and the image to the bounding box
        cropped_mask = mask[rmin:rmax+1, cmin:cmax+1]
        cropped_image = original_image_np[rmin:rmax+1, cmin:cmax+1]

        # Apply the mask
        masked_image = np.where(cropped_mask[:, :, None], cropped_image, 0).astype(np.uint8)
        masked_images.append(Image.fromarray(masked_image))

    return masked_images


def calculate_bounding_boxes(masks: List[np.ndarray]) -> Tuple[List[Dict], List[Tuple[float, float]]]:
    """
    Calculate bounding boxes for each mask in the list separately.

    Args:
        masks: A list of numpy.ndarrays, each representing a mask.

    Returns:
        A list containing dictionaries, each containing the "top", "left", "height", "width" of the bounding box for each mask.
    """
    bounding_boxes = []

    for mask in masks:

        # Find all indices where mask is True
        rows, cols = np.where(mask)
        if len(rows) == 0 or len(cols) == 0:  # In case of an empty mask
            continue

        # Calculate bounding box
        top, left = rows.min(), cols.min()
        height, width = rows.max() - top, cols.max() - left

        # Append data to the lists
        bounding_boxes.append({
            "top": float(top),
            "left": float(left),
            "height": float(height),
            "width": float(width),
        })

    # Sort bounding boxes from left to right, top to bottom
    sorted_indices = sorted(range(len(bounding_boxes)), key=lambda i: (bounding_boxes[i].get("top", float('inf')), bounding_boxes[i].get("left", float('inf'))))
    sorted_bounding_boxes = [bounding_boxes[i] for i in sorted_indices]

    return sorted_bounding_boxes


def calculate_centroid(bbox: dict) -> tuple:
    """Calculate the centroid of a bounding box.

    Args:
        bbox (dict): The bounding box dictionary with 'top', 'left', 'height', and 'width' keys.

    Returns:
        tuple: The (x, y) coordinates of the centroid.
    """
    x_center = bbox['left'] + bbox['width'] / 2
    y_center = bbox['top'] + bbox['height'] / 2
    return (x_center, y_center)


def plot_som(screenshot_filename, bounding_boxes):

    if config.plot_bbox_multi_color == True:
        som_img = plot_som_multicolor(screenshot_filename, bounding_boxes)
    else:
        som_img = plot_som_unicolor(screenshot_filename, bounding_boxes)

    return som_img


def plot_som_multicolor(screenshot_filename, bounding_boxes):

    org_img = Image.open(screenshot_filename)
    font_path = "arial.ttf"
    font_size, padding = 20, 2
    font = ImageFont.truetype(font_path, font_size)

    # Create a color cycle using one of the categorical color palettes in matplotlib
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    text_to_draw = []

    # Determine padding around the image to prevent label overflow
    som_padding_size = config.som_padding_size  # Adjust the size of the padding as needed
    image_with_padding = Image.new("RGB", (org_img.width + som_padding_size * 2, org_img.height + som_padding_size * 2), config.som_padding_color)
    image_with_padding.paste(org_img, (som_padding_size, som_padding_size))
    draw = ImageDraw.Draw(image_with_padding)

    # Draw each bounding box on the image
    for i, bbox in enumerate(bounding_boxes):

        if len(bbox) == 0:
            continue

        # Calculate the bottom-right corner of the bounding box
        top, right, bottom, left = (
            bbox['top'] + som_padding_size,
            bbox['left'] + bbox['width'] + som_padding_size,
            bbox['top'] + bbox['height'] + som_padding_size,
            bbox['left'] + som_padding_size,
        )

        bbox_padding = 0
        bbox_border = 2
        color = color_cycle[i % len(color_cycle)]

        # Expand the bounding box outwards by the border width
        draw.rectangle(
            [
                left - bbox_border,
                top - bbox_border,
                right + bbox_border,
                bottom + bbox_border,
            ],
            outline=color,
            width=bbox_border,
        )

        unique_id = str(i + 1)
        text_width = draw.textlength(unique_id, font=font)
        text_height = font_size

        # Position the text in the top-left corner of the bounding box
        text_position = (
            left - text_width - padding,
            top - text_height - padding,
        )

        new_text_rectangle = [
            text_position[0] - padding,
            text_position[1] - padding,
            text_position[0] + text_width + padding,
            text_position[1] + text_height + padding,
        ]

        text_to_draw.append(
            (new_text_rectangle, text_position, unique_id, color)
        )

    for text_rectangle, text_position, unique_id, color in text_to_draw:
        # Draw a background rectangle for the text
        draw.rectangle(text_rectangle, fill=color)
        draw.text(text_position, unique_id, font=font, fill="white")

    return image_with_padding


def plot_som_unicolor(screenshot_filename, bounding_boxes):
    org_img = Image.open(screenshot_filename)
    font_path = "arial.ttf"
    font_size, padding = 13, 1
    font = ImageFont.truetype(font_path, font_size)

    text_to_draw = []

    draw = ImageDraw.Draw(org_img)

    # Draw each bounding box on the image
    for i, bbox in enumerate(bounding_boxes):
        if len(bbox) == 0:
            continue
        # Calculate the bottom-right corner of the bounding box
        top, right, bottom, left = (
            bbox['top'],
            bbox['left'] + bbox['width'],
            bbox['top'] + bbox['height'],
            bbox['left'],
        )

        bbox_padding = 0
        bbox_border = 2
        color = 'red'

        # Expand the bounding box outwards by the border width
        draw.rectangle(
            [
                left - bbox_border,
                top - bbox_border,
                right + bbox_border,
                bottom + bbox_border,
            ],
            outline=color,
            width=bbox_border,
        )

        unique_id = str(i + 1)
        text_width = draw.textlength(unique_id, font=font)
        text_height = font_size

        # Position the text in the top-left corner of the bounding box
        text_position = (
            left,
            top,
        )

        new_text_rectangle = [
            text_position[0] - padding,
            text_position[1] - padding,
            text_position[0] + text_width + padding,
            text_position[1] + text_height + padding,
        ]

        text_to_draw.append(
            (new_text_rectangle, text_position, unique_id, color)
        )

    for text_rectangle, text_position, unique_id, color in text_to_draw:
        # Draw a background rectangle for the text
        if config.env_name == "CapCut":
            draw.rectangle(text_rectangle, fill="white")
            draw.text(text_position, unique_id, font=font, fill="black")
        else:
            draw.rectangle(text_rectangle, fill="black")
            draw.text(text_position, unique_id, font=font, fill="white")

    return org_img


def crop_grow_image(image_path: str, crop_area_border: tuple[int, int, int, int] = (8, 1, 8, 8), overwrite_flag: bool = False):

    with Image.open(image_path) as img:

        width, height = img.size
        crop_area = (crop_area_border[0], crop_area_border[1], width-crop_area_border[2], height-crop_area_border[3])
        cropped_img = img.crop(crop_area)
        cropped_img = cropped_img.resize(config.DEFAULT_ENV_RESOLUTION, Image.ANTIALIAS)

        if overwrite_flag:
            crop_screenshot_path = image_path
        else:
            crop_screenshot_path = image_path.replace(".jpg", f"_crop.jpg")

        cropped_img.save(crop_screenshot_path)

        return crop_screenshot_path


def remove_redundant_bboxes(bboxes: list) -> list:
    # Remove redundant bounding boxes based on the size and intersection over union (IoU) between each other.
    len_original_bboxes = len(bboxes)

    def calculate_iou(bbox1, bbox2):
        # Calculate the intersection over union (IoU) of two bounding boxes

        # Calculate intersection
        x1 = max(bbox1['left'], bbox2['left'])
        y1 = max(bbox1['top'], bbox2['top'])
        x2 = min(bbox1['left'] + bbox1['width'], bbox2['left'] + bbox2['width'])
        y2 = min(bbox1['top'] + bbox1['height'], bbox2['top'] + bbox2['height'])

        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

        # Calculate areas of the bounding boxes
        bbox1_area = bbox1['width'] * bbox1['height']
        bbox2_area = bbox2['width'] * bbox2['height']

        larger_area = max(bbox1_area, bbox2_area)

        return intersection_area / larger_area  # Return the ratio of the intersection area to the larger bbox's area

    # Remove exact duplicate bounding boxes
    seen = set()
    unique_bboxes = []
    for bbox in bboxes:
        bbox_tuple = (bbox['left'], bbox['top'], bbox['width'], bbox['height'])
        if bbox_tuple not in seen:
            seen.add(bbox_tuple)
            unique_bboxes.append(bbox)

    bboxes = unique_bboxes

    # To keep track of indices of bounding boxes to remove
    to_remove = set()

    # Compare each pair of bounding boxes
    for i in range(len(bboxes)):
        # Remove small bounding boxes
        if bboxes[i]['width'] * bboxes[i]['height'] <= config.min_bbox_area:
            to_remove.add(i)
            continue

        for j in range(i + 1, len(bboxes)):
            iou = calculate_iou(bboxes[i], bboxes[j])
            if iou >= config.max_intersection_rate:
                smaller_bbox_idx = i if bboxes[i]['width'] * bboxes[i]['height'] < bboxes[j]['width'] * bboxes[j]['height'] else j
                to_remove.add(smaller_bbox_idx)

    filtered_bboxes = [bbox for k, bbox in enumerate(bboxes) if k not in to_remove]

    # Log the percentage of bounding boxes removed
    if bboxes:
        percentage_removed = (len(to_remove) / len_original_bboxes) * 100
        logger.write(f"Removed {percentage_removed:.2f}% redundant bounding boxes.")

    return filtered_bboxes


def filter_inner_bounding_boxes(bboxes: List[Dict]) -> List[Dict]:
    # Remove situations in which a bbox contains small bboxes.
    # These small bboxes are adjacent to each other. The sum area of these small bboxes is similar to this big bbox.

    len_original_bboxes = len(bboxes)
    to_remove = set()

    for i in range(len(bboxes)):
        bboxes_in_i = set()
        area_i = bboxes[i]['width'] * bboxes[i]['height']

        max_mergeable_area = 5000
        if area_i > max_mergeable_area:
            continue

        for j in range(i + 1, len(bboxes)):
            if bboxes[i]['left'] <= bboxes[j]['left'] and bboxes[i]['top'] <= bboxes[j]['top'] and \
                bboxes[i]['left'] + bboxes[i]['width'] >= bboxes[j]['left'] + bboxes[j]['width'] and \
                bboxes[i]['top'] + bboxes[i]['height'] >= bboxes[j]['top'] + bboxes[j]['height']:
                bboxes_in_i.add(j)

        if bboxes_in_i:
            sum_area_i = sum(bboxes[k]['width'] * bboxes[k]['height'] for k in bboxes_in_i)
            if abs(sum_area_i - area_i) < (0.3 * area_i):
                to_remove.update(bboxes_in_i)

    # Create the final list excluding the bboxes marked for removal
    final_bboxes = [bboxes[k] for k in range(len(bboxes)) if k not in to_remove]

    # Log the percentage of bounding boxes removed
    if final_bboxes:
        percentage_removed = (len(to_remove) / len_original_bboxes) * 100
        logger.write(f"Removed {percentage_removed:.2f}% merged bounding boxes.")

    return final_bboxes


def looks_like_watermark(org_img, bbox):
    # Crop the image using the bounding box coordinates
    cropped_img = org_img.crop((bbox['left'], bbox['top'], bbox['left'] + bbox['width'], bbox['top'] + bbox['height']))
    cropped_array = np.array(cropped_img)

    # Define the watermark color
    watermark_color = (204, 204, 204)
    white_color = (255, 255, 255)

    # @TODO Overly general check, but it's now only enabled by config and includes the proportion check, it becomes a much less likely issue.
    # Check if all pixels in the bbox are close to the watermark color
    is_all_pixels_watermark_color = np.all((cropped_array >= watermark_color) & (cropped_array <= white_color), axis=-1).all()

     # Check if pixels are within the watermark color range
    in_watermark_range = np.all((cropped_array >= watermark_color) & (cropped_array < white_color), axis=-1)

    # Calculate the proportion of pixels within the watermark color range
    proportion_in_range = np.sum(in_watermark_range) / (cropped_array.size / 3)  # size / 3 for RGB channels

    return is_all_pixels_watermark_color and proportion_in_range >= 0.4


def filter_out_watermarks(org_img, bboxes):
    # Remove watermark bounding boxes based on certain conditions
    len_original_bboxes = len(bboxes)
    bboxes = [bbox for bbox in bboxes if not looks_like_watermark(org_img, bbox)]

    len_remove = len_original_bboxes - len(bboxes)

    percentage_removed = (len_remove / len_original_bboxes) * 100
    logger.write(f"Removed {percentage_removed:.2f}% watermark bounding boxes.")

    return bboxes


def textsize(draw, text, font=None):
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    return text_width, text_height


def draw_axis(image,
               crop_region=None,
               axis_color=COLOURS["black"],
               axis_division=(3, 5),
               axis_linewidth=3,
               font_color=COLOURS["black"],
               font_size=50,
               scale_length=20,
               **kwargs):

    if isinstance(axis_color, str):
        axis_color = COLOURS[axis_color]
    else:
        axis_color = tuple(axis_color)

    if isinstance(font_color, str):
        font_color = COLOURS[font_color]
    else:
        font_color = tuple(font_color)

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


def draw_mask_panel(image: Image, **kwargs):

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
               axis_color=COLOURS["red"],
               axis_division=(3, 5),
               axis_linewidth=3,
               font_color=COLOURS["yellow"],
               font_size=50,
               **kwargs):

    if isinstance(axis_color, str):
        axis_color = COLOURS[axis_color]
    else:
        axis_color = tuple(axis_color)

    if isinstance(font_color, str):
        font_color = COLOURS[font_color]
    else:
        font_color = tuple(font_color)

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
                     left_band_color=COLOURS["blue"],
                     right_band_color=COLOURS["yellow"],
                    **kwargs):

    if isinstance(left_band_color, str):
        left_band_color = COLOURS[left_band_color]
    else:
        left_band_color = tuple(left_band_color)

    if isinstance(right_band_color, str):
        right_band_color = COLOURS[right_band_color]
    else:
        right_band_color = tuple(right_band_color)

    # Create the left and right bands
    left_band = Image.new('RGB', (left_band_width, left_band_height), left_band_color)
    right_band = Image.new('RGB', (right_band_width, right_band_height), right_band_color)

    # Create a new image with the bands and the original image
    image_with_bands = Image.new('RGB', (image.width + left_band.width + right_band.width, image.height))
    image_with_bands.paste(left_band, (0, 0))
    image_with_bands.paste(image, (left_band.width, 0))
    image_with_bands.paste(right_band, (left_band.width + image.width, 0))

    return image_with_bands


def draw_coordinate_axis_on_screenshot(image,
                                       crop_region = None,
                                       axis_color = COLOURS["black"],
                                       axis_division = (3, 5),
                                       axis_linewidth = 3,
                                       font_color = COLOURS["black"],
                                       font_size = 50,
                                       scale_length = 20,
                                       x_y_order = False):
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


def convert_ocr_bbox_format(data):
    # convert ocr bbox format to som bbox format
    bounding_boxes = []

    # Extract the first four coordinates
    for item in data[0]:
        coordinates = item[0]
        top = min(coord[1] for coord in coordinates)
        left = min(coord[0] for coord in coordinates)
        height = max(coord[1] for coord in coordinates) - top
        width = max(coord[0] for coord in coordinates) - left

        bounding_boxes.append({
            "top": float(top),
            "left": float(left),
            "height": float(height),
            "width": float(width),
        })

    return bounding_boxes


def is_within(bbox1, bbox2, padding=5):

    """
    Check if bbox1 is within bbox2 with a padding.
    bbox is expected to be a tuple (x1, y1, x2, y2).
    """

    return (bbox1["left"] >= bbox2["left"] - padding and
            bbox1["top"] >= bbox2["top"] - padding and
            bbox1["left"] + bbox1["width"] <= bbox2["left"] + bbox2["width"] + padding and
            bbox1["top"] + bbox1["height"] <= bbox2["top"] + bbox2["height"] + padding)


def filter_intersecting_rectangles(rectangles1, rectangles2, padding=5):

    """
    Filter out rectangles in rectangles1 that are within any rectangle in rectangles2 with a given padding.
    """

    initial_count = len(rectangles1)
    filtered_rectangles1 = []

    for rect1 in rectangles1:
        within_any_rect2 = any(is_within(rect1, rect2, padding) for rect2 in rectangles2)
        if not within_any_rect2:
            filtered_rectangles1.append(rect1)

    # Combine the filtered rectangles1 with rectangles2
    combined_rectangles = filtered_rectangles1 + rectangles2

    removed_count = initial_count - len(filtered_rectangles1)
    percentage_removed = (removed_count / initial_count) * 100 if initial_count > 0 else 0

    # Log the percentage of intersected bounding boxes removed
    logger.write(f"Removed {percentage_removed:.2f}% intersected bounding boxes.")

    return combined_rectangles


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

    # Create a new directory to store the new icon
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
