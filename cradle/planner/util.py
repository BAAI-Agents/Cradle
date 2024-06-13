from PIL import Image
import mss
import time
import os


def get_attr(attr, key, default=None):

    if isinstance(attr, dict):
        return attr.get(key, default)
    else:
        return getattr(attr, key, default)


def crop_region_from_screen(region, filename):
    region = region
    region = {
        "left": region[0],
        "top": region[1],
        "width": region[2],
        "height": region[3],
    }
    with mss.mss() as sct:
        screen_image = sct.grab(region)
        image = Image.frombytes(
            "RGB", screen_image.size, screen_image.bgra, "raw", "BGRX"
        )
        image.save(filename)


def save_image(source_picture, filename):
    with Image.open(source_picture) as img:
        img.save(filename)


def crop_region_from_picture(source_picture, region, filename):
    region = region
    region = {
        "left": region[0],
        "top": region[1],
        "width": region[2],
        "height": region[3],
    }
    with Image.open(source_picture) as img:
        crop_region = (
            region["left"],
            region["top"],
            region["left"] + region["width"],
            region["top"] + region["height"],
        )
        cropped_image = img.crop(crop_region)
        cropped_image.save(filename)


def template_match_tool():

    main_screen = (0, 0, 2048, 1280)
    second_screen = (main_screen[2], 0, 1920, 1080)
    toolbar_relative_second_screen = (560, 975, 800, 95)
    toolbar = (
        toolbar_relative_second_screen[0] + second_screen[0],
        toolbar_relative_second_screen[1] + second_screen[1],
        toolbar_relative_second_screen[2],
        toolbar_relative_second_screen[3],
    )

    source_file = "res\stardew\samples\game_screenshot.jpg"

    # Crop toolbar
    filename = "runs/crop_test/toolbar.jpg"
    crop_region_from_picture(source_file, toolbar_relative_second_screen, filename)
    source_file = "runs/crop_test/toolbar.jpg"
    filename = "runs/crop_test/blank.jpg"
    crop_region_from_picture(source_file, (0, 0, 798, 90), filename)
    # original wentao version parameter, good for template matching but erase the number of the right lower conner
    # tool_span = 14
    # tool_left = 25
    # tool_top = 25
    # tool_width = 50
    # tool_height = 50
    # modified version with number cooperated
    #  The width of a single cell is 6480. The height of a plain light-colored cell is 72, which is sufficient. The excess amount at the top and bottom is 9. If the extra edges on both sides are fully occupied, then it would be 7272. We can infer that the width of a black border is 2, with the total width at the extreme left and right being 32, so each side is 16.
    tool_span_single = 2
    tool_left = 16
    tool_top = 9
    tool_width = 72
    tool_height = 75

    # dict
    tool_region = {}

    for i in range(1, 13):
        tool_region[str(i)] = (
            # tool_left + (i - 1) * tool_span + (i - 1) * tool_width,
            tool_left + (i - 1) * (tool_width - 4 * tool_span_single),
            tool_top,
            tool_width,
            tool_height,
        )

    source_file = "runs/crop_test/toolbar.jpg"
    if not os.path.exists("runs/crop_test/toolbar"):
        os.makedirs("runs/crop_test/toolbar")
    for key in tool_region:
        region = tool_region[key]
        # Save tool
        filename = f"runs/crop_test/toolbar/{key}.jpg"
        crop_region_from_picture(
            source_file,
            # (toolbar[0] + region[0], toolbar[1] + region[1], region[2], region[3]),
            (region[0], region[1], region[2], region[3]),
            filename,
        )


if __name__ == "__main__":
    template_match_tool()
