import os, time, math
import mss
import cv2
import numpy as np
from MTM import matchTemplates
from cradle.config import Config
from cradle.log import Logger
from cradle.utils.file_utils import assemble_project_path

# @TODO: this is not consistent now
# from cradle.environment.rdr2.atomic_skills.move import turn, move_forward
# from cradle.environment.rdr2.lifecycle.ui_control import take_screenshot
# from cradle.environment.stardew.skill_registry import register_skill

from PIL import Image, ImageDraw, ImageFont
# from cradle.gameio.game_manager import GameManager
from cradle.utils.template_matching import match_template_image

config = Config()
logger = Logger()


DEFAULT_GO_TO_STORE_ITERATIONS = 30
DEFAULT_GO_TO_STORE_TERMINAL_THRESHOLD = 95


import time
from cradle.environment.stardew.atomic_skills.basic_skills import (
    use_tool,
    do_action,
    move_up,
    move_down,
    move_left,
    move_right,
    select_tool,
    mouse_check_do_action,
)
from cradle.environment.stardew.skill_registry import register_skill




@register_skill("go_to_store")
def go_to_store():
    """
    Best way to go to the closest horse. Uses the minimap. Horses are useful to travel mid to long distances.
    """
    go_to_icon("store_door", iterations=DEFAULT_GO_TO_STORE_ITERATIONS, terminal_threshold=DEFAULT_GO_TO_STORE_TERMINAL_THRESHOLD, debug=False, gm=None, character=["up", "down", "left", "right"])


# @register_skill("go_to_icon")
def go_to_icon(target: str = "store", iterations=None, terminal_threshold=None, debug: bool = False, gm=None, character=None):
    """
    Navigates to the closed icon of the target in the minimap.

    Parameters:
    - target: Name of the target icon type on the minimap. The default value is "horse"
    """
    cv_go_to_icon(iterations, template_file=f'./res/{config.env_sub_path}/icons/{target}.jpg', terminal_threshold=terminal_threshold, debug=debug, gm=gm, character=[f'./res/{config.env_sub_path}/icons/{ch}.jpg' for ch in character])

# @TODO: This should be merged with the one in utils/template_matching.py
def match_template(src_file, template_file, template_resize_scale=1, debug=False, character=None):
    srcimg = cv2.imread(assemble_project_path(src_file))
    template = cv2.imread(assemble_project_path(template_file))

    # map the id of character_grid to the pixel localization
    # assume the grid is (2, 3)
    # origin = (srcimg.shape[0]//5*(character_grid[1]+0.5), srcimg.shape[1]//3*(character_grid[0]+0.5))

    if character == None:
        origin = (srcimg.shape[1] // 2, srcimg.shape[0] // 2)
    else:
        max_confidence = -1

        for ch_file in character:
            ch = cv2.imread(assemble_project_path(ch_file))
            detection = matchTemplates([('', cv2.resize(ch, (0, 0), fx=s, fy=s)) for s in [1]],
                                       srcimg,
                                       N_object=1,
                                       method=cv2.TM_CCOEFF_NORMED,
                                       maxOverlap=0.1)

            (_x, _y, _w, _h), confidence = detection['BBox'].iloc[0], detection['Score'].iloc[0]
            if confidence > max_confidence:
                max_confidence = confidence
                x, y, w, h = _x, _y, _w, _h

        origin = (x + w // 2, y + h // 2)

    # resize
    if template_resize_scale != 1:
        template = cv2.resize(template, (0, 0), fx=template_resize_scale, fy=template_resize_scale)

    detection = matchTemplates([('', cv2.resize(template, (0, 0), fx=s, fy=s)) for s in [1]],
                               srcimg,
                               N_object=1,
                               method=cv2.TM_CCOEFF_NORMED,
                               maxOverlap=0.1)
    (x, y, h, w), confidence = detection['BBox'].iloc[0], detection['Score'].iloc[0]

    if confidence < 0.3:
        logger.write(f"Can not find {template_file}")
        return None, None

    center_x = x + w // 2
    center_y = y + h // 2


    # draw the matched icon on the screenshot
    image = Image.open(src_file)
    draw = ImageDraw.Draw(image)

    # Draw the rectangle
    draw.rectangle([x, y, x+h, y+w], outline="blue", width=10)

    # Save the image
    matched_screenshot = src_file[:-4]+'_template_match.jpg'
    image.save(matched_screenshot)



    # go towards it
    # theta = get_theta(*origin, center_x, center_y)
    # dis = np.sqrt((center_x - origin[0]) ** 2 + (center_y - origin[1]) ** 2)
    # print(f'center_x: {center_x} center_y:{center_y} origin: {origin}')
    dis_x = center_x - origin[0]
    dis_y = center_y - origin[1]
    #
    # KalmanFilter threshold = 0.59
    measure = {'confidence': confidence, 'distance': (dis_x, dis_y), 'bounding_box': (x, y, h, w)}
    #
    # if debug:
    #     # logger.debug(f"confidence: {confidence:.3f}, distance: {dis:.3f}, theta: {theta:.3f}")
    #     vis = srcimg.copy()
    #     cv2.rectangle(vis, (x,y), (x + w, y + h), (0, 0, 255), 2)
    #     cv2.putText(vis, f'{confidence:.3f}', (x,y), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1, cv2.LINE_AA)
    #     cv2.arrowedLine(vis, origin, (center_x, center_y), (0, 255, 0), 2, tipLength=0.1)
    #
    #     measure['vis'] = vis
    #     #cv2.imshow("vis", measure['vis'])
    return (dis_x, dis_y), measure


def cv_go_to_icon(
        iterations,
        template_file,
        terminal_threshold=95,
        debug=False,
        gm=None,
        character=None
):

    save_dir = config.work_dir
    # terminal_threshold *= config.resolution_ratio

    check_success, prev_dis, prev_theta, counter, ride_attempt, ride_mod, dis_stat = False, 0, 0, 0, 0, 10, []

    for step in range(iterations):

        logger.write(f'Go to icon iter #{step}')

        if config.ocr_different_previous_text:
            logger.write("The text is different from the previous one.")
            config.ocr_enabled = False # disable ocr
            config.ocr_different_previous_text = False  # reset
            break

        timestep = time.time()

        # gm.unpause_game()
        # cur_screenshot_path, cur_screenshot_path_augmented = gm.capture_screen(draw_axis=config.draw_axis,
        #                                                                               draw_color_band=config.draw_color_band)

        cur_screenshot_path = take_screenshot()
        # if the character is in the center of the cur_screenshot, then the dis_x, dis_y is the distance between character and template,
        # otherwise, you can only use the info["bounding_box"] for the position of the character
        dis_xy, info = match_template(cur_screenshot_path, template_file, debug=debug, character=character)

        if info is None:
            return False

        (dis_x, dis_y) = dis_xy

        dis = np.sqrt(dis_x ** 2 + dis_y ** 2)
        if dis < terminal_threshold:  # begin to settle
            logger.write('Success! Reached the icon.')
            do_action()
            return True

        # 2. Check stuck
        if abs(prev_dis - dis) < 5:
            counter += 1
            if counter >= 1:
                if debug:
                    logger.debug('Move randomly to get unstuck')
                for _ in range(2):
                    duration = np.random.rand()*0.4
                    rand_value = np.random.rand()
                    if rand_value < 0.20:
                        move_right(duration)
                    elif rand_value < 0.4:
                        move_left(duration)
                    elif rand_value < 0.6:
                        move_up(duration)
                    else:
                        move_down(duration)
        else:
            counter = 0

        # 3. Move (needed to cooperated with testing results, how many seconds take to run 100 pixel?
        duration = 0.1
        print(f'step: {step} dis:{(dis_x, dis_y)}')
        gap_threshold = 20
        if dis_x > gap_threshold:
            move_right(duration)
        if dis_x < -gap_threshold:
            move_left(duration)
        if dis_y < gap_threshold:
            move_up(duration)
        if dis_y > -gap_threshold:
            move_down(duration)
        time.sleep(0.5)
        prev_dis = dis

        # gm.pause_game(ide_name=config.IDE_NAME)

    logger.error(f'Go to icon failed to reach icon.')
    return False  # failed

def take_screenshot():
    with mss.mss() as sct:
        current_time = time.time()
        region = config.env_region
        region = {
            "left": region[0],
            "top": region[1],
            "width": region[2],
            "height": region[3],
        }
        screen_image = sct.grab(region)
        image = Image.frombytes("RGB", screen_image.size, screen_image.bgra, "raw", "BGRX")
        image_name = f'screen_go_store_interact_{current_time}.jpg'
        output_dir = config.work_dir
        image_name = os.path.join(output_dir, image_name)
        image.save(image_name)

        # config.update_current_game_screenshot_path(image_name)
        return image_name


__all__ = [
    "go_to_store",
]
