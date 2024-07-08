import time, os
from collections import deque

import numpy as np
import cv2

from cradle.config import Config
from cradle.log import Logger
from cradle.utils.image_utils import minimap_movement_detection
from cradle.environment.rdr2.atomic_skills.move import turn, move_forward
from cradle.environment.rdr2.skill_registry import register_skill
from cradle.utils.image_utils import exec_clip_minimap
from cradle.utils.object_utils import circle_detector_detect
from cradle import constants

config = Config()
logger = Logger()

MAX_FOLLOW_ITERATIONS = 40


@register_skill("follow")
def follow():
    """
    Follow target on the minimap.
    """
    cv_follow_circles(MAX_FOLLOW_ITERATIONS, debug=False)


def cv_follow_circles(
        iterations,
        follow_dis_threshold=50,
        debug=False,
):
    '''
    Prioritize following the yellow circle. If not detected, then follow the gray circle.
    '''
    save_dir = config.work_dir
    follow_dis_threshold *= config.resolution_ratio

    is_move = False

    previous_distance, previous_theta, counter = 0, 0, 0
    max_q_size = 2
    minimap_image_filename_q = deque(maxlen=max_q_size)
    condition_q = deque(maxlen=max_q_size)

    for step in range(iterations):

        if debug:
            logger.write(f'Go into combat #{step}')

        if config.ocr_different_previous_text:
            logger.write("The text is different from the previous one.")
            config.ocr_enabled = False # disable ocr
            config.ocr_different_previous_text = False  # reset
            break

        timestep = time.time()
        screen_image_filename, minimap_image_filename = exec_clip_minimap(timestep,
                                                                          screen_region=config.env_region,
                                                                          minimap_region=config.minimap_region)
        minimap_image_filename_q.append(minimap_image_filename)

        adjacent_minimaps = list(minimap_image_filename_q)[::max_q_size-1] if len(minimap_image_filename_q)>=max_q_size else None

        # Find direction to follow
        follow_theta, follow_info = circle_detector_detect(minimap_image_filename, debug=debug)
        follow_dis = follow_info[constants.DISTANCE_TYPE]

        if abs(follow_theta) <= 360 and step == 0:

            turn(follow_theta)
            move_forward(1) # warm up

        if debug:
            logger.debug(
                f"step {step:03d} | timestep {timestep} done | follow theta: {follow_theta:.2f} | follow distance: {follow_dis:.2f} | follow confidence: {follow_info['confidence']:.3f}")

            cv2.circle(follow_info['vis'], follow_info['center'], 1, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(save_dir, f"minimap_{timestep}_follow_template.jpg"), follow_info['vis'])

        if debug and follow_dis < follow_dis_threshold:
            logger.write('Keep with the companion')

        if abs(follow_theta) <= 360 and step > 0:

            turn(follow_theta)

            if not is_move:
                move_forward(0.8)
                is_move = True
            else:
                move_forward(0.3)
        else:
            is_move = False

        # Check stuck
        if adjacent_minimaps:

            condition, img_matches, average_distance = minimap_movement_detection(*adjacent_minimaps, threshold = 5)

            if debug:
                cv2.imwrite(os.path.join(save_dir, f"minimap_{timestep}_bfmatch.jpg"),img_matches)

            condition_q.append(~condition)
            condition = all(condition_q)
        else:
            condition = abs(previous_distance - follow_dis) < 0.5 and abs(previous_theta - follow_theta) < 0.5

        if condition:
            if debug:
                logger.debug('Move randomly to get unstuck')

            turn(180),move_forward(np.random.randint(1, 6))
            time.sleep(1)  # improve stability
            turn(-90),move_forward(np.random.randint(1, 6))

        previous_distance, previous_theta = follow_dis, follow_theta


__all__ = [
    "follow",
]
