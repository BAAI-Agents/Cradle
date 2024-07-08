import os
import time

import numpy as np
import cv2
import torch
from torchvision.ops import box_convert

from cradle.config import Config
from cradle.log import Logger
from cradle.gameio.io_env import IOEnvironment
from cradle.environment.rdr2.atomic_skills.combat import aim, shoot
from cradle.environment.rdr2.atomic_skills.move import turn
from cradle.environment.rdr2.skill_registry import register_skill
from cradle.utils.image_utils import exec_clip_minimap
from cradle import constants

config = Config()
logger = Logger()
io_env = IOEnvironment()

if config.is_game == True:
    try:
        from groundingdino.util.inference import annotate
    except:
        pass

    from cradle.utils.object_utils import groundingdino_detect, circle_detector_detect

DEFAULT_MAX_SHOOTING_ITERATIONS = 100
SHOOT_PEOPLE_TARGET_NAME = "person"
SHOOT_WOLVES_TARGET_NAME = "wolf"
CONTINUE_NO_ENEMY_FREQ = 5


@register_skill("shoot_people")
def shoot_people():
    """
    Shoot at person-shaped targets, if necessary.
    """
    keep_shooting_target(DEFAULT_MAX_SHOOTING_ITERATIONS, detect_target=SHOOT_PEOPLE_TARGET_NAME, debug=False)


@register_skill("shoot_wolves")
def shoot_wolves():
    """
    Shoot at wolf targets, if necessary.
    """
    keep_shooting_target(DEFAULT_MAX_SHOOTING_ITERATIONS, detect_target=SHOOT_WOLVES_TARGET_NAME, debug=False)


def keep_shooting_target(
        iterations,
        detect_target="wolf",
        debug=True
):
    '''
    Keep shooting the 'detect_target' detected by object detector automatically.
    '''
    POST_WAIT_TIME = 0.1

    save_dir = config.work_dir

    aim()  # aim before detection

    terminal_flags = []
    for step in range(1, 1 + iterations):

        if debug:
            logger.debug(f'Go into combat #{step}')

        if config.ocr_different_previous_text:
            logger.write("The text is different from the previous one.")
            config.ocr_enabled = False # disable ocr
            config.ocr_different_previous_text = False # reset
            break

        timestep = time.time()

        screen_image_filename, minimap_image_filename = exec_clip_minimap(timestep,
                                                                          screen_region=config.env_region,
                                                                          minimap_region=config.minimap_region)
        screen = cv2.imread(screen_image_filename)
        h, w, _ = screen.shape

        # center pointer
        pointer = np.array([h // 2,  w // 2])
        red_range=np.array([[0, 0, 150], [100, 100, 255]])

        is_red = cv2.countNonZero(cv2.inRange(screen[pointer[0],pointer[1]].reshape(1,1,3), red_range[0], red_range[1]))
        if is_red:
            shoot(0.5,0.5)
            time.sleep(POST_WAIT_TIME)
            continue

        if not detect_target.endswith(' .'):
            detect_target += ' .'
        _, boxes, logits, phrases = groundingdino_detect(screen_image_filename, detect_target, box_threshold=0.4)

        # enemy detection
        follow_theta, follow_info = circle_detector_detect(minimap_image_filename,
                                                      detect_mode='red',
                                                      debug=debug)
        logger.debug(f'turn: {follow_theta}')

        if abs(follow_theta)<=360:
            follow_theta = np.sign(follow_theta) * np.clip(abs(follow_theta),0,180)
            terminal_flags.append(0)
        else:
            terminal_flags.append(1)
            # if sum(terminal_flags[-CONTINUE_NO_ENEMY_FREQ:]) == CONTINUE_NO_ENEMY_FREQ:
            #     logger.debug(f'From step {step} to {step-CONTINUE_NO_ENEMY_FREQ} no enemy detected! Shooting is terminated.')
            #     return

        if debug:
            cv2.imwrite(os.path.join(save_dir, f"red_detect_{timestep}.jpg"), follow_info['vis'])

        if not phrases:
            if abs(follow_theta)<=360:
                turn(follow_theta)
                time.sleep(POST_WAIT_TIME)
            continue

        # sort according to areas
        areas = [(b[2]*b[3]).item() for b in boxes]
        area_ascend_index = np.argsort(areas)
        boxes = torch.stack([boxes[i] for i in area_ascend_index])
        logits = torch.stack([logits[i] for i in area_ascend_index])
        phrases = [phrases[i] for i in area_ascend_index]

        if SHOOT_PEOPLE_TARGET_NAME in detect_target.lower():

            if len(boxes) > 1:

                index = 0
                dis = 1.5

                for i in range(len(boxes)):
                    down_mid = (boxes[i, 0], boxes[i, 1] + boxes[i, 3] / 2)
                    distance = torch.sum(torch.abs(torch.tensor(down_mid) - torch.tensor((0.5, 1.0))))

                    if distance < dis:
                        dis = distance
                        index = i

                boxes = torch.cat([boxes[:index], boxes[index + 1:]])
                logits = torch.cat([logits[:index], logits[index + 1:]])
                phrases.pop(index)
                logger.debug(f'dis:{dis}  remove{index}')

            elif len(boxes) == 1:
                boxes = torch.tensor(boxes[1:])
                logits = torch.tensor(logits[1:])
                phrases.pop(0)

        if debug:
            annotated_frame = annotate(image_source=screen, boxes=boxes, logits=logits, phrases=phrases)
            cv2.imwrite(os.path.join(save_dir, f"annotated_{timestep}.jpg"), annotated_frame)


        xyxy = box_convert(boxes=boxes * torch.Tensor([w, h, w, h]), in_fmt="cxcywh", out_fmt="xyxy").numpy().astype(int)
        is_shoot = False

        for j, (detect_xyxy, detect_object, detect_confidence) in enumerate(zip(xyxy, phrases, logits)):

            if debug:
                logger.debug(f'detect_xyxy is {detect_xyxy},detect_object is {detect_object},shoot_xy is {int((detect_xyxy[0] + detect_xyxy[2]) / 2)},{int((detect_xyxy[1] + detect_xyxy[3]) / 2)}')

            # exclude the person occupied large area (threshold: 0.1)
            s_w = constants.SHOOT_PEOPLE_TARGET_NAME in detect_object.lower()  # true represents shoot wolves
            if s_w and boxes[j][2] * boxes[j][3] > 0.06:
                continue

            if detect_object in detect_target:

                shoot_x = boxes[j][0]
                shoot_y = boxes[j][1]

                if debug:
                    cv2.arrowedLine(annotated_frame, (config.env_resolution[0] // 2, config.env_resolution[1] // 2), (
                        int((detect_xyxy[0] + detect_xyxy[2]) / 2), int((detect_xyxy[1] + detect_xyxy[3]) / 2)),(0, 255, 0), 2, tipLength=0.1)
                    cv2.imwrite(os.path.join(save_dir, f"annotated_{detect_object}_{timestep}.jpg"), annotated_frame)

                logger.debug(f'pixel is {shoot_x},{shoot_y}')
                shoot(shoot_x, shoot_y)
                time.sleep(POST_WAIT_TIME)
                is_shoot = True
                break

        if not is_shoot or (is_shoot and np.random.uniform(0,1) < .2): # turn

            follow_theta, follow_info = circle_detector_detect(minimap_image_filename,
                                                          detect_mode='red',
                                                          debug=debug)
            logger.debug(f'turn: {follow_theta}')

            if abs(follow_theta)<=360:
                follow_theta = np.sign(follow_theta) * np.clip(abs(follow_theta),0,180)
                turn(follow_theta)
                time.sleep(POST_WAIT_TIME)
            if debug:
                cv2.imwrite(os.path.join(save_dir, f"red_detect_{timestep}.jpg"), follow_info['vis'])

__all__ = [
    "shoot_people",
    "shoot_wolves"
]
