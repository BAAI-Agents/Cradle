import math

import cv2
import torch
import numpy as np
from torchvision.ops import box_convert
from groundingdino.util.inference import load_model, load_image

from cradle.gameio.lifecycle.ui_control import annotate_with_coordinates, segment_minimap
from cradle.utils import Singleton
from cradle.log import Logger
from cradle import constants

logger = Logger()


def unique_predict(
        model,
        image: torch.Tensor,
        caption: str,
        box_threshold: float,
        device: str = "cuda",
):

    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption = caption + " ."

    model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])

    prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # prediction_logits.shape = (nq, 256)
    prediction_boxes = outputs["pred_boxes"].cpu()[0]  # prediction_boxes.shape = (nq, 4)

    mask = prediction_logits.max(dim=1)[0] > box_threshold
    logits = prediction_logits[mask]  # logits.shape = (n, 256)
    boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)

    # Modified version: recognize seperation and choose the best one with highest probability
    phrases = []
    input_text = caption.split()
    for logit in logits:
        prob = logit[logit > 0][1:-1]
        max_prob, cum_prob, pre_i, label = 0, 0, 0, ''
        for i, (c, p) in enumerate(zip(input_text, prob)):
            if c == '.':
                if cum_prob > max_prob:
                    max_prob = cum_prob
                    label = ' '.join(input_text[pre_i:i])
                cum_prob = 0
                pre_i = i + 1
            else:
                cum_prob += p
        phrases.append(label)

    return boxes, logits.max(dim=1)[0], phrases


class GdProvider(metaclass=Singleton):

    def __init__(self):

        self.detect_model = None

        try:
            self.detect_model = load_model("./cache/GroundingDINO_SwinB_cfg.py", "./cache/groundingdino_swinb_cogcoor.pth")
        except Exception as e:
            logger.error(f"Failed to load the grounding model. Make sure you follow the instructions on README to download the necessary files.\n{e}")


    def detect(self, image_path,
                  text_prompt="wolf .",
                  box_threshold=0.4,
                  device='cuda',
                  ):

        image_source, image = load_image(image_path)

        boxes, logits, phrases = unique_predict(
            model=self.detect_model,
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            device=device
        )

        return image_source, boxes, logits, phrases


    def save_annotate_frame(self, image_source, boxes, logits, phrases, text_prompt, cur_screen_shot_path):

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
                cv2.imwrite(cur_screen_shot_path, annotated_frame)

            elif len(boxes)==1:

                phrases.pop(0)
                boxes_ = torch.tensor(boxes[1:])
                logits_ = torch.tensor(logits[1:])

                annotated_frame = annotate_with_coordinates(image_source=image_source, boxes=boxes_[:,:], logits=logits_[:], phrases=phrases)
                cv2.imwrite(cur_screen_shot_path, annotated_frame)
            else:
                annotated_frame = annotate_with_coordinates(image_source=image_source, boxes=boxes[:,:], logits=logits[:], phrases=phrases)
                cv2.imwrite(cur_screen_shot_path, annotated_frame)

        else:
            annotated_frame = annotate_with_coordinates(image_source=image_source, boxes=boxes[:,:], logits=logits[:], phrases=phrases)
            cv2.imwrite(cur_screen_shot_path, annotated_frame)


    # Process current minimap for detect red points, yellow points and yellow region. return the angle to turn.
    def process_minimap_targets(self, image_path):

        minimap_image, boxes, logits, phrases = self.detect(image_path=segment_minimap(image_path),
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
