import os
from typing import Any, List, Tuple
import time

import numpy as np
import cv2
import easyocr
import PIL
from PIL import Image

# Hack to avoid EasyOCR crash
PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

from cradle.log import Logger
from cradle.config import Config
from cradle.provider import BaseProvider
from cradle.utils.encoding_utils import decode_image
from cradle.utils.file_utils import assemble_project_path

logger = Logger()
config = Config()


class VideoOCRExtractorProvider(BaseProvider):

    def __init__(self):
        super(VideoOCRExtractorProvider, self).__init__()

        self.crop_region = config.DEFAULT_OCR_CROP_REGION
        self.reader = easyocr.Reader(['en'])


    def to_images(self, data: Any) -> Any:

        images = []

        if isinstance(data, (str, Image.Image, np.ndarray, bytes)):
            data = [data]

        for image in data:
            if isinstance(image, str): # path to cv2 image
                if os.path.exists(assemble_project_path(image)):
                    path = assemble_project_path(image)
                    image = cv2.imread(path)
                else: # base64 to cv2 image
                    image_data = decode_image(image)
                    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

            elif isinstance(image, bytes):  # bytes to cv2 image
                image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
            elif isinstance(image, Image.Image):  # PIL to cv2 image
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            elif isinstance(image, np.ndarray):  # cv2 image
                if len(image.shape) == 3 and image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

            # crop the image
            if self.crop_region is not None:
                x1, y1, x2, y2 = self.crop_region
                image = image[y1:y2, x1:x2]

            images.append(image)
        return images


    def extract_text(self, image: Any, return_full: int = 1) -> List[Any]:
        images = self.to_images(image)
        res = []
        for image in images:
            # if full, return the (bounding box, text, prob) tuple
            # else, return text only
            item = self.reader.readtext(image, detail=return_full)
            res.append(item)
        return res


    def extract_text_from_video(self, video_path: str, return_full: int = 1) -> List[Any]:

        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                break
            # Break the loop with 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

        res = self.extract_text(frames, return_full=return_full)
        return res


    def extract_text_from_frames(self, frames: List, return_full: int = 1) -> List[Any]:
        res = self.extract_text(frames, return_full=return_full)
        return res


    def detect_text(self, image: Any) -> Tuple[List[Any], List[bool]]:
        images = self.to_images(image)
        bounding_boxes = []
        for image in images:
            item = self.reader.detect(image)
            item = item[0][0] # list of bounding boxes, (x, y, w, h)
            bounding_boxes.append(item)

        has_text_flag = [True if len(item) >0 else False for item in bounding_boxes ]
        return bounding_boxes, has_text_flag


    def detect_text_from_video(self, video_path: str) -> Tuple[List[Any], List[bool]]:

        cap = cv2.VideoCapture(video_path)
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                break

            # Break the loop with 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

        bounding_boxes, has_text_flag = self.detect_text(frames)

        return bounding_boxes, has_text_flag


    def detect_text_from_frames(self, frames: List) -> Tuple[List[Any], List[bool]]:
        bounding_boxes, has_text_flag = self.detect_text(frames)
        return bounding_boxes, has_text_flag


    def run(self, *args, data: Any, **kwargs) -> Any:
        return self.detect_text(data)
