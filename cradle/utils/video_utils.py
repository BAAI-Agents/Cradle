import io

import cv2
from PIL import Image

from cradle.config import Config
from cradle.gameio import IOEnvironment
from cradle.log import Logger

config = Config()
io_env = IOEnvironment()
logger = Logger()


def get_basic_info(video_path: str):
    """
    Get basic information from video.
    """
    video = cv2.VideoCapture(video_path)

    fps = video.get(cv2.CAP_PROP_FPS)
    fps = int(fps)

    frames = []

    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        img = Image.open(io.BytesIO(buffer.tobytes()))
        frames.append(img)
        #base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    video.release()

    return frames, fps
