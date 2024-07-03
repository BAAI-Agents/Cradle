import base64
import os
from typing import Any, List
import io

import numpy as np
import cv2
from PIL import Image

from cradle.log.logger import Logger
from cradle.utils.file_utils import assemble_project_path
from cradle.utils.string_utils import hash_text_sha256

logger = Logger()


def encode_base64(payload):

    if payload is None:
        raise ValueError("Payload cannot be None.")

    return base64.b64encode(payload).decode('utf-8')


def decode_base64(payload):

    if payload is None:
        raise ValueError("Payload cannot be None.")

    return base64.b64decode(payload)


def encode_image_path(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = encode_image_binary(image_file.read(), image_path)
        return encoded_image


def encode_image_binary(image_binary, image_path=None):
    encoded_image = encode_base64(image_binary)
    if image_path is None:
        image_path = '<$bin_placeholder$>'

    logger.debug(f'|>. img_hash {hash_text_sha256(encoded_image)}, path {image_path} .<|')
    return encoded_image


def decode_image(base64_encoded_image):
    return decode_base64(base64_encoded_image)


def encode_data_to_base64_path(data: Any) -> List[str]:
    encoded_images = []

    if isinstance(data, (str, Image.Image, np.ndarray, bytes)):
        data = [data]

    for item in data:
        if isinstance(item, str):
            if os.path.exists(assemble_project_path(item)):
                path = assemble_project_path(item)
                encoded_image = encode_image_path(path)
                image_type = path.split(".")[-1].lower()
                encoded_image = f"data:image/{image_type};base64,{encoded_image}"
                encoded_images.append(encoded_image)
            else:
                encoded_images.append(item)

            continue

        elif isinstance(item, bytes):  # mss grab bytes
            image = Image.frombytes('RGB', item.size, item.bgra, 'raw', 'BGRX')
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
        elif isinstance(item, Image.Image):  # PIL image
            buffered = io.BytesIO()
            item.save(buffered, format="JPEG")
        elif isinstance(item, np.ndarray):  # cv2 image array
            item = cv2.cvtColor(item, cv2.COLOR_BGR2RGB)  # convert to RGB
            image = Image.fromarray(item)
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
        elif item is None:
            logger.error("Tring to encode None image! Skipping it.")
            continue

        encoded_image = encode_image_binary(buffered.getvalue())
        encoded_image = f"data:image/jpeg;base64,{encoded_image}"
        encoded_images.append(encoded_image)

    return encoded_images
