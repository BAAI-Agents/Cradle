import math

import cv2
import numpy as np

from cradle.config.config import Config
from cradle.provider.base import BaseProvider


config = Config()

class CircleDetectProvider(metaclass=BaseProvider):

    resolution_ratio = None

    def __init__(self):
        super(CircleDetectProvider, self).__init__()
        self.sr_model = None
        self.k = None


    def check_init_model(self, resolution_ratio):
        if resolution_ratio <= .67:  # need super resolution
            self.sr_model = cv2.dnn_superres.DnnSuperResImpl_create()
            self.k = 2 if resolution_ratio <=.5 else 3
            self.sr_model.readModel(f'./res/models/ESPCN_x{self.k}.pb')
            self.sr_model.setModel('espcn', self.k)
        else:
            self.sr_model = None

        self.resolution_ratio = resolution_ratio


    def get_theta(self, origin_x, origin_y, center_x, center_y):
        '''
        The origin of the image coordinate system is usually located in the upper left corner of the image, with the x-axis to the right indicating a positive direction and the y-axis to the down indicating a positive direction. Using vertical upward as the reference line, i.e. the angle between it and the negative direction of the y-axis
        '''
        theta = math.atan2(center_x - origin_x, origin_y - center_y)
        theta = math.degrees(theta)
        return theta


    def run(self,
            img_file,
            yellow_range=np.array([[140, 230, 230], [170, 255, 255]]),
            gray_range=np.array([[165, 165, 165], [175, 175, 175]]),
            red_range=np.array([[0, 0, 170], [30, 30, 240]]),
            detect_mode='yellow & gray',
            debug=False
            ):

        image = cv2.imread(img_file)

        # Init SR model?
        if self.resolution_ratio is None:
            self.check_init_model(config.resolution_ratio)

        # Super resolution according to resolution ratio
        if self.sr_model is not None:
            image = self.sr_model.upsample(image)
            if self.k == 3:
                image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

        origin = (image.shape[0] // 2, image.shape[1] // 2)
        circles = cv2.HoughCircles(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.HOUGH_GRADIENT, 1, 10, param1=200,param2=10, minRadius=5 * 2, maxRadius=8 * 2)
        theta = 0x3f3f3f3f
        measure = {'theta': theta, 'distance': theta, 'color': np.array([0, 0, 0]), 'confidence': 0, 'vis': image,
                   'center': origin}

        circles_info = []
        if circles is not None:

            circles = np.round(circles[0, :]).astype("int")

            for (x, y, r) in circles:

                # Crop the circle from the original image
                circle_img = np.zeros_like(image)
                cv2.circle(circle_img, (x, y), r, (255, 255, 255), thickness=-1)
                circle = cv2.bitwise_and(image, circle_img)

                # Define range for red color and create a mask
                red_mask = cv2.inRange(circle, red_range[0], red_range[1])
                gray_mask = cv2.inRange(circle, gray_range[0], gray_range[1])
                yellow_mask = cv2.inRange(circle, yellow_range[0], yellow_range[1])

                # Count red pixels in the circle
                red_count = cv2.countNonZero(red_mask)
                gray_count = cv2.countNonZero(gray_mask)
                yellow_count = cv2.countNonZero(yellow_mask)

                # Add circle information and color counts to the list
                circles_info.append({
                    "center": (x, y),
                    "radius": r,
                    "red_count": red_count,
                    "gray_count": gray_count,
                    "yellow_count": yellow_count
                })

            # Sort the circles based on yellow_count, gray_count, and red_count
            if 'red' in detect_mode:
                circles_info.sort(key=lambda c: (c['red_count'], c['yellow_count'], c['gray_count']), reverse=True)
                detect_criterion = lambda circle: circle["red_count"] >= 5
            else:
                circles_info.sort(key=lambda c: (c['yellow_count'], c['gray_count'], c['red_count']), reverse=True)
                detect_criterion = lambda circle: circle["gray_count"] >= 5 or circle["yellow_count"] >= 5

            for circle in circles_info:

                center_x, center_y, radius = circle["center"][0], circle["center"][1], circle["radius"]

                if detect_criterion(circle):
                    theta = self.get_theta(*origin, center_x, center_y)
                    dis = np.sqrt((center_x - origin[0]) ** 2 + (center_y - origin[1]) ** 2)
                    measure = {'theta': theta, 'distance': dis,
                               'color': "yellow" if circle["yellow_count"] >= 5 else "gray", 'confidence': 1,
                               'center': (center_x, center_y),
                               'bounding_box': (center_x - radius, center_y - radius, 2 * radius, 2 * radius)}
                    break

            if debug:
                for i, circle in enumerate(circles_info):
                    cv2.circle(image, circle["center"], circle["radius"], (0, 255, 0), 2)
                    cv2.putText(image, str(i + 1), (circle["center"][0] - 5, circle["center"][1] + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                measure['vis'] = image

        return theta, measure
