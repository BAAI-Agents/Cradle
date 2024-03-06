import numpy as np
import cv2

from cradle.config import Config

config = Config()


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
