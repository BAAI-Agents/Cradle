import os

import cv2
import numpy as np
from MTM import matchTemplates

from cradle.config import Config

config = Config()


class IconReplacer:

    def __init__(self, template_path = f'./res/{config.env_sub_path}/icons/keys'):

        if '/-/' in template_path:
            template_path = f'./res/{config.env_sub_path}/icons/keys'

        self.template_paths = [os.path.join(template_path, filename) for filename in os.listdir(template_path)]


    def __call__(self, image_paths):
        return self.replace_icon(image_paths)


    def _drawBoxesOnRGB(self, image, tableHit, boxThickness=2, boxColor=(255, 255, 00), showLabel=False, labelColor=(255, 255, 0), labelScale=0.5):
        """
        Return a copy of the image with predicted template locations as bounding boxes overlaid on the image
        The name of the template can also be displayed on top of the bounding box with showLabel=True

        Parameters
        ----------
        - image  : image in which the search was performed

        - tableHit: list of hit as returned by matchTemplates or findMatches

        - boxThickness: int
                        thickness of bounding box contour in pixels
        - boxColor: (int, int, int)
                    RGB color for the bounding box

        - showLabel: Boolean
                    Display label of the bounding box (field TemplateName)

        - labelColor: (int, int, int)
                    RGB color for the label

        Returns
        -------
        outImage: RGB image
                original image with predicted template locations depicted as bounding boxes
        """
        # Convert Grayscale to RGB to be able to see the color bboxes
        if image.ndim == 2:
            outImage = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) # convert to RGB to be able to show detections as color box on grayscale image
        else:
            outImage = image.copy()

        for _, row in tableHit.iterrows():

            x,y,w,h = row['BBox']
            text = row['TemplateName']

            if showLabel:
                text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, labelScale, 1)
                text_width, text_height = text_size

                rectangle_pos = [(int(x - 0.2 * w), int(y - 0.15 * h)), (int(x + 1.2 * w), int(y + 1.05 * h))]
                cv2.rectangle(outImage, rectangle_pos[0], rectangle_pos[1], color=boxColor, thickness=-1)

                text_x = int((rectangle_pos[0][0] + rectangle_pos[1][0]) / 2 - text_width / 2)
                text_y = int((rectangle_pos[0][1] + rectangle_pos[1][1]) / 2 + text_height / 2)
                cv2.putText(outImage, text=text, org=(text_x, text_y), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=labelScale, color=labelColor, lineType=cv2.LINE_AA,thickness=1)

        return outImage


    def _get_mtm_match(self, image: np.ndarray, template: np.ndarray, template_name):
        detection = matchTemplates([(template_name, cv2.resize(template, (round(template.shape[1] * s), round(template.shape[0] * s)))) for s in [0.9, 1, 1.1]],
                                image,
                                N_object=1,
                                method=cv2.TM_CCOEFF_NORMED,
                                maxOverlap=0.1)

        if detection['Score'].iloc[0] > 0.75:
            image = self._drawBoxesOnRGB(image, detection, boxThickness=-1, showLabel=True, boxColor=(255, 255, 255), labelColor=(0, 0, 0), labelScale=.62)

        return {'info': detection, 'vis': image}


    def _show(self, image, window_name='screen',show=True,save=''):
        if save:
            cv2.imwrite(save, image)
        if show:
            cv2.namedWindow(window_name, 0)
            cv2.imshow(window_name, image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


    # Image augmentation to mitigate VLM issues
    def replace_icon(self, image_paths):

        replaced_image_paths = []

        for image_path in image_paths:
            image = cv2.imread(image_path)

            for template_file in self.template_paths:
                template = cv2.imread(template_file)
                template_name = os.path.splitext(os.path.basename(template_file))[0]

                if 'left_mouse' in template_name:
                    template_name = 'LM'
                elif 'right_mouse' in template_name:
                    template_name = 'RM'
                elif 'mouse' in template_name:
                    template_name = 'MS'
                elif 'enter' in template_name:
                    template_name = 'Ent'

                detection = self._get_mtm_match(image, template, template_name)
                image = detection['vis']

            directory, filename = os.path.split(image_path)
            save_path = os.path.join(directory, "icon_replace_"+filename)

            self._show(image, save=save_path, show=False)

            replaced_image_paths.append(save_path)

        return replaced_image_paths
