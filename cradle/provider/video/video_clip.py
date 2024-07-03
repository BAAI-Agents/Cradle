from typing import Any, Dict
import time

from cradle.provider import BaseProvider
from cradle.log import Logger
from cradle.memory import LocalMemory
from cradle.provider import VideoRecordProvider
from cradle import constants

logger = Logger()


class VideoClipProvider(BaseProvider):

    def __init__(self, gm):

        super(VideoClipProvider, self).__init__()

        self.gm = gm
        self.memory = LocalMemory()
        self.video_recorder = VideoRecordProvider()


    @BaseProvider.write
    def __call__(self,
                 *args,
                 init = False,
                 **kwargs):

        if init:
            start_frame_id = self.video_recorder.get_current_frame_id()
            screenshot_path = self.gm.capture_screen()
            time.sleep(2)
            end_frame_id = self.video_recorder.get_current_frame_id()
            video_clip_path = self.video_recorder.get_video(start_frame_id, end_frame_id)

            logger.write(f"Initiate video clip path from the screen shot by frame id ({start_frame_id}, {end_frame_id}).")

            res_params = {
                "video_clip_path": video_clip_path,
                "start_frame_id": start_frame_id,
                "end_frame_id": end_frame_id,
                constants.IMAGES_MEM_BUCKET: screenshot_path,
            }

        else:
            start_frame_id = self.memory.get_recent_history("start_frame_id")[-1]
            end_frame_id = self.memory.get_recent_history("end_frame_id")[-1]
            video_clip_path = self.video_recorder.get_video(start_frame_id, end_frame_id)

            logger.write(f"Get video clip path from the memory by frame id ({start_frame_id}, {end_frame_id}).")

            res_params = {
                "video_clip_path": video_clip_path,
            }

        self.memory.update_info_history(res_params)

        return res_params
