import threading
import os
import time

import spacy
import numpy as np
import cv2
import mss

from cradle.log import Logger
from cradle.config import Config
from cradle.provider import BaseProvider
from cradle.provider.video.video_ocr_extractor import VideoOCRExtractorProvider

config = Config()
logger = Logger()


class FrameBuffer():

    def __init__(self):
        self.queue = []
        self.lock = threading.Lock()


    def add_frame(self, frame_id, frame):
        with self.lock:
            self.queue.append((frame_id, frame))


    def get_last_frame(self):
        with self.lock:
            if len(self.queue) == 0:
                return None
            else:
                return self.queue[-1]


    def get_frame_by_frame_id(self, frame_id):
        with self.lock:
            for frame in self.queue:
                if frame[0] == frame_id:
                    return frame

        return None


    def get_frames_to_latest(self, frame_id, before_frame_nums=5):
        frames = []
        with self.lock:
            for frame in self.queue:
                if frame[0] >= frame_id - before_frame_nums and frame[0] <= frame_id:
                    frames.append(frame)

        return frames


    def clear(self):
        with self.lock:
            self.queue.clear()


    def get_frames(self, start_frame_id, end_frame_id=None):
        frames = []
        with self.lock:
            for frame in self.queue:
                if frame[0] >= start_frame_id:
                    if end_frame_id is not None and frame[0] > end_frame_id:
                        break
                    frames.append(frame)

        return frames


class VideoRecordProvider(BaseProvider):

    def __init__(self,
                 video_path: str = None,
                 ):

        super(VideoRecordProvider, self).__init__()

        if video_path is None:
            video_path = os.path.join(config.work_dir, 'video.mp4')

        self.fps = config.video_fps
        self.max_size = 10000
        self.video_path = video_path
        self.frames_per_slice = config.frames_per_slice
        self.frames_count = 0
        self.screen_region = config.env_region
        self.frame_size = (self.screen_region[2], self.screen_region[3])

        self.current_frame_id = -1
        self.current_frame = None
        self.frame_buffer = FrameBuffer()
        self.thread_flag = True

        self.thread = threading.Thread(
            target=self.capture_screen,
            args=(self.frame_buffer, ),
            name='Screen Capture'
        )
        self.thread.daemon = True

        self.video_path_dir = os.path.join(os.path.dirname(self.video_path), 'videos')
        os.makedirs(self.video_path_dir, exist_ok=True)
        self.video_splits_dir = os.path.join(os.path.dirname(self.video_path), 'video_splits')
        os.makedirs(self.video_splits_dir, exist_ok=True)

        self.nlp = spacy.load("en_core_web_lg")
        self.pre_text = None

        self.video_ocr_extractor = VideoOCRExtractorProvider()


    def get_frames(self, start_frame_id, end_frame_id = None):
        return self.frame_buffer.get_frames(start_frame_id, end_frame_id)


    def get_frames_to_latest(self, frame_id, before_frame_nums = 5):
        return self.frame_buffer.get_frames_to_latest(frame_id, before_frame_nums)


    def get_video(self, start_frame_id, end_frame_id = None):
        path = os.path.join(self.video_splits_dir, 'video_{:06d}.mp4'.format(start_frame_id))
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, self.frame_size)

        frames = self.get_frames(start_frame_id, end_frame_id)
        for frame in frames:
            writer.write(frame[1])
        writer.release()

        return path


    def clear_frame_buffer(self):
        self.frame_buffer.clear()


    def get_current_frame(self):
        """
        Get the current frame
        """
        return self.current_frame


    def get_current_frame_id(self):
        """
        Get the current frame id
        """
        return self.current_frame_id


    def capture_screen(self, frame_buffer: FrameBuffer):
        logger.write('Screen capture started')

        video_name = os.path.split(self.video_path)[1].split('.')[0]
        video_slice_path = os.path.join(self.video_path_dir, video_name + '_slice_{:06d}.mp4'.format(self.frames_count // self.frames_per_slice))
        video_writer = cv2.VideoWriter(video_slice_path,
                                       cv2.VideoWriter_fourcc(*'mp4v'),
                                       self.fps,
                                       self.frame_size)

        with mss.mss() as sct:
            region = self.screen_region
            region = {
                "left": region[0],
                "top": region[1],
                "width": region[2],
                "height": region[3],
            }

            while self.thread_flag:
                try:
                    frame = sct.grab(region)
                    frame = np.array(frame) # Convert to numpy array

                    # if config.ocr_enabled is true, start ocr and check whether the text is different from the previous one
                    if config.ocr_enabled:
                        cur_text = self.video_ocr_extractor.extract_text(frame, return_full=0)
                        cur_text = cur_text[0]
                        cur_text = " ".join(cur_text)

                        if self.pre_text is None:
                            self.pre_text = cur_text
                        else:
                            emb1 = self.nlp(self.pre_text)
                            emb2 = self.nlp(cur_text)

                            score = emb1.similarity(emb2)

                            if score < config.ocr_similarity_threshold:
                                config.ocr_different_previous_text = True
                            else:
                                config.ocr_different_previous_text = False
                            self.pre_text = cur_text

                    # if config.ocr_enabled is false, the ocr is not enabled, so the pre_text should be None
                    if not config.ocr_enabled and self.pre_text is not None:
                        self.pre_text = None

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    video_writer.write(frame)
                    self.frames_count += 1

                    if self.frames_count % self.frames_per_slice == 0:
                        # Release the previous video writer
                        video_writer.release()

                        # Create a new video writer
                        video_slice_path = os.path.join(self.video_path_dir, video_name + '_slice_{:06d}.mp4'.format(self.frames_count // self.frames_per_slice))
                        video_writer = cv2.VideoWriter(video_slice_path,
                                                       cv2.VideoWriter_fourcc(*'mp4v'),
                                                       self.fps,
                                                       self.frame_size)

                    self.current_frame = frame
                    for i in range(config.duplicate_frames):
                        self.current_frame_id += 1
                        frame_buffer.add_frame(self.current_frame_id, frame)
                    time.sleep(config.duplicate_frames / config.video_fps - 0.05) # 0.05: time for taking a screenshots

                    # Check the flag at regular intervals
                    if not self.thread_flag:
                        break

                except KeyboardInterrupt:
                    logger.write('Screen capture interrupted')
                    self.finish_capture()

            video_writer.release()


    def start_capture(self):
        self.thread.start()


    def finish_capture(self):
        if not self.thread.is_alive():
            logger.write('Screen capture thread is not executing')
        else:
            self.thread_flag = False  # Set the flag to False to signal the thread to stop
            self.thread.join()  # Now we wait for the thread to finish
            logger.write('Screen capture finished')
