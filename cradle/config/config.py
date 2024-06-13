from collections import namedtuple
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from colorama import Fore, Style, init as colours_on

import cradle
from cradle import constants
from cradle.utils.json_utils import load_json
from cradle.utils import Singleton
from cradle.utils.file_utils import assemble_project_path, get_project_root
from cradle.utils.dict_utils import kget

load_dotenv(verbose=True)


class Config(metaclass=Singleton):
    """
    Configuration class.
    """

    DEFAULT_ENV_RESOLUTION = (1920, 1080)
    DEFAULT_ENV_SCREEN_RATIO = (16, 9)

    DEFAULT_TEMPERATURE = 1.0
    DEFAULT_SEED = None

    DEFAULT_FIXED_SEED_VALUE = 42
    DEFAULT_FIXED_TEMPERATURE_VALUE = 0.0

    DEFAULT_POST_ACTION_WAIT_TIME = 3 # Currently in use in multiple places with this value

    DEFAULT_MESSAGE_CONSTRUCTION_MODE = constants.MESSAGE_CONSTRUCTION_MODE_TRIPART
    DEFAULT_OCR_CROP_REGION = (380, 720, 1920, 1080) # x1, y1, x2, y2, from top left to bottom right

    root_dir = '.'
    work_dir = './runs'
    log_dir = './logs'

    # env name
    env_name = "-"
    env_sub_path = "-"
    env_short_name = "-"

    # config for frame extraction
    VideoFrameExtractor_path = "./res/tool/subfinder/VideoSubFinderWXW.exe"
    VideoFrameExtractor_placeholderfile_path = "./res/tool/subfinder/test.srt"


    def __init__(self) -> None:
        """Initialize the Config class"""

        self.debug_mode = False
        self.continuous_mode = False
        self.continuous_limit = 0

        self.temperature = self.DEFAULT_TEMPERATURE
        self.seed = self.DEFAULT_SEED
        self.fixed_seed = False

        if self.fixed_seed:
            self.set_fixed_seed()

        # Default LLM parameters
        self.temperature = float(os.getenv("TEMPERATURE", self.temperature))
        self.max_tokens = int(os.getenv("MAX_TOKENS", "1024"))

        # Memory parameters
        self.memory_backend = os.getenv("MEMORY_BACKEND", "local")
        self.max_recent_steps = 5
        self.event_count = 5
        self.memory_load_path = None

        # Parallel request to LLM parameters
        self.parallel_request_gather_information = True

        # Video
        self.video_fps = 8
        self.duplicate_frames = 4
        self.frames_per_slice = 400

        # Self-reflection image count
        self.max_images_in_self_reflection = 4
        self.self_reflection_image_num = None

        # Decision-making image count
        self.decision_making_image_num = 2

        # Max turn count for process
        self.max_turn_count = 50

        # OCR local checks
        self.ocr_fully_ban = True # whether to fully turn-off OCR checks
        self.ocr_enabled = False # whether to enable OCR during composite skill loop
        self.ocr_similarity_threshold = 0.9  # cosine similarity, smaller than this threshold the text is considered to be different
        self.ocr_different_previous_text = False # whether the text is different from the previous one
        self.ocr_check_composite_skill_names = [
            "shoot_people",
            "shoot_wolves",
            "follow",
            "go_to_horse",
            "navigate_path"
        ]

        # SAM2SOM parameters
        self.use_sam_flag = True
        self.sam_model_name = "default"

        # Default parameters. Can be updated by environment specific configuration.
        self.sam_pred_iou_thresh = 0.4
        self.sam_resize_ratio = 0.4
        self.sam_contrast_level = 0.6
        self.sam_max_area = 0.10
        self.plot_bbox_multi_color = True
        self.min_resom_area = 2000
        self.max_intersection_rate = 0.70
        self.min_bbox_area = 100

        self.show_mouse_in_screenshot = False
        self.som_padding_size = 25
        self.som_padding_color = (0, 0, 0)

        # Disable close app icon flag
        self.disable_close_app_icon = False  # Always enabled by default

        # Enable video capture
        self.enable_videocapture = False

        # Parameters for comparing images
        self.pixel_diff_threshold = 100

        # Just for convenience of testing, should be removed in final version.
        self.use_latest_memory_path = False
        if self.use_latest_memory_path:
            self._set_latest_memory_path()

        self._set_dirs()

        self.env_window = None  # Default value init for window handle


    def load_env_config(self, env_config_path):
        """Load environment specific configuration."""

        path = assemble_project_path(env_config_path)
        self.env_config = load_json(path)

        self.env_name = kget(self.env_config, constants.ENVIRONMENT_NAME, default='')
        self.win_name_pattern = kget(self.env_config, constants.ENVIRONMENT_WINDOW_NAME_PATTERN, default='')
        self.env_sub_path = kget(self.env_config, constants.ENVIRONMENT_SHORT_NAME, default='')
        self.env_short_name = kget(self.env_config, constants.ENVIRONMENT_SUB_PATH, default='')

        # Base resolution and region for the game in 4k, used for angle scaling
        self.base_resolution = (3840, 2160)
        self.base_minimap_region = (112, 1450, 640, 640)

        # Full screen resolution for normalizing mouse movement
        self.screen_resolution = cradle.gameio.gui_utils.get_screen_size()
        self.mouse_move_factor = self.screen_resolution[0] / self.base_resolution[0]

        self._set_env_window_info()

        # Skill retrieval
        self.skill_from_local = True
        self.skill_local_path = './res/' + self.env_sub_path + '/skills/'
        self.skill_retrieval = False
        self.skill_num = 20  # 10
        self.skill_scope = 'Full' #'Full', 'Basic', and None

        # SAM2SOM parameters for specific environment
        default_sam2som_config = {
            constants.SAM_PRED_IOU_THRESH: self.sam_pred_iou_thresh,
            constants.SAM_RESIZE_RATIO: self.sam_resize_ratio,
            constants.SAM_CONTRAST_LEVEL: self.sam_contrast_level,
            constants.SAM_MAX_AREA: self.sam_max_area,
            constants.PLOT_BBOX_MULTI_COLOR: self.plot_bbox_multi_color,
            constants.DISABLE_CLOSE_APP_ICON: self.disable_close_app_icon,
        }

        sam2som_config = kget(self.env_config, constants.SAM2SOM_CONFIG, default = None)
        if sam2som_config is not None:
            for key in default_sam2som_config.keys():
                sam2som_config[key] = kget(sam2som_config, key, default=default_sam2som_config[key])
        else:
            sam2som_config = default_sam2som_config

        self.sam_pred_iou_thresh = sam2som_config[constants.SAM_PRED_IOU_THRESH]
        self.sam_resize_ratio = sam2som_config[constants.SAM_RESIZE_RATIO]
        self.sam_contrast_level = sam2som_config[constants.SAM_CONTRAST_LEVEL]
        self.sam_max_area = sam2som_config[constants.SAM_MAX_AREA]
        self.plot_bbox_multi_color = sam2som_config[constants.PLOT_BBOX_MULTI_COLOR]
        self.disable_close_app_icon = sam2som_config[constants.DISABLE_CLOSE_APP_ICON]


    def set_env_name(self, env_name: str) -> None:
        """Set the environment name."""
        self.env_name = env_name


    def set_fixed_seed(self, is_fixed: bool = True, seed: int = DEFAULT_FIXED_SEED_VALUE, temperature: float = DEFAULT_FIXED_TEMPERATURE_VALUE) -> None:
        """Set the fixed seed values. By default, used the default values. Please avoid using different values."""
        self.fixed_seed = is_fixed
        self.seed = seed
        self.temperature = temperature


    def set_continuous_mode(self, value: bool) -> None:
        """Set the continuous mode value."""
        self.continuous_mode = value


    def _set_dirs(self) -> None:
        """Setup directories needed for one system run."""
        self.root_dir = get_project_root()

        self.work_dir = assemble_project_path(os.path.join(self.work_dir, str(time.time())))
        Path(self.work_dir).mkdir(parents=True, exist_ok=True)

        self.log_dir = os.path.join(self.work_dir, self.log_dir)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)


    def _set_env_window_info(self):

        # Fake target environment window info for testing cases with no running env application
        env_window = namedtuple('A', ['left', 'top', 'width', 'height'])
        env_window.left = 0
        env_window.top = 0
        env_window.width = self.DEFAULT_ENV_RESOLUTION[0]
        env_window.height = self.DEFAULT_ENV_RESOLUTION[1]

        # Get window candidates by name alternatives
        named_windows = cradle.gameio.gui_utils.get_named_windows_fallback(self.env_name, self.win_name_pattern)

        if len(named_windows) == 0:
            self._config_warn(f'-----------------------------------------------------------------')
            self._config_warn(f'Cannot find env window: {self.env_name}|{self.win_name_pattern}. Assuming this is an offline test run!')
            self._config_warn(f'-----------------------------------------------------------------')
        elif len(named_windows) > 1:
            env_window = cradle.gameio.lifecycle.ui_control.select_window(named_windows)
        else:
            env_window = named_windows[0]

        cradle.gameio.gui_utils.check_window_conditions(env_window)

        self.env_window = env_window
        self.env_resolution = (env_window.width, env_window.height)
        self.env_region = (env_window.left, env_window.top, env_window.width, env_window.height)
        self.resolution_ratio = self.env_resolution[0] / self.base_resolution[0]
        self.minimap_region = self._calc_minimap_region(self.env_resolution)
        self.minimap_region[0] += env_window.left
        self.minimap_region[1] += env_window.top
        self.minimap_region = tuple(self.minimap_region)


    def _min_resolution_check(self, env_window):
        return env_window.width == self.DEFAULT_ENV_RESOLUTION[0] and env_window.height == self.DEFAULT_ENV_RESOLUTION[1]


    def _aspect_ration_check(self, env_window):
        return env_window.width * self.DEFAULT_ENV_SCREEN_RATIO[1] == env_window.height * self.DEFAULT_ENV_SCREEN_RATIO[0]


    def _calc_minimap_region(self, screen_region):
        return [int(x * self.resolution_ratio) for x in self.base_minimap_region]


    def _config_warn(self, message):
        colours_on()
        print(Fore.RED + f' >>> WARNING: {message} ' + Style.RESET_ALL)


    def _set_latest_memory_path(self):
        path_list = os.listdir(self.work_dir)
        path_list.sort()
        if len(path_list) != 0:
            self.skill_local_path = os.path.join(self.work_dir, path_list[-1])
            self.memory_load_path = os.path.join(self.work_dir, path_list[-1])
