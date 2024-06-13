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
from cradle.utils.gui_utils import get_screen_size, get_active_window, get_named_windows, get_named_windows_fallback

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

    records_path = None

    # config for frame extraction
    VideoFrameExtractor_path = "./res/tool/subfinder/VideoSubFinderWXW.exe"
    VideoFrameExtractor_placeholderfile_path = "./res/tool/subfinder/test.srt"

    IDE_NAME = "PyCharm"


    def __init__(self) -> None:
        """Initialize the Config class"""

        self.debug_mode = False
        self.continuous_mode = False
        self.continuous_limit = 0

        # Max steps for each task
        self.max_steps = 999999

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
        self.parallel_rquest_gather_information = True

        self.skill_library_with_code = False

        self.parallel_request_gather_information = True

        # Video
        self.video_fps = 8
        self.duplicate_frames = 4
        self.frames_per_slice = 1000

        # Self-reflection image count
        self.max_images_in_self_reflection = 4

        # Decision-making image count
        self.action_planning_image_num = 2
        self.number_of_execute_skills = 1
        self.skill_library_with_code = False

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

        self.show_mouse_in_screenshot = False

        # Just for convenience of testing, will be removed in final version.
        self.use_latest_memory_path = False
        if self.use_latest_memory_path:
            self._set_latest_memory_path()

        self._set_dirs()


    def load_env_config(self, env_config_path):
        """Load environment specific configuration."""

        path = assemble_project_path(env_config_path)
        env_config = load_json(path)

        self.env_name = kget(env_config, constants.ENVIRONMENT_NAME, default='')
        self.win_name_pattern = kget(env_config, constants.ENVIRONMENT_WINDOW_NAME_PATTERN, default='')
        self.env_sub_path = kget(env_config, constants.ENVIRONMENT_SHORT_NAME, default='')
        self.env_short_name = kget(env_config, constants.ENVIRONMENT_SUB_PATH, default='')

        # Base resolution and region for the game in 4k, used for angle scaling
        self.base_resolution = (3840, 2160)
        self.base_minimap_region = (112, 1450, 640, 640)
        self.base_new_icon_region = (30, 2000, 70, 70)
        self.base_new_icon_name_region = (110, 2000, 75, 60)
        self.base_toolbar_region = (1520, 2055, 800, 95)
        self.selection_box_region = (1, 10, 63, 70)
        self.inventory_dict = {
            "tool_span_single": 2,
            "tool_left": 16,
            "tool_top": 9,
            "tool_width": 72,
            "tool_height": 75,
        }

        # Full screen resolution for normalizing mouse movement
        self.screen_resolution = get_screen_size()
        self.mouse_move_factor = self.screen_resolution[0] / self.base_resolution[0]

        self._set_env_window_info()


        self.task_description = env_config.get("task_description", "")

        # Skill retrieval
        self.skill_from_default = False
        self.skill_local_path = './res/' + self.env_sub_path + '/skills/'
        self.skill_num = 10
        self.skill_mode = env_config.get("skill_mode", "Basic")
        self.skill_names_basic = env_config.get("skill_names_basic", [])
        self.skill_names_movement = env_config.get("skill_names_movement", [])
        self.skill_names_map = env_config.get("skill_names_map", [])
        self.skill_names_trade = env_config.get("skill_names_trade", [])
        self.skill_names_allow = env_config.get("skill_names_allow", [])
        self.skill_names_deny = env_config.get("skill_names_deny", [])

        # Provider configs
        self.provider_configs = env_config.get("provider_configs", {})
        self.provider_configs = namedtuple('ProviderConfigs', self.provider_configs.keys())(**self.provider_configs)

        self.planner_params = env_config.get("planner_params", {})

        self._check_ide_window_info()
        self._set_env_window_info()


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

    def _check_ide_window_info(self):
        named_windows = get_named_windows(self.IDE_NAME)
        if len(named_windows) <= 0:
            ide_window = get_active_window()
            self.ide_name = ide_window.title
        else:
            self.ide_name = self.IDE_NAME


    def _set_env_window_info(self):

        # Fake target environment window info for testing cases with no running env application
        env_window = namedtuple('A', ['left', 'top', 'width', 'height'])
        env_window.left = 0
        env_window.top = 0
        env_window.width = self.DEFAULT_ENV_RESOLUTION[0]
        env_window.height = self.DEFAULT_ENV_RESOLUTION[1]

        # Get window candidates by name alternatives
        named_windows = get_named_windows_fallback(self.env_name, self.win_name_pattern)

        if len(named_windows) == 0 or len(named_windows) > 1:
            self._config_warn(f'-----------------------------------------------------------------')
            self._config_warn(f'Cannot find unique env window nor pattern: {self.env_name}|{self.win_name_pattern}. Assuming this is an offline test run!')
            self._config_warn(f'-----------------------------------------------------------------')
        else:
            env_window = named_windows[0]

            # Check if pre-resize is necessary
            if not self._min_resolution_check(env_window) or not self._aspect_ration_check(env_window):
                env_window.resizeTo(self.DEFAULT_ENV_RESOLUTION[0], self.DEFAULT_ENV_RESOLUTION[1])

            assert self._min_resolution_check(env_window), 'The resolution of env window should at least be 1920 X 1080.'
            assert self._aspect_ration_check(env_window), 'The screen ratio should be 16:9.'

        self.env_resolution = (env_window.width, env_window.height)
        self.env_region = (env_window.left, env_window.top, env_window.width, env_window.height)
        self.resolution_ratio = self.env_resolution[0] / self.base_resolution[0]
        self.minimap_region = self._calc_minimap_region(self.env_resolution)
        self.minimap_region[0] += env_window.left
        self.minimap_region[1] += env_window.top
        self.minimap_region = tuple(self.minimap_region)


    def _min_resolution_check(self, env_window):
        return env_window.width >= self.DEFAULT_ENV_RESOLUTION[0] and env_window.height >= self.DEFAULT_ENV_RESOLUTION[1]


    def _aspect_ration_check(self, env_window):
        return env_window.width * self.DEFAULT_ENV_SCREEN_RATIO[1] == env_window.height * self.DEFAULT_ENV_SCREEN_RATIO[0]


    def _calc_minimap_region(self, screen_region):
        return [int(x * self.resolution_ratio ) for x in self.base_minimap_region]

    def _cal_toolbar_region(self):
        height_subsctraction = self.base_resolution[1] * (1 - self.resolution_ratio)
        width_subsctraction = self.base_resolution[0] * (1 - self.resolution_ratio)
        top = self.base_toolbar_region[1] - height_subsctraction
        left = self.base_toolbar_region[0] - width_subsctraction / 2
        return [left, top, self.base_toolbar_region[2], self.base_toolbar_region[3]]

    def _cal_new_icon_region(self):
        height_subsctraction = self.base_resolution[1] * (1 - self.resolution_ratio)
        width_subsctraction = self.base_resolution[0] * (1 - self.resolution_ratio)
        top = self.base_new_icon_region[1] - height_subsctraction
        left = self.base_new_icon_region[0] - width_subsctraction / 2
        return [left, top, self.base_new_icon_region[2], self.base_new_icon_region[3]]

    def _cal_new_icon_name_region(self):
        height_subsctraction = self.base_resolution[1] * (1 - self.resolution_ratio)
        width_subsctraction = self.base_resolution[0] * (1 - self.resolution_ratio)
        top = self.base_new_icon_name_region[1] - height_subsctraction
        left = self.base_new_icon_name_region[0] - width_subsctraction / 2
        return [left, top, self.base_new_icon_name_region[2], self.base_new_icon_name_region[3]]


    def _config_warn(self, message):
        colours_on()
        print(Fore.RED + f' >>> WARNING: {message} ' + Style.RESET_ALL)


    def _set_latest_memory_path(self):
        path_list = os.listdir(self.work_dir)
        path_list.sort()
        if len(path_list) != 0:
            self.skill_local_path = os.path.join(self.work_dir, path_list[-1])
            self.memory_load_path = os.path.join(self.work_dir, path_list[-1])