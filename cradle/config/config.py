from collections import namedtuple
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from colorama import Fore, Style, init as colours_on

import cradle
from cradle import constants
from cradle.log.logger import Logger
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

    # Env parameters
    env_name = "-"
    env_sub_path = "-"
    env_short_name = "-"
    env_shared_runner = None
    is_game = False

    # Dev parameters
    ide_name = os.getenv("IDE_NAME", "")
    auto_ide_switch = False

    # Skill retrieval defaults
    skill_configs = {
        constants.SKILL_CONFIG_FROM_DEFAULT: True,
        constants.SKILL_CONFIG_RETRIEVAL: False,
        constants.SKILL_CONFIG_MAX_COUNT: 20,
        constants.SKILL_CONFIG_MODE: constants.SKILL_LIB_MODE_FULL, # FULL, BASIC, or NONE
        constants.SKILL_CONFIG_NAMES_DENY: [],
        constants.SKILL_CONFIG_NAMES_ALLOW: [],
        constants.SKILL_CONFIG_NAMES_BASIC: [],
        constants.SKILL_CONFIG_NAMES_OTHERS: None,
        constants.SKILL_CONFIG_LOCAL_PATH: None,
        constants.SKILL_CONFIG_REGISTERED_SKILLS: None,
    }

    skill_library_with_code = False

    # Config for frame extraction
    VideoFrameExtractor_path = "./res/tool/subfinder/VideoSubFinderWXW.exe"
    VideoFrameExtractor_placeholderfile_path = "./res/tool/subfinder/test.srt"


    def __init__(self) -> None:
        """Initialize the Config class"""

        self.debug_mode = False
        self.continuous_mode = False
        self.continuous_limit = 0

        # Max steps for each task
        self.max_turn_count = 999999
        self.checkpoint_interval = 1

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

        # Change max steps, Self-reflection image count for software
        self.max_turn_count = 50 if self.is_game == False else 999999
        self.max_images_in_self_reflection = 2 if self.is_game == False else 4

        # SAM2SOM parameters
        self.use_sam_flag = True # @TODO load from config in augmentation configs?
        self.sam_model_name = "default"

        # Default parameters. Can be updated by environment specific configuration.
        self.sam2som_mode = constants.SAM2SOM_DEFAULT_MODE # SAM2SOM_DEFAULT_MODE for only use sam, SAM2SOM_OCR_MODE for sam combine ocr
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

        # @TODO To be moved to augmentation provider, just loaded from config
        self.draw_axis = False
        self.draw_axis_conf = False
        self.draw_color_band = False
        self.draw_color_band_conf = False
        self.draw_coordinate_axis = False
        self.draw_coordinate_axis_conf = False
        self.draw_panel_mask_in_skylines = False

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
        self.is_game = kget(self.env_config, constants.ENVIRONMENT_IS_GAME, default=self.is_game)
        self.env_shared_runner = kget(self.env_config, constants.ENVIRONMENT_SHARED_RUNNER, default=self.env_shared_runner)

        # Base resolution and region for the game in 4k, used for angle scaling
        self.base_resolution = (3840, 2160)

        # @TODO These should all be moved to env-specific settings
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
        self.screen_resolution = cradle.gameio.gui_utils.get_screen_size()
        self.mouse_move_factor = self.screen_resolution[0] / self.base_resolution[0]

        self._check_ide_window_info()
        self._set_env_window_info()

        # Load object names
        self.skill_registry_name = kget(self.env_config, constants.SKILL_REGISTRY_KEY, default=None)
        self.ui_control_name = kget(self.env_config, constants.UI_CONTROL_KEY, default=None)

        # Skill retrieval
        self.skill_local_path = './res/' + self.env_sub_path + '/skills/'
        self.skill_configs[constants.SKILL_CONFIG_LOCAL_PATH] = self.skill_local_path

        self.provider_configs = kget(self.env_config, constants.PROVIDER_CONFIGS_KEY, default={})
        sam2som_config = kget(self.provider_configs, constants.SAM2SOM_CONFIG, default=None)
        self.provider_configs = namedtuple('ProviderConfigs', self.provider_configs.keys())(**self.provider_configs)

        # SAM2SOM parameters for specific environment
        default_sam2som_config = {
            constants.SAM2SOM_MODE: self.sam2som_mode,
            constants.SAM_PRED_IOU_THRESH: self.sam_pred_iou_thresh,
            constants.SAM_RESIZE_RATIO: self.sam_resize_ratio,
            constants.SAM_CONTRAST_LEVEL: self.sam_contrast_level,
            constants.SAM_MAX_AREA: self.sam_max_area,
            constants.PLOT_BBOX_MULTI_COLOR: self.plot_bbox_multi_color,
            constants.DISABLE_CLOSE_APP_ICON: self.disable_close_app_icon,
        }

        if sam2som_config is not None:
            for key in default_sam2som_config.keys():
                sam2som_config[key] = kget(sam2som_config, key, default=default_sam2som_config[key])
        else:
            sam2som_config = default_sam2som_config

        self.sam2som_mode = sam2som_config[constants.SAM2SOM_MODE]
        self.sam_pred_iou_thresh = sam2som_config[constants.SAM_PRED_IOU_THRESH]
        self.sam_resize_ratio = sam2som_config[constants.SAM_RESIZE_RATIO]
        self.sam_contrast_level = sam2som_config[constants.SAM_CONTRAST_LEVEL]
        self.sam_max_area = sam2som_config[constants.SAM_MAX_AREA]
        self.plot_bbox_multi_color = sam2som_config[constants.PLOT_BBOX_MULTI_COLOR]
        self.disable_close_app_icon = sam2som_config[constants.DISABLE_CLOSE_APP_ICON]

        self.planner_params = self.env_config.get("planner_params", {})

        default_skill_configs = self.skill_configs.copy()

        skill_config = kget(self.env_config, constants.SKILL_CONFIGS, default=None)
        if skill_config is not None:
            for key in default_skill_configs.keys():
                skill_config[key] = kget(skill_config, key, default=default_skill_configs[key])
        else:
            skill_config = default_skill_configs

        self.skill_configs = skill_config


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

        Logger.work_dir = self.work_dir


    def _check_ide_window_info(self):
        named_windows = cradle.gameio.gui_utils.get_named_windows(self.ide_name)
        if len(named_windows) <= 0:
            ide_window = cradle.gameio.gui_utils.get_active_window()
            self.ide_name = ide_window.title


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
            try:
                env_window = cradle.gameio.lifecycle.ui_control.select_window(named_windows)
            except Exception as e:
                self._config_warn(f'-----------------------------------------------------------------')
                self._config_warn(f'Issue in non-unique env window: {self.env_name}|{self.win_name_pattern}.')
                self._config_warn(f'-----------------------------------------------------------------')
                pass
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


    def _config_warn(self, message):
        colours_on()
        print(Fore.RED + f' >>> WARNING: {message} ' + Style.RESET_ALL)


    def _set_latest_memory_path(self):
        path_list = os.listdir(self.work_dir)
        path_list.sort()
        if len(path_list) != 0:
            self.skill_local_path = os.path.join(self.work_dir, path_list[-1])
            self.memory_load_path = os.path.join(self.work_dir, path_list[-1])


    # @TODO These should all be moved to env-specific settings
    def _calc_minimap_region(self, screen_region):
        return [int(x * self.resolution_ratio) for x in self.base_minimap_region]


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
