# Environment configuration keys
ENVIRONMENT_NAME = 'env_name'
ENVIRONMENT_WINDOW_NAME_PATTERN = 'win_name_pattern'
ENVIRONMENT_SHORT_NAME = 'env_short_name'
ENVIRONMENT_SUB_PATH = 'sub_path'
ENVIRONMENT_SHARED_RUNNER = 'shared_runner'
ENVIRONMENT_IS_GAME = 'is_game'

# Info passing around tags
PA_PROCESSED_RESPONSE_TAG = 'res_dict'

# Video capture keys
START_FRAME_ID = 'start_frame_id'
END_FRAME_ID = 'end_frame_id'

# Cradle module keys
ACTION_PLANNING_MODULE = 'action_planning'
INFORMATION_GATHERING_MODULE = 'information_gathering'
INFORMATION_TEXT_GATHERING_MODULE = 'information_text_gathering'
SELF_REFLECTION_MODULE = 'self_reflection'
TASK_INFERENCE_MODULE = 'task_inference'
SKILL_CURATION_MODULE = 'skill_curation'
MEMORY_MODULE = 'memory'

# Gather information expected fields
ACTION_GUIDANCE = 'action_guidance'
ITEM_STATUS = 'item_status'
TASK_GUIDANCE = 'task_guidance'
DIALOGUE = 'dialogue'
GATHER_TEXT_REASONING = 'reasoning'
TARGET_OBJECT_NAME = 'target_object_name'
GATHER_INFO_REASONING = 'reasoning_of_object'
SCREEN_CLASSIFICATION = 'screen_classification'
GENERAL_GAME_INTERFACE = 'general game interface without any menu'
TRADE_INTERFACE = 'trade'
MAP_INTERFACE = 'map'
PAUSE_INTERFACE = 'pause'
SATCHEL_INTERFACE = 'satchel'
RADIAL_INTERFACE = 'radial menu'
LAST_TASK_GUIDANCE = 'last_task_guidance'
LAST_TASK_HORIZON = 'task_horizon'
LAST_TASK_DURATION = 'last_task_duration'
CUR_SCREENSHOT_PATH = 'cur_screenshot_path'
MOUSE_POSITION = 'mouse_position'
EMPTY_STRING = ''

# Gather information configurations
GATHER_INFORMATION_CONFIGURATIONS = 'gather_information_configurations'
FRAME_EXTRACTOR = 'frame_extractor'
ICON_REPLACER = 'icon_replacer'
LLM_DESCRIPTION = 'llm_description'
OBJECT_DETECTOR = 'object_detector'

# Parameters for each app task_description
TASK_DESCRIPTION_LIST = 'task_description_list'
TASK_DESCRIPTION = 'task_description'
SUB_TASK_DESCRIPTION = 'sub_task_description'
SUB_TASK_DESCRIPTION_LIST = 'sub_task_description_list'

OBJECT_DETECTOR = 'object_detector'
FRAME_EXTRACTOR = 'frame_extractor'
LLM_DESCRIPTION = 'llm_description'
GET_ITEM_NUMBER = 'get_item_number'

# Skill-related keys
SKILL_CODE_HASH_KEY = "skill_code_base64"
SKILL_FULL_LIB_FILE = "skill_lib.json"
SKILL_BASIC_LIB_FILE = "skill_lib_basic.json"
SKILL_LIB_MODE_BASIC = 'Basic'
SKILL_LIB_MODE_FULL = 'Full'
SKILL_LIB_MODE_NONE = None

# Skill-related config keys
SKILL_CONFIGS = "skill_configs"
SKILL_CONFIG_FROM_DEFAULT = "skill_from_default"
SKILL_CONFIG_RETRIEVAL = "skill_retrieval"
SKILL_CONFIG_MAX_COUNT = "skill_num"
SKILL_CONFIG_MODE = "skill_mode"
SKILL_CONFIG_NAMES_DENY = "skill_names_deny"
SKILL_CONFIG_NAMES_ALLOW = "skill_names_allow"
SKILL_CONFIG_NAMES_BASIC = "skill_names_basic"
SKILL_CONFIG_NAMES_OTHERS = "skill_names_others"
SKILL_CONFIG_LOCAL_PATH = "skill_local_path"
SKILL_CONFIG_REGISTERED_SKILLS = 'skills_registered'

# Class name config keys
SKILL_REGISTRY_KEY = 'skill_registry_name'
UI_CONTROL_KEY = 'ui_control_name'

# Env-specific skill list configs
SKILL_CONFIG_NAMES_MOVEMENT = "skill_names_movement"
SKILL_CONFIG_NAMES_MAP = "skill_names_map"
SKILL_CONFIG_NAMES_TRADE = "skill_names_trade"

# Provider-related keys
PROVIDER_CONFIGS_KEY = 'provider_configs'

# Target environment SAM2SOM parameters
SAM2SOM_CONFIG = 'sam2som_config'
SAM2SOM_MODE = 'sam2som_mode'
SAM_PRED_IOU_THRESH = 'sam_pred_iou_thresh'
SAM_RESIZE_RATIO = 'sam_resize_ratio'
SAM_CONTRAST_LEVEL = 'sam_contrast_level'
SAM_MAX_AREA = 'sam_max_area'
PLOT_BBOX_MULTI_COLOR = 'plot_bbox_multi_color'
DISABLE_CLOSE_APP_ICON = 'disable_close_app_icon'
SAM2SOM_DEFAULT_MODE = 'default'
SAM2SOM_OCR_MODE = 'enable_ocr'

# Two instances of augmentation info dict
PREVIOUS_AUGMENTATION_INFO = "previous_augmentation_info"
CURRENT_AUGMENTATION_INFO = "current_augmentation_info"
CURRENT_IMAGE_DESCRIPTION = 'current_image_description'

# Augmentation info dict attributes
IMAGE_DESCRIPTION = 'image_description'
DESCRIPTION_OF_BOUNDING_BOXES = 'description_of_bounding_boxes'
AUG_MOUSE_X = 'mouse_x'
AUG_MOUSE_Y = 'mouse_y'
AUG_BASE_IMAGE_PATH='image_path'
AUG_SOM_MOUSE_IMG_PATH = 'som_mouse_img_path'
AUG_MOUSE_IMG_PATH = 'mouse_img_path'
AUG_SOM_IMAGE_PATH = 'som_image_path'
AUG_SOM_MAP = 'som_map'
LENGTH_OF_SOM_MAP = 'length_of_som_map'

# Augmentation flags between two image to check whether som_image should be generated again or not
IMAGE_SAME_FLAG = "image_same_flag"
MOUSE_POSITION_SAME_FLAG = "mouse_position_same_flag"

# Tags used in prompt templates
IMAGES_INPUT_TAG_NAME = 'image_introduction'
FEW_SHOTS_INPUT_TAG_NAME = 'few_shots'
IMAGE_INTRO_TAG_NAME = 'introduction'
IMAGE_PATH_TAG_NAME = 'path'
IMAGE_RESOLUTION_TAG_NAME = 'resolution'
IMAGE_RESIZE_TAG_NAME = 'resize'
IMAGE_ASSISTANT_TAG_NAME = 'assistant'
IMAGES_INPUT_TAG = f'<${IMAGES_INPUT_TAG_NAME}$>'

# Minimap information
MINIMAP_INFORMATION = 'minimap_information'
RED_POINTS = 'red points'
YELLOW_POINTS = 'yellow points'
YELLOW_REGION = 'yellow region'
GD_PROMPT = 'red points . yellow points . yellow region .'

# Skill-related keys
DISTANCE_TYPE = 'distance'
SKILL_LIBRARY = 'skill_library'

# Local memory
AUGMENTED_IMAGES_MEM_BUCKET = 'augmented_image'
IMAGES_MEM_BUCKET = 'image'
NO_IMAGE = '[None]'
SKIIL_LIB_MEM_BUCKET = 'skill_library'
SUMMARIZATION_MEM_BUCKET = 'summarization'

# LLM message type constants
MESSAGE_CONSTRUCTION_MODE_TRIPART = 'tripartite'
MESSAGE_CONSTRUCTION_MODE_PARAGRAPH = 'paragraph'

# Prompts when output is None
NONE_TASK_OUTPUT = "null"
NONE_TARGET_OBJECT_OUTPUT = "null"

# Keys in exec_info
EXEC_INFO = 'exec_info'
EXECUTED_SKILLS = 'executed_skills'
LAST_SKILL = 'last_skill'
ERRORS = 'errors'
ERRORS_INFO = 'errors_info'

# Info in pre process action
INVALID_BBOX = 'invalid_bbox'

# Info message in close icon detection
CLOSE_ICON_DETECTED = 'You are trying to close the software window by clicking on the close icon. Action Denied. Don\'t close the software window.'

# Response for self-reflection
SELF_REFLECTION_REASONING = 'self_reflection_reasoning'
SUCCESS_DETECTION = 'success_detection'
PRE_SELF_REFLECTION_REASONING = 'pre_self_reflection_reasoning'
PREVIOUS_SELF_REFLECTION_REASONING = 'previous_self_reflection_reasoning'
PREVIOUS_REASONING = 'previous_reasoning'

# Response for action planning
DECISION_MAKING_REASONING = 'decision_making_reasoning'
ACTION = 'action'
ACTIONS = 'actions'
KEY_REASON_OF_LAST_ACTION = 'key_reason_of_last_action'
PRE_ACTION = 'pre_action'
PREVIOUS_ACTION = 'previous_action'
PRE_DECISION_MAKING_REASONING = 'pre_decision_making_reasoning'
PREVIOUS_ACTION_CALL = 'previous_action_call'
ACTION_CODE = 'action_code'
EXECUTING_ACTION_ERROR = 'executing_action_error'
NUMBER_OF_EXECUTE_SKILLS = 'number_of_execute_skills'
ACTION_ERROR = 'action_error'
SKILL_STEPS = 'skill_steps'
SOM_MAP = 'som_map'

# Information summary
SUMMARIZATION = 'summarization'
SUBTASK_DESCRIPTION = 'subtask_description'
SUBTASK_REASONING = 'subtask_reasoning'
HISTORY_SUMMARY = 'history_summary'
INFO_SUMMARY = 'info_summary'
PREVIOUS_SUMMARIZATION = 'previous_summarization'

# UI Control constants
PAUSE_SCREEN_WAIT = 1

# Standard colours to reuse
COLOURS = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
}
