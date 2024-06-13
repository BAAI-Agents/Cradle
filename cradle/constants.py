# Gather information expected fields
ACTION_GUIDANCE = 'action_guidance'
ITEM_STATUS = 'item_status'
TASK_GUIDANCE = 'task_guidance'
DIALOGUE = 'dialogue'
GATHER_TEXT_REASONING = 'reasoning'
IMAGE_DESCRIPTION = 'description'
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
TASK_DESCRIPTION = 'task_description'
OBJECT_DETECTOR = 'object_detector'
FRAME_EXTRACTOR = 'frame_extractor'
LLM_DESCRIPTION = 'llm_description'
GET_ITEM_NUMBER = 'get_item_number'

# Tags used in prompt templates
IMAGES_INPUT_TAG_NAME = 'image_introduction'
FEW_SHOTS_INPUT_TAG_NAME = 'few_shots'
IMAGE_INTRO_TAG_NAME = 'introduction'
IMAGE_PATH_TAG_NAME = 'path'
IMAGE_RESOLUTION_TAG_NAME = 'resolution'
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

# Local memory
AUGMENTED_IMAGES_MEM_BUCKET = 'augmented_screen_shot_path'
IMAGES_MEM_BUCKET = 'sceen_shot_path'
NO_IMAGE = '[None]'

# LLM message type constants
MESSAGE_CONSTRUCTION_MODE_TRIPART = 'tripartite'
MESSAGE_CONSTRUCTION_MODE_PARAGRAPH = 'paragraph'

# Prompts when output is None
NONE_TASK_OUTPUT = "null"
NONE_TARGET_OBJECT_OUTPUT = "null"

# COLOR dict
COLORS = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
}

# Skill-related keys
SKILL_CODE_HASH_KEY = "skill_code_base64"
SKILL_FULL_LIB_FILE = "skill_lib.json"
SKILL_BASIC_LIB_FILE = "skill_lib_basic.json"

PAUSE_SCREEN_WAIT = 1

# RDR2 skills
DEFAULT_MAX_SHOOTING_ITERATIONS = 100
SHOOT_PEOPLE_TARGET_NAME = "person"
SHOOT_WOLVES_TARGET_NAME = "wolf"
CONTINUE_NO_ENEMY_FREQ = 5
MAX_FOLLOW_ITERATIONS = 20
DEFAULT_GO_TO_ICON_ITERATIONS = 20
DEFAULT_GO_TO_HORSE_ITERATIONS = DEFAULT_GO_TO_ICON_ITERATIONS
DEFAULT_NAVIGATION_ITERATIONS = 500
NAVIGATION_TERMINAL_THRESHOLD = 500

ENVIRONMENT_NAME = 'env_name'
ENVIRONMENT_WINDOW_NAME_PATTERN = 'win_name_pattern'
ENVIRONMENT_SHORT_NAME = 'env_short_name'
ENVIRONMENT_SUB_PATH = 'sub_path'
