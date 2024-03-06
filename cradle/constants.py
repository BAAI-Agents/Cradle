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

# Tags used in prompt templates
IMAGES_INPUT_TAG_NAME = 'image_introduction'
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
AUGMENTED_IMAGES_MEM_BUCKET = 'augmented_image'
IMAGES_MEM_BUCKET = 'image'
NO_IMAGE = '[None]'

# LLM message type constants
MESSAGE_CONSTRUCTION_MODE_TRIPART = 'tripartite'
MESSAGE_CONSTRUCTION_MODE_PARAGRAPH = 'paragraph'

# Prompts when output is None
NONE_TASK_OUTPUT = "null"
NONE_TARGET_OBJECT_OUTPUT = "null"
