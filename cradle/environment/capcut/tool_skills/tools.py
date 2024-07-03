import json
import re

from cradle.config import Config
from cradle.log import Logger
from cradle.gameio import IOEnvironment
from cradle.environment.capcut.skill_registry import register_skill
from cradle.utils.json_utils import parse_semi_formatted_text
from cradle.utils.video_utils import get_basic_info

config = Config()
logger = Logger()
io_env = IOEnvironment()


@register_skill("get_information_from_video")
def get_information_from_video(event):
    """
    Get information from video by ask gpt-4o.

    Parameters:
    - event: The event you want to the exact second from video.
    """

    frames, fps = get_basic_info(r'./res/capcut/videos/soccer.mp4')

    image_introduction = []
    for i in frames[0::fps]:
        image_introduction.append(
            {
                "introduction": "",
                "path": i,
                "resize": 768,
                "assistant": "",
            })

    magic_prompt = """
        Please help me to analyze the video below and answer the relevant questions.

        Input frames:
        <$image_introduction$>

        These are frames for the video I want to analyze.
        Please answer at which frame does "<$event$>" happen.
        Please only answer with a number. (The first frame should be numbered as 0. For example, if you think <$event$> in the 3rd frame, you should answer 2.)
        """

    input = {}
    input['image_introduction'] = image_introduction
    input['event'] = event

    try:

        message_prompts = io_env.llm_provider.assemble_prompt(template_str = magic_prompt, params=input)

        logger.write(f'{logger.UPSTREAM_MASK}{json.dumps(message_prompts, ensure_ascii=False)}\n')

        response, info = io_env.llm_provider.create_completion(message_prompts)

        logger.debug(f'{logger.DOWNSTREAM_MASK}{response}\n')

        # Convert the response to dict
        processed_response = parse_semi_formatted_text(response)

        processed_response = processed_response[None] # answer is like {None: number}
        numbers = re.findall(r'\d+', processed_response)

        processed_response = f'The second of "{event}" is {numbers[0]}' if numbers else 'No response from GPT-4o. Maybe try again later.'

    except Exception as e:
        logger.error(f"Error in get_information_from_video: {e}")

    return processed_response


__all__ = [
    "get_information_from_video",
]
