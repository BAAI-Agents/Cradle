import os
import re
from typing import (
    Any,
    Dict,
    List,
    Tuple,
)
import asyncio

import backoff
from openai import OpenAI, APIError, RateLimitError, APITimeoutError

from cradle import constants
from cradle.provider.base import LLMProvider, EmbeddingProvider
from cradle.config import Config
from cradle.log import Logger
from cradle.utils.json_utils import load_json
from cradle.utils.file_utils import assemble_project_path

config = Config()
logger = Logger()

PROVIDER_SETTING_KEY_VAR = "key_var"
PROVIDER_SETTING_EMB_MODEL = "emb_model"
PROVIDER_SETTING_COMP_MODEL = "comp_model"
PROVIDER_SETTING_BASE_URL = "base_url"

# MiniMax temperature must be in (0.0, 1.0]
MINIMAX_TEMP_MIN = 0.01
MINIMAX_TEMP_MAX = 1.0


def _clamp_temperature(temperature: float) -> float:
    """Clamp temperature to MiniMax's accepted range (0.01, 1.0]."""
    if temperature <= 0:
        return MINIMAX_TEMP_MIN
    return min(temperature, MINIMAX_TEMP_MAX)


class MiniMaxProvider(LLMProvider, EmbeddingProvider):
    """LLM provider for MiniMax models via OpenAI-compatible API.

    MiniMax provides OpenAI-compatible chat completions at
    https://api.minimax.io/v1. Supported models include MiniMax-M2.7
    and MiniMax-M2.5.
    """

    client: Any = None
    llm_model: str = ""
    embedding_model: str = ""

    def __init__(self) -> None:
        self.retries = 5

    def init_provider(self, provider_cfg) -> None:
        self.provider_cfg = self._parse_config(provider_cfg)

    def _parse_config(self, provider_cfg) -> dict:
        conf_dict = dict()

        if isinstance(provider_cfg, dict):
            conf_dict = provider_cfg
        else:
            path = assemble_project_path(provider_cfg)
            conf_dict = load_json(path)

        key_var_name = conf_dict[PROVIDER_SETTING_KEY_VAR]
        key = os.getenv(key_var_name)
        base_url = conf_dict.get(PROVIDER_SETTING_BASE_URL, "https://api.minimax.io/v1")

        self.client = OpenAI(api_key=key, base_url=base_url)
        self.llm_model = conf_dict[PROVIDER_SETTING_COMP_MODEL]
        self.embedding_model = conf_dict.get(PROVIDER_SETTING_EMB_MODEL, "")

        return conf_dict

    def create_completion(
        self,
        messages: List[Dict[str, str]],
        model: str | None = None,
        temperature: float = config.temperature,
        seed: int = config.seed,
        max_tokens: int = config.max_tokens,
    ) -> Tuple[str, Dict[str, int]]:
        if model is None:
            model = self.llm_model

        temperature = _clamp_temperature(temperature)

        if config.debug_mode:
            logger.debug(f"Creating MiniMax completion with model {model}, temperature {temperature}, max_tokens {max_tokens}")
        else:
            logger.write(f"Requesting {model} completion...")

        @backoff.on_exception(
            backoff.constant,
            (APIError, RateLimitError, APITimeoutError),
            max_tries=self.retries,
            interval=10,
        )
        def _generate_response_with_retry(
            messages: List[Dict[str, str]],
            model: str,
            temperature: float,
            seed: int = None,
            max_tokens: int = 512,
        ) -> Tuple[str, Dict[str, int]]:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            if response is None:
                logger.error("Failed to get a response from MiniMax. Try again.")
                logger.double_check()

            message = response.choices[0].message.content

            # Strip thinking tags that MiniMax M2.5/M2.7 may include
            if message and "<think>" in message:
                message = re.sub(r"<think>.*?</think>\s*", "", message, flags=re.DOTALL)

            info = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "system_fingerprint": getattr(response, "system_fingerprint", None),
            }

            logger.write(f"Response received from {model}.")
            return message, info

        return _generate_response_with_retry(messages, model, temperature, seed, max_tokens)

    async def create_completion_async(
        self,
        messages: List[Dict[str, str]],
        model: str | None = None,
        temperature: float = config.temperature,
        seed: int = config.seed,
        max_tokens: int = config.max_tokens,
    ) -> Tuple[str, Dict[str, int]]:
        if model is None:
            model = self.llm_model

        temperature = _clamp_temperature(temperature)

        if config.debug_mode:
            logger.debug(f"Creating MiniMax async completion with model {model}, temperature {temperature}, max_tokens {max_tokens}")
        else:
            logger.write(f"Requesting {model} completion...")

        @backoff.on_exception(
            backoff.constant,
            (APIError, RateLimitError, APITimeoutError),
            max_tries=self.retries,
            interval=10,
        )
        async def _generate_response_with_retry_async(
            messages: List[Dict[str, str]],
            model: str,
            temperature: float,
            seed: int = None,
            max_tokens: int = 512,
        ) -> Tuple[str, Dict[str, int]]:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            if response is None:
                logger.error("Failed to get a response from MiniMax. Try again.")
                logger.double_check()

            message = response.choices[0].message.content

            if message and "<think>" in message:
                message = re.sub(r"<think>.*?</think>\s*", "", message, flags=re.DOTALL)

            info = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "system_fingerprint": getattr(response, "system_fingerprint", None),
            }

            logger.write(f"Response received from {model}.")
            return message, info

        return await _generate_response_with_retry_async(messages, model, temperature, seed, max_tokens)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text using MiniMax embedding API."""
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts using the OpenAI-compatible embedding endpoint."""
        response = self.client.embeddings.create(
            input=texts,
            model=self.embedding_model,
        )
        return [d.embedding for d in response.data]

    def get_embedding_dim(self) -> int:
        """Get the embedding dimension for the configured model."""
        if self.embedding_model == "text-embedding-ada-002":
            return 1536
        # Default for MiniMax or other models: return a standard dimension
        return 1536

    def assemble_prompt(self, template_str: str = None, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        if config.DEFAULT_MESSAGE_CONSTRUCTION_MODE == constants.MESSAGE_CONSTRUCTION_MODE_TRIPART:
            return self._assemble_prompt_tripartite(template_str=template_str, params=params)
        raise NotImplementedError("Only tripartite mode is supported for MiniMax provider.")

    def _assemble_prompt_tripartite(self, template_str: str = None, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Assemble prompt in the tripartite format, same as OpenAI provider."""
        import json
        from cradle.utils.encoding_utils import encode_data_to_base64_path

        pattern = re.compile(r"(.+?)(?=\n\n|$)", re.DOTALL)
        paragraphs = re.findall(pattern, template_str)
        filtered_paragraphs = [p for p in paragraphs if p.strip() != '']

        system_content = filtered_paragraphs[0]
        system_message = {
            "role": "system",
            "content": [{"type": "text", "text": f"{system_content}"}]
        }

        image_introduction_paragraph_index = None
        image_introduction_paragraph = None
        for i, paragraph in enumerate(filtered_paragraphs):
            if constants.IMAGES_INPUT_TAG in paragraph:
                image_introduction_paragraph_index = i
                image_introduction_paragraph = paragraph
                break

        user_messages_part1_paragraphs = filtered_paragraphs[1:image_introduction_paragraph_index]
        user_messages_part2_paragraphs = filtered_paragraphs[image_introduction_paragraph_index + 1:]

        # Part 1
        user_messages_part1_contents = []
        search_placeholder_pattern = re.compile(r"<\$[^\$]+\$>")
        for paragraph in user_messages_part1_paragraphs:
            placeholder = re.search(search_placeholder_pattern, paragraph)
            if not placeholder:
                user_messages_part1_contents.append(paragraph)
            else:
                placeholder = placeholder.group()
                placeholder_name = placeholder.replace("<$", "").replace("$>", "")
                paragraph_input = params.get(placeholder_name, None)
                if paragraph_input is None or paragraph_input == "" or paragraph_input == []:
                    continue
                if isinstance(paragraph_input, str):
                    user_messages_part1_contents.append(paragraph.replace(placeholder, paragraph_input))
                elif isinstance(paragraph_input, list):
                    user_messages_part1_contents.append(paragraph.replace(placeholder, json.dumps(paragraph_input)))
                else:
                    raise ValueError(f"Unexpected input type: {type(paragraph_input)}")

        user_messages_part1 = None
        if len(user_messages_part1_contents) > 0:
            user_messages_part1 = {
                "role": "user",
                "content": [{"type": "text", "text": "\n\n".join(user_messages_part1_contents)}]
            }

        # Image introduction
        image_introduction_messages = []
        paragraph_input = params.get(constants.IMAGES_INPUT_TAG_NAME, [])
        if paragraph_input and paragraph_input != "" and paragraph_input != []:
            paragraph_content_pre = image_introduction_paragraph.replace(constants.IMAGES_INPUT_TAG, "")
            image_introduction_messages.append({
                "role": "user",
                "content": [{"type": "text", "text": f"{paragraph_content_pre}"}]
            })
            for item in paragraph_input:
                introduction = item.get(constants.IMAGE_INTRO_TAG_NAME, None)
                path = item.get(constants.IMAGE_PATH_TAG_NAME, None)
                assistant = item.get(constants.IMAGE_ASSISTANT_TAG_NAME, None)
                resolution = item.get(constants.IMAGE_RESOLUTION_TAG_NAME, None)

                message = {"role": "user", "content": []}
                if introduction:
                    message["content"].append({"type": "text", "text": f"{introduction}"})
                if path:
                    encoded_images = encode_data_to_base64_path(path)
                    for encoded_image in encoded_images:
                        msg_content = {"type": "image_url", "image_url": {"url": f"{encoded_image}"}}
                        if resolution:
                            msg_content["image_url"]["detail"] = resolution
                        message["content"].append(msg_content)
                if len(message["content"]) > 0:
                    image_introduction_messages.append(message)
                if assistant:
                    image_introduction_messages.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": f"{assistant}"}]
                    })

        # Part 2
        user_messages_part2_contents = []
        for paragraph in user_messages_part2_paragraphs:
            placeholder = re.search(search_placeholder_pattern, paragraph)
            if not placeholder:
                user_messages_part2_contents.append(paragraph)
            else:
                placeholder = placeholder.group()
                placeholder_name = placeholder.replace("<$", "").replace("$>", "")
                paragraph_input = params.get(placeholder_name, None)
                if paragraph_input is None or paragraph_input == "" or paragraph_input == []:
                    continue
                if isinstance(paragraph_input, str):
                    user_messages_part2_contents.append(paragraph.replace(placeholder, paragraph_input))
                elif isinstance(paragraph_input, (bool, int, float)):
                    user_messages_part2_contents.append(paragraph.replace(placeholder, str(paragraph_input)))
                elif isinstance(paragraph_input, list):
                    user_messages_part2_contents.append(paragraph.replace(placeholder, json.dumps(paragraph_input)))
                else:
                    raise ValueError(f"Unexpected input type: {type(paragraph_input)}")

        user_messages_part2 = {
            "role": "user",
            "content": [{"type": "text", "text": "\n\n".join(user_messages_part2_contents)}]
        }

        if user_messages_part1 is None:
            return [system_message] + image_introduction_messages + [user_messages_part2]
        return [system_message] + [user_messages_part1] + image_introduction_messages + [user_messages_part2]
