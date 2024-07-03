import os
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)
import json
import re
import asyncio

import backoff
import tiktoken
import numpy as np
from openai import OpenAI, AzureOpenAI, APIError, RateLimitError, APITimeoutError

from cradle import constants
from cradle.provider.base import LLMProvider, EmbeddingProvider
from cradle.config import Config
from cradle.log import Logger
from cradle.utils.json_utils import load_json
from cradle.utils.encoding_utils import encode_data_to_base64_path
from cradle.utils.file_utils import assemble_project_path

config = Config()
logger = Logger()

MAX_TOKENS = {
    "gpt-3.5-turbo-0301": 4097,
    "gpt-3.5-turbo-0613": 4097,
    "gpt-3.5-turbo-16k-0613": 16385,
}

PROVIDER_SETTING_KEY_VAR = "key_var"
PROVIDER_SETTING_EMB_MODEL = "emb_model"
PROVIDER_SETTING_COMP_MODEL = "comp_model"
PROVIDER_SETTING_IS_AZURE = "is_azure"
PROVIDER_SETTING_BASE_VAR = "base_var"       # Azure-speficic setting
PROVIDER_SETTING_API_VERSION = "api_version" # Azure-speficic setting
PROVIDER_SETTING_DEPLOYMENT_MAP = "models"   # Azure-speficic setting


class OpenAIProvider(LLMProvider, EmbeddingProvider):
    """A class that wraps a given model"""

    client: Any = None
    llm_model: str = ""
    embedding_model: str = ""

    allowed_special: Union[Literal["all"], Set[str]] = set()
    disallowed_special: Union[Literal["all"], Set[str], Sequence[str]] = "all"
    chunk_size: int = 1000
    embedding_ctx_length: int = 8191
    request_timeout: Optional[Union[float, Tuple[float, float]]] = None
    tiktoken_model_name: Optional[str] = None

    """Whether to skip empty strings when embedding or raise an error."""
    skip_empty: bool = False


    def __init__(self) -> None:
        """Initialize a class instance

        Args:
            cfg: Config object

        Returns:
            None
        """
        self.retries = 5


    def init_provider(self, provider_cfg ) -> None:
        self.provider_cfg = self._parse_config(provider_cfg)


    def _parse_config(self, provider_cfg) -> dict:
        """Parse the config object"""

        conf_dict = dict()

        if isinstance(provider_cfg, dict):
            conf_dict = provider_cfg
        else:
            path = assemble_project_path(provider_cfg)
            conf_dict = load_json(path)

        key_var_name = conf_dict[PROVIDER_SETTING_KEY_VAR]

        if conf_dict[PROVIDER_SETTING_IS_AZURE]:

            key = os.getenv(key_var_name)
            endpoint_var_name = conf_dict[PROVIDER_SETTING_BASE_VAR]
            endpoint = os.getenv(endpoint_var_name)

            self.client = AzureOpenAI(
                api_key = key,
                api_version = conf_dict[PROVIDER_SETTING_API_VERSION],
                azure_endpoint = endpoint
            )
        else:
            key = os.getenv(key_var_name)
            self.client = OpenAI(api_key=key)

        self.embedding_model = conf_dict[PROVIDER_SETTING_EMB_MODEL]
        self.llm_model = conf_dict[PROVIDER_SETTING_COMP_MODEL]

        return conf_dict

    @property
    def _emb_invocation_params(self) -> Dict:

        openai_args = {
            "model": self.embedding_model,
        }

        if self.provider_cfg[PROVIDER_SETTING_IS_AZURE]:
            engine = self._get_azure_deployment_id_for_model(self.embedding_model)
            openai_args = {
                "model": self.embedding_model,
            }

        return openai_args

    def embed_with_retry(self, **kwargs: Any) -> Any:
        """Use backoff to retry the embedding call."""

        @backoff.on_exception(
            backoff.expo,
            (
                APIError,
                RateLimitError,
                APITimeoutError,
            ),
            max_tries=self.retries,
            max_value=10,
            jitter=None,
        )
        def _embed_with_retry(**kwargs: Any) -> Any:
            response = self.client.embeddings.create(**kwargs)
            if any(len(d.embedding) == 1 for d in response.data):
                raise RuntimeError("OpenAI API returned an empty embedding")
            return response

        return _embed_with_retry(**kwargs)


    def _get_len_safe_embeddings(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        embeddings: List[List[float]] = [[] for _ in range(len(texts))]
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "Could not import tiktoken python package. "
                "This is needed in order to for OpenAIEmbeddings. "
                "Please install it with `pip install tiktoken`."
            )

        tokens = []
        indices = []
        model_name = self.tiktoken_model_name or self.embedding_model
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            logger.warn("Warning: model not found. Using cl100k_base encoding.")
            model = "cl100k_base"
            encoding = tiktoken.get_encoding(model)
        for i, text in enumerate(texts):
            token = encoding.encode(
                text,
                allowed_special=self.allowed_special,
                disallowed_special=self.disallowed_special,
            )
            for j in range(0, len(token), self.embedding_ctx_length):
                tokens.append(token[j : j + self.embedding_ctx_length])
                indices.append(i)

        batched_embeddings: List[List[float]] = []
        _chunk_size = self.chunk_size
        _iter = range(0, len(tokens), _chunk_size)

        for i in _iter:
            response = self.embed_with_retry(
                input=tokens[i : i + self.chunk_size],
                **self._emb_invocation_params,
            )
            batched_embeddings.extend(r.embedding for r in response.data)

        results: List[List[List[float]]] = [[] for _ in range(len(texts))]
        num_tokens_in_batch: List[List[int]] = [[] for _ in range(len(texts))]
        for i in range(len(indices)):
            if self.skip_empty and len(batched_embeddings[i]) == 1:
                continue
            results[indices[i]].append(batched_embeddings[i])
            num_tokens_in_batch[indices[i]].append(len(tokens[i]))

        for i in range(len(texts)):
            _result = results[i]
            if len(_result) == 0:
                average = self.embed_with_retry(
                    input="",
                    **self._emb_invocation_params,
                ).data[0].embedding
            else:
                average = np.average(_result, axis=0, weights=num_tokens_in_batch[i])
            embeddings[i] = (average / np.linalg.norm(average)).tolist()

        return embeddings

    def embed_documents(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        """Call out to OpenAI's embedding endpoint for embedding search docs.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        # NOTE: to keep things simple, we assume the list may contain texts longer
        #       than the maximum context and use length-safe embedding function.
        return self._get_len_safe_embeddings(texts)


    def embed_query(self, text: str) -> List[float]:
        """Call out to OpenAI's embedding endpoint for embedding query text.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        return self.embed_documents([text])[0]


    def get_embedding_dim(self) -> int:
        """Get the embedding dimension."""
        if self.embedding_model == "text-embedding-ada-002":
            embedding_dim = 1536
        else:
            raise ValueError(f"Unknown embedding model: {self.embedding_model}")
        return embedding_dim


    def create_completion(
        self,
        messages: List[Dict[str, str]],
        model: str | None = None,
        temperature: float = config.temperature,
        seed: int = config.seed,
        max_tokens: int = config.max_tokens,
    ) -> Tuple[str, Dict[str, int]]:
        """Create a chat completion using the OpenAI API

        Supports both GPT-4 and GPT-4V).

        Example Usage:
        image_path = "path_to_your_image.jpg"
        base64_image = encode_image(image_path)
        response, info = self.create_completion(
            model="gpt-4-vision-preview",
            messages=[
              {
                "role": "user",
                "content": [
                  {
                    "type": "text",
                    "text": "What’s in this image?"
                  },
                  {
                    "type": "image_url",
                    "image_url": {
                      "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                  }
                ]
              }
            ],
        )
        """

        if model is None:
            model = self.llm_model

        if config.debug_mode:
            logger.debug(f"Creating chat completion with model {model}, temperature {temperature}, max_tokens {max_tokens}")
        else:
            logger.write(f"Requesting {model} completion...")

        @backoff.on_exception(
            backoff.constant,
            (
                APIError,
                RateLimitError,
                APITimeoutError),
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

            """Send a request to the OpenAI API."""
            if self.provider_cfg[PROVIDER_SETTING_IS_AZURE]:
                response = self.client.chat.completions.create(model=model,
                messages=messages,
                temperature=temperature,
                seed=seed,
                max_tokens=max_tokens,)
            else:
                response = self.client.chat.completions.create(model=model,
                messages=messages,
                temperature=temperature,
                seed=seed,
                max_tokens=max_tokens,)

            if response is None:
                logger.error("Failed to get a response from OpenAI. Try again.")
                logger.double_check()

            message = response.choices[0].message.content

            info = {
                "prompt_tokens" : response.usage.prompt_tokens,
                "completion_tokens" : response.usage.completion_tokens,
                "total_tokens" : response.usage.total_tokens,
                "system_fingerprint" : response.system_fingerprint,
            }

            logger.write(f'Response received from {model}.')

            return message, info

        return _generate_response_with_retry(
            messages,
            model,
            temperature,
            seed,
            max_tokens,
        )

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

        if config.debug_mode:
            logger.debug(
                f"Creating chat completion with model {model}, temperature {temperature}, max_tokens {max_tokens}")
        else:
            logger.write(f"Requesting {model} completion...")

        @backoff.on_exception(
            backoff.constant,
            (
                    APIError,
                    RateLimitError,
                    APITimeoutError),
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

            """Send a request to the OpenAI API."""
            if self.provider_cfg[PROVIDER_SETTING_IS_AZURE]:
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    seed=seed,
                    max_tokens=max_tokens,
                )
            else:
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    seed=seed,
                    max_tokens=max_tokens,
                )

            if response is None:
                logger.error("Failed to get a response from OpenAI. Try again.")
                logger.double_check()

            message = response.choices[0].message.content

            info = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "system_fingerprint": response.system_fingerprint,
            }

            logger.write(f'Response received from {model}.')

            return message, info

        return await _generate_response_with_retry_async(
            messages,
            model,
            temperature,
            seed,
            max_tokens,
        )


    def num_tokens_from_messages(self, messages, model):
        """Return the number of tokens used by a list of messages.
        Borrowed from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        """
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            logger.debug("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        if model in {
            "gpt-4-1106-vision-preview",
        }:
            raise ValueError("We don't support counting tokens of GPT-4V yet.")

        if model in {
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
            "gpt-4-1106-preview",
        }:
            tokens_per_message = 3
            tokens_per_name = 1
        elif model == "gpt-3.5-turbo-0301":
            tokens_per_message = (
                4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            )
            tokens_per_name = -1  # if there's a name, the role is omitted
        else:
            raise NotImplementedError(
                f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
            )

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name

        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>

        return num_tokens


    def _get_azure_deployment_id_for_model(self, model_label) -> list:
        return self.provider_cfg[PROVIDER_SETTING_DEPLOYMENT_MAP][model_label]


    def assemble_prompt_tripartite(self, template_str: str = None, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:

        """
        A tripartite prompt is a message with the following structure:
        <system message>

        <user message part 1 before image introduction>
        <image introduction>
        <user message part 2 after image introduction>
        """
        pattern = re.compile(r"(.+?)(?=\n\n|$)", re.DOTALL)

        paragraphs = re.findall(pattern, template_str)

        filtered_paragraphs = [p for p in paragraphs if p.strip() != '']

        system_content = filtered_paragraphs[0]  # the system content defaults to the first paragraph of the template
        system_message = {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": f"{system_content}"
                }
            ]
        }

        # segmenting "paragraphs"
        image_introduction_paragraph_index = None
        image_introduction_paragraph = None
        for i, paragraph in enumerate(filtered_paragraphs):
            if constants.IMAGES_INPUT_TAG in paragraph:
                image_introduction_paragraph_index = i
                image_introduction_paragraph = paragraph
                break

        user_messages_part1_paragraphs = filtered_paragraphs[1:image_introduction_paragraph_index]
        user_messages_part2_paragraphs = filtered_paragraphs[image_introduction_paragraph_index + 1:]

        # assemble user messages part 1
        user_messages_part1_contents = []
        for paragraph in user_messages_part1_paragraphs:
            search_placeholder_pattern = re.compile(r"<\$[^\$]+\$>")

            placeholder = re.search(search_placeholder_pattern, paragraph)
            if not placeholder:
                user_messages_part1_contents.append(paragraph)
            else:
                placeholder = placeholder.group()
                placeholder_name = placeholder.replace("<$", "").replace("$>", "")

                paragraph_input = params.get(placeholder_name, None)
                if paragraph_input is None or paragraph_input == "" or paragraph_input == []:
                    continue
                else:
                    if isinstance(paragraph_input, str):
                        paragraph_content = paragraph.replace(placeholder, paragraph_input)
                        user_messages_part1_contents.append(paragraph_content)
                    elif isinstance(paragraph_input, list):
                        paragraph_content = paragraph.replace(placeholder, json.dumps(paragraph_input))
                        user_messages_part1_contents.append(paragraph_content)
                    else:
                        raise ValueError(f"Unexpected input type: {type(paragraph_input)}")

        if len(user_messages_part1_contents) > 0:

            user_messages_part1_content = "\n\n".join(user_messages_part1_contents)

            user_messages_part1 = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{user_messages_part1_content}"
                    }
                ]
            }

        else:
            user_messages_part1 = None

        # assemble image introduction messages
        image_introduction_messages = []

        paragraph_input = params.get(constants.IMAGES_INPUT_TAG_NAME, []) # 'image_introduction'

        if paragraph_input is None or paragraph_input == "" or paragraph_input == []:
            image_introduction_messages = []
        else:
            paragraph_content_pre = image_introduction_paragraph.replace(constants.IMAGES_INPUT_TAG, "")
            message = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{paragraph_content_pre}"
                    }
                ]
            }

            image_introduction_messages.append(message)

            for item in paragraph_input:
                introduction = item.get(constants.IMAGE_INTRO_TAG_NAME, None)
                path = item.get(constants.IMAGE_PATH_TAG_NAME, None)
                assistant = item.get(constants.IMAGE_ASSISTANT_TAG_NAME, None)
                resolution = item.get(constants.IMAGE_RESOLUTION_TAG_NAME, None)
                resize = item.get(constants.IMAGE_RESIZE_TAG_NAME, None)

                message = {
                    "role": "user",
                    "content": [],
                }

                if introduction is not None and introduction != "":
                    message["content"].append(
                        {
                            "type": "text",
                            "text": f"{introduction}"
                        })

                if path is not None and path != "":
                    encoded_images = encode_data_to_base64_path(path)

                    for encoded_image in encoded_images:
                        msg_content = {
                                "type": "image_url",
                                "image_url":
                                    {
                                        "url": f"{encoded_image}"
                                    }
                            }

                        if resolution is not None and resolution != "":
                            msg_content["image_url"]["detail"] = resolution

                        if resize is not None and resize != "":
                            msg_content["image_url"]["resize"] = resize

                        message["content"].append(msg_content)

                if len(message["content"]) > 0:
                    image_introduction_messages.append(message)

                if assistant is not None and assistant != "":
                    message = {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": f"{assistant}"
                            }
                        ]
                    }
                    image_introduction_messages.append(message)

        # assemble user messages part 2
        user_messages_part2_contents = []
        for paragraph in user_messages_part2_paragraphs:
            search_placeholder_pattern = re.compile(r"<\$[^\$]+\$>")

            placeholder = re.search(search_placeholder_pattern, paragraph)
            if not placeholder:
                user_messages_part2_contents.append(paragraph)
            else:
                placeholder = placeholder.group()
                placeholder_name = placeholder.replace("<$", "").replace("$>", "")

                paragraph_input = params.get(placeholder_name, None)
                if paragraph_input is None or paragraph_input == "" or paragraph_input == []:
                    continue
                else:
                    if isinstance(paragraph_input, str):
                        paragraph_content = paragraph.replace(placeholder, paragraph_input)
                        user_messages_part2_contents.append(paragraph_content)
                    elif isinstance(paragraph_input, bool) or isinstance(paragraph_input, int) or isinstance(paragraph_input, float):
                        paragraph_content = paragraph.replace(placeholder, str(paragraph_input))
                        user_messages_part2_contents.append(paragraph_content)
                    elif isinstance(paragraph_input, list):
                        paragraph_content = paragraph.replace(placeholder, json.dumps(paragraph_input))
                        user_messages_part2_contents.append(paragraph_content)
                    else:
                        raise ValueError(f"Unexpected input type: {type(paragraph_input)}")

        user_messages_part2_content = "\n\n".join(user_messages_part2_contents)
        user_messages_part2 = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"{user_messages_part2_content}"
                }
            ]
        }

        if user_messages_part1 is None:
            return [system_message] + image_introduction_messages + [user_messages_part2]
        else:
            return [system_message] + [user_messages_part1] + image_introduction_messages + [user_messages_part2]


    def assemble_prompt_paragraph(self, template_str: str = None, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        raise NotImplementedError("This method is not implemented yet.")


    def assemble_prompt(self, template_str: str = None, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        if config.DEFAULT_MESSAGE_CONSTRUCTION_MODE == constants.MESSAGE_CONSTRUCTION_MODE_TRIPART:
            return self.assemble_prompt_tripartite(template_str=template_str, params=params)
        elif config.DEFAULT_MESSAGE_CONSTRUCTION_MODE == constants.MESSAGE_CONSTRUCTION_MODE_PARAGRAPH:
            return self.assemble_prompt_paragraph(template_str=template_str, params=params)
