import os
import re
from typing import Dict, Any
from functools import wraps

from cradle.utils import Singleton
from cradle.log import Logger
from cradle.utils.file_utils import assemble_project_path, read_resource_file
from cradle.utils.json_utils import parse_semi_formatted_text

logger = Logger()

class BaseProvider(metaclass=Singleton):

    def __init__(self, *args, **kwargs):
        pass


    @staticmethod
    def write(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Wrap the original logger.write method
            original_write = logger.write

            def new_write(message):
                if self.__class__.__name__ in message:
                    full_message = message
                else:
                    full_message = f"# {self.__class__.__name__} # {message}"
                original_write(full_message)

            # Replace logger.write with new_write
            logger.write = new_write
            try:
                result = func(self, *args, **kwargs)
            finally:
                # Restore the original logger.write method
                logger.write = original_write

            return result

        return wrapper


    @staticmethod
    def debug(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Wrap the original logger.debug method
            original_debug = logger.debug

            def new_debug(message):
                if self.__class__.__name__ in message:
                    full_message = message
                else:
                    full_message = f"# {self.__class__.__name__} # {message}"
                original_debug(full_message)

            # Replace logger.debug with new_debug
            logger.debug = new_debug
            try:
                result = func(self, *args, **kwargs)
            finally:
                # Restore the original logger.debug method
                logger.debug = original_debug

            return result

        return wrapper


    @staticmethod
    def error(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Wrap the original logger.error method
            original_error = logger.error

            def new_error(message):
                if self.__class__.__name__ in message:
                    full_message = message
                else:
                    full_message = f"# {self.__class__.__name__} # {message}"
                original_error(full_message)

            # Replace logger.error with new_error
            logger.error = new_error
            try:
                result = func(self, *args, **kwargs)
            finally:
                # Restore the original logger.error method
                logger.error = original_error

            return result

        return wrapper


class BaseModuleProvider(BaseProvider):

    def __init__(self, *args, template_path, **kwargs):
        super().__init__(*args, **kwargs)
        self.template_path = template_path
        self.template, self.input_keys, self.output_keys = self._extract_keys_from_template()


    @BaseProvider.write
    def _extract_keys_from_template(self):
        template_path = assemble_project_path(self.template_path)
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template file {template_path} does not exist")

        template = read_resource_file(template_path)

        # <$few_shots$> -> few_shots
        parse_input_keys = re.findall(r'<\$(.*?)\$>', template)
        input_keys = [key.strip() for key in parse_input_keys]
        logger.write(f"Recommended input parameters: {input_keys}")

        # TODO: Extract output text should be general
        start_output_line_index = template.find('You should only respond')
        output_text = template[start_output_line_index + 1:]
        output = parse_semi_formatted_text(output_text)
        output_keys = list(output.keys())
        logger.write(f"Recommended output parameters: {output_keys}")

        return template, input_keys, output_keys


    @BaseProvider.write
    def _check_input_keys(self, params: Dict[str, Any]):
        for key in self.input_keys:
            if key not in params:
                logger.write(f"Key {key} is not in the input parameters")
                params[key] = None


    @BaseProvider.error
    def _check_output_keys(self, response: Dict[str, Any]):
        for key in self.output_keys:
            if key not in response:
                logger.error(f"Key {key} is not in the response")
                response[key] = None
