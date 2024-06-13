from cradle.utils import Singleton
from functools import wraps
from cradle.log import Logger

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