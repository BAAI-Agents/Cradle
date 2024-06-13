import json
import logging
import os
import re
import sys
import ast
import psutil
from pathlib import PureWindowsPath

from colorama import Fore, Back, Style, init as colours_on

from cradle.utils import Singleton
from cradle.config import Config
from cradle.utils.encoding_utils import decode_base64
from cradle.utils.string_utils import hash_text_sha256


config = Config()
colours_on(autoreset=True)


class CPUMemFormatter(logging.Formatter):
    def format(self, record):
        cpu_usage = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        memory_usage = memory.percent

        record.cpu_usage = cpu_usage
        record.memory_usage = memory_usage

        return super().format(record)

class CPUMemColorFormatter(logging.Formatter):

    # Change your colours here. Should use extra from log calls.
    COLORS = {
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "DEBUG": Fore.GREEN,
        "INFO": Fore.WHITE,
        "CRITICAL": Fore.RED + Back.WHITE
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, "")
        if color:
            record.name = color + record.name
            record.msg = record.msg + Style.RESET_ALL

        record.cpu_usage = psutil.cpu_percent(interval=None)
        record.memory_usage = psutil.virtual_memory().percent

        cpu_usage = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        memory_usage = memory.percent

        record.cpu_usage = cpu_usage
        record.memory_usage = memory_usage

        return super().format(record)


class Logger(metaclass=Singleton):

    log_file = 'cradle.log'

    DOWNSTREAM_MASK = "\n>> Downstream - A:\n"
    UPSTREAM_MASK = "\n>> Upstream - R:\n"

    def __init__(self):
        self.to_file = False
        self._configure_root_logger()


    def _configure_root_logger(self):

        format = '%(asctime)s - %(name)s - CPU: %(cpu_usage)s%%, Memory: %(memory_usage)s%% - %(levelname)s - %(message)s'

        formatter = CPUMemFormatter(format)
        c_formatter = CPUMemColorFormatter(format)

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.INFO)
        stdout_handler.setFormatter(c_formatter)

        stderr_handler = logging.StreamHandler()
        stderr_handler.setLevel(logging.ERROR)
        stderr_handler.setFormatter(c_formatter)

        file_handler = logging.FileHandler(filename=os.path.join(config.log_dir, self.log_file), mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        logging.basicConfig(level=logging.DEBUG, handlers=[stdout_handler, stderr_handler, file_handler])
        self.logger = logging.getLogger("UAC Logger")


    def _log(
            self,
            title="",
            title_color=Fore.WHITE,
            message="",
            level=logging.INFO
        ):

        if message:
            if isinstance(message, list):
                message = " ".join(message)

        self.logger.log(level, message, extra={"title": title, "color": title_color})

    def critical(
            self,
            message,
            title=""
        ):

        self._log(title, Fore.RED + Back.WHITE, message, logging.ERROR)

    def error(
            self,
            message,
            title=""
        ):

        self._log(title, Fore.RED, message, logging.ERROR)

    def debug(
            self,
            message,
            title="",
            title_color=Fore.GREEN,
        ):

        self._log(title, title_color, message, logging.DEBUG)

    def write(
            self,
            message="",
            title="",
            title_color=Fore.WHITE,
        ):

        self._log(title, title_color, message, logging.INFO)

    def warn(
            self,
            message,
            title="",
            title_color=Fore.YELLOW,
        ):

        self._log(title, title_color, message, logging.WARN)


    def error_ex(self, exception: Exception):
        traceback = exception.__traceback__
        while traceback:
            self.error("{}: {}".format(traceback.tb_frame.f_code.co_filename, traceback.tb_lineno))
            traceback = traceback.tb_next


#
# Functions below this line are just auxiliary to code to process a log file to facilitate debug. Code is very brittle.
#
def _extract_text_between_tokens(text, start_token="(?<=;base64,)", end_token="(?=\")", escape = False):

    # Escape the tokens if they contain special regex characters
    if escape is True:
        start_token = re.escape(start_token)
        end_token = re.escape(end_token)

    # Regex pattern to capture text between start_token and end_token
    pattern = rf'{start_token}(.*?){end_token}'

    # Extracting all occurrences
    extracted_texts = re.findall(pattern, text)

    return extracted_texts

def _extract_image_between_tokens(text, start_token=";base64,", end_tokens=["\"", "'"], escape=False):
    """
    Extracts text between the specified start and end tokens.

    Args:
        text (str): The input text from which to extract the text.
        start_token (str): The token indicating the start of the text to extract.
        end_tokens (list): A list of tokens indicating the end of the text to extract.
        escape (bool): Whether to escape the tokens if they contain special regex characters.

    Returns:
        list: A list of extracted text segments.
    """
    # Escape the tokens if they contain special regex characters
    if escape:
        start_token = re.escape(start_token)
        end_tokens = [re.escape(token) for token in end_tokens]

    # Combine end tokens into a single pattern for non-greedy matching
    end_token_pattern = '|'.join(end_tokens)

    # Regex pattern to capture text between start_token and any of the end tokens
    pattern = rf'{start_token}(.*?)(?={end_token_pattern})'

    # Extracting all occurrences
    extracted_texts = re.findall(pattern, text)

    return extracted_texts


def _replacer(text, encoded_images, image_paths, work_dir):

    if image_paths is None or len(image_paths) == 0:
        image_paths = ['<$img_placeholder$>']

    for i in range(len(encoded_images)):
        if len(image_paths) == 1 and len(encoded_images) > len(image_paths):
            paths_idx = 0
            text = text.replace(encoded_images[i], image_paths[paths_idx])
        else:
            key = hash_text_sha256(encoded_images[i])

            encoded_image = encoded_images[i]

            if key not in image_paths.keys() or image_paths[key] == '<$bin_placeholder$>':
                # Re-construct image, then replace
                file_name = f"base64_rec_{i}.jpg"
                path = os.path.join(work_dir, file_name)

                with open(path, "wb") as f:
                    f.write(decode_base64(encoded_image))

                image_paths[key] = path
            else:
                path = image_paths[key]
                if not os.path.exists(path):
                    with open(path, "wb") as f:
                        f.write(decode_base64(encoded_image))

            text = text.replace(encoded_image, json.dumps(image_paths[key], ensure_ascii=False).strip('"'))

    return text


def _extract_image_hashes(text):

    map = dict()

    hash_list = _extract_text_between_tokens(text, "|>.", ".<|", escape = True)
    for i in range(len(hash_list)):
        # Extract mapping info
        hash_info = hash_list[i].split(" ")
        map[hash_info[2].split(",")[0]] = hash_info[4]

        # Remove line from log
        text = text.replace(hash_list[i], "")

    return (map, text)


def process_string(input_str):

    processed_str = input_str.replace("\\", "\\\\")
    processed_str = processed_str.replace("'text': \"", "'text': '").replace(".\"}]}", ".'}]}")
    processed_str = processed_str.replace("\"", "\\\"")

    try:
        msgs = ast.literal_eval(processed_str)

        json_obj = msgs
    except SyntaxError as e:
        print(f"Syntax error: {e}")
        print("Error may be near:", processed_str[max(0, e.offset - 10):e.offset + 10])
    except json.JSONDecodeError as e: #SyntaxError
        print()
        error_position = e.pos
        start = max(0, error_position - 10)  # Show 10 characters before the error position
        end = min(len(processed_str), error_position + 10)  # Show 10 characters after the error position
        span = processed_str[start:end]
        print(f'Error near: {span}')
        print(f'Exact error position: {error_position}')

    return json_obj


def process_log_messages(work_dir):

    log_path = os.path.join(work_dir, "logs/cradle.log")

    with open(log_path, "r", encoding="utf-8") as fd:
        log = fd.read()

    filter_list = ['|>..<|', 'httpcore.http11 - DEBUG', 'httpcore.connection - DEBUG', 'asyncio - DEBUG', 'httpx - DEBUG', 'matplotlib.pyplot - DEBUG', 'openai._base_client - DEBUG - Request options:']

    log_lines = []
    for line in log.split('\n'):
        if any(substring in line for substring in filter_list):
            continue
        log_lines.append(line)
    log = '\n'.join(log_lines)

    hash_file_maps, log = _extract_image_hashes(log)
    encoded_images = _extract_image_between_tokens(log)
    log = _replacer(log, encoded_images, hash_file_maps, work_dir)

    md_log = []
    img_start_token = ';base64,'
    img_end_token = 'g"'

    msg_start_token = '[{"role": "system",'
    msg_end_token = '}]}]'

    for line in log.split('\n'):

        if msg_start_token in line:
            candidates = _extract_text_between_tokens(line, msg_start_token, msg_end_token, escape=True)
            if len(candidates) > 0:
                msgs = f'{msg_start_token}{candidates[0]}{msg_end_token}'#.replace('\\', '\\\\')
                obj = json.loads(msgs)
                obj_str = json.dumps(obj, indent=4, ensure_ascii=False)
                line = "\n````text\n" + obj_str + "\n````\n\n"

        if img_start_token in line:
            candidates = _extract_text_between_tokens(line, img_start_token, img_end_token)
            for candidate in candidates:
                norm_path = os.path.normpath(candidate+'g')
                norm_work_dir = PureWindowsPath(os.path.normpath(work_dir)).as_posix()
                rel_path = PureWindowsPath(os.path.relpath(norm_path, norm_work_dir)).as_posix()
                new_substxt = "\n````\n" + f'![{norm_path}](../{rel_path})'.replace('\\','/').replace('//','/') + "\n````text\n"
                line = line.replace(candidate, new_substxt)
        elif re.match('^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}', line):
            line = f'{line}\n'

        if any(substring in line for substring in filter_list):
            continue

        md_log.append(line)

    log = '\n'.join(md_log)

    return log
