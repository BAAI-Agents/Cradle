import logging
import os
import psutil
from pathlib import Path
import sys

from colorama import Fore, Back, Style, init as colours_on

from cradle.utils import Singleton

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
    COLOURS = {
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "DEBUG": Fore.GREEN,
        "INFO": Fore.WHITE,
        "CRITICAL": Fore.RED + Back.WHITE
    }

    def format(self, record):
        color = self.COLOURS.get(record.levelname, "")
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

    log_dir = './logs'
    work_dir = None

    DOWNSTREAM_MASK = "\n>> Downstream - A:\n"
    UPSTREAM_MASK = "\n>> Upstream - R:\n"

    def __init__(self, work_dir=None):

        self.to_file = False

        if work_dir is not None:
            self.work_dir = work_dir

        self._configure_root_logger()


    def _configure_root_logger(self):

        # format = f'%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        format = '%(asctime)s - %(name)s - CPU: %(cpu_usage)s%%, Memory: %(memory_usage)s%% - %(levelname)s - %(message)s'

        formatter = CPUMemFormatter(format)
        c_formatter = CPUMemColorFormatter(format)

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.INFO)
        stdout_handler.setFormatter(c_formatter)

        stderr_handler = logging.StreamHandler()
        stderr_handler.setLevel(logging.ERROR)
        stderr_handler.setFormatter(c_formatter)

        handlers = [stdout_handler, stderr_handler]

        if self.work_dir is not None:
            self.log_dir = os.path.join(self.work_dir, self.log_dir)
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(filename=os.path.join(self.log_dir, self.log_file), mode='w', encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)

            handlers.append(file_handler)

        logging.basicConfig(level=logging.DEBUG, handlers=handlers)
        self.logger = logging.getLogger("UAC Logger")

        if len(handlers) == 2:
            self.logger.warn('Work directory not set. Logging to console only???')


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
