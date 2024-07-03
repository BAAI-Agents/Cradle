from typing import Dict, Any, List, Tuple
import platform
import ctypes
import re

import pyautogui

from cradle.utils.string_utils import strip_anchor_chars
from cradle.config import Config


config = Config()


def _isMac():
    return platform.system() == "Darwin"


def _isWin():
    return platform.system() == "Windows"


if _isWin():
    from ahk import AHK
    import pydirectinput

    # Windows API constants
    MOUSEEVENTF_MOVE = 0x0001
    MOUSEEVENTF_ABSOLUT = 0x8000
    WIN_NORM_MAX = 65536 # int max val

    ahk = AHK()

    # PyDirectInput is only used for key pressing, so no need for mouse checks
    pydirectinput.FAILSAFE = False

    PUL = ctypes.POINTER(ctypes.c_ulong)


    class KeyBdInput(ctypes.Structure):
        _fields_ = [
            ("wVk", ctypes.c_ushort),
            ("wScan", ctypes.c_ushort),
            ("dwFlags", ctypes.c_ulong),
            ("time", ctypes.c_ulong),
            ("dwExtraInfo", PUL),
        ]


    class HardwareInput(ctypes.Structure):
        _fields_ = [
            ("uMsg", ctypes.c_ulong),
            ("wParamL", ctypes.c_short),
            ("wParamH", ctypes.c_ushort),
        ]


    class MouseInput(ctypes.Structure):
        _fields_ = [
            ("dx", ctypes.c_long),
            ("dy", ctypes.c_long),
            ("mouseData", ctypes.c_ulong),
            ("dwFlags", ctypes.c_ulong),
            ("time", ctypes.c_ulong),
            ("dwExtraInfo", PUL),
        ]


    class Input_I(ctypes.Union):
        _fields_ = [("ki", KeyBdInput), ("mi", MouseInput), ("hi", HardwareInput)]


    class Input(ctypes.Structure):
        _fields_ = [("type", ctypes.c_ulong), ("ii", Input_I)]


    def _mouse_coord_to_abs_win(coord, width_or_height):
        abs_coord = ((WIN_NORM_MAX * coord) / width_or_height) + (-1 if coord < 0 else 1)
        return int(abs_coord)

elif _isMac():
    import Quartz


class TargetWindow():

    def __init__(self, window):
        self.window = window
        self.system = platform.system()
        self._set_sizes(window)


    def __repr__(self):
        return f"EnvWindow({self.title}, {self.left}, {self.top}, {self.width}, {self.height})"


    def __str__(self):
        return f"EnvWindow({self.title}, {self.left}, {self.top}, {self.width}, {self.height})"


    def _set_sizes(self, window):

        if _isWin():
            self.left = window.left
            self.top = window.top
            self.width = window.width
            self.height = window.height
            self.title = window.title
        elif _isMac():
            bounds = self.window.get("kCGWindowBounds", "")
            self.left = int(bounds["X"])
            self.top = int(bounds["Y"])
            self.width = int(bounds["Width"])
            self.height = int(bounds["Height"])
            self.title = window.get("kCGWindowOwnerName", "")
        else:
            raise ValueError(f"Platform {self.system} not supported yet")


    def _run_applescript(self, applescript_command, capture_output=False):

        if _isMac():
            import subprocess
            if capture_output is True:
                result = subprocess.run(["osascript", "-e", applescript_command],
                                        capture_output = capture_output,
                                        text=True)
                return result.stdout
            else:
                subprocess.run(["osascript", "-e", applescript_command])
                return True


    def activate(self):
        """
        Activate the target window
        """
        if _isWin():
            self.window.activate()
        elif _isMac():
            env_name = self.title
            if ":" in env_name:
                env_name = env_name.split(":")[0]

            applescript_command = f"""
            tell application "{env_name}"
                activate
            end tell
            """
            self._run_applescript(applescript_command)


    def deactivate(self):
        """
        Deactivate the target window
        """
        if _isWin():
            self.window.hide()
        elif _isMac():
            env_name = self.title
            applescript_command = f"""
                                  tell application "System Events" to tell process "{env_name}"
                                      set visible to false
                                  end tell"""
            self._run_applescript(applescript_command)


    def is_active(self):
        """
        Check if the target window is active
        """
        if _isWin():
            return self.window.isActive
        elif _isMac():
            env_name = self.title
            applescript_command = f"""
            tell application "System Events" to tell process "{env_name}"
                return frontmost
            end tell"""
            result = self._run_applescript(applescript_command, capture_output=True)
            return result.strip() == "true"


    def minimize(self):
        """
        Minimize the target window
        """
        if _isWin():
            self.window.minimize()
        elif _isMac():
            env_name = self.title
            applescript_command = f"""
            tell application "System Events" to tell process "{env_name}"
                set frontmost to true
                click button 3 of window 1
            end tell"""
            self._run_applescript(applescript_command)


    def maximize(self):
        """
        Maximize the target window
        """
        if _isWin():
            self.window.maximize()
        elif _isMac():
            env_name = self.title
            applescript_command = f"""
            tell application "System Events" to tell process "{env_name}"
                set frontmost to true
                click button 2 of window 1
            end tell"""
            self._run_applescript(applescript_command)


    def hide(self):
        """
        Hide the target window
        """
        if _isWin():
            self.window.hide()
        elif _isMac():
            env_name = self.title
            applescript_command = f"""
            tell application "System Events" to tell process "{env_name}"
                set visible to false
            end tell"""
            self._run_applescript(applescript_command)


    def show(self):
        """
        Show the target window
        """
        if _isWin():
            self.window.show()
        elif _isMac():
            env_name = self.title
            applescript_command = f"""
            tell application "System Events" to tell process "{env_name}"
                set visible to true
            end tell"""
            self._run_applescript(applescript_command)


    def is_visible(self):
        """
        Check if the target window is visible
        """
        if _isWin():
            return self.window.isShowing
        elif _isMac():
            env_name = self.title
            applescript_command = f"""
            tell application "System Events" to tell process "{env_name}"
                return visible
            end tell"""
            result = self._run_applescript(applescript_command, capture_output=True)
            return result.strip() == "true"


    def moveTo(self, set_left, set_top):
        """
        Move the target window to the specified coordinates
        """
        if _isWin():
            self.window.moveTo(set_left, set_top)
        elif _isMac():
            env_name = self.title

            window = get_named_windows(env_name)[0]

            left = window.left
            top = window.top
            width = window.width
            height = window.height

            pyautogui.moveTo(left + width // 2, top + 10, duration=1.0)
            pyautogui.mouseDown()
            pyautogui.moveRel(-left + set_left, -top + set_top, duration=1.0)
            pyautogui.mouseUp()

        # Because the desktop may have bar limitations, the window should be reacquired
        window = get_named_windows(self.title)[0]
        self._set_sizes(window)
        return window


    def resizeTo(self, set_width, set_height):
        """
        Resize the target window to the specified width and height
        """

        if _isWin():
            if self.window.isMaximized:
                self.window.restore()
            self.window.resizeTo(set_width, set_height)

        elif _isMac():
            env_name = self.title

            window = get_named_windows(env_name)[0]

            left = window.left
            top = window.top
            width = window.width
            height = window.height

            right, bottom = left + width, top + height

            pyautogui.moveTo(right, bottom, duration=1.0)
            pyautogui.mouseDown()
            pyautogui.moveRel(set_width - width, set_height - height, duration=1.0)
            pyautogui.mouseUp()

        # Because the desktop may have bar limitations, the window should be reacquired
        window = get_named_windows(self.title)[0]
        self._set_sizes(window)
        return window


def mouse_button_down(button):
    if _isWin() and config.is_game is True:
        ahk.click(button=button, direction='D')
    else:
        pyautogui.mouseDown(button=button, duration=0.2)


def mouse_button_up(button):
    if _isWin() and config.is_game is True:
        ahk.click(button=button, direction='U')
    else:
        pyautogui.mouseUp(button=button, duration=0.2)


def mouse_click(click_count, button, relative=False):
    if _isWin() and config.is_game is True:
        ahk.click(click_count=click_count, button=button, relative=relative)
    else:
        for i in range(click_count):
            mouse_button_down(button)
            mouse_button_up(button)


def mouse_move_to(x, y, duration = -1, relative = False, screen_resolution = None, env_region = None):

    if _isWin():

        if duration > -1:
            raise ValueError("Duration is not yet supported on Windows.")

        timestamp = 0
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()

        # logger.debug(f'game coord x {x} y {y} relative {relative}')

        event_flag = MOUSEEVENTF_MOVE

        if relative is False:
            event_flag = MOUSEEVENTF_ABSOLUT | MOUSEEVENTF_MOVE
            x = x + env_region[0]
            y = y + env_region[1]

            # logger.debug(f'screen x {x} y {y}')

            x = _mouse_coord_to_abs_win(x, screen_resolution[0])
            y = _mouse_coord_to_abs_win(y, screen_resolution[1])

            # logger.debug(f'windows x {x} y {y}')

        ii_.mi = MouseInput(int(x), int(y), 0, event_flag, timestamp, ctypes.pointer(extra))

        command = Input(ctypes.c_ulong(0), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(command), ctypes.sizeof(command))

    else:
        if relative == True:
            pyautogui.move(x, y, duration=duration)
        else:
            pyautogui.moveTo(x, y, duration=duration)


def mouse_wheel_scroll(amount):
    pyautogui.scroll(clicks = amount)


def get_mouse_location(absolute = False):
    if _isWin() and config.is_game is True:
        return ahk.get_mouse_position()
    else:
        p = pyautogui.position()

        if absolute is True:
            return (p.x, p.y)

        x = p.x - config.env_window.left
        y = p.y - config.env_window.top

        return (x, y)


def key_down(key):
    if _isWin():
        pydirectinput.keyDown(key)
    else:
        pyautogui.keyDown(key)


def key_up(key):
    if _isWin():
        pydirectinput.keyUp(key)
    else:
        pyautogui.keyUp(key)


def type_keys(keys):
    pyautogui.typewrite(keys, interval=0.2)


def get_screen_size():
    return pyautogui.size()


def check_window_conditions(env_window: TargetWindow):
    # Check if pre-resize is necessary
    if not config._min_resolution_check(env_window) or not config._aspect_ration_check(env_window):
        env_window = env_window.resizeTo(config.DEFAULT_ENV_RESOLUTION[0], config.DEFAULT_ENV_RESOLUTION[1])

    # Workaround for dialogs
    if env_window.width >= 300 and env_window.width <= 355 and env_window.height >= 130 and env_window.height <= 200:
        pass # Likely a dialog?
    else:
        assert config._min_resolution_check(env_window), 'The resolution of env window should at least be 1920x1080.'
        assert config._aspect_ration_check(env_window), 'The screen ratio should be 16:9.'


def get_active_window():

    if platform.system() == "Windows":
        try:
            window = pyautogui.getActiveWindow()
            window = TargetWindow(window)
            return window

        except AttributeError:
            return None

    elif platform.system() == "Darwin":
        window_list = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionAll,
            Quartz.kCGNullWindowID
        )

        frontmost_app_pid = Quartz.CGWindowListCopyWindowInfo(Quartz.kCGWindowListOptionOnScreenOnly, 0)[0]['kCGWindowOwnerPID']

        for window in window_list:
            if window['kCGWindowOwnerPID'] == frontmost_app_pid:
                if 'kCGWindowName' in window and window['kCGWindowName']:
                    window = TargetWindow(window)
                    return window
        return None

    else:
        raise ValueError(f"Platform {platform.system()} not supported yet")


def get_named_windows(env_name) -> List[TargetWindow]:

    clean_name = strip_anchor_chars(env_name)

    if _isWin():
        windows = pyautogui.getWindowsWithTitle(clean_name)

        if clean_name == env_name:
            windows = [TargetWindow(window) for window in windows]
        else:
            windows = [TargetWindow(window) for window in windows if re.search(env_name, window.title)]
        return windows

    elif _isMac():
        windows = []

        window_list = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionAll,
            Quartz.kCGNullWindowID
            )

        for window in window_list:
            owner_name = window.get("kCGWindowOwnerName", "")
            if owner_name == clean_name or re.search(env_name, owner_name):
                windows.append(TargetWindow(window))
        return windows

    else:
        raise ValueError(f"Platform {platform.system()} not supported yet")


def get_named_windows_fallback(win_name, win_name_pattern) -> List[TargetWindow]:
    # Get window candidates by name alternatives
    named_windows = get_named_windows(win_name)

    if (len(named_windows) == 0 or len(named_windows) > 1) and len(win_name_pattern) > 0:
        named_windows = get_named_windows(win_name_pattern)

    return named_windows


def is_top_level_window(window_handle: int) -> Tuple[bool, int]:

    if _isWin():
        parent_window_handle = get_parent_window_handle(window_handle)
        return (parent_window_handle == 0), parent_window_handle
    else:
        raise ValueError(f"Platform {platform.system()} not supported yet")


def get_parent_window_handle(window_handle: int) -> int:

    if _isWin():
        import win32gui
        return win32gui.GetParent(window_handle)
    else:
        raise ValueError(f"Platform {platform.system()} not supported yet")


def _get_active_window():
    return pyautogui.getActiveWindow()
