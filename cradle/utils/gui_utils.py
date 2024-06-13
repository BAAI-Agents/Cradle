import platform
import ctypes
import re

import pyautogui

from cradle.utils.string_utils import contains_regex_characters, strip_anchor_chars


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
                result = subprocess.run(["osascript", "-e", applescript_command, capture_output])
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
            return result.stdout == b"true\n"


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
            return result.stdout == b"true\n"


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
    if _isWin():
        ahk.click(button=button, direction='D')
    else:
        pyautogui.mouseDown(button=button, duration=0.2)


def mouse_button_up(button):
    if _isWin():
        ahk.click(button=button, direction='U')
    else:
        pyautogui.mouseUp(button=button, duration=0.2)


def mouse_click(click_count, button, relative=False):
    if _isWin():
        ahk.click(click_count=click_count, button=button, relative=relative)
    else:
        for i in range(click_count):
            mouse_button_down(button)
            mouse_button_up(button)


def mouse_move_to(x, y, duration = -1, relative = False, screen_resolution = None, env_region = None):

    if _isWin():
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

        ii_.mi = MouseInput(int(x), int(y), 0, event_flag, 0, ctypes.pointer(extra))

        command = Input(ctypes.c_ulong(0), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(command), ctypes.sizeof(command))
    else:
        if relative == True:
            pyautogui.move(x, y, duration=duration)
        else:
            pyautogui.moveTo(x, y, duration=duration)


def get_mouse_location():
    if _isWin():
        return ahk.get_mouse_position()
    else:
        p = pyautogui.position()
        return (p.x, p.y)


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


def get_screen_size():
    return pyautogui.size()

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


def get_named_windows(env_name):

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


def get_named_windows_fallback(win_name, win_name_pattern):
    # Get window candidates by name alternatives
    named_windows = get_named_windows(win_name)

    if (len(named_windows) == 0 or len(named_windows) > 1) and len(win_name_pattern) > 0:
        named_windows = get_named_windows(win_name_pattern)

    return named_windows
