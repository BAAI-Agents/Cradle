from typing import (
    Any,
    Dict,
    List,
    Tuple,
)
import os
import time

from cradle.utils import Singleton
from cradle.config import Config
from cradle.log import Logger
from cradle.gameio.gui_utils import get_named_windows, get_named_windows_fallback, get_screen_size, mouse_button_down, mouse_button_up, key_down, key_up, mouse_wheel_scroll, type_keys, mouse_click, get_mouse_location, mouse_move_to

config = Config()
logger = Logger()

class IOEnvironment(metaclass=Singleton):
    """
    Wrapper for resources to interact with the game to make sure they're available where needed and multiple instances are not created.
    """
    # Constants
    RIGHT_MOUSE_BUTTON = 'Right'
    LEFT_MOUSE_BUTTON = 'Left'
    MIDDLE_MOUSE_BUTTON = 'Middle'
    right_mouse_button = RIGHT_MOUSE_BUTTON
    left_mouse_button = LEFT_MOUSE_BUTTON
    middle_mouse_button = MIDDLE_MOUSE_BUTTON
    WHEEL_UP_MOUSE_BUTTON = 'WU'
    WHEEL_DOWN_MOUSE_BUTTON = 'WD'

    MIN_DURATION = 2 # In seconds

    HOLD_DEFAULT_BLOCK_TIME = 2
    RELEASE_DEFAULT_BLOCK_TIME = 0.5

    MAX_ITERATIONS = 3

    KEY_KEY = 'key'
    BUTTON_KEY = 'button'
    EXPIRATION_KEY = 'expiration'

    # All key interactions are now tracked and use the same calling structure
    # - Release is equivalent to keyUp. I.e., release a key that was pressed or held.
    # - Hold is equivalent to keyDown. I.e., hold a key for a certain duration, probably while something else happens.
    # - Press is equivalent to keyDown followed by keyUp, after a delay. I.e., press a key for a short duration.
    ACTION_PRESS = 'press' # Equivalent to click on the mouse
    ACTION_HOLD = 'hold'
    ACTION_RELEASE = 'release'
    MOUSE_TYPE = 'is_mouse'
    KEY_TYPE = 'is_keyboard'

    # List of keys currently held. To be either released by specific calls or after timeout (max iterations).
    # {
    #     self.KEY_KEY: key,
    #     self.EXPIRATION_KEY: self.MAX_ITERATIONS
    # }
    held_keys = []
    held_buttons = []

    # Used currently due to an issue with pause in RDR2
    backup_held_keys = []
    backup_held_buttons = []

    # Temporary handle to main initialized LLM provider
    llm_provider = None

    def __init__(self) -> None:
        """Initialize the IO environment class"""
        pass

    def pop_held_button(self, button):

        self._mouse_button_up(button)

        # Remove from held list
        for i in range(len(self.held_buttons)):
            if self.held_buttons[i][self.BUTTON_KEY] == button:
                self.held_buttons.pop(i)
                break

        time.sleep(self.RELEASE_DEFAULT_BLOCK_TIME)

        self._to_message(self.held_buttons, self.ACTION_RELEASE, self.MOUSE_TYPE)


    def put_held_button(self, button):

        for e in self.held_buttons:
            if e[self.BUTTON_KEY] == button:
                logger.warn(f'Button {button} already being held.')
                return
        else:
            entry = {
                self.BUTTON_KEY: button,
                self.EXPIRATION_KEY: self.MAX_ITERATIONS
            }
            self.held_buttons.append(entry)

            self._mouse_button_down(button)

            time.sleep(self.HOLD_DEFAULT_BLOCK_TIME)

            self._to_message(self.held_buttons, self.ACTION_HOLD, self.MOUSE_TYPE)


    def _mouse_button_down(self, button):
        mouse_button_down(button)


    def _mouse_button_up(self, button):
        mouse_button_up(button)


    def pop_held_keys(self, key):

        if self.check_held_keys(keys = [key]):
            key_up(key)
            time.sleep(self.RELEASE_DEFAULT_BLOCK_TIME)
            self.held_keys.pop()
        else:
            key_up(key) # Just as a guarantee to up an untracked key
            logger.warn(f'Key {key} was not being held at top.')

        self._to_message(self.held_keys, self.ACTION_RELEASE, self.KEY_TYPE)


    def put_held_keys(self, key):

        top_key = _safe_list_get(self.held_keys, -1, self.KEY_KEY)
        if key == top_key:
            logger.warn(f'Key {key} already being held.')
        else:
            entry = {
                self.KEY_KEY: key,
                self.EXPIRATION_KEY: self.MAX_ITERATIONS
            }
            self.held_keys.append(entry)

            key_down(key)

            time.sleep(self.HOLD_DEFAULT_BLOCK_TIME)

            self._to_message(self.held_keys, self.ACTION_HOLD, self.KEY_TYPE)


    def check_held_keys(self, keys):

        result = False

        if keys is not None and len(keys) != 0:
            for e in self.held_keys:
                k = e[self.KEY_KEY]
                if k in keys:
                    result = True
                    break

        return result


    def _to_message(self, list, purpose, type):

        if type == self.KEY_TYPE:
            vals = ', '.join(f'{e[self.KEY_KEY]}:{e[self.EXPIRATION_KEY]}' for e in list)
            msg = f'Held keys after {purpose}: {vals}'
        elif type == self.MOUSE_TYPE:
            vals = ', '.join(f'{e[self.BUTTON_KEY]}:{e[self.EXPIRATION_KEY]}' for e in list)
            msg = f'Held button after {purpose}: {vals}'

        logger.write(msg)

        return msg


    def update_timeouts(self):

        if self.held_keys is None or len(self.held_keys) == 0:
            return

        tmp_list = []

        for e in self.held_keys:

            t = e[self.EXPIRATION_KEY] - 1
            if t <= 0:
                key = e[self.KEY_KEY]
                logger.warn(f'Releasing key {key} after timeout.')
                key_up(key)
                time.sleep(0.1)

            else:
                e[self.EXPIRATION_KEY] = t
                tmp_list.append(e)

        self.held_keys = tmp_list.copy()
        del tmp_list

        tmp_list = []

        for e in self.held_buttons:

            t = e[self.EXPIRATION_KEY] - 1
            if t <= 0:
                button = e[self.BUTTON_KEY]
                logger.warn(f'Releasing mouse button {button} after timeout.')
                self._mouse_button_up(button)
                time.sleep(0.1)

            else:
                e[self.EXPIRATION_KEY] = t
                tmp_list.append(e)

        self.held_buttons = tmp_list.copy()
        del tmp_list


    def handle_hold_in_pause(self):
        self.backup_held_keys = self.held_keys.copy()
        if self.backup_held_keys is not None and self.backup_held_keys != []:
            for e in self.backup_held_keys:
                key_up(e[self.KEY_KEY])

        self.held_keys = []

        self.backup_held_buttons = self.held_buttons.copy()
        if self.backup_held_buttons is not None and self.backup_held_buttons != []:
            for e in self.backup_held_buttons:
                self._mouse_button_up(e[self.BUTTON_KEY])

        self.held_buttons = []


    def handle_hold_in_unpause(self):
        buttons_hold = False
        keys_hold = False
        if self.backup_held_buttons is not None and self.backup_held_buttons != []:
            for e in self.backup_held_buttons:
                self._mouse_button_down(e[self.BUTTON_KEY])

            buttons_hold = True

            self.held_buttons = self.backup_held_buttons.copy()

        time.sleep(.1)

        if self.backup_held_keys is not None and self.backup_held_keys != []:
            for e in self.backup_held_keys:
                key_down(e[self.KEY_KEY])

            keys_hold = True

            self.held_keys = self.backup_held_keys.copy()

        if buttons_hold or keys_hold:
            time.sleep(1)


    def list_session_screenshots(self, session_dir: str = config.work_dir):

        # List all files in dir starting with "screen"
        screenshots = [f for f in os.listdir(session_dir) if os.path.isfile(os.path.join(session_dir, f)) and f.startswith("screen")]

        # Sort list by creation time
        screenshots.sort(key=lambda x: os.path.getctime(os.path.join(session_dir, x)))

        return screenshots


    def mouse_move_normalized(self, x, y, relative = False, from_center = False):

        logger.debug(f'Normalized game coord x {x} y {y} relative {relative} fc {from_center}')

        w, h = config.env_resolution

        offset = 0
        if from_center is True:
            offset = .5 # Center of the game screen in normalized coordinates

        gx = int((x - offset) * w)
        gy = int((y - offset) * h)

        self.mouse_move(x = gx, y = gy, relative = relative)


    # If either relative or not, always pass in-game coordinates
    # This implementation is not fully functional and was intended to address game-category specific issues first
    def mouse_move(self, x, y, duration = -1, relative = False):
        mouse_move_to(x, y, duration, relative, config.screen_resolution, config.env_region)


    def mouse_drag(self, x1, y1, x2, y2):
        self.mouse_move(x1, y1)
        self.mouse_hold('left')
        self.mouse_move(x2, y2)
        self.mouse_release('left')


    def mouse_move_horizontal_angle(self, theta):
        distance = _theta_calculation(theta)
        self.mouse_move(distance, 0, relative=True)


    def mouse_click_button(self, button, duration = None, clicks=1):

        button = self.map_button(button)

        if duration is None:
            mouse_click(click_count=clicks, button=button, relative=False)
        else:
            self._mouse_button_down(button)
            time.sleep(duration)
            self._mouse_button_up(button)


    def mouse_hold(self, button, duration = None):

        if duration is None:
            self.mouse_hold_button(button)
        else:
            self._mouse_button_down(button)
            time.sleep(duration)
            self._mouse_button_up(button)


    def mouse_hold_button(self, button):

        button = self.map_button(button)

        self.put_held_button(button)


    def mouse_release(self, button):
        self.mouse_release_button(button)


    def mouse_release_button(self, button):

        button = self.map_button(button)

        self.pop_held_button(button)


    def get_mouse_position(self, absolute = False) -> Tuple[int, int]:
        return get_mouse_location(absolute)


    def clip_check_horizonal_angle(self, theta):
        result = False

        pixels = _theta_calculation(theta)
        mx, _ = self.get_mouse_position()

        if pixels > 0 and mx + pixels > config.env_resolution[0]:
            result = True
        elif pixels < 0 and mx + pixels < 0:
            result = True

        return result


    def mouse_scroll(self, direction, distance):

        # Distance is based on click units and varies by OS. On Windows, one click in the wheel circumference is 120 units.

        # Positive/negative scrolls may be affected by OS config.
        if direction == self.WHEEL_DOWN_MOUSE_BUTTON:
            distance = -distance

        mouse_wheel_scroll(amount=distance)


    def _check_multi_key(self, input):

        if input is not None and len(input) > 1:
            if type(input) is list:
                return (True, input)
            else:
                key_tokens = input.split(',')
                keys = []
                for k in key_tokens:
                    k = k.strip()
                    if k != '':
                        k = self.map_key(k)
                        keys.append(k)

                if len(keys) == 0:
                    return (False, None)
                elif len(keys) == 1:
                    return (False, keys[0])
                else:
                    return (True, keys)

        else:
            return (False, None)


    # Special function to facilitate multi-key combos from GPT-4V like "io_env.key_hold('w,space')", which are commonly generated
    def _multi_key_action(self, keys, action, duration = 2):

        actions = [self.ACTION_PRESS, self.ACTION_HOLD, self.ACTION_RELEASE]

        if action not in actions:
            logger.warn(f'Invalid action: {action}. Ignoring it.')

        # Act in order, release in reverse
        for key in keys:

            # Special case to facilitate multi-key combos
            if key != keys[-1]:
                action = self.ACTION_HOLD

            if action == self.ACTION_PRESS:
                self.key_press(key)
            elif action == self.ACTION_HOLD:
                self.key_hold(key)

        if duration is None:
            duration = 0.3

        time.sleep(duration)

        for key in reversed(keys):
            self.key_release(key)


    def key_press(self, key, duration=None):

        if type(key) is not list:
            if key in self.ALIASES_MOUSE_REDIRECT:
                self.mouse_click_button(key, duration)

            key = self.map_key(key)

        f, keys = self._check_multi_key(key)
        if f == True:
            self._multi_key_action(keys, self.ACTION_PRESS, duration)
        else:

            if duration is None:
                key_down(key)
                time.sleep(.2)
                key_up(key)
            else:
                key_down(key)
                time.sleep(duration)
                key_up(key)


    def key_hold(self, key, duration=None):

        if key in self.ALIASES_MOUSE_REDIRECT:
            self.mouse_hold_button(key, duration)

        key = self.map_key(key)

        f, keys = self._check_multi_key(key)
        if f == True:
            self._multi_key_action(keys, self.ACTION_HOLD, duration)
        else:

            if duration is not None:
                key_down(key)
                time.sleep(duration)
                key_up(key)
            else:
                self.put_held_keys(key)


    def key_release(self, key):

        if key in self.ALIASES_MOUSE_REDIRECT:
            self.mouse_release_button(key)

        key = self.map_key(key)

        self.pop_held_keys(key)


    def keys_type(self, text):
        if text is not None and text != '':
            type_keys(text)


    def release_held_keys(self):
        for i in range(len(self.held_keys)):
            self.held_keys.pop()


    def release_held_buttons(self):
        for i in range(len(self.held_buttons)):
            self._mouse_button_up(self.held_buttons[i][self.BUTTON_KEY])


    ALIASES_RIGHT_MOUSE = ['right', 'rightbutton', 'rightmousebutton', 'r', 'rbutton', 'rmouse', 'rightmouse', 'rm', 'mouseright', 'mouserightbutton']
    ALIASES_LEFT_MOUSE = ['left', 'leftbutton', 'leftmousebutton', 'l', 'lbutton', 'lmouse', 'leftmouse', 'lm', 'mouseleft', 'mouseleftbutton']
    ALIASES_CENTER_MOUSE = ['middle', 'middelbutton', 'middlemousebutton', 'm', 'mbutton', 'mmouse', 'middlemouse', 'center', 'c', 'centerbutton', 'centermouse', 'cm', 'mousecenter', 'mousecenterbutton']
    ALIASES_MOUSE_REDIRECT = set()  # set(ALIASES_RIGHT_MOUSE + ALIASES_LEFT_MOUSE + ALIASES_CENTER_MOUSE) - set(['r', 'l', 'm', 'c'])

    # @TODO mapping can be improved
    def map_button(self, button):

        if button is None or button == '':
            logger.error('Empty Button.')
            raise Exception(f'Empty mouse button IO: {button}')

        if len(button) > 1:
            button = button.lower().replace('_', '').replace(' ', '')

        if button in self.ALIASES_RIGHT_MOUSE:
            return self.RIGHT_MOUSE_BUTTON
        elif button in self.ALIASES_LEFT_MOUSE:
            return self.LEFT_MOUSE_BUTTON
        elif button in self.ALIASES_CENTER_MOUSE:
            return self.MIDDLE_MOUSE_BUTTON

        return button


    ALIASES_RIGHT_SHIFT_KEY = ['rshift', 'right shift', 'rightshift', 'shift right', 'shiftright']
    ALIASES_LEFT_SHIFT_KEY = ['lshift', 'left shift', 'leftshift', 'shift left', 'shiftleft']
    ALIASES_SHIFT_KEY = ALIASES_RIGHT_SHIFT_KEY + ALIASES_LEFT_SHIFT_KEY

    ALIASES_RIGHT_ALT_KEY = ['ralt', 'right alt', 'rightalt', 'alt right', 'altright']
    ALIASES_LEFT_ALT_KEY = ['lalt', 'left alt', 'leftalt', 'alt left', 'altleft']
    ALIASES_ALT_KEY = ALIASES_RIGHT_ALT_KEY + ALIASES_LEFT_ALT_KEY

    ALIASES_RIGHT_CONTROL_KEY = ['rctrl', 'right ctrl', 'rightctrl', 'ctrl right', 'ctrlright', 'rcontrol', 'right control', 'rightcontrol', 'control right', 'contorlright']
    ALIASES_LEFT_CONTROL_KEY = ['lctrl', 'left ctrl', 'leftctrl', 'ctrl left', 'ctrlleft', 'lcontrol', 'left control', 'leftcontrol', 'control left', 'contorlleft']
    ALIASES_CONTROL_KEY = ALIASES_RIGHT_CONTROL_KEY + ALIASES_LEFT_CONTROL_KEY

    ALIASES_SPACE_KEY = [' ', 'whitespace', 'spacebar', 'space bar']

    # @TODO mapping can be improved
    def map_key(self, key):

        if key is None or key == '':
            logger.error('Empty key.')
            raise Exception(f'Empty key IO: {key}')

        if len(key) > 1:
            key = key.lower().replace('_', '').replace('-', '')
        elif len(key) == 1:
            key = key.lower()

        if key in self.ALIASES_LEFT_SHIFT_KEY:
            return 'shift'
        elif key in self.ALIASES_RIGHT_SHIFT_KEY:
            return 'shift'

        if key in self.ALIASES_LEFT_ALT_KEY:
            return 'alt'
        elif key in self.ALIASES_RIGHT_ALT_KEY:
            return 'alt'

        if key in self.ALIASES_LEFT_CONTROL_KEY:
            return 'ctrl'
        elif key in self.ALIASES_RIGHT_CONTROL_KEY:
            return 'ctrl'

        if key in self.ALIASES_SPACE_KEY:
            return 'space'

        return key

    def get_display_size(self):
        return get_screen_size()


    def get_windows_by_name(self, env_name):
        return get_named_windows(env_name)


    def get_active_window(self):
        return get_active_window()


    def get_windows_by_config(self):
        return get_named_windows_fallback(config.env_name, config.win_name_pattern)


def _theta_calculation(theta):
    """
    Calculates the adjusted theta value based on the configured mouse move factor.

    Parameters:
    - theta: The original theta value to be adjusted.
    """
    return theta * (150 / 9)


def _safe_list_get(list, idx, key = None, default = None):
    try:
        return list[idx][key]
    except IndexError:
        return default
