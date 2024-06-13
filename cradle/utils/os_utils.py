import psutil
import platform

if platform.system() == "Windows":
    import win32gui
    import win32process


def getProcessIDByName(process_name):
    pids = []

    for proc in psutil.process_iter():
        if process_name in proc.name():
            pids.append(proc.pid)

    return pids


def getProcessNameByPID(process_id):
    for proc in psutil.process_iter():
        if process_id == proc.pid:
            return proc.name()

    return None


def getProcessIDByWindowHandle(window_handle):

    pid = None

    if platform.system() == "Windows":
        _, pid = win32process.GetWindowThreadProcessId(window_handle)
    else:
        raise NotImplementedError("This function is only implemented for Windows.")

    return pid


def getParentWindowHandle(window_handle):
    if platform.system() == "Windows":
        # If it returns 0, it meand windows is top level window
        return win32gui.GetParent(window_handle)
    else:
        raise NotImplementedError("This function is only implemented for Windows.")


def getParentWindowHandle(window_handle):
    if platform.system() == "Windows":
        child_handles = []
        win32gui.EnumChildWindows(window_handle, lambda hwnd,p: child_handles.append(hwnd), None)
        return child_handles
    else:
        raise NotImplementedError("This function is only implemented for Windows.")


def getWindowText(window_handle):
    if platform.system() == "Windows":
        return win32gui.GetWindowText(window_handle)
    else:
        raise NotImplementedError("This function is only implemented for Windows.")
