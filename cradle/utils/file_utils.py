import glob
import os
import shutil

def assemble_project_path(path):
    """Assemble a path relative to the project root directory"""
    if not os.path.isabs(path):
        path = os.path.join(get_project_root(), path)
    return path


def gen_relative_project_path(path):

    root = get_project_root()

    if root not in path:
        raise ValueError('Path to convert should be within the project root.')

    path = path.replace(root, '.').replace('.\\', '')
    return path


def exists_in_project_path(path):
    return os.path.exists(assemble_project_path(path))


def get_project_root():
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.dirname(path) # get to parent, outside of project code path"
    return path


def read_resource_file(path):

    assert "./res/" in path, 'Path should include ./res/'

    with open(assemble_project_path(path), "r", encoding="utf-8") as fd:
        return fd.read()


def get_latest_directories_in_path(path, count = 1):

    # Get a list of all directories in the given path
    directories = glob.glob(os.path.join(path, '*/'))

    # Find the latest directory based on modification time
    if directories:
        #return max(directories, key=os.path.getmtime)
        return sorted(directories, key=os.path.getmtime, reverse=True)[:count]

    else:
        return None


def copy_file(source_file: str, destination_file: str):

    if not os.path.exists(os.path.dirname(destination_file)):
        raise FileNotFoundError(f"Destination directory does not exist: {os.path.dirname(destination_file)}")

    if not os.path.exists(destination_file):
        shutil.copy2(source_file, destination_file)
