import os


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
