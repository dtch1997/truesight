import pathlib

def get_current_dir(curr_file: str):
    """Get the current directory of the file.

    Example usage: get_current_dir(__file__)
    """
    return pathlib.Path(curr_file).parent
