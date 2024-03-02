import os
from urllib.parse import urlparse


# TODO: Can be removed
def get_file_extension_from_url(url: str) -> str:
    """
    Extracts the file extension from a given URL.
    """
    path = urlparse(url).path
    ext = os.path.splitext(path)[1]
    return ext
