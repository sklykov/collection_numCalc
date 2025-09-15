# -*- coding: utf-8 -*-
"""
Collect images in a folder and subfolders.

@author: @sklykov

"""
from pathlib import Path
from datetime import datetime


# %% Function def.
def get_images_paths(root_path: Path, image_extensions: list = ["tiff", "tif"]) -> list:
    """
    Get paths of images and their creation time in a list.

    Parameters
    ----------
    root_path : Path
        Root path for searching recursively for images.
    image_extensions : list, optional
        List with image extensions for searching. The default is ["tiff", "tif"].

    Returns
    -------
    list
        List in format [(Path, timestamp), ...] with paths to found images and timestamps with time of their creation.

    """
    images_paths = []
    if isinstance(root_path, Path) and root_path.exists() and root_path.is_dir() and len(image_extensions) > 0:
        image_extensions = [ext.lower() for ext in image_extensions]  # make all extensions low
        # Collect found images in a folder along with their creation time
        for file in root_path.rglob("*"):
            if file.is_file() and (file.suffix.lower().replace(".", "") in image_extensions or file.suffix.lower() in image_extensions):
                images_paths.append((file, file.stat().st_ctime))  # add tuple with image path and raw time of creation
        # Sorting images on a creation time (newest - first)
        if len(images_paths) > 0:
            images_paths = sorted(images_paths, key=lambda image_data: image_data[1], reverse=True)
    return images_paths


# %% Testing as the main script
if __name__ == "__main__":
    root = Path(__file__).parent
    found_images = get_images_paths(root, ["jpeg", "jpg", "png"])
    if len(found_images) > 0:
        for found_image in found_images:
            print(found_image[0].name, ":", datetime.fromtimestamp(found_image[1]))
