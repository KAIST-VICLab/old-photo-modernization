import numpy as np
import torch
import torchvision.utils
from PIL import Image
import os
from collections import OrderedDict
from pathlib import Path
import logging

log = logging.getLogger(__name__)


def save_image(image_numpy: np.array, image_path: os.path, aspect_ratio: int = 1.0):
    """Save a numpy image to the disk
    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
        aspect_ratio -- aspect ratio of the saved image
    """
    image_pil = Image.fromarray(image_numpy.astype(np.uint8))
    w, h = image_pil.size

    if aspect_ratio is None:
        pass
    elif aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    elif aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def tensor2im(input_image: torch.Tensor, imtype: type = np.uint8) -> np.array:
    """Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
                             -- can be torch.Tensor or np.ndarray
                             --  NxCxHxW
        imtype (type)        --  the desired type of the converted numpy array
    """
    image_tensor = input_image
    if isinstance(image_tensor, torch.Tensor):
        if image_tensor.ndim == 4 and image_tensor.size(0) > 1:
            image_numpy = []
            for batch_idx in range(image_tensor.size(0)):
                curr_image = image_tensor[batch_idx]
                curr_image_np = curr_image.detach().cpu().float().numpy()
                if curr_image_np.shape[-1] == 1:
                    curr_image_np = curr_image_np.squeeze(axis=-1)
                curr_image_np = curr_image_np.astype(imtype)[np.newaxis, :]
                image_numpy.append(curr_image_np)
            image_numpy = np.concatenate(image_numpy, axis=0)
            image_numpy = tile_images(image_numpy)
        else:
            image_tensor = image_tensor[0]
            image_numpy = image_tensor.detach().cpu().float().numpy()
            if image_numpy.shape[-1] == 1:  # handling H x W x 1 channel
                image_numpy = image_numpy.squeeze(axis=-1)
            image_numpy = image_numpy.astype(imtype)
    elif isinstance(
        image_tensor, np.ndarray
    ):  # if it is a numpy array, select the first element
        if image_tensor.shape[-1] == 1:
            image_numpy = image_tensor.squeeze(axis=-1)
        image_numpy = image_numpy.astype(imtype)
    else:
        image_numpy = None
        raise NotImplementedError
    return image_numpy


def tile_images(imgs, picturesPerRow=4):
    """Code borrowed from
    https://stackoverflow.com/questions/26521365/cleanly-tile-numpy-array-of-images-stored-in-a-flattened-1d-format/26521997
    """

    # Padding
    if imgs.shape[0] % picturesPerRow == 0:
        rowPadding = 0
    else:
        rowPadding = picturesPerRow - imgs.shape[0] % picturesPerRow
    if rowPadding > 0:
        imgs = np.concatenate(
            [imgs, np.zeros((rowPadding, *imgs.shape[1:]), dtype=imgs.dtype)], axis=0
        )

    # Tiling Loop (The conditionals are not necessary anymore)
    tiled = []
    for i in range(0, imgs.shape[0], picturesPerRow):
        tiled.append(
            np.concatenate([imgs[j] for j in range(i, i + picturesPerRow)], axis=1)
        )

    tiled = np.concatenate(tiled, axis=0)
    return tiled


class ImageWriter:
    def __init__(self, config: OrderedDict):
        self.config = config
        self.root_dir = self.config["root_dir"]
        self.output_visual = self.config["output_visual"]
        self.image_dir_paths = {}
        log.info("Initializing image writer")

    def is_output_visual(self) -> bool:
        return self.output_visual

    def write(self, output: dict, step: str, filename: str):
        if len(self.image_dir_paths) == 0:
            output_names = output.keys()
            for name in output_names:
                dir_path = os.path.join(self.root_dir, step, name)
                self.image_dir_paths[name] = dir_path
                Path(dir_path).mkdir(parents=True, exist_ok=True)

        for name, out in output.items():
            img_name = f"{filename}.png"
            img_path = os.path.join(self.image_dir_paths[name], img_name)
            torchvision.utils.save_image(out, img_path)
            # img_numpy = tensor2im(out)
            # save_image(img_numpy, img_path)

    def write_batch(self, output: dict, step: str, filenames: list):
        if len(self.image_dir_paths) == 0:
            output_names = output.keys()
            for name in output_names:
                dir_path = os.path.join(self.root_dir, step, name)
                self.image_dir_paths[name] = dir_path
                Path(dir_path).mkdir(parents=True, exist_ok=True)

        for name, batch_out in output.items():
            for filename, out in zip(filenames, batch_out):
                img_name = f"{filename}.png"
                img_path = os.path.join(self.image_dir_paths[name], img_name)
                out = out.unsqueeze(0)
                torchvision.utils.save_image(out, img_path)
