from PIL import Image
from torchvision import transforms
from io import BytesIO

import cv2
import numpy as np
import random


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if ow > tw or oh > th:
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def get_paired_transforms(configs, params=None):
    global_transform_params = configs["transform_params"]

    transform_list = []
    transform_list.append(
        transforms.Lambda(
            lambda img: __crop(
                img, params["crop_pos"], global_transform_params["crop_size"]
            )
        )
    )

    return transforms.Compose(transform_list)


def get_params(configs, size):
    transform_params_config = configs["transform_params"]
    new_h = new_w = transform_params_config["resize_size"]

    x = random.randint(0, np.maximum(0, new_w - transform_params_config["crop_size"]))
    y = random.randint(0, np.maximum(0, new_h - transform_params_config["crop_size"]))

    flip = random.random() > 0.5

    return {"crop_pos": (x, y), "flip": flip}


def synthesize_ud(img: np.array):
    tasks = list(range(4))
    random.shuffle(tasks)  # inplace operation

    # unstructured degradation
    for task in tasks:
        if task == 0 and random.uniform(0, 1) < 0.5:  # blur generation
            img = gaussian_blur(img)
        if task == 1 and random.uniform(0, 1) < 0.5:  # noise generation
            flag = random.choice([1, 2])
            if flag == 1:
                img = synthesize_gaussian(img, 5, 10)
            if flag == 2:
                img = synthesize_speckle(img, 5, 20)
        if task == 2 and random.uniform(0, 1) < 0.5:  # HR->LR->HR artifact generation
            img = synthesize_low_resolution(img)
        if task == 3 and random.uniform(0, 1) < 0.5:  # jpeg artifact generation
            img = convert_to_jpeg(img, random.randint(40, 100))
    return img


def synthesize_gaussian(img: np.array, std_l: float, std_r: float) -> np.array:
    """
    img: np.array range [0, 1]
    std_l, std_r: float in range [0, 255]
    """
    std = random.uniform(std_l / 255.0, std_r / 255.0)
    noise = np.random.normal(loc=0, scale=std, size=img.shape)
    img = img + noise
    img = np.clip(img, 0.0, 1.0)
    return img


def synthesize_speckle(img: np.array, std_l: float, std_r: float) -> np.array:
    """
    img: np.array range [0, 1]
    std_l, std_r: float in range [0, 255]
    """
    std = random.uniform(std_l / 255.0, std_r / 255.0)
    noise = np.random.normal(loc=0, scale=std, size=img.shape)
    img = img + img * noise
    img = np.clip(img, 0.0, 1.0)
    return img


def synthesize_low_resolution(img: np.array) -> np.array:
    """
    img: np.array range [0, 1] (H, W, C)
    """
    img = Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8))

    w, h = img.size

    new_w = random.randint(int(w / 2), w)
    new_h = random.randint(int(h / 2), h)

    img = img.resize((new_w, new_h), Image.BICUBIC)

    if random.uniform(0, 1) < 0.5:
        img = img.resize((w, h), Image.NEAREST)
    else:
        img = img.resize((w, h), Image.BILINEAR)

    img = np.array(img) / 255.0
    img = np.clip(img, 0.0, 1.0)
    return img


def convert_to_jpeg(img: np.array, quality: int) -> np.array:
    img = Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8))
    with BytesIO() as f:
        img.save(f, format="JPEG", quality=quality)
        f.seek(0)
        img = Image.open(f).convert("RGB")
        return np.clip(np.array(img) / 255.0, 0.0, 1.0)


def gaussian_blur(img: np.array) -> np.array:
    x = np.array(img) * 255  # range from 0 - 255
    kernel_size_candidate = [[3, 3], [5, 5], [7, 7]]
    kernel_size = random.sample(kernel_size_candidate, 1)[0]
    std = random.uniform(1.0, 5.0)

    blur = cv2.GaussianBlur(x, kernel_size, std)
    blur = blur / 255.0
    return blur


def style_invariant_transform(
    style_img_list: list, style_mask_list: list, style_annotation_list: list
):
    assert len(style_img_list) == len(style_mask_list)
    assert len(style_img_list) == len(style_annotation_list)

    k = random.randint(0, 3)
    for i in range(len(style_img_list)):
        style_img = style_img_list[i]
        style_mask = style_mask_list[i]
        style_annotation = style_annotation_list[i]

        style_img = np.rot90(style_img, k).copy()
        style_mask = np.rot90(style_mask, k).copy()
        style_annotation = np.rot90(style_annotation, k).copy()

        style_img_list[i] = style_img
        style_mask_list[i] = style_mask
        style_annotation_list[i] = style_annotation

    # Translation
    for i in range(len(style_mask_list)):
        # preprocess
        style_img = style_img_list[i]
        style_mask = style_mask_list[i]
        style_annotation = style_annotation_list[i]
        style_annotation = np.float32(style_annotation[..., np.newaxis])  # preprocess

        # translate
        style_mask, style_img, style_annotation = translate_img(
            style_mask, style_img, style_annotation
        )

        # post process
        style_annotation = np.int64(np.squeeze(style_annotation, axis=2))  # postprocess

        style_img_list[i] = style_img
        style_mask_list[i] = style_mask
        style_annotation_list[i] = style_annotation

    # Flip
    for i in range(len(style_mask_list)):
        style_img = style_img_list[i]
        style_mask = style_mask_list[i]
        style_annotation = style_annotation_list[i]

        # fliplr
        k = random.randint(0, 1)
        if k == 1:
            style_img = np.fliplr(style_img).copy()
            style_mask = np.fliplr(style_mask).copy()
            style_annotation = np.fliplr(style_annotation).copy()

        # flipud
        k = random.randint(0, 1)
        if k == 1:
            style_img = np.flipud(style_img).copy()
            style_mask = np.flipud(style_mask).copy()
            style_annotation = np.flipud(style_annotation).copy()

        style_img_list[i] = style_img
        style_mask_list[i] = style_mask
        style_annotation_list[i] = style_annotation

    return style_img_list, style_mask_list, style_annotation_list


def translate_img(mask, img, annotation):
    height, width, _ = mask.shape

    # corner computation
    non_zero_idx = np.argwhere(mask)
    if len(non_zero_idx) > 0:
        top = np.min(non_zero_idx[:, 0])  # top
        bottom = np.max(non_zero_idx[:, 0])  # bottom
        left = np.min(non_zero_idx[:, 1])  # left
        right = np.max(non_zero_idx[:, 1])  # right
    else:
        top, bottom, left, right = 0, height - 1, 0, width - 1

    top_shift = abs(0 - top)
    bottom_shift = abs((height - 1) - bottom)
    left_shift = abs(0 - left)
    right_shift = abs((width - 1) - right)

    # shift distance computation
    x_shift_distance = 0
    y_shift_distance = 0
    if left_shift != 0 and right_shift != 0:
        k = random.randint(0, 1)
        if k == 0:  # go left
            x_shift_distance = -random.randint(0, left_shift)
        else:  # go right
            x_shift_distance = random.randint(0, right_shift)
    elif left_shift != 0:
        x_shift_distance = -random.randint(0, left_shift)
    else:
        x_shift_distance = random.randint(0, right_shift)

    if top_shift != 0 and bottom_shift != 0:
        k = random.randint(0, 1)
        if k == 0:  # top shift
            y_shift_distance = -random.randint(0, top_shift)
        else:  # bottom shift
            y_shift_distance = random.randint(0, bottom_shift)
    elif top_shift != 0:
        y_shift_distance = -random.randint(0, top_shift)
    else:
        y_shift_distance = random.randint(0, bottom_shift)
    # translation
    transformation_matrix = np.float32(
        [[1, 0, x_shift_distance], [0, 1, y_shift_distance]]
    )

    mask = cv2.warpAffine(mask, transformation_matrix, (width, height))
    img = cv2.warpAffine(img, transformation_matrix, (width, height))
    annotation = cv2.warpAffine(annotation, transformation_matrix, (width, height))

    # post process
    mask = mask[..., np.newaxis]
    annotation = annotation[..., np.newaxis]
    return mask, img, annotation


def fill_unmasked_region(
    style_img_list: list,
    style_mask_list: list,
    style_annotation_list: list,
    style_filler_list: list,
    style_annotation_filler_list: list,
):
    new_style_img_list = []
    new_style_annotation_list = []
    for i, mask in enumerate(style_mask_list):
        old_img = style_img_list[i]
        old_annotation = style_annotation_list[i]

        filler_img = style_filler_list[i]
        filler_annotation = style_annotation_filler_list[i]

        new_img = old_img * mask + (1 - mask) * filler_img
        new_img = new_img.astype(np.uint8)

        mask = np.squeeze(mask, axis=-1)
        new_annotation = old_annotation * mask + (1 - mask) * filler_annotation
        new_annotation = new_annotation.astype(np.int64)

        new_style_img_list.append(new_img)
        new_style_annotation_list.append(new_annotation)
    return new_style_img_list, new_style_annotation_list
