import numpy as np
import os
import random
import matplotlib

from ..base import BaseDataset
from collections import OrderedDict
from src.util.visualizer.palette import COCOSTUFF_PALETTE
from src.data.util.modernization import (
    synthesize_ud,
    style_invariant_transform,
    get_params,
    get_paired_transforms,
    fill_unmasked_region,
)
from pathlib import Path
from PIL import Image
from torchvision import transforms

matplotlib.use("TkAgg")


class ETRIMultiReferenceDataset(BaseDataset):
    def __init__(self, dataset_config: OrderedDict):
        super().__init__(dataset_config)

        self.data_mode = self.dataset_config["params"]["data_mode"]
        self.n_styles = self.dataset_config["params"]["n_styles"]

        self.root_content_dir = os.path.join(self.root_dir, self.data_mode, "content")
        self.root_style_dir = os.path.join(self.root_dir, self.data_mode, "style")

        content_img_list = os.listdir(self.root_content_dir)

        self.content_path_list = [
            os.path.join(self.root_content_dir, filename)
            for filename in content_img_list
        ]
        self.content_path_list = sorted(self.content_path_list)

        # Filter based on the min number of styles
        self.style_paths_dict = {}
        deleted_paths = []
        for content_path in self.content_path_list:
            stem = Path(content_path).stem

            style_dir_path = os.path.join(self.root_style_dir, stem)
            style_files = sorted(os.listdir(style_dir_path))
            if len(style_files) < self.n_styles:
                deleted_paths.append(content_path)
            else:
                style_paths = []
                for file in style_files[: self.n_styles]:
                    style_file_path = os.path.join(style_dir_path, file)
                    style_paths.append(style_file_path)
                self.style_paths_dict[stem] = style_paths

        # Delete content path that doesn't contain minimum number of style images
        self.content_path_list = [
            content_path
            for content_path in self.content_path_list
            if content_path not in deleted_paths
        ]

    def __getitem__(self, index):
        content_path = self.content_path_list[index]
        content_name = Path(content_path).stem

        style_paths = self.style_paths_dict[content_name]

        content_img = Image.open(content_path).convert("RGB")
        style_img_list = []
        for style_path in style_paths:
            style_img = Image.open(style_path).convert("RGB")
            style_img_list.append(style_img)
        # Specific transformations to prepare data for model
        to_tensor = transforms.ToTensor()

        content_img = to_tensor(content_img)
        style_img_list = [to_tensor(style_img) for style_img in style_img_list]

        return {
            "content": content_img,
            "style_img_list": style_img_list,
            "img_path": content_path,
        }

    def __len__(self):
        return len(self.content_path_list)


class ETRIMultiReferenceSemanticsDataset(BaseDataset):
    def __init__(self, dataset_config: OrderedDict):
        super().__init__(dataset_config)

        self.n_styles = self.dataset_config["params"]["n_styles"]
        self.data_mode = self.dataset_config["params"]["data_mode"]

        self.root_content_dir = os.path.join(self.root_dir, self.data_mode, "content")
        self.root_style_dir = os.path.join(self.root_dir, self.data_mode, "style")
        self.content_semantic_dir = os.path.join(
            self.root_dir, "{}-{}".format(self.data_mode, "semantic"), "content"
        )
        self.style_semantic_dir = os.path.join(
            self.root_dir, "{}-{}".format(self.data_mode, "semantic"), "style"
        )

        # Content image
        content_img_list = os.listdir(self.root_content_dir)

        self.content_path_list = [
            os.path.join(self.root_content_dir, filename)
            for filename in content_img_list
        ]
        self.content_path_list = sorted(self.content_path_list)

        # Content semantics
        content_semantic_filenames = os.listdir(self.content_semantic_dir)

        self.content_semantic_path_list = [
            os.path.join(self.content_semantic_dir, filename)
            for filename in content_semantic_filenames
        ]
        self.content_semantic_path_list = sorted(self.content_semantic_path_list)

        # Create style paths and style semantic paths, then filter based on the number of min styles
        self.style_paths_dict = {}
        self.style_semantic_paths_dict = {}

        deleted_content_paths = []
        deleted_content_semantic_paths = []
        for content_path, content_semantic_path in zip(
            self.content_path_list, self.content_semantic_path_list
        ):
            stem = Path(content_path).stem

            style_dir_path = os.path.join(self.root_style_dir, stem)
            style_files = sorted(os.listdir(style_dir_path))

            style_semantic_dir_path = os.path.join(self.style_semantic_dir, stem)
            style_semantic_files = sorted(os.listdir(style_semantic_dir_path))
            if len(style_files) < self.n_styles:
                deleted_content_paths.append(content_path)
                deleted_content_semantic_paths.append(content_semantic_path)
            else:
                style_paths = []
                for file in style_files[: self.n_styles]:
                    style_file_path = os.path.join(style_dir_path, file)
                    style_paths.append(style_file_path)
                self.style_paths_dict[stem] = style_paths

                style_semantic_paths = []
                for file in style_semantic_files[: self.n_styles]:
                    style_semantic_path = os.path.join(style_semantic_dir_path, file)
                    style_semantic_paths.append(style_semantic_path)
                self.style_semantic_paths_dict[stem] = style_semantic_paths

        # Delete content path that doesn't contain minimum number of style images
        self.content_path_list = [
            content_path
            for content_path in self.content_path_list
            if content_path not in deleted_content_paths
        ]
        self.content_semantic_path_list = [
            content_semantic_path
            for content_semantic_path in self.content_semantic_path_list
            if content_semantic_path not in deleted_content_semantic_paths
        ]

        # Palette2label to convert palette color into label
        self.palette2label = {}
        for label, palette_value in enumerate(COCOSTUFF_PALETTE):
            palette_value = (palette_value[0], palette_value[1], palette_value[2])
            self.palette2label[palette_value] = label

        # Ignore label
        self.ignore_label = 182
        self.dummy_label = 183

    def __getitem__(self, index):
        content_path = self.content_path_list[index]
        content_semantic_path = self.content_semantic_path_list[index]

        content_name = Path(content_path).stem

        style_paths = self.style_paths_dict[content_name]
        style_semantic_paths = self.style_semantic_paths_dict[content_name]

        content_img = Image.open(content_path).convert("RGB")
        content_semantic = Image.open(content_semantic_path).convert("RGB")
        content_semantic = np.array(content_semantic).astype(np.int64)

        style_img_list = []
        style_semantic_list = []
        for style_path, style_semantic_path in zip(style_paths, style_semantic_paths):
            style_img = Image.open(style_path).convert("RGB")
            style_semantic = Image.open(style_semantic_path).convert("RGB")
            style_semantic = np.array(style_semantic).astype(np.int64)

            style_img_list.append(style_img)
            style_semantic_list.append(style_semantic)

        # Convert semantic to annotation
        content_annotation = np.zeros(content_semantic.shape[:2], dtype=np.int64)
        content_annotation = self.ignore_label  # ignore label
        style_annotation_list = []
        for style_semantic in style_semantic_list:
            style_annotation = np.zeros(style_semantic.shape[:2], dtype=np.int64)
            style_annotation = self.ignore_label
            style_annotation_list.append(style_annotation)

        for palette_value, label in self.palette2label.items():
            palette_value = [palette_value[0], palette_value[1], palette_value[2]]
            content_annotation = np.where(
                np.all(content_semantic == palette_value, axis=2),
                label,
                content_annotation,
            )

            for i, (style_semantic, style_annotation) in enumerate(
                zip(style_semantic_list, style_annotation_list)
            ):
                style_annotation = np.where(
                    np.all(style_semantic == palette_value, axis=2),
                    label,
                    style_annotation,
                )
                style_annotation_list[i] = style_annotation

        # Specific transformations to prepare data for model
        to_tensor = transforms.ToTensor()

        content_img = to_tensor(content_img)
        style_img_list = [to_tensor(style_img) for style_img in style_img_list]
        label_set = np.unique(content_annotation)

        # Mask
        content_mask_list = []
        for style_annotation in style_annotation_list:
            style_annotation[
                np.logical_not(np.isin(style_annotation, label_set))
            ] = self.dummy_label
            style_annotation_list[i] = style_annotation

            style_label_set = np.unique(style_annotation)
            content_mask = content_annotation.copy()
            content_mask[
                np.logical_not(np.isin(content_annotation, style_label_set))
            ] = self.dummy_label
            content_mask = np.where(content_mask != self.dummy_label, 1.0, 0.0)
            content_mask = content_mask.astype(np.float32)
            content_mask_list.append(content_mask)
        content_mask_list = [
            to_tensor(content_mask) for content_mask in content_mask_list
        ]

        return {
            "A": content_img,
            "content_annotation": content_annotation,
            "style_img_list": style_img_list,
            "style_annotation_list": style_annotation_list,
            "img_path": content_path,
            "label_set": label_set,
            "content_mask_list": content_mask_list,
        }

    def __len__(self):
        return len(self.content_path_list)


class COCOStuffMultiStyleV2Dataset(BaseDataset):
    def __init__(self, dataset_config: OrderedDict):
        super().__init__(dataset_config)
        self.dataset_params = self.dataset_config["params"]

        self.data_dir = self.root_dir
        self.annotation_dir = self.dataset_params["annotation_dir"]

        img_list_paths = os.listdir(self.data_dir)
        annotation_list_paths = os.listdir(self.annotation_dir)

        img_list_paths = sorted(img_list_paths)
        annotation_list_paths = sorted(annotation_list_paths)

        self.img_list_paths = [
            os.path.join(self.data_dir, path) for path in img_list_paths
        ]
        self.annotation_list_paths = [
            os.path.join(self.annotation_dir, path) for path in annotation_list_paths
        ]

        self.index_list = list(range(len(self.img_list_paths)))

        self.label_count = self.dataset_params["label_count"]
        self.n_style_images = self.dataset_params["n_styles"]
        self.ignore_label = self.dataset_params["ignore_label"]  # unlabeled
        self.dummy_label = self.dataset_params["dummy_label"]  # additional processing

        self.color_jitter = transforms.ColorJitter(
            **self.dataset_params["color_jitter"]
        )  # Follow SimCLR parameter

    def __getitem__(self, index):
        content_img_path = self.img_list_paths[index]
        content_annotation_path = self.annotation_list_paths[index]

        # GT
        GT = Image.open(content_img_path).convert("RGB")

        # Initial content image
        content_img = Image.open(content_img_path).convert("RGB")
        content_annotation = Image.open(content_annotation_path)

        # Style Filling
        style_img_filler_list = []
        style_annotation_filler_list = []
        sampled_idxs = random.sample(
            self.index_list[:index] + self.index_list[index + 1 :], self.n_style_images
        )
        for style_idx in sampled_idxs:
            style_img_path = self.img_list_paths[style_idx]
            style_annotation_path = self.annotation_list_paths[style_idx]

            style_img = Image.open(style_img_path).convert("RGB")
            style_annotation = Image.open(style_annotation_path)

            style_img_filler_list.append(style_img)
            style_annotation_filler_list.append(style_annotation)

        # augmentation: resize and crop
        dummy_transforms = {"transform_params": {"resize_size": 286, "crop_size": 256}}
        transform_params = get_params(dummy_transforms, content_img.size)
        crop_transform = get_paired_transforms(dummy_transforms, transform_params)
        image_rs_transform = transforms.Resize(
            dummy_transforms["transform_params"]["resize_size"],
            interpolation=transforms.InterpolationMode.BILINEAR,
        )
        annotation_rs_transform = transforms.Resize(
            dummy_transforms["transform_params"]["resize_size"],
            interpolation=transforms.InterpolationMode.NEAREST,
        )

        GT = crop_transform(image_rs_transform(GT))
        content_img = crop_transform(image_rs_transform(content_img))
        content_annotation = crop_transform(annotation_rs_transform(content_annotation))

        style_img_filler_list = [
            crop_transform(image_rs_transform(style_img))
            for style_img in style_img_filler_list
        ]
        style_annotation_filler_list = [
            crop_transform(annotation_rs_transform(style_annotation))
            for style_annotation in style_annotation_filler_list
        ]

        # Preprocess Annotation
        content_annotation = np.asarray(content_annotation).astype(np.int64)
        content_annotation[
            content_annotation == 255
        ] = self.ignore_label  # TODO: should be dummy label
        for i in range(len(style_annotation_filler_list)):
            style_annotation = style_annotation_filler_list[i]
            style_annotation = np.asarray(style_annotation).astype(np.int64)
            style_annotation[
                style_annotation == 255
            ] = self.ignore_label  # TODO: should be dummy label
            style_annotation_filler_list[i] = style_annotation
        label_set = np.unique(content_annotation)

        # [Pass Test] Style variant transformation for each content label
        for label in label_set:
            masked_cont_annotation = content_annotation.copy()
            masked_cont_annotation[
                masked_cont_annotation != label
            ] = self.dummy_label  # we ignore this
            mask = np.where(masked_cont_annotation != self.dummy_label, 1.0, 0.0)[
                ..., np.newaxis
            ]

            # masking and style distortion (for now only color jitter)
            masked_content_img = np.array(content_img) / 255.0
            masked_content_img = mask * masked_content_img
            masked_content_img = np.clip(masked_content_img * 255, 0, 255).astype(
                np.uint8
            )
            masked_content_img = Image.fromarray(masked_content_img)
            masked_content_img = self.color_jitter(masked_content_img)
            masked_content_img = np.array(masked_content_img) / 255.0

            # numpy manipulation for degradation
            masked_content_img = synthesize_ud(masked_content_img)

            content_img = np.array(content_img) / 255.0
            content_img = (content_img * (1 - mask)) + (masked_content_img * mask)
            content_img = np.clip(content_img * 255, 0, 255).astype(np.uint8)
            content_img = Image.fromarray(content_img)

        # [Passed Test] - Generate style by using style invariant transformation
        # (split: spatial or class label split)
        style_img_list = []
        style_mask_list = []
        style_annotation_list = []

        # img_width = 256
        img_height = 256
        random_choice = 1
        if random_choice < 0.5:
            # Spatial Split
            # [Passed Test] Horizontal Split
            step_size = img_height // self.n_style_images
            start_indices = [step_size * i for i in range(self.n_style_images)]
            end_indices = [start_index + step_size for start_index in start_indices]
            end_indices[-1] = max(end_indices[-1], img_height)

            for start_index, end_index in zip(start_indices, end_indices):
                style_img = np.array(GT) / 255.0
                style_annotation = content_annotation.copy()
                style_annotation[start_index:end_index, :] = self.dummy_label

                # masking
                style_mask = np.where(style_annotation != self.dummy_label, 1.0, 0.0)[
                    ..., np.newaxis
                ]

                style_img = style_img * style_mask
                style_img = np.clip(style_img * 255, 0, 255).astype(np.uint8)

                style_img_list.append(style_img)
                style_mask_list.append(style_mask.astype(np.float32))
                style_annotation_list.append(style_annotation)
        else:
            # [Passed Test] Class Split
            k, m = divmod(len(label_set), self.n_style_images)
            splitted_label_set = [
                label_set[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]
                for i in range(self.n_style_images)
            ]

            for style_label_set in splitted_label_set:
                style_img = np.array(GT) / 255.0
                style_annotation = content_annotation.copy()
                style_annotation[
                    np.logical_not(np.isin(content_annotation, style_label_set))
                ] = self.dummy_label  # we ignore this

                # masking
                style_mask = np.where(style_annotation != self.dummy_label, 1.0, 0.0)[
                    ..., np.newaxis
                ]

                style_img = style_img * style_mask
                style_img = np.clip(style_img * 255, 0, 255).astype(np.uint8)

                style_img_list.append(style_img)
                style_mask_list.append(style_mask.astype(np.float32))
                style_annotation_list.append(style_annotation)

            # content mask list
            content_mask_list = []
            for style_label_set in splitted_label_set:
                masked_cont_annotation = content_annotation.copy()
                masked_cont_annotation[
                    np.logical_not(np.isin(content_annotation, style_label_set))
                ] = self.dummy_label  # we ignore this
                mask = np.where(masked_cont_annotation != self.dummy_label, 1.0, 0.0)[
                    ..., np.newaxis
                ]

                content_mask_list.append(mask.astype(np.float32))
        # Important: Don't use data from the output,
        # instead use the data from the original one for the testing
        # Style 1 and style 2 is outputted from the network
        (
            style_img_list,
            style_mask_list,
            style_annotation_list,
        ) = style_invariant_transform(
            style_img_list, style_mask_list, style_annotation_list
        )

        # [Passed Test] - fill unknown region with filler image, remove filler first
        style_img_filler_list = [
            np.array(style_img).astype(np.uint8) for style_img in style_img_filler_list
        ]
        style_img_list, style_annotation_list = fill_unmasked_region(
            style_img_list,
            style_mask_list,
            style_annotation_list,
            style_img_filler_list,
            style_annotation_filler_list,
        )

        # identity image
        k = random.randint(0, 3)
        identity_content_img = content_img.copy()
        identity_content_img = np.rot90(identity_content_img, k).copy()
        identity_content_annotation = content_annotation.copy()
        identity_content_annotation = np.rot90(identity_content_annotation, k).copy()

        # DEBUGGING: rotated GT
        k = random.randint(0, 3)
        rotated_GT = GT.copy()
        rotated_GT = np.rot90(rotated_GT, k).copy()
        rotated_GT_annotation = content_annotation.copy()
        rotated_GT_annotation = np.rot90(rotated_GT_annotation, k)

        # transformations specific for img and annotations
        to_tensor = transforms.ToTensor()
        GT = to_tensor(GT)

        # DEBUGGING
        rotated_GT = to_tensor(rotated_GT)
        rotated_GT_annotation = np.asarray(rotated_GT_annotation).astype(np.int64)

        content_img = to_tensor(content_img)
        content_annotation = np.asarray(content_annotation).astype(np.int64)

        style_img_list = [to_tensor(style_img) for style_img in style_img_list]
        style_annotation_list = [
            np.asarray(style_annotation).astype(np.int64)
            for style_annotation in style_annotation_list
        ]

        style_mask_list = [to_tensor(style_mask) for style_mask in style_mask_list]
        content_mask_list = [
            to_tensor(content_mask) for content_mask in content_mask_list
        ]

        identity_content_img = to_tensor(identity_content_img)
        identity_content_annotation = np.asarray(identity_content_annotation).astype(
            np.int64
        )

        return {
            "content": content_img,
            "content_annotation": content_annotation,
            "style_img_list": style_img_list,
            "style_annotation_list": style_annotation_list,
            "GT": GT,
            "img_path": content_img_path,
            "label_set": label_set,
            "content_mask_list": content_mask_list,
            "style_mask_list": style_mask_list,
            "identity_content_img": identity_content_img,
            "identity_content_annotation": identity_content_annotation,
            "rotated_GT": rotated_GT,
            "rotated_GT_annotation": rotated_GT_annotation,
        }

    def __len__(self):
        return len(self.img_list_paths)


if __name__ == "__main__":
    data_configs = {
        "transforms": None,
        "data_root": "external/dataset/ETRI_MultiRef512x512",
        "data_dir": "external/dataset/ETRI_MultiRef512x512",
        "n_styles": 2,
    }
    dataset = ETRIMultiReferenceSemanticsDataset(data_configs, {})
    data_iterator = iter(dataset)
    data = next(data_iterator)

    # DEBUG
    # END DEBUG
