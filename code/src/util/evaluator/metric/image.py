from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import numpy as np


class ImagePSNR:
    def __init__(self):
        pass

    def __call__(self, output: dict):
        # Must be in HWC channels or NHWC
        image_out = output["image_out"]
        image_GT = output["image_gt"]

        assert image_out.shape == image_GT.shape

        if image_out.ndim > 3:
            batch_result = 0
            batch_size = image_out.shape[0]
            for i in range(batch_size):
                out = image_out[i]
                gt = image_GT[i]
                batch_result += peak_signal_noise_ratio(gt, out)
            return batch_result / batch_size
        else:
            return peak_signal_noise_ratio(image_GT, image_out)

    def __repr__(self):
        return "ImagePSNR"


class ImageSSIM:
    def __init__(self):
        pass

    def __call__(self, output: dict):
        # Must be in HWC or NHWC channels
        image_out = output["image_out"]
        image_GT = output["image_gt"]

        assert type(image_out) == np.ndarray
        assert type(image_GT) == np.ndarray
        assert image_out.shape == image_GT.shape

        if image_out.ndim > 3:
            batch_result = 0
            batch_size = image_out.shape[0]
            for i in range(batch_size):
                out = image_out[i]
                gt = image_GT[i]
                batch_result += structural_similarity(gt, out, channel_axis=-1)
            return batch_result / batch_size
        elif image_out.ndim == 3:
            return structural_similarity(image_GT, image_out, channel_axis=-1)
        elif image_out.ndim == 2:
            return structural_similarity(image_GT, image_out)

    def __repr__(self):
        return "ImageSSIM"
